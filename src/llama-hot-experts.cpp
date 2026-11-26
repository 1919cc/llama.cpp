#include "llama-hot-experts.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// the experts of one physical layer: the layers that share these tensors share the bank
struct hot_group {
    std::vector<ggml_tensor *> ws;        // host expert tensors [.., .., n_expert]
    ggml_backend_dev_t         dev = nullptr;
    size_t                     expert_bytes = 0;
    std::vector<int64_t>       count;     // [n_expert] selections in the profile
    std::vector<int32_t>       hot;       // the hot experts, in bank slot order
    std::vector<ggml_tensor *> banks;     // [ws.size()] on dev
    // F32 [1, n_expert] tables on the device (its GET_ROWS reads F32), so that the lookup runs in the split of the router
    ggml_tensor *              hot_map  = nullptr; // slot or -1
    ggml_tensor *              cold_map = nullptr; // -1 or the expert

    // adaptation to the conversation: decayed selections per expert, counted by a CPU op in the graph
    std::vector<float>         score;     // [n_expert]
    std::vector<uint32_t>      recent;    // [n_expert] selections since the last rebalance
    std::vector<int32_t>       slot_of;   // [n_expert] bank slot, -1 if cold
};

class hot_experts {
public:
    static hot_experts & get() {
        static hot_experts h;
        return h;
    }

    // the group of layer il, built with the banks on the first call for an allocated model; nullptr if none
    hot_group * group(const llama_model & model, int il) {
        if (budget == 0 || model.no_alloc()) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mtx);
        if (!built) {
            built = true;
            build(model);
        }
        const ggml_tensor * key = layer_key(model, il);
        auto it = by_key.find(key);
        return it == by_key.end() || it->second->hot.empty() ? nullptr : it->second;
    }

    ggml_tensor * bank(const ggml_tensor * w) const {
        auto it = by_tensor.find(w);
        return it == by_tensor.end() ? nullptr : it->second;
    }

    // between two graph computes: every `period` calls, swap the hot experts that the conversation no longer uses
    // for the cold ones it uses most, up to max_swaps experts (each is copied and repacked on the device)
    void maintain() {
        if (!built || groups.empty() || ++n_calls % period != 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx);
        int n_swaps = 0;
        for (auto & g : groups) {
            if (g->hot.empty()) {
                continue;
            }
            const int32_t n_expert = (int32_t) g->score.size();
            for (int32_t e = 0; e < n_expert; ++e) {
                g->score[e]  = 0.5f*g->score[e] + (float) g->recent[e];
                g->recent[e] = 0;
            }
            // the strongest cold experts against the weakest hot ones
            std::vector<int32_t> cold;
            for (int32_t e = 0; e < n_expert; ++e) {
                if (g->slot_of[e] < 0 && g->score[e] > 0.0f) {
                    cold.push_back(e);
                }
            }
            std::sort(cold.begin(), cold.end(), [&](int32_t a, int32_t b) { return g->score[a] > g->score[b]; });
            std::vector<int32_t> hot = g->hot;
            std::sort(hot.begin(), hot.end(), [&](int32_t a, int32_t b) { return g->score[a] < g->score[b]; });
            for (size_t i = 0; i < cold.size() && i < hot.size() && n_swaps < max_swaps; ++i) {
                const int32_t e_in = cold[i];
                const int32_t e_out = hot[i];
                // hysteresis: an expert moves only when clearly busier, so that the bank does not churn
                if (g->score[e_in] < 1.5f*g->score[e_out] + 1.0f) {
                    break;
                }
                swap(*g, e_out, e_in);
                n_swaps++;
            }
        }
        n_swapped += n_swaps;
    }

    // CPU op: counts the experts selected in generation (a prompt selects most experts of a layer)
    static void count_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);
        const ggml_tensor * ids = dst->src[0];
        if (ith != 0 || ids->ne[1] > 4) {
            return;
        }
        hot_group * g = (hot_group *) userdata;
        for (int64_t t = 0; t < ids->ne[1]; ++t) {
            const int32_t * row = (const int32_t *) ((const char *) ids->data + t*ids->nb[1]);
            for (int64_t i = 0; i < ids->ne[0]; ++i) {
                if (row[i] >= 0 && row[i] < (int32_t) g->recent.size()) {
                    g->recent[row[i]]++;
                    get().n_sel++;
                    get().n_hit += g->slot_of[row[i]] >= 0;
                }
            }
        }
    }

private:
    hot_experts() {
        const char * mb   = getenv("LLAMA_HOT_EXPERTS_MB");
        const char * prof = getenv("LLAMA_HOT_EXPERTS_PROFILE");
        if (mb && atol(mb) > 0) {
            if (!prof) {
                LLAMA_LOG_WARN("%s: LLAMA_HOT_EXPERTS_MB needs LLAMA_HOT_EXPERTS_PROFILE (a LLAMA_EXPERT_TRACE file)\n", __func__);
                return;
            }
            budget  = (size_t) atol(mb) << 20;
            profile = prof;
            if (const char * p = getenv("LLAMA_HOT_EXPERTS_PERIOD")) {
                period = atoi(p);
            }
            if (const char * p = getenv("LLAMA_HOT_EXPERTS_SWAPS")) {
                max_swaps = atoi(p);
            }
            if (period <= 0) {
                period = INT32_MAX; // static bank
            }
            atexit([] {
                const auto & h = hot_experts::get();
                if (h.built && !h.groups.empty()) {
                    // the logger may be gone at exit
                    fprintf(stderr, "hot experts: %.1f%% of %lld generation selections hot, %lld experts swapped in %lld decode calls\n",
                            h.n_sel ? 100.0*h.n_hit/h.n_sel : 0.0, (long long) h.n_sel, (long long) h.n_swapped, (long long) h.n_calls);
                }
            });
        }
    }

    static std::vector<ggml_tensor *> layer_experts(const llama_layer & l) {
        std::vector<ggml_tensor *> ws;
        for (ggml_tensor * w : { l.ffn_gate_up_exps, l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps }) {
            if (w) {
                ws.push_back(w);
            }
        }
        return ws;
    }

    static bool is_host_weight(const ggml_tensor * w) {
        return w->buffer && ggml_backend_buffer_is_host(w->buffer) && ggml_n_dims(w) == 3;
    }

    static const ggml_tensor * layer_key(const llama_model & model, int il) {
        if (il < 0 || il >= (int) model.layers.size()) {
            return nullptr;
        }
        const auto ws = layer_experts(model.layers[il]);
        return ws.empty() ? nullptr : ws.back();
    }

    void build(const llama_model & model) {
        // one group per physical set of expert tensors, on the device of the first layer that uses it
        for (int il = 0; il < (int) model.layers.size(); ++il) {
            const auto ws = layer_experts(model.layers[il]);
            if (ws.empty() || !std::all_of(ws.begin(), ws.end(), is_host_weight) || by_key.count(ws.back())) {
                continue;
            }
            ggml_backend_dev_t dev = model.dev_layer(il);
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            auto g = std::make_unique<hot_group>();
            g->ws  = ws;
            g->dev = dev;
            for (const ggml_tensor * w : ws) {
                g->expert_bytes += w->nb[2];
            }
            g->count.assign(ws[0]->ne[2], 0);
            by_key[ws.back()] = g.get();
            groups.push_back(std::move(g));
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no MoE layer with host experts on an accelerator, hot experts disabled\n", __func__);
            return;
        }

        // the decode selections of the profile: [layer, n_tokens, n_used, ids[n_tokens * n_used]] int16 records
        FILE * f = fopen(profile.c_str(), "rb");
        if (!f) {
            LLAMA_LOG_WARN("%s: cannot open the profile %s, hot experts disabled\n", __func__, profile.c_str());
            return;
        }
        int64_t n_sel = 0;
        n_prof_tokens = 0;
        int16_t hdr[3];
        std::vector<int16_t> ids;
        while (fread(hdr, sizeof(int16_t), 3, f) == 3) {
            const size_t n = (size_t) std::max<int>(hdr[1], 0) * std::max<int>(hdr[2], 0);
            ids.resize(n);
            if (fread(ids.data(), sizeof(int16_t), n, f) != n) {
                break;
            }
            // prompts select most experts of a layer; the hot set is for generation
            if (hdr[1] > 4) {
                continue;
            }
            if (hdr[0] == 0) {
                n_prof_tokens += hdr[1];
            }
            auto it = by_key.find(layer_key(model, hdr[0]));
            if (it == by_key.end()) {
                continue;
            }
            for (int16_t e : ids) {
                if (e >= 0 && e < (int16_t) it->second->count.size()) {
                    it->second->count[e]++;
                    n_sel++;
                }
            }
        }
        fclose(f);

        // the most selected experts of all layers, until the budget is used
        struct cand { hot_group * g; int32_t e; int64_t n; };
        std::vector<cand> cands;
        for (auto & g : groups) {
            for (int32_t e = 0; e < (int32_t) g->count.size(); ++e) {
                if (g->count[e] > 0) {
                    cands.push_back({ g.get(), e, g->count[e] });
                }
            }
        }
        std::stable_sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.n > b.n; });
        size_t used = 0;
        int64_t n_hot_sel = 0;
        for (const cand & c : cands) {
            if (used + c.g->expert_bytes > budget) {
                continue;
            }
            used += c.g->expert_bytes;
            c.g->hot.push_back(c.e);
            n_hot_sel += c.n;
        }

        // the banks, one buffer per device, and the maps in host memory
        std::map<ggml_backend_dev_t, std::vector<hot_group *>> by_dev;
        size_t n_tensors = 0;
        for (auto & g : groups) {
            if (!g->hot.empty()) {
                by_dev[g->dev].push_back(g.get());
                n_tensors += g->ws.size() + 2;
            }
        }
        for (auto & [dev, gs] : by_dev) {
            ggml_init_params ip = { (n_tensors + 1)*ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx_dev  = ggml_init(ip);
            ggml_context * ctx_host = ggml_init(ip);
            ctxs.emplace_back(ctx_dev);
            ctxs.emplace_back(ctx_host);
            for (hot_group * g : gs) {
                for (const ggml_tensor * w : g->ws) {
                    ggml_tensor * b = ggml_new_tensor_3d(ctx_dev, w->type, w->ne[0], w->ne[1], (int64_t) g->hot.size());
                    ggml_format_name(b, "%s.hot", w->name);
                    g->banks.push_back(b);
                }
                g->hot_map  = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, 1, (int64_t) g->count.size());
                g->cold_map = ggml_new_tensor_2d(ctx_host, GGML_TYPE_F32, 1, (int64_t) g->count.size());
            }
            ggml_backend_buffer_t buf_dev  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, ggml_backend_dev_buffer_type(dev));
            // the maps are small and not weights: a plain device buffer, which the device reads without DMA
            ggml_backend_buffer_t buf_host = ggml_backend_alloc_ctx_tensors_from_buft(ctx_host, ggml_backend_dev_buffer_type(dev));
            if (!buf_dev || !buf_host) {
                LLAMA_LOG_WARN("%s: could not allocate %.1f MiB of hot experts on %s, hot experts disabled\n", __func__,
                        used/1048576.0, ggml_backend_dev_name(dev));
                for (hot_group * g : gs) {
                    g->hot.clear();
                }
                continue;
            }
            // as weights: the ops that read them run on the device, and the device may repack them
            ggml_backend_buffer_set_usage(buf_dev,  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            bufs.emplace_back(buf_dev);
            bufs.emplace_back(buf_host);

            for (hot_group * g : gs) {
                std::vector<float> hmap(g->count.size(), -1.0f);
                std::vector<float> cmap(g->count.size());
                for (size_t e = 0; e < cmap.size(); ++e) {
                    cmap[e] = (float) e;
                }
                for (size_t s = 0; s < g->hot.size(); ++s) {
                    hmap[g->hot[s]] = (float) s;
                    cmap[g->hot[s]] = -1.0f;
                }
                // the profile is the starting point of the adaptation, in selections per rebalance period
                g->recent.assign(g->count.size(), 0);
                g->slot_of.assign(g->count.size(), -1);
                g->score.resize(g->count.size());
                for (size_t e = 0; e < g->count.size(); ++e) {
                    g->score[e] = n_prof_tokens > 0 ? (float) g->count[e]*(float) std::min(period, 1024)/(float) n_prof_tokens : 0.0f;
                }
                for (size_t s = 0; s < g->hot.size(); ++s) {
                    g->slot_of[g->hot[s]] = (int32_t) s;
                }
                ggml_backend_tensor_set(g->hot_map,  hmap.data(), 0, hmap.size()*sizeof(float));
                ggml_backend_tensor_set(g->cold_map, cmap.data(), 0, cmap.size()*sizeof(float));
                for (size_t k = 0; k < g->ws.size(); ++k) {
                    const ggml_tensor * w = g->ws[k];
                    for (size_t s = 0; s < g->hot.size(); ++s) {
                        ggml_backend_tensor_set(g->banks[k], (const char *) w->data + (size_t) g->hot[s]*w->nb[2],
                                s*g->banks[k]->nb[2], w->nb[2]);
                    }
                    by_tensor[w] = g->banks[k];
                }
            }
        }

        size_t n_hot = 0;
        for (auto & g : groups) {
            n_hot += g->hot.size();
        }
        LLAMA_LOG_INFO("%s: %zu hot experts of %zu layers in %.1f MiB on the device: %.1f%% of the %lld profiled selections\n",
                __func__, n_hot, groups.size(), used/1048576.0, n_sel ? 100.0*n_hot_sel/n_sel : 0.0, (long long) n_sel);
    }

    // expert e_in of g takes the bank slot of e_out
    void swap(hot_group & g, int32_t e_out, int32_t e_in) {
        const int32_t s = g.slot_of[e_out];
        for (size_t k = 0; k < g.ws.size(); ++k) {
            const ggml_tensor * w = g.ws[k];
            ggml_backend_tensor_set(g.banks[k], (const char *) w->data + (size_t) e_in*w->nb[2], (size_t) s*g.banks[k]->nb[2], w->nb[2]);
        }
        const float slot = (float) s;
        const float none = -1.0f;
        const float out  = (float) e_out;
        ggml_backend_tensor_set(g.hot_map,  &none, (size_t) e_out*sizeof(float), sizeof(float));
        ggml_backend_tensor_set(g.hot_map,  &slot, (size_t) e_in *sizeof(float), sizeof(float));
        ggml_backend_tensor_set(g.cold_map, &out,  (size_t) e_out*sizeof(float), sizeof(float));
        ggml_backend_tensor_set(g.cold_map, &none, (size_t) e_in *sizeof(float), sizeof(float));
        g.hot[s]        = e_in;
        g.slot_of[e_in] = s;
        g.slot_of[e_out] = -1;
    }

    std::mutex  mtx;
    int         period    = 0;  // tokens between rebalances (LLAMA_HOT_EXPERTS_PERIOD, 0 = static: adapting measured no gain on the PLR110)
    int         max_swaps = 16; // experts moved per rebalance (LLAMA_HOT_EXPERTS_SWAPS)
    int64_t     n_calls   = 0;
    int64_t     n_swapped = 0;
    int64_t     n_sel     = 0; // generation selections counted, and the hot ones among them
    int64_t     n_hit     = 0;
    size_t      budget = 0;
    std::string profile;
    bool        built  = false;
    int64_t     n_prof_tokens = 0;

    std::vector<std::unique_ptr<hot_group>> groups;
    std::unordered_map<const ggml_tensor *, hot_group *>    by_key;    // the last expert tensor of a layer
    std::unordered_map<const ggml_tensor *, ggml_tensor *>  by_tensor; // host expert tensor -> bank
    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

} // namespace

llama_hot_experts_ids llama_hot_experts_split(ggml_context * ctx, ggml_cgraph * gf, const llama_model & model, int il, ggml_tensor * selected) {
    hot_group * g = hot_experts::get().group(model, il);
    if (!g) {
        return {};
    }
    // [n_used, n_tokens] -> rows of the [1, n_expert] maps
    ggml_tensor * sel = ggml_is_contiguous(selected) ? selected : ggml_cont(ctx, selected);
    sel = ggml_reshape_1d(ctx, sel, ggml_nelements(sel));
    llama_hot_experts_ids res;
    res.hot  = ggml_reshape_2d(ctx, ggml_cast(ctx, ggml_get_rows(ctx, g->hot_map,  sel), GGML_TYPE_I32), selected->ne[0], selected->ne[1]);
    res.cold = ggml_reshape_2d(ctx, ggml_cast(ctx, ggml_get_rows(ctx, g->cold_map, sel), GGML_TYPE_I32), selected->ne[0], selected->ne[1]);
    // the lookups stay in the device split of the router, the counting op starts the CPU split that follows
    ggml_build_forward_expand(gf, res.hot);
    ggml_build_forward_expand(gf, res.cold);
    if (sel->ne[0] <= 4*selected->ne[0]) {
        ggml_tensor * args[1] = { selected };
        ggml_build_forward_expand(gf, ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args, 1, hot_experts::count_op, 1, g));
    }
    return res;
}

void llama_hot_experts_maintain() {
    hot_experts::get().maintain();
}

ggml_tensor * llama_hot_experts_bank(const ggml_tensor * w) {
    return hot_experts::get().bank(w);
}
