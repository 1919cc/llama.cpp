#include "llama-expert-cache.h"

#include "llama-impl.h"
#include "llama-weight-stream.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#define LLAMA_EC_SUPPORTED 1
#endif

namespace {

#ifdef LLAMA_EC_SUPPORTED

constexpr int    EC_MAX_LAYERS  = 512;
constexpr int    EC_MAX_EXPERTS = 1024;
constexpr size_t EC_SLOT        = 2u << 20; // one expert of a layer (all its tensors); 125B experts are 1.56-1.77 MB

// the expert-address table of one expert tensor, attached as its extra
struct ec_table {
    ggml_cpu_expert_remap                remap;
    std::unique_ptr<const char * volatile[]> ptr;
    int il;
    size_t part_off; // offset of this tensor's slice inside a slot
};

class expert_cache {
public:
    static expert_cache * get() {
        static expert_cache * c = [] () -> expert_cache * {
            const char * s = getenv("LLAMA_EXPERT_CACHE_MB");
            const long mb = s ? atol(s) : 0;
            if (mb <= 0) {
                return nullptr;
            }
            auto * c = new expert_cache((size_t) mb << 20);
            if (!c->arena) {
                delete c;
                return nullptr;
            }
            return c;
        }();
        return c;
    }

    bool has(int il, int e) const {
        return il >= 0 && il < EC_MAX_LAYERS && e >= 0 && e < EC_MAX_EXPERTS && slot_of[il * EC_MAX_EXPERTS + e] >= 0;
    }

    bool use(int il, const int32_t * ids, size_t n, const ggml_tensor * const * ws, int n_ws) {
        if (il < 0 || il >= EC_MAX_LAYERS || n_ws <= 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<ec_table *> tabs;
        size_t slot_bytes = 0;
        for (int k = 0; k < n_ws; ++k) {
            ec_table * t = table_of(ws[k], il, slot_bytes);
            if (!t) {
                return false;
            }
            tabs.push_back(t);
            slot_bytes += ws[k]->nb[2];
        }
        if (slot_bytes > EC_SLOT || n > (size_t) n_slots) {
            return false;
        }

        // hits first: they become the most recently used, so the evictions below never take this layer's experts
        std::vector<int32_t> misses;
        for (size_t i = 0; i < n; ++i) {
            const int e = ids[i];
            const int s = slot_of[il * EC_MAX_EXPERTS + e];
            if (s >= 0 && tabs[0]->ptr[e]) {
                lru.splice(lru.end(), lru, pos[s]);
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        std::vector<llama_read_req> reads;
        std::vector<std::pair<int32_t, int>> placed; // expert, slot
        for (int32_t e : misses) {
            // a slot with this key but no table entry belongs to tensors of another model: reuse it
            int s = slot_of[il * EC_MAX_EXPERTS + e];
            if (s < 0 && !free_slots.empty()) {
                s = free_slots.back();
                free_slots.pop_back();
                pos[s] = lru.insert(lru.end(), s);
            } else {
                if (s < 0) {
                    s = lru.front();
                    n_evict++;
                }
                lru.splice(lru.end(), lru, pos[s]);
                // the old expert is read from the mmap again from now on
                const int64_t old = key_of[s];
                slot_of[old] = -1;
                for (ec_table * t : owner_tabs[s]) {
                    t->ptr[old % EC_MAX_EXPERTS] = nullptr;
                }
            }
            key_of[s] = (int64_t) il * EC_MAX_EXPERTS + e;
            slot_of[key_of[s]] = s;
            owner_tabs[s] = tabs;
            char * base = arena + (size_t) s * EC_SLOT;
            for (int k = 0; k < n_ws; ++k) {
                reads.push_back({ (const char *) ws[k]->data + (size_t) e * ws[k]->nb[2], base + tabs[k]->part_off, (size_t) ws[k]->nb[2] });
            }
            placed.emplace_back(e, s);
            n_miss++;
        }
        if (!llama_read_into(reads)) {
            // not file-backed: undo, the experts stay where they are
            for (auto [e, s] : placed) {
                slot_of[(int64_t) il * EC_MAX_EXPERTS + e] = -1;
                lru.erase(pos[s]);
                free_slots.push_back(s);
            }
            return false;
        }
        for (auto [e, s] : placed) {
            for (ec_table * t : tabs) {
                t->ptr[e] = arena + (size_t) s * EC_SLOT + t->part_off;
            }
        }
        return true;
    }

private:
    explicit expert_cache(size_t budget) {
        n_slots = (int) (budget / EC_SLOT);
        void * p = n_slots > 0 ? mmap(nullptr, (size_t) n_slots * EC_SLOT, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) : MAP_FAILED;
        if (p == MAP_FAILED) {
            LLAMA_LOG_WARN("%s: could not allocate the expert cache\n", __func__);
            return;
        }
        arena = (char *) p;
        slot_of.assign((size_t) EC_MAX_LAYERS * EC_MAX_EXPERTS, -1);
        key_of.assign(n_slots, -1);
        pos.resize(n_slots);
        owner_tabs.resize(n_slots);
        for (int s = n_slots - 1; s >= 0; --s) {
            free_slots.push_back(s);
        }
        LLAMA_LOG_INFO("%s: expert cache: %d slots of %zu MiB (%zu MiB)\n", __func__, n_slots, EC_SLOT >> 20, ((size_t) n_slots * EC_SLOT) >> 20);
        atexit([] {
            if (auto * c = expert_cache::get()) {
                const size_t tot = c->n_hit + c->n_miss;
                LLAMA_LOG_INFO("expert cache: hits %zu / %zu (%.1f%%), evictions %zu\n", c->n_hit, tot, tot ? 100.0 * c->n_hit / tot : 0.0, c->n_evict);
            }
        });
    }

    // the table of tensor w, created on first use and attached as w->extra (plain CPU buffers keep extra unused)
    ec_table * table_of(const ggml_tensor * w, int il, size_t part_off) {
        auto it = tables.find(w);
        if (it != tables.end()) {
            if (w->extra == &it->second->remap) {
                return it->second.get();
            }
            // the address was reused by a new tensor; slots can still point to the old table, so keep it
            retired.push_back(std::move(it->second));
            tables.erase(it);
        }
        if (w->extra != nullptr || w->ne[2] > EC_MAX_EXPERTS) {
            return nullptr;
        }
        auto t = std::make_unique<ec_table>();
        t->ptr.reset(new const char * volatile[w->ne[2]]);
        for (int64_t e = 0; e < w->ne[2]; ++e) {
            t->ptr[e] = nullptr;
        }
        t->remap.magic    = GGML_CPU_EXPERT_REMAP_MAGIC;
        t->remap.n_expert = (int32_t) w->ne[2];
        t->remap.ptr      = t->ptr.get();
        t->il       = il;
        t->part_off = part_off;
        const_cast<ggml_tensor *>(w)->extra = &t->remap;
        ec_table * r = t.get();
        tables.emplace(w, std::move(t));
        return r;
    }

    std::mutex mtx;
    char *     arena   = nullptr;
    int        n_slots = 0;

    std::vector<int32_t>                     slot_of;    // [layer * EC_MAX_EXPERTS + expert] -> slot, -1 if not cached
    std::vector<int64_t>                     key_of;     // [slot] -> layer * EC_MAX_EXPERTS + expert
    std::list<int>                           lru;        // slots, least recently used first
    std::vector<std::list<int>::iterator>    pos;        // [slot] -> position in lru
    std::vector<std::vector<ec_table *>>     owner_tabs; // [slot] -> the tables pointing into it
    std::vector<int>                         free_slots;
    std::unordered_map<const ggml_tensor *, std::unique_ptr<ec_table>> tables;
    std::vector<std::unique_ptr<ec_table>>   retired;

    size_t n_hit = 0, n_miss = 0, n_evict = 0;
};

#endif // LLAMA_EC_SUPPORTED

} // namespace

bool llama_expert_cache_enabled() {
#ifdef LLAMA_EC_SUPPORTED
    return expert_cache::get() != nullptr;
#else
    return false;
#endif
}

bool llama_expert_cache_has(int il, int e) {
#ifdef LLAMA_EC_SUPPORTED
    const auto * c = expert_cache::get();
    return c && c->has(il, e);
#else
    GGML_UNUSED(il);
    GGML_UNUSED(e);
    return false;
#endif
}

bool llama_expert_cache_use(int il, const int32_t * ids, size_t n, const ggml_tensor * const * ws, int n_ws) {
#ifdef LLAMA_EC_SUPPORTED
    auto * c = expert_cache::get();
    return c && c->use(il, ids, n, ws, n_ws);
#else
    GGML_UNUSED(il);
    GGML_UNUSED(ids);
    GGML_UNUSED(n);
    GGML_UNUSED(ws);
    GGML_UNUSED(n_ws);
    return false;
#endif
}
