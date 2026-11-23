// hui_infer Android engine: llama.cpp (Hexagon NPU backend, O_DIRECT weight streaming, MoE expert
// paging, block-sparse FFN, MTP / n-gram speculative decoding) behind a small JNI surface, the
// Kotlin object com.hui.infer.HuiEngine (java/com/hui/infer/HuiEngine.kt next to this file).
//
// Merged from the two app-side copies that existed before: EdgeAgent's engine.cpp (prompt cache,
// chat template render / parse for tool calling, llama-bench style runs, GGUF model info) and
// super_ai_com's edge_jni.cpp (suffix trimming for context reuse, stop during prefill, NPU polling
// per load, DEBUG log filtering). Apps only build llama.cpp command lines and consume this library.
//
// One engine per process. A deployment scheme is expressed as llama.cpp command line arguments,
// parsed by common_params_parse, so every scheme goes through the same option handling as
// llama-cli. Engine knobs read from the environment (LLAMA_WEIGHT_STREAM, LLAMA_EXPERT_PREFETCH,
// LLAMA_FFN_SPARSE_BLOCKS, LLAMA_LAYER_PREFETCH, GGML_HEXAGON_*) are set by the app before load().

#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#define TAG "HuiEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// JNI names of the Kotlin binding
#define HUI_JNI(name) Java_com_hui_infer_HuiEngine_##name

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct stats {
    double prefill_ms  = 0;
    int    prefill_tok = 0;
    double decode_ms   = 0;
    int    decode_tok  = 0;
    double ttft_ms     = 0;
    int    drafted     = 0;
    int    accepted    = 0;
    int    passes      = 0; // target forward passes during decode
    int    reused_tok  = 0; // prompt tokens served from the context (not evaluated again)
    int    cache_tok   = 0; // of which restored from the prompt cache file
    double cache_ms    = 0; // time to restore (or to evaluate and save) the cached prefix
};

struct engine {
    std::mutex mtx;

    common_params                      params;
    common_init_result_ptr             init;
    common_speculative_init_result_ptr spec_init;
    llama_model *                      model    = nullptr;
    llama_context *                    ctx      = nullptr;
    llama_context *                    ctx_dft  = nullptr;
    common_speculative *               spec     = nullptr;
    bool                               use_spec = false;

    llama_tokens ctx_tokens;           // the tokens whose state is in the target context
    bool         needs_reset = false;  // the context holds positions ctx_tokens does not describe

    // what llama_memory_seq_rm can do on each context, probed once at load (the probe clears it)
    common_context_seq_rm_type seq_rm_tgt = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type seq_rm_dft = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // prompt cache: the evaluated state of a fixed prompt prefix (the system turn), saved to
    // cache_path and restored into a cold context instead of evaluating the prefix again
    std::string  cache_path;
    llama_tokens cache_tokens;

    // the model's own chat template (jinja, from the GGUF); chat = the parser state of the last
    // prompt rendered by chatRender, for chatParse
    common_chat_templates_ptr            tmpls;
    std::unique_ptr<common_chat_session> chat;
    bool                                 chatml = false;

    stats       last;
    std::string model_desc;
    double      load_ms = 0;

    std::atomic<bool> stop{false};

    void unload() {
        if (spec) {
            common_speculative_free(spec);
        }
        spec = nullptr;
        spec_init.reset();
        init.reset();
        model   = nullptr;
        ctx     = nullptr;
        ctx_dft = nullptr;
        ctx_tokens.clear();
        cache_path.clear();
        cache_tokens.clear();
        chat.reset();
        tmpls.reset();
        chatml      = false;
        use_spec    = false;
        needs_reset = false;
        seq_rm_tgt  = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        seq_rm_dft  = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    // The drafters keep what they saw (ngram-mod pools every n-gram of earlier replies): restart
    // them so they draft only from what is still in the context, which generate_spec hands them.
    // Without this, asking again had Gemma 4 12B draft 64 tokens of the dropped reply, 7 accepted.
    void restart_spec() {
        if (!spec) {
            return;
        }
        common_speculative_free(spec);
        spec = common_speculative_init(params.speculative, 1);
    }

    // Drops positions [n, end) from the contexts, when they can drop a suffix and the sliding
    // window of the next token is still cached; false leaves them for reset_context().
    bool trim(size_t n) {
        const bool part = seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_PART &&
                          (!ctx_dft || seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_PART);
        if (!part || n == 0) {
            return false;
        }
        const auto     p   = static_cast<llama_pos>(n);
        llama_memory_t mem = llama_get_memory(ctx);
        if (!llama_memory_seq_rm(mem, 0, p, -1)) {
            return false;
        }
        if (ctx_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, p, -1)) {
            return false;
        }
        // token p attends to the positions above p - n_swa: they must not have been evicted
        // (-1: the window held only removed positions)
        const int       n_swa   = llama_model_n_swa(model);
        const llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
        if (n_swa > 0 && (pos_min < 0 || pos_min > std::max(0, p - n_swa + 1))) {
            return false;
        }
        ctx_tokens.resize(n);
        needs_reset = false;
        restart_spec();
        return true;
    }

    void reset_context() {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (ctx_dft) {
            llama_memory_clear(llama_get_memory(ctx_dft), true);
        }
        restart_spec();
        ctx_tokens.clear();
        needs_reset = false;
    }
};

engine g_engine;

// llama.cpp's DEBUG messages are dropped: llama_graph_n_input_tensors writes one per input tensor
// use every time a graph is built, thousands per token, on the decode thread. With them Gemma 4
// 12B decoded at 0.14-0.8 tok/s in an app; without, 4.8. A CONT line belongs to the message before.
void log_cb(ggml_log_level level, const char * text, void *) {
    static thread_local bool dropping = false;
    if (level == GGML_LOG_LEVEL_DEBUG || (level == GGML_LOG_LEVEL_CONT && dropping)) {
        dropping = true;
        return;
    }
    dropping       = false;
    const int prio = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
                     level == GGML_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN  :
                                                     ANDROID_LOG_INFO;
    __android_log_print(prio, "llama", "%s", text);
}

std::string jstr(JNIEnv * env, jstring s) {
    if (!s) {
        return {};
    }
    const char * c = env->GetStringUTFChars(s, nullptr);
    std::string  r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}

jstring jstring_or_null(JNIEnv * env, const std::string & s) {
    return env->NewStringUTF(s.c_str());
}

// Longest prefix of buf that ends on a UTF-8 character boundary.
size_t utf8_complete_prefix(const std::string & buf) {
    const size_t n = buf.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const unsigned char c = buf[n - back];
        if ((c & 0xC0) == 0x80) {
            continue; // continuation byte
        }
        const size_t len = (c & 0x80) == 0    ? 1 :
                           (c & 0xE0) == 0xC0 ? 2 :
                           (c & 0xF0) == 0xE0 ? 3 :
                           (c & 0xF8) == 0xF0 ? 4 :
                                                1;
        return back >= len ? n : n - back;
    }
    return n;
}

// Streams whole UTF-8 characters to the Kotlin listener's onBytes(byte[]): Boolean.
struct emitter {
    JNIEnv *    env;
    jobject     cb;
    jmethodID   on_bytes;
    std::string pending;

    bool emit(const std::string & piece) {
        pending += piece;
        const size_t n = utf8_complete_prefix(pending);
        if (n == 0) {
            return true;
        }
        jbyteArray arr = env->NewByteArray(static_cast<jsize>(n));
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(n), reinterpret_cast<const jbyte *>(pending.data()));
        const jboolean go = env->CallBooleanMethod(cb, on_bytes, arr);
        env->DeleteLocalRef(arr);
        pending.erase(0, n);
        return go && !env->ExceptionCheck();
    }
};

// Evaluates inp[from, to) into the target (and the speculative state), n_batch at a time.
// Stops between batches when asked to (a long prompt prefills for minutes on a large model): what
// was evaluated stays cached and listed. A failure leaves the contexts holding positions
// ctx_tokens does not list (earlier ubatches of the batch, or the target when only the drafter
// failed): the next prompt trims them or starts over instead of writing past them.
bool eval_tokens(engine & e, const llama_tokens & inp, size_t from, size_t to) {
    const size_t n_batch = llama_n_batch(e.ctx);
    common_batch batch(e.ctx);
    for (size_t i = from; i < to; i += n_batch) {
        if (e.stop) {
            LOGI("prefill stopped at token %zu of %zu", i, to);
            return false;
        }
        const size_t end = std::min(to, i + n_batch);
        batch.clear();
        for (size_t j = i; j < end; ++j) {
            batch.add(inp[j], static_cast<llama_pos>(j), 0, j + 1 == to);
        }
        if (llama_process(e.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOGE("prefill failed at token %zu", i);
            e.needs_reset = true;
            return false;
        }
        if (e.spec && !common_speculative_process(e.spec, batch)) {
            LOGE("speculative prefill failed");
            e.needs_reset = true;
            return false;
        }
        e.ctx_tokens.insert(e.ctx_tokens.end(), inp.begin() + i, inp.begin() + end);
    }
    return true;
}

// Restores the cached prefix state into a cold context; false (context cleared) if the file is
// missing or stale.
bool cache_restore(engine & e) {
    llama_tokens got(e.cache_tokens.size() + 1);
    size_t       n = 0;
    if (llama_state_seq_load_file(e.ctx, e.cache_path.c_str(), 0, got.data(), got.size(), &n) == 0 ||
        n != e.cache_tokens.size() || !std::equal(e.cache_tokens.begin(), e.cache_tokens.end(), got.begin())) {
        e.reset_context();
        return false;
    }
    if (e.ctx_dft) {
        const std::string dft = e.cache_path + ".dft";
        if (llama_state_seq_load_file(e.ctx_dft, dft.c_str(), 0, got.data(), got.size(), &n) == 0 ||
            n != e.cache_tokens.size()) {
            e.reset_context();
            return false;
        }
    }
    e.ctx_tokens = e.cache_tokens;
    return true;
}

// Evaluates the prefix into a cold context and saves its state for the next cold start.
bool cache_build(engine & e) {
    if (!eval_tokens(e, e.cache_tokens, 0, e.cache_tokens.size())) {
        return false;
    }
    const std::string tmp = e.cache_path + ".tmp";
    bool ok = llama_state_seq_save_file(e.ctx, tmp.c_str(), 0, e.cache_tokens.data(), e.cache_tokens.size()) > 0;
    if (ok && e.ctx_dft) {
        const std::string dft = e.cache_path + ".dft";
        ok = llama_state_seq_save_file(e.ctx_dft, dft.c_str(), 0, e.cache_tokens.data(), e.cache_tokens.size()) > 0;
    }
    // the target file goes in place last: its presence means both are complete
    if (ok && rename(tmp.c_str(), e.cache_path.c_str()) == 0) {
        LOGI("prompt cache: saved %zu tokens to %s", e.cache_tokens.size(), e.cache_path.c_str());
    } else {
        remove(tmp.c_str());
        LOGE("prompt cache: could not save %s", e.cache_path.c_str());
    }
    return true;
}

// Brings the context to inp[0, n_keep), reusing the longest prefix a previous reply already
// evaluated, at most max_reuse tokens of it. Apps re-render the last reply from its text, which
// need not tokenize as it was generated (and drops its thinking), so the cached tokens often stop
// matching a few tokens before their end: contexts that can drop a suffix keep the common part;
// hybrid (recurrent) ones reuse an exact prefix only. A cold context whose prompt starts with the
// prompt cache prefix restores it from flash (or evaluates it and saves it once).
bool prepare_prompt(engine & e, const llama_tokens & inp, size_t n_keep, size_t max_reuse) {
    const size_t lim = std::min({ e.ctx_tokens.size(), n_keep, max_reuse });
    size_t       n   = 0;
    while (n < lim && e.ctx_tokens[n] == inp[n]) {
        n++;
    }
    const size_t cached  = e.ctx_tokens.size();
    const bool   exact   = !e.needs_reset && n == cached;
    const bool   trimmed = !exact && e.trim(n);
    if (!exact && !trimmed) {
        e.reset_context();
    }
    LOGI("prompt of %zu tokens: %zu of %zu cached tokens reused (%s)", inp.size(), e.ctx_tokens.size(), cached,
         exact ? "prefix" : trimmed ? "trimmed" : "reset");

    size_t built = 0;
    if (e.ctx_tokens.empty() && !e.cache_path.empty() && !e.cache_tokens.empty() &&
        e.cache_tokens.size() <= std::min(n_keep, max_reuse) &&
        std::equal(e.cache_tokens.begin(), e.cache_tokens.end(), inp.begin())) {
        const auto tc = clk::now();
        if (cache_restore(e)) {
            e.last.cache_tok = static_cast<int>(e.cache_tokens.size());
            LOGI("prompt cache: restored %zu tokens in %.0f ms", e.cache_tokens.size(), ms_since(tc));
        } else {
            if (!cache_build(e)) {
                return false;
            }
            built = e.cache_tokens.size(); // evaluated now: counts as prefill, not as reuse
        }
        e.last.cache_ms = ms_since(tc);
    }
    e.last.reused_tok = static_cast<int>(e.ctx_tokens.size() - built);
    return eval_tokens(e, inp, e.ctx_tokens.size(), n_keep);
}

int generate_plain(engine & e, const llama_tokens & inp, int n_predict, emitter & out, clk::time_point t0) {
    const llama_vocab * vocab = llama_model_get_vocab(e.model);
    common_sampler_ptr  smpl(common_sampler_init(e.model, e.params.sampling));

    const auto tp = clk::now();
    // the last prompt token is evaluated again: its logits give the first reply token
    if (!prepare_prompt(e, inp, inp.size(), inp.size() - 1)) {
        return e.stop ? 0 : -1;
    }
    e.last.prefill_ms  = ms_since(tp);
    e.last.prefill_tok = static_cast<int>(inp.size()) - e.last.reused_tok;

    const auto   td = clk::now();
    common_batch batch(e.ctx);
    int          n_gen = 0;
    while (n_gen < n_predict && !e.stop) {
        const llama_token id = common_sampler_sample(smpl.get(), e.ctx, -1);
        common_sampler_accept(smpl.get(), id, true);
        if (n_gen == 0) {
            e.last.ttft_ms = ms_since(t0);
        }
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        n_gen++;
        if (!out.emit(common_token_to_piece(e.ctx, id))) {
            break;
        }
        batch.clear();
        batch.add(id, static_cast<llama_pos>(e.ctx_tokens.size()), 0, true);
        if (llama_process(e.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOGE("decode failed");
            e.needs_reset = true;
            break;
        }
        e.ctx_tokens.push_back(id);
        e.last.passes++;
    }
    e.last.decode_ms  = ms_since(td);
    e.last.decode_tok = n_gen;
    return n_gen;
}

// Speculative decoding (MTP, n-gram): the loop of examples/speculative-simple on a persistent context.
int generate_spec(engine & e, const llama_tokens & inp, int n_predict, emitter & out, clk::time_point t0) {
    const llama_vocab * vocab = llama_model_get_vocab(e.model);
    common_sampler_ptr  smpl(common_sampler_init(e.model, e.params.sampling));
    const llama_seq_id  seq      = 0;
    const bool          ckpt_tgt = e.seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool          ckpt_dft = e.ctx_dft && e.seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    const auto tp = clk::now();
    if (!prepare_prompt(e, inp, inp.size() - 1, inp.size() - 1)) {
        return e.stop ? 0 : -1;
    }
    e.last.prefill_ms  = ms_since(tp);
    e.last.prefill_tok = static_cast<int>(inp.size()) - 1 - e.last.reused_tok;

    llama_tokens & prompt_tgt = e.ctx_tokens;
    llama_token    id_last    = inp.back();
    int            n_past     = static_cast<int>(prompt_tgt.size());
    common_speculative_begin(e.spec, seq, prompt_tgt);

    const auto               td = clk::now();
    common_batch             batch_tgt(e.ctx);
    llama_tokens             draft;
    common_prompt_checkpoint ckpt;
    int                      n_gen = 0;
    bool                     done  = false;
    while (!done && !e.stop) {
        if (draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(), llama_memory_seq_pos_min(llama_get_memory(e.ctx), seq),
                            llama_memory_seq_pos_max(llama_get_memory(e.ctx), seq));
            if (ckpt_dft) {
                ckpt.update_dft(e.ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            const int room  = static_cast<int>(llama_n_ctx(e.ctx)) - n_past - 2;
            const int n_max = std::max(0, std::min(room, n_predict - n_gen - 1));
            common_speculative_get_draft_params(e.spec, seq) = {
                /* .drafting = */ true,
                /* .n_max    = */ n_max,
                /* .pos0     = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tgt,
                /* .result   = */ &draft,
            };
            common_speculative_draft(e.spec);
            if (!draft.empty() && ckpt_tgt) {
                ckpt.update_tgt(e.ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (e.ctx_dft) {
                if (ckpt_dft) {
                    ckpt.load_dft(e.ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                llama_memory_seq_rm(llama_get_memory(e.ctx_dft), seq, ckpt.pos_max + 1, -1);
            }
        }

        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], n_past + static_cast<int>(i), seq, true);
        }
        if (llama_process(e.ctx, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0 ||
            !common_speculative_process(e.spec, batch_tgt)) {
            LOGE("speculative decode failed");
            e.needs_reset = true;
            break;
        }
        e.last.passes++;

        common_sampler_ptr smpl_save;
        if (ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }
        const size_t n_draft = draft.size();
        auto         ids     = common_sampler_sample_and_accept_n(smpl.get(), e.ctx, draft);

        if (ckpt_tgt && ids.size() - 1 < n_draft) {
            // partial acceptance on a context that cannot drop a suffix: restore, replay the accepted part
            draft = std::move(ids);
            ckpt.load_tgt(e.ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(e.ctx), seq, ckpt.pos_max + 1, -1);
            if (e.ctx_dft) {
                ckpt.load_dft(e.ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(e.ctx_dft), seq, ckpt.pos_max + 1, -1);
            }
            prompt_tgt.resize(ckpt.n_tokens);
            smpl   = std::move(smpl_save);
            n_past = static_cast<int>(prompt_tgt.size());
            e.last.drafted += static_cast<int>(n_draft);
            continue;
        }

        common_speculative_accept(e.spec, seq, ids.size() - 1);
        n_past += static_cast<int>(ids.size()) - 1;
        e.last.drafted += static_cast<int>(n_draft);
        e.last.accepted += static_cast<int>(ids.size()) - 1;
        for (const llama_token id : ids) {
            prompt_tgt.push_back(id_last);
            id_last = id;
            if (n_gen == 0) {
                e.last.ttft_ms = ms_since(t0);
            }
            if (llama_vocab_is_eog(vocab, id_last) || n_gen >= n_predict) {
                done = true;
                break;
            }
            n_gen++;
            if (!out.emit(common_token_to_piece(e.ctx, id_last))) {
                done = true;
                break;
            }
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(e.ctx), seq, n_past, -1);
        if (e.ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(e.ctx_dft), seq, n_past, -1);
        }
    }
    // a run that ended inside an accepted draft leaves positions prompt_tgt does not list
    if (prompt_tgt.size() != static_cast<size_t>(n_past)) {
        e.needs_reset = true;
    }
    e.last.decode_ms  = ms_since(td);
    e.last.decode_tok = n_gen;
    return n_gen;
}

// Shared by generate / generateChat: checks the fit, then runs the plain or speculative loop.
int run_generation(JNIEnv * env, engine & e, const llama_tokens & inp, int n_predict, jobject cb, clk::time_point t0) {
    if (inp.size() < 2) {
        return 0;
    }
    if (inp.size() + n_predict + 8 > llama_n_ctx(e.ctx)) {
        LOGE("prompt of %zu tokens does not fit the context of %u", inp.size(), llama_n_ctx(e.ctx));
        return -2;
    }
    jmethodID on_bytes = env->GetMethodID(env->GetObjectClass(cb), "onBytes", "([B)Z");
    emitter   out      = { env, cb, on_bytes, {} };
    return e.use_spec ? generate_spec(e, inp, n_predict, out, t0) : generate_plain(e, inp, n_predict, out, t0);
}

// Renders OpenAI-style messages (a JSON array) with the model's own chat template, ending with an
// open assistant turn, and tokenizes it part by part: special tokens are parsed in the template's
// text only, never in what the user wrote.
llama_tokens render_messages(engine & e, const std::string & messages, bool thinking) {
    common_chat_templates_inputs in;
    in.messages                     = common_chat_msgs_parse_oaicompat(common_json::parse(messages));
    in.enable_thinking              = thinking;
    in.add_generation_prompt        = true;
    in.use_jinja                    = true;
    const common_chat_params chat   = common_chat_templates_apply(e.tmpls.get(), in);
    llama_tokens             out;
    for (const auto & part : chat.prompt.parts) {
        const llama_tokens t = common_tokenize(e.ctx, part.text, out.empty(), !part.is_input);
        out.insert(out.end(), t.begin(), t.end());
    }
    return out;
}

// NPU bytes of a weight after repack (bits per weight of its Hexagon tile); 0 = stays on the CPU.
double tile_bpw(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return 4.5;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
            return 5.0;
        case GGML_TYPE_Q8_0:
            return 8.5;
        case GGML_TYPE_MXFP4:
            return 4.25;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return 4.0;
        case GGML_TYPE_Q5_K:
            return 6.0;
        case GGML_TYPE_Q6_K:
            return 7.0;
        case GGML_TYPE_F16:
            return 16.0;
        case GGML_TYPE_F32:
            return 32.0;
        default:
            return 0.0;
    }
}

// Every file of a (possibly split) model, from the path of its first split.
std::vector<std::string> model_split_paths(const std::string & p0) {
    std::vector<std::string> paths = { p0 };
    int                      idx   = 0;
    int                      count = 0;
    const size_t             pos   = p0.rfind("-00001-of-");
    if (pos != std::string::npos && sscanf(p0.c_str() + pos, "-%05d-of-%05d.gguf", &idx, &count) == 2 && count > 1) {
        const std::string base = p0.substr(0, pos);
        for (int k = 2; k <= count; ++k) {
            char split[PATH_MAX];
            llama_split_path(split, sizeof(split), base.c_str(), k - 1, count); // split_no is 0-based
            paths.emplace_back(split);
        }
    }
    return paths;
}

int64_t gguf_int(gguf_context * g, const std::string & key) {
    const int64_t k = gguf_find_key(g, key.c_str());
    if (k < 0) {
        return 0;
    }
    switch (gguf_get_kv_type(g, k)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, k);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, k);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(g, k));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(g, k);
        default:               return 0;
    }
}

std::string gguf_str(gguf_context * g, const std::string & key) {
    const int64_t k = gguf_find_key(g, key.c_str());
    return k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING ? gguf_get_val_str(g, k) : "";
}

// The Hexagon backend reads GGML_HEXAGON_OPPOLL once, at registration; its opt_oppoll global is
// read on every wait, so the per-model value from the environment is applied on each load.
// Polling (spin instead of sleep while the NPU works) measured 57 -> 74 tok/s on MiniCPM-V 1B;
// paging schemes keep sleeping, a spinning core competes with their I/O threads.
void apply_npu_poll() {
    // already loaded as a dependency of libggml; NOLOAD only returns its handle
    void * hexagon = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_NOLOAD);
    auto * poll    = static_cast<int *>(dlsym(hexagon ? hexagon : RTLD_DEFAULT, "opt_oppoll"));
    if (hexagon) {
        dlclose(hexagon);
    }
    if (!poll) {
        LOGI("npu polling unavailable");
        return;
    }
    const char * v = getenv("GGML_HEXAGON_OPPOLL");
    *poll          = v ? atoi(v) : 0;
    LOGI("npu polling %d", *poll);
}

size_t npu_bytes() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("HTP");
    if (!reg) {
        return 0;
    }
    auto fn = reinterpret_cast<size_t (*)()>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_rpcmem_bytes"));
    return fn ? fn() : 0;
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL HUI_JNI(init)(JNIEnv *, jclass) {
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    LOGI("backend initialized, %zu devices", ggml_backend_dev_count());
}

// Loads the model the llama.cpp arguments (without argv[0]) describe; null on success, otherwise
// what went wrong.
JNIEXPORT jstring JNICALL HUI_JNI(load)(JNIEnv * env, jclass, jobjectArray jargs) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    engine &                    e = g_engine;
    e.unload();
    apply_npu_poll();

    std::vector<std::string> args = { "hui-engine" };
    for (jsize i = 0; i < env->GetArrayLength(jargs); ++i) {
        auto * a = static_cast<jstring>(env->GetObjectArrayElement(jargs, i));
        args.push_back(jstr(env, a));
        env->DeleteLocalRef(a);
    }
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }

    e.params = common_params();
    if (!common_params_parse(static_cast<int>(argv.size()), argv.data(), e.params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return env->NewStringUTF("invalid engine arguments");
    }
    const auto limits = common_speculative_get_output_limits(e.params.n_batch, e.params.n_parallel,
                                                             common_speculative_n_max(&e.params.speculative));
    e.params.n_outputs_max         = limits.total;
    e.params.n_outputs_max_per_seq = limits.per_seq;

    const auto t0 = clk::now();
    e.init        = common_init_from_params(e.params);
    e.model       = e.init ? e.init->model() : nullptr;
    e.ctx         = e.init ? e.init->context() : nullptr;
    if (!e.model || !e.ctx) {
        e.unload();
        return env->NewStringUTF("failed to load the model (see logcat tag 'llama')");
    }

    const auto & types = e.params.speculative.types;
    e.use_spec = std::any_of(types.begin(), types.end(), [](auto t) { return t != COMMON_SPECULATIVE_TYPE_NONE; });
    if (e.use_spec) {
        common_params params_dft           = common_base_params_to_speculative(e.params);
        e.spec_init                        = common_speculative_init_from_params(params_dft, e.model, e.ctx);
        e.params.speculative.draft.ctx_tgt = e.ctx;
        e.params.speculative.draft.ctx_dft = e.spec_init->context();
        e.ctx_dft                          = e.params.speculative.draft.ctx_dft;
        e.spec                             = common_speculative_init(e.params.speculative, 1);
        if (!e.spec) {
            LOGE("speculative decoding unavailable, continuing without it");
            e.use_spec = false;
        }
    }
    e.seq_rm_tgt = common_context_can_seq_rm(e.ctx);
    if (e.ctx_dft) {
        e.seq_rm_dft = common_context_can_seq_rm(e.ctx_dft);
    }
    e.load_ms = ms_since(t0);

    e.tmpls  = common_chat_templates_init(e.model, "");
    e.chatml = e.tmpls && common_chat_templates_source(e.tmpls.get()).find("<|im_start|>") != std::string::npos;

    char desc[256];
    llama_model_desc(e.model, desc, sizeof(desc));
    e.model_desc = desc;
    LOGI("loaded %s in %.0f ms, spec %d, seq_rm %d/%d", desc, e.load_ms, static_cast<int>(e.use_spec),
         static_cast<int>(e.seq_rm_tgt), static_cast<int>(e.seq_rm_dft));
    return nullptr;
}

JNIEXPORT void JNICALL HUI_JNI(unload)(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    g_engine.unload();
}

// Stops the running prefill or reply at its next batch / token. Not serialized on the engine.
JNIEXPORT void JNICALL HUI_JNI(stop)(JNIEnv *, jclass) {
    g_engine.stop = true;
}

JNIEXPORT void JNICALL HUI_JNI(resetContext)(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    if (g_engine.ctx) {
        g_engine.reset_context();
    }
}

// Continues a fully rendered prompt (the app renders it, e.g. ChatML kept byte-exact for context
// reuse). Returns the number of tokens (0 when stopped during the prefill), -1 when the engine
// failed, -2 when the prompt does not fit the context.
JNIEXPORT jint JNICALL HUI_JNI(generate)(JNIEnv * env, jclass, jstring jprompt, jint n_predict, jobject cb) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    engine &                    e = g_engine;
    if (!e.ctx) {
        return -1;
    }
    e.stop        = false;
    e.last        = stats();
    const auto t0 = clk::now();
    // a cold context gets BOS; a prompt that extends the context is tokenized as its continuation
    const llama_tokens inp = common_tokenize(e.ctx, jstr(env, jprompt), true, true);
    return run_generation(env, e, inp, n_predict, cb, t0);
}

// Answers OpenAI-style messages (JSON array) rendered with the model's own chat template; same
// return values as generate.
JNIEXPORT jint JNICALL HUI_JNI(generateChat)(JNIEnv * env, jclass, jstring jmessages, jboolean thinking, jint n_predict,
                                            jobject cb) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    engine &                    e = g_engine;
    if (!e.ctx || !e.tmpls) {
        return -1;
    }
    e.stop        = false;
    e.last        = stats();
    const auto   t0 = clk::now();
    llama_tokens inp;
    try {
        inp = render_messages(e, jstr(env, jmessages), thinking);
    } catch (const std::exception & ex) {
        LOGE("chat render: %s", ex.what());
        return -1;
    }
    return run_generation(env, e, inp, n_predict, cb, t0);
}

// {"chatml": the template is ChatML, "thinking": it supports enable_thinking}
JNIEXPORT jstring JNICALL HUI_JNI(chatInfo)(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    const engine &              e = g_engine;
    common_json j = {
        { "chatml",   e.chatml },
        { "thinking", e.tmpls && common_chat_templates_support_enable_thinking(e.tmpls.get()) },
    };
    return jstring_or_null(env, j.dump());
}

// Renders OpenAI-style messages and tools (JSON arrays) with the model's chat template, opening the
// assistant turn (addGeneration) or not (a conversation prefix); null on error. The parser for the
// reply is kept for chatParse.
JNIEXPORT jstring JNICALL HUI_JNI(chatRender)(JNIEnv * env, jclass, jstring jmessages, jstring jtools, jboolean thinking,
                                             jboolean add_generation) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    engine &                    e = g_engine;
    if (!e.tmpls) {
        return nullptr;
    }
    try {
        common_chat_templates_inputs in;
        in.messages              = common_chat_msgs_parse_oaicompat(common_json::parse(jstr(env, jmessages)));
        const std::string tools  = jstr(env, jtools);
        in.tools                 = tools.empty() ? std::vector<common_chat_tool>() :
                                                   common_chat_tools_parse_oaicompat(common_json::parse(tools));
        in.enable_thinking       = thinking;
        in.add_generation_prompt = add_generation;
        in.use_jinja             = true;
        // like llama-server: the reply's thinking (Qwen <think>, Gemma 4 <|channel>thought...<channel|>) goes to
        // reasoning_content, not into the content the user sees
        in.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        e.chat = std::make_unique<common_chat_session>(e.tmpls.get(), llama_model_get_vocab(e.model), in);
        return jstring_or_null(env, e.chat->prompt().str());
    } catch (const std::exception & ex) {
        LOGE("chat render: %s", ex.what());
        return nullptr;
    }
}

// Parses a complete reply to the last rendered prompt: {"role", "content", "reasoning_content",
// "tool_calls": [...]}; on failure the text is returned as the content.
JNIEXPORT jstring JNICALL HUI_JNI(chatParse)(JNIEnv * env, jclass, jstring jtext) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    engine &                    e    = g_engine;
    const std::string           text = jstr(env, jtext);
    common_json                 j;
    try {
        if (!e.chat) {
            throw std::runtime_error("no rendered prompt");
        }
        j = e.chat->finish(common_chat_input(text)).to_json_oaicompat();
    } catch (const std::exception & ex) {
        LOGE("chat parse: %s", ex.what());
        j = { { "role", "assistant" }, { "content", text } };
    }
    return jstring_or_null(env, j.dump());
}

// The prompt prefix whose evaluated state is kept in path (empty path or prefix = off); takes
// effect at the next cold start of the context.
JNIEXPORT void JNICALL HUI_JNI(setPromptCache)(JNIEnv * env, jclass, jstring jpath, jstring jprefix) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    engine &                    e = g_engine;
    if (!e.ctx) {
        return;
    }
    e.cache_path              = jstr(env, jpath);
    const std::string prefix  = jstr(env, jprefix);
    // tokenized as the start of a prompt for a cold context (see generate)
    e.cache_tokens = prefix.empty() || e.cache_path.empty() ? llama_tokens() : common_tokenize(e.ctx, prefix, true, true);
}

// See HuiEngine.Stats for the indices.
JNIEXPORT jdoubleArray JNICALL HUI_JNI(stats)(JNIEnv * env, jclass) {
    const engine & e = g_engine;
    const double   v[] = {
        e.last.prefill_ms,
        static_cast<double>(e.last.prefill_tok),
        e.last.decode_ms,
        static_cast<double>(e.last.decode_tok),
        e.last.ttft_ms,
        static_cast<double>(e.last.drafted),
        static_cast<double>(e.last.accepted),
        static_cast<double>(e.last.passes),
        static_cast<double>(e.last.reused_tok),
        static_cast<double>(e.ctx_tokens.size()),
        e.ctx ? static_cast<double>(llama_n_ctx(e.ctx)) : 0.0,
        e.load_ms,
        e.use_spec ? 1.0 : 0.0,
        static_cast<double>(e.last.cache_tok),
        e.last.cache_ms,
        static_cast<double>(npu_bytes()),
    };
    const jsize  n   = sizeof(v) / sizeof(v[0]);
    jdoubleArray arr = env->NewDoubleArray(n);
    env->SetDoubleArrayRegion(arr, 0, n, v);
    return arr;
}

JNIEXPORT jstring JNICALL HUI_JNI(describe)(JNIEnv * env, jclass) {
    return jstring_or_null(env, g_engine.model_desc);
}

JNIEXPORT jint JNICALL HUI_JNI(tokenCount)(JNIEnv * env, jclass, jstring text) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    if (!g_engine.ctx) {
        return -1;
    }
    return static_cast<jint>(common_tokenize(g_engine.ctx, jstr(env, text), false, true).size());
}

// rpcmem currently allocated for the NPU (Hexagon backend), 0 without it.
JNIEXPORT jlong JNICALL HUI_JNI(npuBytes)(JNIEnv *, jclass) {
    return static_cast<jlong>(npu_bytes());
}

// NPU bytes of each transformer layer after repack, plus the output head as the last entry, read
// from every split (the first one may hold only metadata). With dense_only the routed experts
// (*_exps), which stay on flash, are not counted. Null when a file cannot be read.
JNIEXPORT jlongArray JNICALL HUI_JNI(layerNpuBytes)(JNIEnv * env, jclass, jstring jpath, jint n_layer, jboolean dense_only) {
    if (n_layer < 0) {
        return nullptr;
    }
    std::map<int, double> bytes;
    double                out_bytes = 0;
    for (const auto & path : model_split_paths(jstr(env, jpath))) {
        gguf_init_params ip = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
        gguf_context *   g  = gguf_init_from_file(path.c_str(), ip);
        if (!g) {
            return nullptr;
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const std::string name = gguf_get_tensor_name(g, i);
            if (dense_only && name.find("_exps") != std::string::npos) {
                continue;
            }
            const ggml_type t = gguf_get_tensor_type(g, i);
            const double    n = static_cast<double>(gguf_get_tensor_size(g, i)) / ggml_type_size(t) * ggml_blck_size(t);
            const double    b = n * tile_bpw(t) / 8.0;
            int             il = -1;
            if (sscanf(name.c_str(), "blk.%d.", &il) == 1) {
                bytes[il] += b;
            } else if (name == "output.weight") {
                out_bytes += b;
            }
        }
        gguf_free(g);
    }
    std::vector<jlong> v(n_layer + 1, 0);
    for (int il = 0; il < n_layer; ++il) {
        v[il] = static_cast<jlong>(bytes[il]);
    }
    v[n_layer]     = static_cast<jlong>(out_bytes);
    jlongArray arr = env->NewLongArray(static_cast<jsize>(v.size()));
    env->SetLongArrayRegion(arr, 0, static_cast<jsize>(v.size()), v.data());
    return arr;
}

// The number of transformer layers of a model (its block_count); 0 when it cannot be read.
JNIEXPORT jint JNICALL HUI_JNI(layerCount)(JNIEnv * env, jclass, jstring jpath) {
    gguf_init_params ip = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context *   g  = gguf_init_from_file(jstr(env, jpath).c_str(), ip);
    if (!g) {
        return 0;
    }
    const std::string arch = gguf_str(g, "general.architecture");
    const jint        n    = arch.empty() ? 0 : static_cast<jint>(gguf_int(g, arch + ".block_count"));
    gguf_free(g);
    return n;
}

// GGUF metadata of a model (all splits): {"arch", "name", "size_label", "n_layer", "nextn",
// "params", "bytes", "splits", "quant" (type holding most weight bytes), "ctx_train"}; null if a
// file cannot be read.
JNIEXPORT jstring JNICALL HUI_JNI(modelInfo)(JNIEnv * env, jclass, jstring jpath) {
    const std::vector<std::string> paths = model_split_paths(jstr(env, jpath));
    std::string                    arch, name, size_label;
    int64_t                        n_layer = 0, nextn = 0, ctx_train = 0;
    double                         params = 0, bytes = 0;
    std::map<std::string, double>  by_type;

    for (size_t f = 0; f < paths.size(); ++f) {
        gguf_init_params ip = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
        gguf_context *   g  = gguf_init_from_file(paths[f].c_str(), ip);
        if (!g) {
            return nullptr;
        }
        if (f == 0) {
            arch       = gguf_str(g, "general.architecture");
            name       = gguf_str(g, "general.name");
            size_label = gguf_str(g, "general.size_label");
            n_layer    = gguf_int(g, arch + ".block_count");
            nextn      = gguf_int(g, arch + ".nextn_predict_layers");
            ctx_train  = gguf_int(g, arch + ".context_length");
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const ggml_type t  = gguf_get_tensor_type(g, i);
            const double    sz = static_cast<double>(gguf_get_tensor_size(g, i));
            params += sz / ggml_type_size(t) * ggml_blck_size(t);
            bytes += sz;
            by_type[ggml_type_name(t)] += sz;
        }
        gguf_free(g);
    }
    std::string quant;
    double      best = -1;
    for (const auto & kv : by_type) {
        if (kv.second > best) {
            best  = kv.second;
            quant = kv.first;
        }
    }
    common_json j = {
        { "arch", arch },       { "name", name },     { "size_label", size_label },
        { "n_layer", n_layer }, { "nextn", nextn },   { "params", params },
        { "bytes", bytes },     { "splits", static_cast<int64_t>(paths.size()) },
        { "quant", quant },     { "ctx_train", ctx_train },
    };
    return jstring_or_null(env, j.dump());
}

// llama-bench style measurement on the loaded model: prefill n_pp random tokens (in n_batch
// chunks, threads_batch threads; 0 = skip), then n_tg single-token decode steps (threads threads),
// on a cleared context; the conversation context is cleared too. Returns
// [pp tok/s, tg tok/s, pp ms, tg ms, n_pp, n_tg] or null.
JNIEXPORT jdoubleArray JNICALL HUI_JNI(bench)(JNIEnv * env, jclass, jint n_pp, jint n_tg, jint threads, jint threads_batch) {
    std::lock_guard<std::mutex> lock(g_engine.mtx);
    engine &                    e = g_engine;
    if (!e.ctx) {
        return nullptr;
    }
    e.stop = false;
    e.reset_context();
    llama_set_n_threads(e.ctx, threads, threads_batch);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(e.model));
    const int n_ctx   = static_cast<int>(llama_n_ctx(e.ctx));
    n_tg              = std::max(0, std::min<int>(n_tg, n_ctx - 4));
    n_pp              = std::max(0, std::min<int>(n_pp, n_ctx - n_tg - 4)); // 0: decode only
    std::mt19937 rng(1234);

    const size_t n_batch = llama_n_batch(e.ctx);
    common_batch batch(e.ctx);
    bool         ok = true;

    const auto t0 = clk::now();
    for (int i = 0; ok && i < n_pp && !e.stop; i += static_cast<int>(n_batch)) {
        const int end = std::min<int>(n_pp, i + static_cast<int>(n_batch));
        batch.clear();
        for (int j = i; j < end; ++j) {
            batch.add(static_cast<llama_token>(rng() % n_vocab), j, 0, j + 1 == end);
        }
        ok = llama_process(e.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    }
    llama_synchronize(e.ctx);
    const double pp_ms = ms_since(t0);

    const auto t1     = clk::now();
    int        n_done = 0;
    for (; ok && n_done < n_tg && !e.stop; ++n_done) {
        batch.clear();
        batch.add(static_cast<llama_token>(rng() % n_vocab), n_pp + n_done, 0, true);
        ok = llama_process(e.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    }
    llama_synchronize(e.ctx);
    const double tg_ms = ms_since(t1);

    e.reset_context();
    llama_set_n_threads(e.ctx, e.params.cpuparams.n_threads, e.params.cpuparams_batch.n_threads);
    if (!ok) {
        LOGE("bench: decode failed");
        return nullptr;
    }
    const double v[] = {
        n_pp > 0 && pp_ms > 0 ? n_pp * 1000.0 / pp_ms : 0.0,
        tg_ms > 0 && n_done > 0 ? n_done * 1000.0 / tg_ms : 0.0,
        pp_ms,
        tg_ms,
        static_cast<double>(n_pp),
        static_cast<double>(n_done),
    };
    jdoubleArray arr = env->NewDoubleArray(6);
    env->SetDoubleArrayRegion(arr, 0, 6, v);
    return arr;
}

} // extern "C"
