// Compare logits of the same tokens decoded one by one and in batches of size N.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using logits_t = std::vector<std::vector<float>>;

static logits_t run(llama_context * ctx, const std::vector<llama_token> & toks, int n_prompt, int bs, int n_vocab) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch b = llama_batch_init(std::max(n_prompt, bs), 0, 1);
    auto add = [&](int i, bool out) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i; b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = out; b.n_tokens++;
    };
    b.n_tokens = 0;
    for (int i = 0; i < n_prompt; i++) add(i, i == n_prompt - 1);
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
    logits_t out;
    for (int i = n_prompt; i < (int) toks.size();) {
        b.n_tokens = 0;
        int n = std::min(bs, (int) toks.size() - i);
        for (int j = 0; j < n; j++) add(i + j, true);
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
        for (int j = 0; j < n; j++) { float * l = llama_get_logits_ith(ctx, j); out.emplace_back(l, l + n_vocab); }
        i += n;
    }
    llama_batch_free(b);
    return out;
}

static void compare(const char * name, const logits_t & a, const logits_t & c) {
    double max_abs = 0, sum2 = 0, n = 0, min_margin = 1e9, sum_pos = 0; int flips = 0, bits = 0;
    for (size_t p = 0; p < a.size(); p++) {
        int ta = 0, tc = 0; double pm = 0;
        for (size_t v = 0; v < a[p].size(); v++) {
            double d = std::fabs(a[p][v] - c[p][v]);
            pm = std::max(pm, d); sum2 += d * d; n++;
            if (a[p][v] != c[p][v]) bits++;
            if (a[p][v] > a[p][ta]) ta = v;
            if (c[p][v] > c[p][tc]) tc = v;
        }
        max_abs = std::max(max_abs, pm); sum_pos += pm;
        float second = -1e30f;
        for (size_t v = 0; v < a[p].size(); v++) if ((int) v != ta) second = std::max(second, a[p][v]);
        min_margin = std::min(min_margin, (double) (a[p][ta] - second));
        if (ta != tc) flips++;
    }
    if (getenv("LC_POS")) {
        printf("   per-position max:");
        for (size_t p = 0; p < a.size(); p++) { double pm = 0; for (size_t v = 0; v < a[p].size(); v++) pm = std::max(pm, (double) std::fabs(a[p][v] - c[p][v])); printf(" %.3f", pm); }
        printf("\n");
    }
    printf("%-10s positions=%zu top1_flips=%d max_abs=%.5f mean_pos_max=%.5f rms=%.6f differing=%.2f%% min_margin=%.4f\n", name, a.size(), flips, max_abs, sum_pos / a.size(), std::sqrt(sum2 / n), 100.0 * bits / n, min_margin);
}

static void dump(const std::string & path, const logits_t & l) {
    std::ofstream o(path, std::ios::binary);
    for (const auto & r : l) o.write((const char *) r.data(), r.size() * sizeof(float));
}

static logits_t load(const std::string & path, int n_vocab) {
    std::ifstream i(path, std::ios::binary);
    logits_t l;
    std::vector<float> r(n_vocab);
    while (i.read((char *) r.data(), r.size() * sizeof(float))) l.push_back(r);
    return l;
}

int main(int argc, char ** argv) {
    if (std::string(argv[1]) == "cmp") {
        compare(argv[4], load(argv[2], 262144), load(argv[3], 262144));
        return 0;
    }
    // usage: logit_cmp <model> <prompt file> <device> <n_prompt> <n_test>
    const std::string dev = argv[3];
    const int n_prompt = atoi(argv[4]), n_test = atoi(argv[5]);
    llama_backend_init();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_name(dev.c_str()), nullptr };
    if (dev != "none") { mp.devices = devs; mp.n_gpu_layers = 99; } else { mp.n_gpu_layers = 0; }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 2; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); const std::string text = ss.str();
    std::vector<llama_token> toks(text.size() + 2);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, true);
    toks.resize(std::min(n, n_prompt + n_test));
    const int n_vocab = llama_vocab_n_tokens(vocab);
    logits_t ref = run(ctx, toks, n_prompt, 1, n_vocab);
    compare("repeat_b1", ref, run(ctx, toks, n_prompt, 1, n_vocab));
    if (argc > 6) {
        dump(std::string(argv[6]) + ".b1", ref);
        dump(std::string(argv[6]) + ".b8", run(ctx, toks, n_prompt, 8, n_vocab));
    }
    for (int bs : {2, 3, 4, 5, 6, 7, 8, 16}) {
        char name[32]; snprintf(name, sizeof(name), "batch_%d", bs);
        compare(name, ref, run(ctx, toks, n_prompt, bs, n_vocab));
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
