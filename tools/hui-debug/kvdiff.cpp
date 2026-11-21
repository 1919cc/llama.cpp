#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
using std::vector;
static llama_model * model;
static int n_vocab;
static llama_context * mk(bool unified, int fa) {
    auto p = llama_context_default_params();
    p.n_ctx = 8192; p.n_batch = 2048; p.n_ubatch = 512; p.n_seq_max = 4; p.kv_unified = unified;
    p.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.n_threads = p.n_threads_batch = 8;
    return llama_init_from_model(model, p);
}
static void dec(llama_context * c, const vector<llama_token> & t, int a, int b, int seq) {
    llama_batch ba = llama_batch_init(b - a, 0, 1);
    for (int i = a; i < b; i++) { int k = ba.n_tokens++; ba.token[k] = t[i]; ba.pos[k] = i; ba.n_seq_id[k] = 1; ba.seq_id[k][0] = seq; ba.logits[k] = i == b - 1; }
    int r = llama_decode(c, ba); if (r) { printf("decode failed %d\n", r); exit(1); }
    llama_batch_free(ba);
}
static vector<float> lg(llama_context * c) { float * l = llama_get_logits_ith(c, -1); return vector<float>(l, l + n_vocab); }
static void cmp(const char * tag, const vector<float> & a, const vector<float> & b) {
    float d = 0; size_t ia = 0, ib = 0;
    for (size_t i = 0; i < a.size(); i++) { d = std::max(d, std::fabs(a[i] - b[i])); if (a[i] > a[ia]) ia = i; if (b[i] > b[ib]) ib = i; }
    printf("  %-34s max|dlogit|=%.6f top %zu/%zu\n", tag, d, ia, ib); fflush(stdout);
}
int main(int argc, char ** argv) {
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    model = llama_model_load_from_file(argv[1], mp);
    const int L = argc > 2 ? atoi(argv[2]) : 200;
    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 g(1);
    vector<llama_token> P(L), X(100);
    for (auto & t : P) t = 1000 + g() % 20000;
    for (auto & t : X) t = 1000 + g() % 20000;
    int K = L * 3 / 4;
    auto * c0 = mk(0, 0); dec(c0, P, 0, L, 0); auto exact = lg(c0); llama_free(c0);
    auto * c1 = mk(0, 1); dec(c1, P, 0, L, 0); auto fa1 = lg(c1); cmp("pristine fa=1 vs exact fa=0", exact, fa1);
    auto * m1 = llama_get_memory(c1);
    llama_memory_clear(m1, true); dec(c1, P, 0, K, 0); dec(c1, P, K, L, 0); cmp("fa=1 two batches vs exact", exact, lg(c1)); llama_free(c1);
    for (int off : {1, 4, 8, 16, 32, 64, 100, 128}) {
        auto * c = mk(1, 1);
        vector<llama_token> X2(off, 1000);
        dec(c, X2, 0, off, 0); dec(c, P, 0, L, 1); auto v = lg(c);
        std::string a = "offset " + std::to_string(off) + " vs fa=1"; cmp(a.c_str(), fa1, v);
        a = "offset " + std::to_string(off) + " vs exact"; cmp(a.c_str(), exact, v);
        llama_free(c);
    }
    return 0;
}
