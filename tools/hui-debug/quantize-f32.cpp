// Requantize a GGUF to all-F32, as the reference for kvdiff / logit-cmp.
#include "llama.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.gguf> <out-f32.gguf>\n", argv[0]);
        return 1;
    }
    auto p = llama_model_quantize_default_params();
    p.ftype            = LLAMA_FTYPE_ALL_F32;
    p.allow_requantize = true;
    p.nthread          = 8;
    return llama_model_quantize(argv[1], argv[2], &p);
}
