#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#include <stddef.h>
#include <stdint.h>

#include "ggml.h"
#include "htp/htp-tensor.h"

// codebook i-quants: transcoded on the host into the tile of another type, no flat kernels on the DSP
static inline bool htp_is_iq_tiled_type(enum ggml_type type) {
    return type == GGML_TYPE_IQ4_XS  ||
           type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ1_S   || type == GGML_TYPE_IQ1_M  ||
           type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
}

// type of the DSP tile a tiled weight is stored in
static inline enum ggml_type htp_tiled_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:    return GGML_TYPE_F16;
        case GGML_TYPE_IQ4_XS:  return GGML_TYPE_IQ4_NL;
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:   return GGML_TYPE_IQ2_XXS;
        case GGML_TYPE_IQ1_M:   return GGML_TYPE_IQ1_S;
        default:                return type;
    }
}

static inline bool htp_is_tiled_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16  || htp_is_iq_tiled_type(type);
}

static inline bool htp_is_tiled_weight_tensor(const struct ggml_tensor * t) {
    if (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        return false;
    }
    if (!htp_is_tiled_weight_type(t->type)) {
        return false;
    }
    if (t->ne[0] < 32) {
        return false;
    }
    if (!ggml_is_quantized(t->type)) {
        return (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    return t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
static inline size_t htp_tiled_row_size(enum ggml_type type, int64_t ne0) {
    type = htp_tiled_type(type);
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        return (size_t) ne0 * sizeof(ggml_fp16_t);
    }
    if (type == GGML_TYPE_Q6_K) {
        return (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    if (type == GGML_TYPE_Q2_0) {
        return (size_t) (ne0 / 64) * 18;
    }
    if (type == GGML_TYPE_Q4_K) {
        return (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    if (type == GGML_TYPE_Q5_K) {
        return (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    if (type == GGML_TYPE_Q3_K) {
        return (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    if (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S) {
        return (size_t) (ne0 / 32) * (htp_tensor_get_tile_size(type) / 32);
    }
    return ggml_row_size(type, ne0);
}

static inline bool htp_is_hmx_weight_type(enum ggml_type type) {
    return htp_is_tiled_weight_type(type);
}

void htp_repack_tensor_tiled(struct ggml_tensor * t, const void * data, size_t size);
void htp_unpack_tensor_tiled(void * data, const struct ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
