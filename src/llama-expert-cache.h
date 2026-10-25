#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// LLAMA_EXPERT_CACHE_MB=<n>: keeps the most recently used routed experts of mmap-ed MoE layers in an n MB arena of
// anonymous memory. The experts of a 125B MoE on flash are picked with a strong per-conversation skew (the top 10% of
// a layer's experts take ~2/3 of its selections), which the page cache does not exploit: the expert pages are read
// once per token and evicted again. Cached experts are found by the CPU MUL_MAT_ID through a per-tensor table of
// expert addresses (ggml_cpu_expert_remap, attached as the tensor's extra); a miss is read from the model file
// straight into its slot. No mapping is ever changed. Linux only; a no-op when the variable is unset.

bool llama_expert_cache_enabled();

// whether expert e of layer il is resident in the cache
bool llama_expert_cache_has(int il, int e);

// makes the distinct experts ids[0..n) of layer il resident before the layer's MUL_MAT_ID runs: hits become the
// most recently used, misses evict the least recently used slots and are read into them (waits for the reads).
// ws are the layer's expert tensors [.., .., n_expert]. Must run in graph order (it reuses slots of other layers).
// Returns false if the experts could not be cached (they are then read from the mmap as before).
bool llama_expert_cache_use(int il, const int32_t * ids, size_t n, const ggml_tensor * const * ws, int n_ws);
