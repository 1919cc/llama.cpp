#pragma once

#include "ggml.h"

#include <cstddef>
#include <utility>
#include <vector>

struct llama_model;
struct llama_weight_stream;

// LLAMA_WEIGHT_STREAM=1: the mmap-ed weights of the layers that run from host memory are read layer by layer with
// O_DIRECT into two rotating buffers, the next layer while the current one computes, instead of being paged in by
// page faults. Linux only; nullptr when disabled, unsupported, or the model has no file-backed host layers.
llama_weight_stream * llama_weight_stream_get(const llama_model & model);

// frees the stream of a model (called when the model is destroyed)
void llama_weight_stream_release(const llama_model & model);

// whether layer il is streamed, and the user data of its op
void * llama_weight_stream_layer(llama_weight_stream * ws, int il);

// ggml custom op placed at the start of a streamed layer: makes the layer's weights point at its loaded buffer
void llama_weight_stream_op(ggml_tensor * dst, int ith, int nth, void * userdata);

// Fills the page cache for mmap-ed file ranges with parallel buffered preads in 4 MiB chunks on a pool of I/O
// threads (LLAMA_FILL_THREADS, default 6), so the mmap reads that follow hit RAM instead of faulting page by page.
// wait = true returns once the data is in the page cache. Ranges that are not file-backed are skipped. Linux only.
void llama_page_fill(const std::vector<std::pair<const void *, size_t>> & ranges, bool wait);

// one read for llama_read_into: len bytes at the mmap-ed address src, into dst
struct llama_read_req {
    const void * src;
    void *       dst;
    size_t       len;
};

// reads mmap-ed file ranges into the given buffers on the page-fill thread pool (buffered preads: pages already in
// the page cache come from RAM) and waits for them; false if a range is not file-backed. Linux only.
bool llama_read_into(const std::vector<llama_read_req> & reqs);
