#pragma once

#include "ggml.h"

struct llama_model;

// Hot experts on the accelerator (LLAMA_HOT_EXPERTS_MB=<n>, LLAMA_HOT_EXPERTS_PROFILE=<file>).
//
// A MoE whose routed experts stay mmap-ed in host memory computes all of them on the CPU while the accelerator that
// runs the dense part of the layer waits. The experts are picked with a strong skew (Inkling 2 Mini: the top 8% of
// the experts take half of the selections), so the most used ones are copied once into a bank on the device of
// their layer, up to n MB: the device computes the selected hot experts while the CPU computes the others, and the
// two partial results are added (each side reads the other's experts as id -1, which both backends compute as zero).
//
// The experts are ranked by their selections in a routing profile, a trace written by LLAMA_EXPERT_TRACE on a
// representative run. With LLAMA_HOT_EXPERTS_PERIOD=<n> the bank also follows the conversation, swapping hot experts
// that fall out of use for the busiest cold ones every n tokens (Strata's adaptive expert cache; off by default, it
// measured no gain on the PLR110). Layers that share their expert tensors (the loop passes of Inkling 2 Mini) share
// one bank.
// No-op when the variables are unset.

// the expert ids of the hot (on the device) and cold (in host memory) part of a layer, from its selected experts
struct llama_hot_experts_ids {
    ggml_tensor * hot  = nullptr; // [n_expert_used, n_tokens] bank slot of each selected expert, -1 if cold
    ggml_tensor * cold = nullptr; // [n_expert_used, n_tokens] the selected expert, -1 if hot
};

// splits the selected experts of layer il between its bank and host memory; empty ids if the layer has no hot experts
// in generation it also adds the op that counts the selections for the adaptation
llama_hot_experts_ids llama_hot_experts_split(ggml_context * ctx, ggml_cgraph * gf, const llama_model & model, int il, ggml_tensor * selected);

// adapts the banks to the conversation (LLAMA_HOT_EXPERTS_PERIOD tokens, LLAMA_HOT_EXPERTS_SWAPS experts per step):
// call between two graph computes, never while one runs
void llama_hot_experts_maintain();

// the bank of the hot experts of the host expert tensor w, nullptr if none
ggml_tensor * llama_hot_experts_bank(const ggml_tensor * w);
