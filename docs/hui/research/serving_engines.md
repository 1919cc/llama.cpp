# Server-side LLM serving engines (vLLM, SGLang, TensorRT-LLM, TGI, Dynamo) and what transfers to a phone NPU

Evidence tags used on every number:
- **[Paper]**: peer-reviewed or arXiv paper with stated evaluation. Not independently replicated by me.
- **[Vendor]**: engine author's blog or docs. Self-benchmarked, and usually compared with a competitor.
- **[Secondary]**: third-party blog or mirror. Weak.
- **[Inference]**: my own reasoning. Only appears under "Inferences" or is labelled there.

Reading limits: paper numbers come from abstracts and the arXiv HTML text, extracted by a fetch tool. The vLLM PDF was not machine-readable, and no figures were read, so per-workload values shown only in figures are missing. "Up to" figures are peaks, not typical results.

## How does each optimization work, and which engine uses it?

### Takeaway
Most server techniques answer one of two problems: GPU memory fragmentation, or wasted batch capacity. The ideas that address a phone's problems are prefix reuse, phase-specific graphs, speculative decoding, and multi-token structured decoding. The memory-manager and fleet-routing machinery largely does not apply.

### Cited Findings

**Engine status**
- TGI is in maintenance mode, and the repo was archived on 2026-03-21. It points users to vLLM and SGLang, plus llama.cpp and MLX for local use — [TGI repo](https://github.com/huggingface/text-generation-inference). Treat TGI as a historical reference, not a live competitor.

**Continuous batching (iteration-level scheduling)**
- Principle: the scheduler runs one iteration at a time, so finished requests return and new ones join without waiting for the whole batch. Selective batching applies batching to only a subset of operations — [Orca, OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu).
- TensorRT-LLM calls this in-flight batching: finished sequences are evicted immediately and new requests start while others run — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/).

**PagedAttention**
- Principle: split each sequence's KV cache into fixed-size, non-contiguous blocks, mapped by a block table and allocated on demand, like OS virtual memory. Reference counts and copy-on-write allow sharing between sequences — [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm).
- Motivation: KV cache reaches up to 1.7 GB per sequence on LLaMA-13B. The authors say existing systems waste 60-80% of memory, and that paging limits waste to under 4% (last block only) — [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm).

**Prefix caching / RadixAttention**
- Principle: keep prompt and output KV after a request finishes. A radix tree on the CPU maps token sequences to paged KV on the GPU (one token per page). Eviction is LRU, from leaves — [LMSYS blog](https://lmsys.org/blog/2024-01-17-sglang/).
- TGI v3 uses its own lookup structure, with about 5-6 µs overhead per lookup — [TGI docs](https://huggingface.co/docs/text-generation-inference/conceptual/chunking).

**Cache-aware scheduling and routing**
- Within a node, SGLang's scheduling policy raises the hit rate, reaching about 96% of optimal on average — [SGLang paper](https://arxiv.org/html/2312.07104v2).
- Across nodes, SGLang's load balancer routes requests to workers likely to hold the prefix — [SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/). Dynamo offers KV-aware routing — [Dynamo docs](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture).

**Chunked prefill and stall-free batching**
- Principle: split long prefills into roughly equal chunks, and add them to batches without pausing ongoing decodes. Uniform iterations also reduce pipeline bubbles — [Sarathi-Serve, OSDI'24](https://www.usenix.org/conference/osdi24/presentation/agrawal).
- TGI credits its own chunking code with finer resource control and lower VRAM use — [TGI docs](https://huggingface.co/docs/text-generation-inference/conceptual/chunking).

**Disaggregation**
- Principle: run prefill and decode on separate GPUs, which removes interference and lets each phase have its own resources and parallelism — [DistServe](https://arxiv.org/abs/2401.09670).
- Dynamo also tiers KV across GPU, CPU, SSD, and object storage — [Dynamo docs (search excerpt)](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture).

**Speculative decoding**
- vLLM documents four proposers: a draft model, n-gram prompt lookup, MLP speculators, and EAGLE-based drafts — [vLLM v0.7 docs](https://docs.vllm.ai/en/v0.7.0/features/spec_decode.html).
- Principle of the load dependence: proposing and verifying adds compute, which hurts when the system is already compute-bound — [vLLM blog](https://vllm.ai/blog/2024-10-17-spec-decode).
- TensorRT-LLM reportedly uses a modified EAGLE-3 with one linear draft sequence per request instead of a tree, and also n-gram — [Mintlify mirror of TensorRT-LLM docs](https://www.mintlify.com/NVIDIA/TensorRT-LLM/features/speculative-decoding) [Secondary].

**Structured output**
- SGLang's compressed FSM decodes multiple tokens at once. Reusing the preprocessed state machine across requests matters — [SGLang paper](https://arxiv.org/html/2312.07104v2).
- XGrammar splits the vocabulary into context-independent tokens, checked ahead of time, and context-dependent tokens, checked at runtime. It runs grammar work in parallel with GPU execution — [XGrammar](https://arxiv.org/abs/2411.15100).

**Kernels and graph compilation**
- TensorRT-LLM combines the TensorRT compiler with fused FlashAttention and masked multi-head attention kernels, for separate context and generation phases — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/).
- SGLang names FlashInfer kernels, `torch.compile`, and CUTLASS FP8 kernels (the last also used by vLLM) — [LMSYS blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/).
- TGI says every kernel launch costs several milliseconds, so fusing bookkeeping kernels helps. This matters most on small models on oversized hardware. It also drops prefill logits by default, which would otherwise take 25.6 GB for 100k tokens on Llama 3.1 8B — [TGI docs](https://huggingface.co/docs/text-generation-inference/conceptual/chunking).

**Quantization**
- TensorRT-LLM converts to FP8 for the H100 Transformer Engine, with kernels compiled automatically. NVIDIA claims better accuracy than INT8 or INT4, but gives no INT8 or INT4 benchmark — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/).
- TGI lists bitsandbytes, GPTQ, EETQ, AWQ, Marlin, and fp8 — [TGI repo](https://github.com/huggingface/text-generation-inference).

### Inferences
- Server quantization is mainly about using a hardware FP8 unit and fitting more in VRAM. On a phone, its first-order effect should be fewer weight bytes per decode step, which targets the DRAM bottleneck directly. I found no server source that measures this.

### Gaps
- I did not confirm the current Dynamo doc page for the storage-tier statement beyond the search excerpt, and it gave no numbers for tiering.
- I found no primary TensorRT-LLM doc page for its speculative design (only the mirror).

## What speedups are reported, and under what conditions?

### Takeaway
Headline multipliers (37x, 13x, 6.4x) are peaks measured against weak baselines or on the most favourable workload. They cannot be compared with each other. Where a like-for-like or decomposed number exists, it is much smaller: 1.1-1.3x for scheduler work, 1.3x-2x for disaggregation on few nodes, and speculative decoding that turns into a slowdown at high load.

### Cited Findings

| Technique / system | Reported gain | Conditions | Baseline | Tag | Source |
|---|---|---|---|---|---|
| Orca continuous batching | 36.9x throughput at same latency | GPT-3 175B | FasterTransformer | Paper | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) |
| vLLM PagedAttention | 2-4x throughput at same latency; larger with long sequences, large models, complex decoding | Popular LLMs | FasterTransformer, Orca | Paper | [vLLM](https://arxiv.org/abs/2309.06180) |
| vLLM | 2.2-2.5x (1 completion); 3.3-3.5x (3 completions) | LLaMA-7B on A10G, LLaMA-13B on A100, ShareGPT lengths | TGI (2023) | Vendor | [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm) |
| vLLM sharing | Up to 55% memory saving, up to 2.2x throughput | Parallel sampling, beam search | No sharing | Vendor | [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm) |
| SGLang (full system) | Up to 6.4x throughput; up to 3.7x latency | Llama-7B, A10G; hit rates 50-99% | Other engines | Paper | [SGLang](https://arxiv.org/html/2312.07104v2) |
| RadixAttention in production | Hit rate 52.4% (LLaVA-Next-34B), 74.1% (Vicuna-33B); Vicuna first-token latency 1.7x lower on average | Chatbot Arena, 1 month | Not cached | Paper | [SGLang](https://arxiv.org/html/2312.07104v2) |
| RadixAttention with no hits | Under 0.3% overhead | 0.2 s of 74.3 s | Disabled | Paper | [SGLang](https://arxiv.org/html/2312.07104v2) |
| SGLang kernel + scheduler | Up to 3.1x on Llama-70B; prefix cache disabled for all engines | Hardware for this figure not stated | vLLM 0.5.2 | Vendor | [LMSYS](https://lmsys.org/blog/2024-07-25-sglang-llama3/) |
| SGLang zero-overhead scheduler | 1.1x vs v0.3, 1.3x vs others; largest on small models and large TP | Default on | v0.3, others | Vendor | [v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) |
| Cache-aware load balancer | Up to 1.9x (82,665 to 158,596 tok/s); hit rate 20% to 75% | Multiple long prefix groups, perfectly balanced; grows with worker count | v0.3 | Vendor | [v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) |
| Sarathi-Serve | 2.6x capacity (Mistral-7B, 1 A100); up to 3.7x (Yi-34B, 2 A100); up to 5.6x (Falcon-180B, pipeline parallel) | Under tail-latency constraints | vLLM | Paper | [Sarathi](https://www.usenix.org/conference/osdi24/presentation/agrawal) |
| DistServe | 7.4x more requests or 12.6x tighter SLOs | Rate per GPU, over 90% within limits | Colocated | Paper | [DistServe](https://arxiv.org/abs/2401.09670) |
| Dynamo disaggregation | 30% throughput/GPU (1 node); over 2x (2 nodes) | H100, R1 Distilled Llama 70B FP8, vLLM, 3K in/150 out | Colocated | Vendor | [Dynamo](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture) |
| Dynamo KV-aware routing | 3x TTFT; 2x lower average latency | 100K R1 requests, 2 H100 nodes, 4K in/800 out | Not stated in excerpt | Vendor | [Dynamo](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture) |
| Dynamo KVBM CPU offload | 2.2-12x TTFT, depending on QPS | Qwen3-8B, H100, 20K in/100 out, multi-turn | No offload | Vendor | [Dynamo](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture) |
| vLLM draft model | Up to 1.5x at QPS=1; **1.4x slowdown at high QPS** | Llama3-70B, 4xH100, ShareGPT, Qwama-0.5B draft | No speculation | Vendor | [vLLM blog](https://vllm.ai/blog/2024-10-17-spec-decode) |
| vLLM n-gram | Up to 2.8x at QPS=1; **1.8x slowdown at high QPS** | Llama3-70B, 4xH100, CNN/DailyMail | No speculation | Vendor | [vLLM blog](https://vllm.ai/blog/2024-10-17-spec-decode) |
| EAGLE-3 | Up to 6.5x speedup, about 1.4x over EAGLE-2; 1.38x throughput at batch 64 in SGLang | Five tasks; hardware not on abstract page | Vanilla | Paper | [EAGLE-3](https://arxiv.org/abs/2503.01840) |
| Compressed FSM | 1.6x JSON decoding throughput; 2.4x lower without FSM reuse | SGLang evaluation | No compression | Paper | [SGLang](https://arxiv.org/html/2312.07104v2) |
| XGrammar | Up to 100x over other grammar engines; claims near-zero end-to-end overhead | Grammar engine speed | Other engines | Paper | [XGrammar](https://arxiv.org/abs/2411.15100) |
| SGLang + XGrammar | Up to 10x for JSON decoding | Must enable the backend flag | Other open-source | Vendor | [v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) |
| TensorRT-LLM | GPT-J-6B: about 8x vs A100, of which H100 + FP8 alone is about 4x. Llama 2 70B: 4.6x | CNN/DailyMail | A100, FP16, PyTorch eager | Vendor | [NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/) |
| TGI v3 | 13x on long prompts (27.5 s to 2 s) | 8xH100, Llama 3.1 70B, 20 requests, 200k tokens, 2nd run | vLLM | Vendor | [TGI docs](https://huggingface.co/docs/text-generation-inference/conceptual/chunking) |

**Results that qualify the headlines**
- TGI's table shows the prefix-cache gain depends on spare KV space (run 1 to run 2, long test) — [TGI docs](https://huggingface.co/docs/text-generation-inference/conceptual/chunking):
  - 8xH100: 12.1 s to 2 s.
  - 4xL4: 47 s to 3.2 s.
  - Single L4 (24 GB): 49.8 s to 53 s, which is no gain.
- TGI claims "up to 30x without prefix caching", but its first-run table shows at most 28.3/12.1 = 2.3x. I could not reconcile these — [TGI docs](https://huggingface.co/docs/text-generation-inference/conceptual/chunking).
- TGI is not always faster: small test on 4xL4, first run, 13 s against vLLM's 12.6 s — [TGI docs](https://huggingface.co/docs/text-generation-inference/conceptual/chunking).
- RadixAttention gives almost no speedup on multi-turn chat with long outputs, because decode dominates — [SGLang paper](https://arxiv.org/html/2312.07104v2).
- SGLang's lead over vLLM narrows on Llama-405B because GPU kernel time dominates. These runs used dummy weights — [LMSYS](https://lmsys.org/blog/2024-07-25-sglang-llama3/).
- TensorRT-LLM headline gains are hardware-plus-software comparisons. NVIDIA's software-only statement is that in-flight batching and kernels at least double throughput on H100 — [NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/).
- vLLM v0.7 docs warned that speculative decoding "does not usually yield inter-token latency reductions" for all prompts and sampling settings, and that EAGLE speedup was below the reference implementation — [vLLM v0.7 docs](https://docs.vllm.ai/en/v0.7.0/features/spec_decode.html). This is old and may no longer hold.
- A hands-on blog reports n-gram at 74 to 251 tok/s for one user (3.4x) and 2x at 32 users, on L4 with Qwen2.5-1.5B, and the author calls the task a copy-heavy best case — [Level Up](https://levelup.gitconnected.com/what-happens-when-an-llm-guesses-ahead-70514d25164d) [Secondary].

### Inferences
- The 36.9x for Orca reflects how poor request-level batching is at the same latency target. It says nothing about a system that is already serving one stream.
- The single-L4 TGI row is the closest server analogue to a phone: a small memory budget can remove the prefix-cache benefit altogether. The docs list constrained KV cache as a caveat but do not explain this row specifically, so the causal link is mine.

### Gaps
- I found no published TensorRT-LLM benchmark of EAGLE-3 or n-gram with in-flight batching on H100.
- I found no independent replication of Dynamo numbers.
- Per-workload SGLang hit rates are in Figure 13, which I could not read.
- Batch-size-versus-speedup figures for speculation (2.31x at batch 1 to 1.4x at batch 8) appear only on blogs with inconsistent hardware labels, so I excluded them — [Prem AI](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026) [Secondary].

## Which methods raise per-stream speed, and which only aggregate throughput?

### Takeaway
Only four groups raise a lone stream's token rate or TTFT: prefix reuse (TTFT only), speculative decoding (decode, when there is idle compute), multi-token structured decoding (decode, on structured output only), and kernels or quantization (both phases). Batching, paging, routing, and disaggregation raise capacity. They help an individual stream only by reducing queueing or interference from other streams.

### Cited Findings
The table is my classification [Inference], based on mechanisms cited above. "Peak" means an uncontended stream. "Under load" means a stream sharing the hardware.

| Method | Peak per-stream decode | Peak per-stream TTFT | Per-stream under load | Aggregate | Basis |
|---|---|---|---|---|---|
| Continuous batching | No | No | Yes, less queueing | Yes | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) |
| PagedAttention | No | No | Indirect, via larger batches | Yes | [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm) credits larger batches |
| Prefix cache | No | **Yes**, on a hit | Yes | Yes, frees prefill compute | [LMSYS](https://lmsys.org/blog/2024-01-17-sglang/); chat result in [SGLang](https://arxiv.org/html/2312.07104v2) |
| Cache-aware scheduling or routing | No | Yes, through hit rate | Yes | Yes | [v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) |
| Chunked prefill | No | No, can be slightly worse | **Yes**, fewer decode stalls | Yes | [Sarathi](https://www.usenix.org/conference/osdi24/presentation/agrawal) |
| Disaggregation | No | No | Yes, less interference | Yes | [DistServe](https://arxiv.org/abs/2401.09670) |
| KV offload tiers | No | Yes, if hit would otherwise be evicted | Yes | Yes | [Dynamo](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture) |
| Speculative decoding | **Yes**, at low load | No | **Can be negative** | Can be negative | [vLLM blog](https://vllm.ai/blog/2024-10-17-spec-decode) |
| Compressed FSM | **Yes**, structured output only | No | Yes | Yes | [SGLang](https://arxiv.org/html/2312.07104v2) |
| Fast grammar engine | Only if grammar was the bottleneck | No | Same | Same | [XGrammar](https://arxiv.org/abs/2411.15100) |
| Kernels, compilation | **Yes** | **Yes** | Yes | Yes | [NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/) |
| Quantization | **Yes**, if bytes or the math unit limit speed | **Yes** | Yes | Yes | [NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/) |

### Inferences
- Reported throughput multipliers for continuous batching, paging, routing, and disaggregation should not be used to predict single-stream phone speed.
- Batching under 2-4 streams has a subtle effect. It cannot exceed the single-stream rate. It can still beat time-slicing, in which each of N streams gets at most 1/N of the single-stream rate, because the per-stream rate becomes (batched aggregate)/N.

### Gaps
- None of the server sources directly report a single-stream peak alongside aggregate figures for all techniques, so this table is reasoning rather than a measured comparison.

## What transfers to a phone NPU with 1-4 streams and limited DRAM bandwidth?

### Takeaway
Priority order: (1) prefix cache, with persistence and alignment to NPU chunk sizes; (2) separate prefill and decode graphs with cheap switching; (3) retrieval-based speculative decoding, the one server technique with a phone measurement behind it; (4) quantization and fused kernels; (5) small-scale chunk interleaving and batching for 2-4 streams. Skip fleet routing, a paged KV allocator, and model-based drafters by default.

### Cited Findings

**Direct phone evidence (sd.npu)**
- Tested on Snapdragon 8 Gen 2, 8 Gen 3, and 8 Gen Elite with Hexagon NPU, using Qwen2.5-0.5B, Qwen2.5-1.5B, and LLaMA3.2-3B — [sd.npu](https://arxiv.org/html/2510.15312v4).
- Result: 1.06-3.81x per-token latency and 1.07-4.71x energy across tasks. Decoding-stage latency drops 5.25-8.15x on OnePlus 13 versus mllm NPU. Qwen2.5-1.5B on the K60 Pro falls from 57.41 to 15.08 ms/token on a summary task — [sd.npu](https://arxiv.org/html/2510.15312v4) [Paper]. These are the authors' own measurements, with no independent replication found. The range spans models, tasks, and devices.
- Decode is memory-bound: one-token generation has low arithmetic intensity, starving the matrix unit and pulling throughput down to about CPU levels. HMX has a minimum 32-token tile — [sd.npu](https://arxiv.org/html/2510.15312v4).
- Prompt lookup averaged 2.53 accepted tokens per step, and about 22% of verification cycles go to rejected drafts. Those accept-length figures were measured on A100 cloud GPUs, not phones. About 38.5-40% of rejected tokens reappear later and can be reused — [sd.npu](https://arxiv.org/html/2510.15312v4).
- Prompts are split into fixed-size chunks such as 256 tokens, to match NPU static shapes. Switching from prefill to decode graph takes 29.0% of total time on 1K-context RAG. Reusing the prefill graph with padding slows decode by up to 8x — [sd.npu](https://arxiv.org/html/2510.15312v4).
- Model-based drafters need auxiliary models, which raises mobile memory cost — [sd.npu](https://arxiv.org/html/2510.15312v4). AHASD and SpecMemo also target memory-limited speculation, but I read only search excerpts — [AHASD](https://arxiv.org/pdf/2604.25326), [SpecMemo](https://arxiv.org/html/2506.01986v1).

**Transfer verdicts** [Inference, with the mechanism citations above]

| Idea | Verdict | Reasoning for the phone |
|---|---|---|
| Prefix cache | **Transfer, top priority** | Prefill is compute-heavy and fully avoidable on a hit. See the model below. |
| KV offload tiers (Dynamo) | **Transfer: RAM to flash** | Persist the system-prompt KV. Restoring is a read rather than a recomputation, which is the same argument as Dynamo's. |
| Separate phase graphs (from disaggregation) | **Transfer the idea** | Prefill and decode want different shapes. Separate silicon does not help as much, since CPU and NPU share DRAM, so bandwidth interference remains. |
| Speculation, retrieval-based | **Transfer, high value** | Few streams means spare compute, and a memory-bound step fits extra tokens at low extra cost, so the server's low-QPS regime is the phone's normal regime. Depends on the text being repetitive. |
| Speculation, draft model | **Default no** | Costs DRAM bandwidth and memory, on the very resource that is scarce. Evaluate only if a tiny drafter shares the NPU budget. |
| Structured output, multi-token | **Transfer** | Forced tokens behave like a draft with guaranteed acceptance, which suits tool calls and JSON. Grammar masking must stay cheap on the CPU. |
| Grammar engine speed | **Transfer, modest** | Avoids per-token CPU spikes across a large vocabulary. Ahead-of-time token classification suits a phone. |
| Quantization | **Transfer, core** | Weight bytes per step bound decode rate. Accuracy must be validated per model. |
| Fused kernels, graph compile | **Transfer, in spirit** | TGI's several-ms launch overhead argument applies to NPU graph dispatch and bookkeeping ops. Dropping prefill logits should transfer directly. |
| Chunked prefill | **Partial** | At one stream there is no scheduling gain, and static shapes force chunking anyway. At 2-4 streams, alternating a chunk with decode steps limits stalls, with chunk duration setting the worst stall. |
| Continuous batching | **Partial, at 2-4 streams** | Weight reads are shared within a batch, which should raise aggregate in a memory-bound regime. It does not help at one stream. Static shapes make join and leave harder. |
| PagedAttention | **Mostly no** | Solves fragmentation at dozens of sequences. Static NPU buffers and 1-4 streams leave little to gain. Sharing a read-only prefix across sessions is the one useful part. |
| Cache-aware fleet routing | **No** | There is one device. The single useful remainder is a local rule: when choosing what to run or evict, keep the system-prompt prefix. |

**Rough ceiling for decode** [Inference]
- tok/s ≤ B / (W + KV_read), where B is usable DRAM bandwidth, W is weight bytes read per step, and KV_read is KV bytes read per step. As a purely illustrative case with assumed values, 2 GB of weights and 30 GB/s of bandwidth give about 15 tok/s. Halving W roughly doubles this ceiling.
- Speculation works against this limit by turning one read pass into more than one token. At 2.53 accepted tokens per step (a GPU-measured figure, used only for scale), the ceiling would rise about 2.5x before verification and rejection costs.

### Inferences
Everything in the table and ceiling formula above is inference. The main risk is thermal and power behaviour: none of the server evidence measures sustained phone load, though sd.npu reports energy.

### Gaps
- I found no source that measures system-prompt prefix caching on a phone NPU together with speculation and DRAM bandwidth.
- I found no measured phone data on 2-4 concurrent stream batching on Hexagon.
- This lab's own data should replace the assumed numbers here; I did not read `results/` or `report/` in hexagon-lab.

## What does a mobile system-prompt prefix hit imply for effective prefill?

### Takeaway
A hit does not speed up prefill computation. It removes it, so TTFT follows Amdahl's law in the miss fraction: about 10x at a 90% hit in the ideal case. On a static-shape NPU, chunk rounding, KV restore, and graph switching bring a 94% hit from an ideal 17x down to about 5x in the worked example (assumed parameters). The decode rate does not change, so report TTFT, not an "effective t/s".

### Cited Findings
- Server evidence supports the direction. The hit benefit is first-token latency, and production hit rates were 52.4-74.1% — [SGLang](https://arxiv.org/html/2312.07104v2). There was no gain when output dominates — [SGLang](https://arxiv.org/html/2312.07104v2). Restoring from CPU memory beats recompute by 2.2-12x TTFT on long prompts — [Dynamo](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture).
- Graph switching is a large fixed cost on a phone NPU: 29.0% of total time on 1K-context RAG — [sd.npu](https://arxiv.org/html/2510.15312v4).

### Inferences
All of the following is my model, not a measurement.

**Model.** N = prompt tokens, h = hit fraction, R = cold prefill tok/s, C = NPU chunk size, s = KV bytes per token, B_r = restore bandwidth, F = fixed cost (graph switch, setup).

- Continuous form: `TTFT = (1-h)·N/R + h·N·s/B_r + F`
- NPU form: `TTFT = ceil((1-h)·N/C)·(C/R) + h·N·s/B_r + F`
- Effective prefill rate: `R_eff = N / TTFT`
- With r = restore cost per token / compute cost per token, and F=0: `speedup = 1 / ((1-h) + h·r)`

| Hit rate h | Ideal, r=0 (KV already in RAM) | r=0.05 (e.g. restoring from flash) |
|---|---|---|
| 50% | 2.0x | 1.9x |
| 80% | 5.0x | 4.2x |
| 90% | 10x | 6.9x |
| 95% | 20x | 10.3x |
| 99% | 100x | 16.8x |

**Estimating s.** KV bytes per token = 2 × layers × KV heads × head dim × element bytes.
- Sanity check: LLaMA-13B (40 layers, 5120 hidden, FP16, config from memory) gives 819 KB/token, or 1.68 GB at 2048 tokens. That matches vLLM's stated 1.7 GB — [vLLM blog](https://vllm.ai/blog/2023-06-20-vllm).
- With the config values below, also from memory and to be verified:
  - Llama 3.2 3B-like (28 layers, 8 KV heads, 128 dim, FP16): about 115 KB/token, or 92 MB for 800 tokens.
  - Qwen2.5 1.5B-like (28 layers, 2 KV heads, 128 dim, FP16): about 29 KB/token, or 23 MB for 800 tokens.
- Quantizing KV would change these numbers.

**Worked example** with assumed values: 800-token system prompt, 50-token user turn (N=850, h=94%), R=500 tok/s, C=128 (so 0.256 s per chunk), F=100 ms, s=115 KB, flash read 2 GB/s (46 ms for 92 MB).

| Case | Prefill work | Restore | F | TTFT | Speedup vs cold | R_eff |
|---|---|---|---|---|---|---|
| Cold | 7 chunks, 1.792 s | none | 0.1 s | 1.892 s | 1x | 449 tok/s |
| Hit, ideal without chunks or F | 0.100 s | none | none | 0.100 s | 17x vs 1.7 s | 8,500 tok/s |
| Hit, KV in RAM | 1 chunk, 0.256 s | none | 0.1 s | 0.356 s | 5.3x | 2,388 tok/s |
| Hit, KV from flash | 1 chunk, 0.256 s | 0.046 s | 0.1 s | 0.402 s | 4.7x | 2,114 tok/s |

What this shows:
- **Chunk rounding is the biggest loss.** A 50-token suffix pays for a full 128-token chunk. Smaller prefill graphs for the tail, such as a 32-token one, would recover much of it, at the cost of holding more graphs.
- **Restore is cheap compared with compute**, so a flash-persisted prefix is worthwhile, including for the cold-start first message of an app session. KV in RAM costs memory continuously (92 MB in this example) and may be killed by the OS.
- **F sets the floor.** At high hit rates, graph switching dominates TTFT.
- **Prefix stability matters more than cache design.** Anything early in the prompt that varies, such as a timestamp, device state, or a reordered tool list, breaks the prefix match and sends h towards 0. Volatile fields should go last.
- **Hit types are different.** The system prompt gives a one-time saving per session. Multi-turn reuse saves more over time as history grows, but only if the template renders the history identically.
- **End-to-end effect is bounded.** With long outputs, decode dominates total time, as in SGLang's chat result. For a 300-token reply at 20 tok/s (15 s), saving 1.5 s of TTFT is a 10% total gain, but a large perceived gain.
- **Streams.** 1-4 sessions with the same system prompt could share one read-only prefix if buffers permit, which saves memory rather than time. If each stream's KV must be contiguous in a static buffer, sharing becomes a copy, which costs DRAM bandwidth.

### Gaps
- Worked-example values (R, C, F, flash speed) are placeholders. Replace them with hexagon-lab measurements: cold prefill by prompt length, the graph-switch time, and KV restore time on 8845 or the target chip.
- I did not find whether current Hexagon runtimes allow restoring KV into NPU-visible memory without a copy, nor whether prefill can start at an arbitrary token offset in a static graph. Both affect the model directly.
