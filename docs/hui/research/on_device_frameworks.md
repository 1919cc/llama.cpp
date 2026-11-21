# On-device LLM inference engines: optimization methods and published performance (Hexagon focus)

Evidence labels: **[VENDOR]** published by the framework or chip vendor. **[AUTHOR]** self-reported by a fork or paper author about their own system. **[3RD]** independent measurement. **[COMMUNITY]** unverified user report. Nothing here was run for these notes. Several numbers come from page extractions by a fetch tool, and charts on those pages were not readable.

Device mapping used below (background knowledge, not from the cited pages): Galaxy S26 Ultra is taken to be Snapdragon 8 Elite Gen 5 for Galaxy. LiteRT's page does not name the chip. No source found mentions SM8845 specifically, so all v81 data is from SM8850-class parts.

## Q1. Published decode and prefill speeds for ~1-4B models on 8 Elite / Elite Gen 5 class hardware

### Takeaway
Published NPU decode for a 2-4B model clusters at about 25-35 t/s on Hexagon v75-v81, and I found no published result above that for a real ~2B model on Hexagon. The 30 t/s baseline is therefore at the published ceiling. Google's LiteRT figures on a Galaxy S26 Ultra put CPU (47 t/s) and GPU (52 t/s) above any published NPU decode for Gemma 4 E2B, while NPU's clearest lead is prefill.

### Cited Findings

**NPU**

| Model | Quant | Device / HTP | Engine | Prefill t/s | Decode t/s | Label | Source |
|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | not stated on card | Dragonwing IQ-8275, 4096 ctx | LiteRT-LM | 3,747 | 31.7 | VENDOR | [HF card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |
| Gemma 4 E2B | w4a16; KV int8 | IQ-8275 (v75), QLI 2.0, 49-token prompt | LiteRT-LM | ~1,520 | ~32 (~200 tokens), ~32.5 (~800) | VENDOR | [Qualcomm](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8) |
| Gemma 4 E2B | w4a16 | IQ-8275, Ubuntu prototype | LiteRT-LM | ~1,240 | ~28 (~200), ~25 (~800); ~16 without HTP burst patch | VENDOR | [Qualcomm](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8) |
| Gemma 4 E2B | Q4_K lm_head | 8 Elite (v79), ctx 8192 | ggml-hexagon, FastRPC fork / official dspqueue | 668 / 490 | 27.00 / 24.99 | AUTHOR | [llama.cpp #26227](https://github.com/ggml-org/llama.cpp/discussions/26227) |
| Qwen3.5-2B | Q6_K lm_head | 8 Elite (v79) | FastRPC fork / dspqueue | 485 / 435 | 26.99 / 14.95 | AUTHOR | [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227) |
| Llama 3.2 1B | Q4_K lm_head | 8 Elite (v79) | FastRPC fork / dspqueue | 1,006 / 1,063 | 41.64 / 28.33 | AUTHOR | [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227) |
| MiniCPM5 1B | Q4_0 lm_head | 8 Elite (v79) | FastRPC fork / dspqueue | 1,162 / 1,017 | 51.95 / 36.14 | AUTHOR | [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227) |
| Qwen3-4B | w4a16, 4096 ctx | 8 Elite Gen 5 Galaxy / 8 Elite Galaxy | Genie | TTFT 0.054-1.71 s / 0.065-2.08 s | 28.64 / 26.13 | VENDOR | [AI Hub card](https://huggingface.co/qualcomm/Qwen3-4B) |
| Qwen3-4B | w4a16 | 8 Elite Gen 5 / 8 Elite, 512 ctx | GENIEX_QAIRT | TTFT 0.068-0.272 s / 0.061-0.244 s | 23.06 / 23.40 | VENDOR | [AI Hub card](https://huggingface.co/qualcomm/Qwen3-4B) |
| Qwen3-4B | w4a16 | 8 Elite Gen 5 / 8 Elite, 4096 ctx | GENIEX_QAIRT | TTFT up to 3.0 s / 3.24 s | 15.19 / 17.54 | VENDOR | [AI Hub card](https://huggingface.co/qualcomm/Qwen3-4B) |
| Qwen3.5-0.8B | q4_0 | 8 Elite Gen 5 / 8 Elite, 4096 ctx | GENIEX_LLAMACPP | not split | best rows 51.6 / 52.2 (3 unlabelled rows per chip, 38-52) | VENDOR | [AI Hub card](https://huggingface.co/qualcomm/Qwen3.5-0.8B) |
| FastVLM-0.5B | int8 weights, int16 activations | 8 Elite Gen 5 | LiteRT QNN | >11,000 | >100 | VENDOR | [Google](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/) |
| Qwen3-0.6B | not stated | SM8850 | MNN fork, QNN context binary | 12.8-20.7 | ~32 | AUTHOR | [fork releases](https://github.com/pisces312/MNN/releases) |
| Qwen3-0.6B | not stated | SM8850 | MNN fork, custom Hexagon ops | ~49.7 | ~1.5 | AUTHOR | [fork releases](https://github.com/pisces312/MNN/releases) |
| Qwen3-4B | not stated | not restated in extraction | MNN fork, custom Hexagon ops, later release | 60.53 | 12.12 | AUTHOR | [fork releases](https://github.com/pisces312/MNN/releases) |
| Llama 3.2 3B Instruct | not stated | 8 Elite | Genie | n/a | ~10 | 3RD (blog) | [Grape Up](https://grapeup.com/blog/running-llms-on-device-with-qualcomm-snapdragon-8-elite) |
| Llama 3.2 1B | Q4_0 | device not clearly stated | llama.cpp Hexagon | 136.21 | 51.57 | COMMUNITY | [HF](https://huggingface.co/Eternalgrey/llama-snapdragon-models) |

**CPU and GPU (comparison on the same class of phone)**

| Model | Device | Backend | Prefill t/s | Decode t/s | Peak CPU mem | Label | Source |
|---|---|---|---|---|---|---|---|
| Gemma 4 E2B (2.58 GB; scheme "mixture of 2bit, 4bit and 8 bit") | Galaxy S26 Ultra | LiteRT CPU | 557 | 47 | 1,733 MB | VENDOR | [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview), [HF card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |
| same | Galaxy S26 Ultra | LiteRT GPU (OpenCL) | 3,808 | 52 | 676 MB | VENDOR | [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview) |
| same, with MTP | Galaxy S26 Ultra | LiteRT GPU | n/a | up to 2.2x decode | n/a | VENDOR | [Google blog](https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/) |
| Gemma 4 E2B | 8 Elite phone | LiteRT GPU | 1,411.74 | 34.74 | n/a | COMMUNITY | [Reddit snapshot](https://reddit.sentinel-team.org/posts/1tc3czb/snapshots/2026-05-15T23%3A40%3A01.25497Z) |
| Qwen3-0.6B | SM8850 | MNN fork CPU fp16, 4 threads | ~306 | ~70 | n/a | AUTHOR | [fork releases](https://github.com/pisces312/MNN/releases) |
| Gemma 2B (2024, GPU) | Snapdragon 8 Gen 3 | Transformer-Lite | 330 | 30 | n/a | AUTHOR | [arXiv 2403.20041](https://arxiv.org/pdf/2403.20041) |

**Headline claims with no reproducible setup**
- Qualcomm "up to 220 tokens per second" on 8 Elite Gen 5 is a marketing claim tied to a 3B-parameter model, seen via secondary coverage — [AOL](https://www.aol.com/articles/qualcomms-snapdragon-8-elite-gen-203000844.html).
- Qualcomm "more than 70 tokens per second" on 8 Elite, with model size not disclosed — [Android Authority](https://www.androidauthority.com/snapdragon-8-elite-3491805/).
- RunAnywhere/QHexRT: a search snippet attributed 12,540 tok/s prefill on v81 vs 871 on llama.cpp CPU for LFM2.5-230M, but the fetched [model card](https://huggingface.co/runanywhere/lfm2_5_230m_HNPU) showed no performance numbers. Treat as unverified. The vendor's cards are also inconsistent on chip naming, for example ["Samsung S25 (SM8850)"](https://huggingface.co/runanywhere/gemma3n_e4b_HNPU).

**Independent stage-level measurement (v75)**
- llama.cpp b7588 on 8 Gen 3, Hexagon v75, Q4_0 models: per-invocation MUL_MAT is 1.27-1.62x slower on NPU in prefill and 1.55-1.67x faster in decode, but end-to-end decode gains only 1.05-1.20x. Full NPU offload raises battery drain 22-51% on Llama-3.2-3B as prompt grows from ~321 to ~1,281 tokens — [arXiv 2605.27435](https://arxiv.org/html/2605.27435). Absolute t/s values are only in its Fig. 2, not extracted.

**Conflicts and quality notes**
- Qwen3-4B Genie on 8 Elite Gen 5 is 28.64 on the [HF card](https://huggingface.co/qualcomm/Qwen3-4B) and ~29.2 on an [aggregator copy](https://app.alphaneural.io/assets/qualcomm/Qwen3-4B), probably different card revisions.
- AI Hub's GENIEX_LLAMACPP tables repeat chipset rows with no distinguishing configuration, so mapping values to CPU, GPU or NPU is ambiguous — [Qwen3.5-0.8B](https://huggingface.co/qualcomm/Qwen3.5-0.8B), [Qwen3-4B](https://huggingface.co/qualcomm/Qwen3-4B).
- AI Hub response rate is measured on a short prompt with a long response, and TTFT lower/upper bounds correspond to a prompt of up to 128 tokens and a full-context prompt. The card also says Genie support "will be deprecated soon" — [Qwen3-4B card](https://huggingface.co/qualcomm/Qwen3-4B).
- E2B NPU file size differs: 2,967 MB on the [HF card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), 3.29 GB on the [Qualcomm page](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8) (may be GB vs GiB or a different build). The HF card also says the QCS8275 file's quantization and benchmarks are not established by that card.
- The FastRPC-fork table is one author on one device, with no maintainer response in the thread — [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227).

### Inferences
- **The two E2B prefill figures reconcile.** 49 tokens at 1,520 t/s is ~32 ms and 128 tokens at 3,747 t/s is ~34 ms. That is what a fixed `prefill_128` graph would produce, since a short prompt still pays for a full chunk. Static-graph prefill numbers should be compared per chunk. My calculation, not stated by either source.
- **Generation-over-generation scaling is near flat.** Gen 5 vs 8 Elite for Qwen3-4B is +9.6% on Genie and -1.5% to -13% on GENIEX_QAIRT. This fits a memory bound more than a compute bound, though sample sizes and thermal conditions are unknown.
- **Your baseline reads as a bandwidth figure.** 41 GB/s ÷ 30 t/s = ~1.37 GB read per token, close to the 1.17 GB persistent w4a16 weight buffer Qualcomm reports for E2B. LiteRT's CPU reaching 47 t/s on a mixed 2/4/8-bit build suggests fewer weight bytes per token may be a large factor, but the CPU row's exact bit mix and bandwidth are unpublished, so this is a hypothesis.
- **IQ-8275 (v75) at ~32 t/s matching a v79/v81 result** suggests HTP generation is not the limiter for E2B. Differences in DDR, thermals and quant make this suggestive only.

### Gaps
- No published LiteRT-LM NPU E2B figure on 8 Elite or Elite Gen 5 was found, only IQ-8275.
- No clearly labelled NPU prefill/decode for Qwen3.5-2B on v81 from any vendor. #26227 did not test v81.
- ExecuTorch QNN: no t/s found on 8 Elite class. [Docs](https://docs.pytorch.org/executorch/1.0/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.html) give none. A Vulkan figure of 44.6 / 29.8 t/s for Llama 3.2 1B has an unspecified device — [Arm](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/executorch-vulkan-learning-path/how-to-5-deploy-run-and-measure/).
- MNN-LLM paper absolute t/s are only in Fig. 5, not extractable. Thermal-sustained NPU numbers on phones are largely unpublished.

## Q2. Which techniques each engine uses

### Takeaway
QNN-derived stacks (Genie, LiteRT QNN, ExecuTorch QNN) share one recipe: offline static graphs, separate prefill and decode graphs, w4a16 block quantization with a higher-precision LM head, int8 KV, and model splitting. ggml and MNN CPU paths use group-wise INT4 with runtime dispatch and instead compete on kernels and memory placement. Low-bit (INT2) is explicitly documented only for LiteRT.

### Cited Findings

"NF" means not found in sources read, not confirmed absent.

| Technique | Genie / QNN | LiteRT-LM | ExecuTorch QNN | llama.cpp Hexagon | MNN-LLM | PowerInfer-2 | NeuroPilot | CoreML / MLX | ncnn |
|---|---|---|---|---|---|---|---|---|---|
| Static/AOT graph | Yes, context binaries | Yes, AOT or JIT | Yes, static Llama | No, per-op dispatch | CPU/GPU no; QNN yes | Yes, many precompiled | Yes via LiteRT, AOT or online | Core ML conversion; MLX NF | NF |
| Prefill + decode graphs | Yes (prompt processor / token generator: see Gaps) | Yes, `prefill_128` + decode | Yes, hybrid AR-N + KV | Single dynamic graph | NF | Yes, by batch size | NF | NF | Prefill/Decode in TTS port |
| Weight sharing across graphs | Reference weight sharing | NF explicit | NF | Inherent, one tensor set | LoRA shares base weights | NF | NF | NF | NF |
| Stateful KV | Per-dialog KV manager | KV int8 36 MiB; save/restore | smart_mask / shift_pointer | ggml KV buffers | int4/int8 keys, fp8 values | fixed region cache | NF | Core ML state buffers | Extractor-held KV |
| Weight bits | w4a16, LPBQ | mix of 2/4/8 bit | 16a4w block | Q4_0, Q4_K, Q6_K | int4/int8 | INT4 + INT8 outliers | NF | 4-bit on Core ML | NF |
| mmap | `use-mmap` | XNNPACK weight cache | NF | NF for Hexagon path | Flash hybrid | Neuron cache I/O | NF | NF | NF |
| Heterogeneous | HTP-centric | GPU/NPU fallback, partial delegation | NF | Fallbacks, FLASH_ATTN on CPU | NF | CPU+NPU split | NPU+GPU/CPU fallback | ANE routing | NF |
| Splitting | ctx-bins (4 files in example) | 2 partitions | `num_sharding=4` | No, but >4 GiB problem | NF | Neuron clusters | NF | NF | NF |

**Genie / QNN**
- Config exposes `use-mmap` (should typically be on), `mmap-budget` (~25 MB chunks), `spill-fill-bufsize` (depends on VTCM, must exceed each context binary's requirement), `n-threads` (KV-cache update threads), `poll`, `cpu-mask`, `kv-dim`, `ctx-bins`, `shared-engine` — [Genie JSON](https://docs.qualcomm.com/bundle/publicresource/80-63442-100/topics/json.md).
- Reference weight sharing stores the first context binary's weight blob once for later binaries, DLC workflow only, via `QNN_HTP_CONTEXT_CONFIG_OPTION_REFERENCE_WEIGHT_SHARING_ENABLED` — [QNN docs](https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/htp_multi_soc_dlc.md?facet=Qualcomm+AI+Engine+Direct+SDK&product=1601111740013095).
- HTP multi-graph switching keeps multiple graphs loaded and retained simultaneously — [revision history](https://docs.qualcomm.com/doc/80-63442-50/topic/revision_history.html).
- LPBQ is Qualcomm's alternative encoding for block quantization, supported only on the NPU backend. ONNX Runtime folds DQ→Q→DQ on Gemm weights into a QNN BlockExpansion encoding — [ORT commit](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/91e91186aa8ab67da4785e24c69e11303ddaa25d), [ORT commit 2](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/ecc358f069488a79c5abc16c5ddfbc4bd5b3c771).
- Earlier QNN per-channel INT4 lost heavy accuracy: GSM8K 22.26 vs 75.24 for PowerInfer-2 on Qwen2 7B — [PowerInfer-2](https://arxiv.org/html/2406.06282v3).

**LiteRT-LM / LiteRT QNN**
- E2B on IQ-8275: 2 QNN partitions, fixed shapes, 128-token prefill chunks, 4096 context fixed at compile time, 1.17 GB weight buffer, NPU-only with no CPU graph, version-locked to one QAIRT release. Path is LiteRT `DISPATCH_OP` → Dispatch delegate → QNN HTP → FastRPC — [Qualcomm](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8).
- AOT recommended for large models because on-device compile raises init time and peak memory. FastVLM uses int8 weights with int16 activations to hit int16 kernels. Partial delegation available — [Google](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/).
- XNNPACK weight caching runs the 2.58 GB model with a 607 MB physical footprint on Apple mobile CPUs. MTP drafter and main model share KV and activations on one unit, avoiding cross-processor transfers. PLEs stay out of memory, and encoders load on demand — [Google blog](https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/).

**ExecuTorch QNN**
- 16a4w block with group size 32, LM head 16a8w, 8-bit KV, `num_sharding=4`, optional SeqMSE — [docs 1.0](https://docs.pytorch.org/executorch/1.0/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.html).
- Hybrid mode uses AR-N for prompt then KV for generation, with `smart_mask` (default) or `shift_pointer` updaters. Hybrid export takes 1-4 hours — [docs 1.2](https://docs.pytorch.org/executorch/1.2/_sources/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.md.txt).

**llama.cpp Hexagon**
- Experimental backend from PR #16547, for v73/v75/v79/v81, HTP library with op dispatcher and kernels — [docs mirror](https://git.peisongxiao.com/CS348Project/llama.cpp/src/commit/0775df768f96ee9376a0e57e8c91a01d4c644319/docs/backend/hexagon/developer.md).
- FastRPC fork vs dspqueue: single shared mempool vs per-buffer shared buffers; first-touch invalidate for resident weights vs full DSP L2 flush per batch; lm_head offloadable on the fork. Two HW threads reserved (HMX queue, FastRPC listener). Author-stated 4 GiB user VA limit on V79 causes ~500 ms/token mirror fallback. Call overhead ~21.4 ms vs ~0.1 ms — [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227).

**MNN-LLM**
- Int4/int8 asymmetric weights, LM head prioritized for int8, W4A8 on CPU and W4A16 on GPU, KV keys int4/int8 and values fp8. Embedding in Flash saves ~2.18 GB on Qwen-7B. KV spills to Flash with prefetch. ISA-specific tiling for sdot and i8mm (smmla twice sdot throughput). Big-little-aware balancing, LoRA sharing base weights — [arXiv 2506.10443](https://arxiv.org/html/2506.10443v1).
- Paper vs llama.cpp on Xiaomi 14: CPU prefill 8.6x and decode 2.3x; GPU up to 25.3x and 7.1x. **[AUTHOR]** — [arXiv 2506.10443](https://arxiv.org/html/2506.10443v1).
- Fork release notes say upstream made QNN, HiAI and NeuroPilot configurable plugins, and upstream Hexagon decode fusion and prefill GEMM rewrites are merged to MNN 3.6.2 — [fork releases](https://github.com/pisces312/MNN/releases). I did not confirm this in upstream docs.

**PowerInfer-2**
- Prefill merges tokens into one NPU cluster. Decode splits attention and FFN across CPU and NPU, ~70% NPU at large batch falling to ~50% at batch 1. Precompiles graphs per batch size and hot-neuron ratio, switching ~10 KB graphs asynchronously. INT4 with INT8 outliers. Neuron cache, I/O pipeline. Requires root (mlock). Sparse ops on NPU slower than CPU. 8 Gen 3 / 8+ Gen 1 only — [arXiv 2406.06282](https://arxiv.org/html/2406.06282v3).
- Bamboo-7B in memory: 2.24x llama.cpp, 2.48x MLC-LLM, 1.86x QNN decode; prefill >700 t/s comparable to QNN. TurboSparse-Mixtral-47B 11.68 t/s at 19 GB. **[AUTHOR]** — [same](https://arxiv.org/html/2406.06282v3).

**NeuroPilot (MediaTek)**
- Dimensity 9500 (Vivo X300 Pro), Gemma 3n E2B: >1,600 prefill, 28 decode at 4K context. AOT recommended, as online compile can take over a minute even for Gemma 3 270M. Zero-copy AHardwareBuffer and SSBO interop. "Special" kernels unspecified. **[VENDOR]** — [Google](https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/).

**Apple**
- Core ML stateful KV buffers declared at conversion. If unconfigured, state update ops fall back silently to GPU — [dev.to guide, low authority](https://dev.to/software_mvp-factory/wiring-apples-neural-engine-to-core-mls-stateful-models-24m1).
- iPhone 17 Pro ANE: Qwen3.5-0.8B ~48 t/s, 2B ~27 t/s. E2B 4-bit 33 t/s burst falling to 22 t/s over 10 minutes. **[COMMUNITY]** — [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM).
- LiteRT on iPhone 17 Pro: GPU 2,878 / 56, CPU 532 / 25 — [LiteRT-LM](https://developers.google.com/edge/litert-lm/overview).
- MLX has rotating KV cache and prompt cache files, no paged attention (Mac-focused paper) — [arXiv 2511.05502](https://arxiv.org/pdf/2511.05502).

**ncnn**
- The only LLM-adjacent evidence is a Qwen3-TTS port with prefill/decode stages and Extractor-held KV, x86 CPU results only, Vulkan listed as future work — [ncnn #6867](https://github.com/Tencent/ncnn/discussions/6867).

### Inferences
- Qualcomm's own `GENIEX_LLAMACPP` q4_0 packaging alongside a stated Genie deprecation suggests ggml-format weights are becoming a supported path on its platform, though the card does not say which processor those rows use.
- LPBQ appears to be the industry answer to the per-channel INT4 accuracy loss PowerInfer-2 reported, making a 2024 "QNN is inaccurate" critique likely dated. No source directly states this link.
- Prefer int8 LM head across engines (MNN, ExecuTorch) for accuracy, at a bandwidth cost that scales with vocabulary × dim per token. On large-vocab Gemma, weight tying with embedding saves storage only, not LM-head read traffic.

### Gaps
- HMX/matmul scheduling details are not documented publicly for Genie/QNN. Only #26227 and arXiv 2605.27435 describe HMX, VTCM and tiling for ggml.
- I found no primary doc for Genie's prompt-processor/token-generator weight sharing mechanics inside a single context binary. The split itself is widely described but not sourced here.
- INT2 on any QNN/HTP path: no source found. LiteRT's 2-bit statement is not tied to its NPU file.
- Weight-tying specifics not found for any engine.

## Q3. Why QNN/Genie differs from a ggml-style dynamic graph backend in memory bandwidth and speed

### Takeaway
Measured evidence supports overhead and scheduling explanations, not a direct bandwidth one. ggml dispatches per op and pays scheduling, cache-maintenance and fallback costs, while QNN compiles a whole-graph plan ahead of time. However, published decode is close across stacks (Genie Qwen3-4B ~28.6 t/s; LiteRT QNN E2B ~32 t/s; ggml E2B ~25-27 t/s and your ~30 t/s), though models and chips differ, so a shared DDR bound seems to dominate decode. No source measures DDR traffic for both.

### Cited Findings
- Decode is memory-bound and prefill compute-bound on edge devices — [MNN-LLM](https://arxiv.org/pdf/2506.10443).
- ggml scheduling tax: RMS_NORM and ADD call time is 8-22x op time. Communication rises to 9.9-13.0% of NPU time in decode vs 0.2-3.2% in prefill. FLASH_ATTN_EXT has no NPU support and runs on CPU — [arXiv 2605.27435](https://arxiv.org/html/2605.27435).
- NPU decode advantage comes from VTCM and DMA streaming weights that are never reused. 8 MiB VTCM limits tiling for large prefill matrices, and CPU cache reuse helps prefill — [same](https://arxiv.org/html/2605.27435).
- Coherency: official dspqueue does a role-blind full DSP L2 flush/invalidate per batch, whereas the fork invalidates resident weights once. The fork as a whole (coherency is one of several differences, alongside mempool and control path) moved Qwen3.5-2B decode 14.95→26.99 t/s (+80.5%), but Qwen1.5 and large models regressed — [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227).
- Same HVX/HMX kernels with different plumbing produce such swings, so runtime plumbing is a first-order variable in ggml backends — [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227).
- QNN side: static shapes, precompiled binaries, 128-token chunks, context fixed at compile time. Burst vote alone moved decode ~16 → ~28 t/s on IQ-8275 — [Qualcomm](https://dragonwingdocs.qualcomm.com/blogs/gemma-litert-lm-on-iq8). Compile-time planning uses a spill-fill buffer tied to VTCM — [Genie JSON](https://docs.qualcomm.com/bundle/publicresource/80-63442-100/topics/json.md).
- QNN costs: long export (1-4 h on ExecuTorch hybrid) and high host RAM, plus rigid context — [ExecuTorch docs](https://docs.pytorch.org/executorch/1.2/_sources/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.md.txt). Genie decode falls with larger context on GENIEX_QAIRT: 23.06 (512) → 15.19 (4096) for Qwen3-4B on Gen 5 — [AI Hub](https://huggingface.co/qualcomm/Qwen3-4B).
- Prefill gap is wide on Hexagon: LiteRT QNN ~3,747 t/s vs ggml fork 668 t/s for E2B, on different chips and quant, with the QNN number on the weaker v75 — [HF card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm), [#26227](https://github.com/ggml-org/llama.cpp/discussions/26227).

### Inferences
- **Decode:** compute is cheap relative to fetch, so both stacks converge to bytes-per-token ÷ bandwidth. QNN's likely advantages are lower-overhead streaming and a smaller or more compressible byte budget, not a faster bus. Testing this needs DDR counters on identical weights.
- **Prefill:** a whole-graph compiler can tile a 128-token batch across layers and reuse VTCM with planned spill-fill. Per-op dispatch cannot schedule across ops, which plausibly explains the ~5x gap, though quant and device differ.
- **Context dependence:** QNN decode slowing with context suggests KV traffic or attention cost scales there. Compare with 36 MiB int8 KV for E2B at 4096 — small against 1.17 GB weights, so KV bytes alone seem unlikely to explain it for E2B-size models.
- **Practical implication for 41 GB/s:** at fixed bandwidth, cutting weight bytes per token beats scheduling work once overhead is removed. Ceiling estimates: 1.37 GB/token → 30 t/s; 1.0 GB → 41 t/s; 0.7 GB → ~58 t/s. These assume all weights are read once per token with no KV cost, so treat them as upper bounds.

### Gaps
- No published side-by-side DDR bandwidth for Genie vs ggml on the same model and weights. This is the core gap, and a direct `simpleperf`/DDR counter experiment would settle it.
- Whether QNN inline-decompresses LPBQ differently or stores 4-bit weights more efficiently in memory is not public.
- arXiv 2605.27435 covers v75 and does not test QNN.

## Q4. Concurrent multi-stream requests on a phone

### Takeaway
Only llama.cpp documents true parallel-sequence batching. Genie documents sharing one engine across dialogs but not whether they run simultaneously. LiteRT-LM documents session save/restore, not parallel execution. I found no published phone-scale multi-stream throughput for any engine.

### Cited Findings
- **Genie:** engine sharing lets multiple dialogs (Eaglet, SSD, Basic, mixable) use one engine instance on the same context binaries with per-dialog KV manager, IO buffers and LoRA, on Qnn-HTP only. It reduces memory. Concurrency is not specified — [Genie docs](https://docs.qualcomm.com/doc/80-63442-100/topic/engine_sharing.html). `shared-engine` needs `allow-async-init` — [JSON](https://docs.qualcomm.com/bundle/publicresource/80-63442-100/topics/json.md).
- **llama.cpp:** `-np/--parallel` sets parallel sequences; features list parallel decoding and continuous batching — [server README copy](https://huggingface.co/Aliguinga01/rule_violation2/blob/main/llama.cpp/tools/server/README.md). `--kv-unified` shares one KV buffer across sequences — [dev.to](https://dev.to/multigrid/serving-parallel-requests-with-llamacpps-server-slots-c90). Defaults changed across versions — [issue #17989](https://github.com/ggml-org/llama.cpp/issues/17989).
- **LiteRT-LM:** KV state can be serialized and restored across sessions, skipping prefill. Session cloning and multi-session are not covered on the page read — [Google blog](https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/).
- **PowerInfer-2:** batch-size-specific NPU graphs exist, but the batches are prompt or neuron-cluster batches, not user streams — [arXiv 2406.06282](https://arxiv.org/html/2406.06282v3).
- **MNN-LLM:** LoRA adapters share base weights, with no multi-stream scheme found — [arXiv 2506.10443](https://arxiv.org/html/2506.10443v1).
- **ExecuTorch, NeuroPilot, MLX on iPhone, ncnn:** no phone multi-stream material found in this search.
- Larger batches raise arithmetic intensity and allow weight reuse in the prefill stage — [arXiv 2605.27435](https://arxiv.org/html/2605.27435).

### Inferences
- **Batching is the only route found that scales aggregate decode against a weight-read wall.** With B streams in one pass, weights are read once per step while KV and activation traffic scale with B. So at 41 GB/s, aggregate could approach B × 30 t/s minus KV and compute costs, whereas per-stream round-robin stays about 30 t/s total. This is unmeasured reasoning for your setup.
- **Static stacks need a batch-specific graph.** A QNN decode graph compiled for batch 1 implies time-slicing (per-dialog KV, serial execution), so Genie engine sharing likely saves memory rather than increases throughput. Docs do not confirm this.
- **Ggml's dynamic shapes are an actual advantage here,** though VTCM tiling, HMX scheduling and per-batch flush costs on Hexagon may limit it, and 2605.27435's prefill finding suggests CPU cache may catch up at larger batches.
- Multi-stream KV budget is small relative to weights at E2B scale (36 MiB per 4096-context stream at int8), so memory should not be the first bound for 2-4 streams.

### Gaps
- No published aggregate t/s for N concurrent streams on any phone NPU, CPU or GPU, and no doc on whether two Genie dialogs can execute at once.
- llama.cpp Hexagon behaviour with `-np > 1` and thermal impact are not published. Official `llama_decode` batch semantics were not verified from primary headers in this search.
