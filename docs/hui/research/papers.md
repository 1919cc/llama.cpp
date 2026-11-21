# Accelerating LLM decode toward 500-1000 tokens/s on a phone: literature notes

Evidence labels used throughout:

- **[MEASURED]** run on real hardware.
- **[SIM]** simulator or synthesis, no silicon.
- **[MODEL]** analytical model.
- **[ARITH]** my own first-order arithmetic from cited inputs. It is an upper bound, not a prediction.
- **[VENDOR]** self-reported by the company.
- **[SNIPPET]** seen only in a search-result summary, not checked against the full paper.

Most table values were extracted from arXiv HTML by an automated summarizer. Check any number against the paper before quoting it as final.

## Weight-read reduction: low-bit quantization, LUT/VQ, pruning, low-rank, compressed weights

### Takeaway
2 bits per weight (bpw) is the lowest rate with solid quality data. It costs a clear but moderate perplexity increase on 7B+ models, and I found no 2-bit data at 2B scale. On weight traffic alone, 500 t/s at 40 GB/s and 2B needs 0.32 bpw. No paper found comes near that, so quantization gives about 2-4x over 4-bit and cannot reach the target alone.

### Cited Findings

**Quality at 2 bits and below**

| Method | Model | Baseline PPL or accuracy | Low-bit result | Evidence | Source |
|---|---|---|---|---|---|
| WINT2/AFP16 | Llama-2-7B | acc 62.7%, PPL 5.47 | acc 56.4%, PPL 7.68 | Measured model eval | [LUT Tensor Core](https://arxiv.org/html/2408.06003) |
| BitNet b1.58 (natively trained, INT2 storage) | 3B | Llama-3B FP16 acc 49.7% | acc 49.4% | Different models compared, not the same model quantized | [LUT Tensor Core](https://arxiv.org/html/2408.06003) |
| EVA VQ | Llama-2-7B | PPL 5.12 | 4-bit 5.43, 2-bit 6.69 | WikiText-2 | [EVA, ISCA 2026](https://arxiv.org/html/2605.24144) |
| EVA VQ | Llama-2-13B | PPL 4.57 | 4-bit 4.76, 2-bit 5.63 | WikiText-2 | [EVA](https://arxiv.org/html/2605.24144) |
| GLVQ-32D / GLVQ-8D | Llama-2-7B | PPL 5.12 | 2-bit 5.41 / 5.69 | WikiText-2 | [GLVQ](https://arxiv.org/html/2510.20984) |
| GLVQ-32D / GLVQ-8D | Llama-2-70B | PPL 3.12 | 2-bit 3.36 / 3.62 | WikiText-2 | [GLVQ](https://arxiv.org/html/2510.20984) |
| INT2 AWQ (no VQ) | Llama-2-7B | PPL 5.12 | PPL 2.2e5 (broken) | Shows that naive 2-bit fails | [EVA](https://arxiv.org/html/2605.24144) |
| OmniQuant 2-bit | Llama-1-7B | PPL 5.68 | PPL 15.5 | Baseline in GLVQ | [GLVQ](https://arxiv.org/html/2510.20984) |

- Quantizer choice can matter more than bit width on small models. Llama 3.2 1B at W4A16 with QNN per-channel scored MATH500 2.1, GSM8K 3.4, PPL 28.99. AWQ per-group 4-bit scored 15.9, 32.6, 19.42. Qwen2.5-1.5B Q4_0-style reached PPL about 10.2 versus 9.80 for FP16 — [Scaling LLM Test-Time Compute, EuroSys '26](https://arxiv.org/html/2509.23324).
- bitnet.cpp's "100% lossless" means kernel outputs exactly match the FP32 kernel for the same BitNet model (1,000 WildChat prompts, 700M model). It does not mean quality equal to a higher-precision model. The paper also calls its benchmark models "dummy setups" — [bitnet.cpp](https://arxiv.org/html/2410.16144).
- A QAT claim exists that 2-bit beats 4-bit at the same memory budget, but I saw it only as a statement in the LUT Tensor Core search snippet, and that paper's full text has no direct 2-bit vs 4-bit table — [LUT Tensor Core](https://arxiv.org/pdf/2408.06003) [SNIPPET].

**Speed from low-bit kernels**

- **bitnet.cpp on Apple M2, unlimited threads [MEASURED, laptop, not phone]** (t/s, bitnet.cpp vs llama.cpp) — [bitnet.cpp](https://arxiv.org/html/2410.16144):
  - 125M: 593 vs 434
  - 350M: 282 vs 187
  - 700M: 194 vs 114
  - 1B: 169 vs 86
  - 1.5B: 119 vs 61
  - 2.5B: 96 vs 40
  - At 2 threads: 1B 79, 2.5B 37
- **Vec-LUT [MEASURED per abstract]:** a unified LUT across parallel tokens, up to 4.2x over baselines across 5 edge devices and 3 LLMs. Directly relevant to batched streams and verification. Absolute t/s was not on the abstract page — [Vec-LUT](https://arxiv.org/abs/2512.06443).
- **EVA [SIM]:** does VQ dot products against the codebook, converting GEMV to GEMM.
  - Speedup is 11.17x over FIGLUT and about 32x over SA, ANT and FIGNA, on FC layers only. No t/s reported.
  - 28 nm, 500 MHz, 528 KB SRAM, Verilog plus simulator, 64 GB/s DDR4 assumed — [EVA](https://arxiv.org/html/2605.24144).
  - The result needs new silicon and is not available on Hexagon.
- **LUT Tensor Core [SIM]:** 2.06-5.51x. BitNet 3B decode goes from 21.70 ms to 11.41 ms (LUT-4X) or 7.47 ms (LUT-8X) on a simulated A100 — [LUT Tensor Core](https://arxiv.org/html/2408.06003).
- **ELUTQ:** 2-bit Llama 3.1-8B, 1.5x over AWQ on RTX 3090 — [ELUTQ](https://arxiv.org/pdf/2510.19482) [SNIPPET].
- **On Hexagon today:** Q4_0-style weights cost 4.5 bpw with a 2-byte scale per group of 32. The LUT INT4-to-FP16 path gives up to 19x on mixed-precision GEMM over the baseline, about 27% slower than a no-dequant upper bound [MEASURED] — [EuroSys '26](https://arxiv.org/html/2509.23324).

**Pruning, sparsity and attention approximation**

- **DIP plus cache-aware masking [SIM, A18 configuration, DRAM and flash only, INT4]:**
  - Phi-3-Medium: 46% lower DRAM footprint, and 40/55/93% throughput at PPL +0.1/+0.2/+0.5.
  - Absolute: 0.29 to 0.56 t/s. Llama-3-8B: 0.57 to 0.89. Mistral-7B: 0.54 to 1.39.
  - At 50% MLP sparsity on Phi-3-Medium: PPL 4.29 to 5.52 (5.01 with LoRA), MMLU 78.14 to 75.54.
  - These gains reflect a model larger than DRAM paging from flash, so they do not transfer directly to a fully resident 2B model — [DIP](https://arxiv.org/html/2412.01380v2).
- **HyperAttention:** ChatGLM2 at 32k context, 50% faster with PPL 5.6 to 6.3. Single-layer gains are 5x in the arXiv abstract and 22x in the ICLR version, an unresolved discrepancy — [arXiv](https://arxiv.org/abs/2310.05869v2), [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/ab5aa940590399350401c57cdf52ce78-Paper-Conference.pdf).
  - It targets attention compute and KV access at long context, and is not a way to cut weight reads.

### Inferences
- **[ARITH] Weights-only ceiling at 40 GB/s for 2B:**

| bpw | Bytes per token | Ceiling (t/s) |
|---|---|---|
| 4.5 (Q4_0) | 1.125 GB | 36 |
| 4 | 1.0 GB | 40 |
| 2 | 0.5 GB | 80 |
| 1.58 (entropy-coded ternary, hypothetical) | 0.395 GB | 101 |
| 1 | 0.25 GB | 160 |
| 0.64 | 0.16 GB | 250 |
| 0.32 | 0.08 GB | 500 |

- Moving from Q4_0 to a 2-bit VQ or BitNet format is the largest single lever available: about 2.25x. Expect roughly PPL +0.3 to +1.5 on 7B-13B models, depending on method. Degradation at 2B is probably worse than at 7B, but that is an extrapolation with no direct data.
- BitNet needs native training and has little model selection. VQ retrofits existing models but needs a decode path. EVA's trick shows that the dot-product form avoids reconstruction, though only with dedicated hardware. A HMX version of this is unproven.
- Activation sparsity is limited to under 2x even on the MLP share at 50% sparsity. Masks differ per token, so a batch reads the union of masks. That should erode the gain under speculation or multi-stream batching, and I found no paper measuring this.

### Gaps
- No quality data for ≤2-bit QAT or VQ on an actual 2B model, and no credible quality-preserving method below 1 bpw at this scale.
- No measured phone result for 2-bit VQ on Hexagon.
- Low-rank decomposition and structured weight pruning for decode throughput were not covered with numbers in this round.
- Weight compression with hardware decompression: no Hexagon or UBWC-for-weights source found. LLM.265 appears only as a comparison baseline inside EVA, and I did not read it.

## Speculation: assisted generation, Medusa, EAGLE-3, lookahead, parallel decoding, Skeleton-of-Thought

### Takeaway
On general text, realistic learned speculation yields about 3 tokens per step and 1.7-3x wall-clock at batch 1. N-gram or retrieval methods give about 2 tokens per step on chat and only exceed 4-7 on rewriting or high-overlap tasks. The widely quoted 6.5x is a single best case, and independent runs land well below the authors' numbers.

### Cited Findings

**Tokens per step by workload (accepted + 1 bonus) [MEASURED, H100, vLLM]** — [Speculative Decoding: Performance or Illusion?](https://arxiv.org/html/2601.11580)

| Dataset | n-gram | EAGLE | EAGLE-3 |
|---|---|---|---|
| InstructCoder (rewrite, repeated text) | 7.27 | 4.24 | 4.18 |
| CNN/DailyMail | 2.33 | 2.32 | 3.08 |
| ShareGPT (general chat) | 2.02 | 2.82 | 3.06 |
| GSM8K | 1.41 | 2.56 | 3.02 |
| GPQA-Main | 4.20 | n/a | 2.78 |
| AIME22-24 | 3.53 | n/a | 2.62 |

- **Wall-clock in the same study [MEASURED]** — [same source](https://arxiv.org/html/2601.11580):
  - Llama3.1-8B GSM8K EAGLE: 1.73x at batch 1, falling to 1.21x at batch 128.
  - Llama3.1-8B ShareGPT: 1.68x to 1.61x (batch 32).
  - Llama3-70B ShareGPT: 1.96x to 1.72x. Tree k=21 gives 2.03x at batch 1.
  - Qwen3-8B-Thinking: EAGLE-3 1.64-1.80x, n-gram 1.50-1.58x.
  - InstructCoder n-gram: about 2.1x, oracle about 2.75x.
  - Trees help only at batch 1 and fall below 1x at batch 64.
  - Verification is 42-95% of runtime. EAGLE drafting is 12-20% at batch 1. A separate small draft model is 21-47%, and it raises KV per token 1.77x on Qwen3-8B.
- **EAGLE-3 authors' numbers (hardware unstated for main table)** — [EAGLE-3](https://arxiv.org/html/2503.01840):
  - 8B T=0: mean 4.44x (τ 6.23). 8B T=1: mean 3.45x (τ 4.92).
  - The 6.47x peak is Vicuna 13B, T=0, HumanEval (τ 7.54).
  - H100 SGLang, batch 1, 8B MT-Bench [MEASURED]: 158 to 373 t/s (2.36x), with chain length 3 at larger batches.
  - Multi-batch: 1.82x at batch 4, 1.38x at batch 64 (SGLang); 1.68x at batch 4, 1.01x at batch 56 (vLLM). The paper's caption and text disagree on vLLM hardware.
- **Independent replication:** Llama-3.1-8B EAGLE-3 with K=7, KL baseline — T=0: τ 3.75-4.82, 2.60-3.30x. T=1: τ 3.39-4.31, 2.29-3.00x. LK losses add only about 1-9% τ — [LK Losses](https://arxiv.org/html/2602.23881). This contradicts EAGLE-3's own τ of 6.23 (T=0) and 4.92 (T=1), the discrepancy being unresolved in what I read. MoE models fare worse: gpt-oss-120b 1.19-1.82x.
- **Mobile, the most relevant result: sd.npu [MEASURED, Redmi K60 Pro, Snapdragon 8 Gen 2, summary task, ms/token]** — [sd.npu](https://arxiv.org/html/2510.15312v4):

| Model | NPU | +SD | Full sd.npu | Equivalent t/s |
|---|---|---|---|---|
| Qwen2.5-0.5B | 24.46 | 18.75 | 7.17 | 139 |
| Qwen2.5-1.5B | 57.41 | 35.04 | 15.08 | 66 |
| Llama 3.2-3B | 71.71 | 47.93 | 41.62 | 24 |

- sd.npu continued:
  - Retrieval-based (SAM), no draft model. Speedup 1.06-3.81x, and 5.25-8.15x vs mllm(NPU) on OnePlus 13 summary. Energy 1.07-4.71x.
  - Accepted length on A100 Qwen2.5-1.5B summary: 2.91 with PLD, 4.21 with SAM. Strongest on summary, where context overlaps output.
  - States decode on the NPU is memory-bound, and verification is near-constant cost up to saturation.
  - Task-selected and retrieval-friendly, so not a general-text number.
- **Medusa [MEASURED, Vicuna 7B/13B]:** Medusa-1 2.18-2.33x, Medusa-2 2.83x with joint training (later paper version 2.3-2.8x) — [Medusa](https://arxiv.org/html/2401.10774v3), [third-party summary of the same](https://www.alphaxiv.org/overview/2401.10774v3).
- **LayerSkip [MEASURED]:** self-speculation up to 2.16x (CNN/DM summarization), 1.82x (coding), 2.0x (TOPv2). Requires training with layer dropout — [LayerSkip, ACL 2024](https://preview.aclanthology.org/setup/2024.acl-long.681).
- **Lookahead n-gram from Jacobi trajectories [MEASURED, WAN split inference, about 78 ms RTT]:** tokens per step for code 1.43-1.57, conversational 1.12-1.17, creative 1.06-1.12. Break-even about 1.15-1.20. Speedups are 0.91-1.30x, but WAN setup and only 2 prompts per category limit comparability — [Split inference](https://arxiv.org/html/2602.16760).
- **Original vLLM case:** code refactor 1.84x, general chat 0.99x, seen only in a secondary source — [Glukhov](https://glukhov.org/llm-performance/optimization/speculative-decoding/) [SNIPPET; page returned 403 on fetch].
- **Jacobi Forcing:** HumanEval 3.86x, accuracy 87.8% to 83.5%, authors' own number — [OpenReview PDF](https://openreview.net/pdf/1d090e3fa4001d8e55ecf12b0ea2f54b9eb055f3.pdf) [SNIPPET].
- **Skeleton-of-Thought:** up to 2.39x across 12 LLMs, not suited to all question types so requires a router. The MSR 3.72x figure is one demo question — [SoT](https://arxiv.org/pdf/2307.15337), [MSR](https://www.microsoft.com/en-us/research/?p=983583).
- **Limit:** LP-Spec simulated PIM reports 13.21x over mobile NPU, but requires PIM hardware — [LP-Spec](https://arxiv.org/abs/2508.07227).

### Inferences
- Speculation is especially suited to a memory-bound phone. A verify step reads the weights once however many drafted tokens it contains, and spends compute that is otherwise idle. Tokens per step (about 3 on chat) should translate better to speedup than the 1.7-2x on H100, where verification compute dominates. sd.npu's near-constant-cost claim supports this but does not prove it for general text.
- Plan on **S ≈ 2-3 for general text and ≥4 only for rewriting or summary**. Using S ≥ 4 for the headline would overstate the product. Repeated-text numbers must be reported separately from chat.
- MoE models fare worse under EAGLE-3 per LK Losses, so MoE and speculation may not multiply cleanly.
- Speculation does not change output distribution when verified correctly, so its quality cost is zero. Its costs are memory (EAGLE-3 3-5% weights), draft compute and variance. Jacobi Forcing, SoT and block diffusion are different, because they change output.

### Gaps
- No measured EAGLE-3 or Medusa on a phone NPU found.
- No acceptance data for draft heads on a 1-2B target. The published 8B-13B acceptance may not transfer, and I would expect heads relative to a smaller target to gain less, which is unverified.
- Original Lookahead Decoding (LMSYS) numbers not retrieved.

## Alternative architectures: diffusion LLMs, RWKV/state-space, very small models, early exit, layer skipping, distillation

### Takeaway
Diffusion models reach 1000+ t/s only on datacenter GPUs. Measured phone results are about 8 t/s on an 8B model. RWKV removes the KV but not the weight wall. Small models are the only measured route above 500 t/s, and they pay directly in quality.

### Cited Findings

**Diffusion**

- **Mercury Coder [independently measured by Artificial Analysis per the paper; H100]:** Mini 1109 t/s, Small 737 t/s. Claimed comparable quality, second on Copilot Arena quality — [Mercury](https://arxiv.org/abs/2506.17298). Closed model.
- **Mercury 2 [VENDOR]:** about 1000 t/s — [Business Wire](https://www.businesswire.com/news/home/20260224034496/en).
- **Gemini Diffusion:** demo 1000-2000 t/s, no paper. One analysis cites 1,479 t/s excluding 0.84 s overhead — [Simon Willison](https://simonwillison.net/b/8700) [SNIPPET].
- **DiffusionGemma quality (secondary):** AIME 2026 69.1% vs Gemma 4 88.3% and Mercury 2 90%. GPQA 73.2% vs Mercury 2 77% — [Decrypt](https://decrypt.co/371722/inception-labs-mercury-2-ai-beats-googles-diffusiongemma). So the open diffusion model here lost 19 AIME points to its AR sibling.
- **Fast-dLLM [MEASURED, A100, batch 1, block 32, threshold 0.9]:** ICLR 2026 — [Fast-dLLM](https://arxiv.org/html/2505.22618v1):
  - LLaDA GSM8K-256: 6.7 to 54.4 t/s, accuracy 79.3 to 78.5.
  - HumanEval-256: 30.5 to 114.1 t/s.
  - Headline 27.6x is a best case with 8-shot prefill and length 1024 (accuracy 77.3 to 76.0).
  - Even on a server GPU it reached 35-114 t/s, far from Mercury.
- **llada.cpp on phone [MEASURED, Snapdragon 8 Elite, LLaDA-8B Q4_0]** — [Wang et al. 2026](https://arxiv.org/html/2606.13740):
  - One denoising pass, 32 tokens: CPU 4239.6 ms, NPU 478.1 ms. 128 tokens: 1531.6 ms on NPU.
  - 128-token GSM8K: CPU 2996 s, NPU 341 s, full pipeline **16.1 s, about 8 t/s** [ARITH on measured].
  - Up to 3.9x over an equal-size AR model on its own task, but Llama is faster at 128 tokens on SM8650.
  - Quality (200-sample subsets): GSM8K 43.5 vs Llama NPU 31.0, ARC-C 85.0 vs 78.5, BoolQ 80.5 vs 77.5, **HellaSwag 49.5 vs 60.0**.
  - Block-wise decoding costs 3.5-6.5 points vs vanilla dLLM.

**RWKV and state-space**

- RWKV-7 1.5B: 25.2 t/s on iPhone 17 Pro, community self-report, FFN and LM head weight-only int8 — [HF model card](https://huggingface.co/mlboydaisuke/RWKV7-Goose-1.5B-CoreAI).
- RWKV-7 2.9B was slower at first-turn t/s than gemma3:4b on a mini-PC, a single informal test — [InsiderLLM](https://insiderllm.com/guides/beyond-transformers-5-architectures/).
- No head-to-head quality benchmark against same-size Transformers found.

**Very small models**

- **bitnet.cpp, trained status aside [MEASURED M2]:** 125M 593 t/s, 350M 282 t/s — [bitnet.cpp](https://arxiv.org/html/2410.16144). These are benchmark models, so no quality applies.
- **Llama 3.2 1B on QNN/GENIE [VENDOR measured on QRD]:** w4a16 65.2 t/s on 8 Elite Gen 5, 67.2 on 8 Elite, 83.6 on X2 Elite, at 4096 context. w4 runs about half: 36.3 and 34.5 — [Qualcomm HF](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct). Newer SoC gains nothing, consistent with a DRAM wall. The search snippet showed different figures for the same card (53.6, 64), so numbers change between card revisions.
- **Quality floor:**
  - Llama 3.2 1B at 4-bit AWQ: MATH500 15.9, GSM8K 32.6 — [EuroSys '26](https://arxiv.org/html/2509.23324).
  - Qwen2.5-1.5B: WinoGrande about 62.6, MMLU about 35.5 — [same](https://arxiv.org/html/2509.23324).
  - Gemma 3 1B: MMLU 26.5, GSM8K 1.36 [SNIPPET, suspiciously low, verify] — [Ollama](https://registry.ollama.ai/library/gemma3:1b).
  - Gemma 3 270M: only a qualitative QAT statement found — [LM Studio](https://www.lmstudio.ai/google/gemma-3-270m).

**Early exit, layer skipping, distillation**

- TIDE per-token early exit: DeepSeek R1 Distill 8B on A100, +6.6% throughput — [TIDE](https://papers.cool/arxiv/2603.21365) [SNIPPET].
- LayerSkip Llama 2 13B: 43.64 vs 28.38 t/s (1.54x) on A100 — [third-party listing](https://featherless.ai/models/facebook/layerskip-llama2-13B) [SNIPPET].

### Inferences
- **Diffusion is not a phone route today.** H100 has about 84x the bandwidth of 40 GB/s (3.35 TB/s vs 0.04 TB/s). Mercury's 1109 t/s scales to about 13 t/s at the same size if purely memory-bound [ARITH], and the phone compute gap makes it no better. The 8B llada.cpp pass (32 tokens, 478 ms) is about 67 token-positions per second and looks compute-bound, since 4.5 GB at 40 GB/s would be about 112 ms. Even ideally scaling to 2B, this needs many passes per token. Mercury's model size and step count were not disclosed in what I read, so this scaling is approximate.
- Block diffusion does have the right structure: weights read once per pass, with many tokens per pass. Credible phone-speed use would require an ultra-low-step 1-2B dLLM. None has been published with quality data, and its accuracy profile is uneven (better GSM8K, worse HellaSwag).
- RWKV removes KV traffic, which has real value at long context, but it reads all weights per token, so at 2B it hits the same ceilings. It will not yield 500 t/s.
- Early exit saves little in a measured post-training setting (6.6%). Its better form is self-speculation, already counted under speculation. It should not be multiplied again.
- Low-bit 125M-350M CPU data suggests compute becomes the limit below about 1B on CPU. At 593 t/s × 31 MB, 125M moves only about 18 GB/s; 7B at 52 t/s × 1.75 GB moves about 92 GB/s [ARITH, weights only, assuming 2 bpw].

### Gaps
- No distillation paper found with a measured t/s vs quality trade-off for sub-1B phone deployment.
- No 0.1-0.5B accuracy benchmark table obtained (SmolLM2, Gemma 3 270M). This should be checked directly with model cards.
- Mamba LLM on-device: no text-LLM evidence. The only search hit was brain-signal decoding.
- Energy data for llada.cpp is in figures only.

## Memory-wall hardware: on-chip memory, TCM, NPU DMA, Hexagon HMX/VTCM, LPDDR limits

### Takeaway
Decode on a phone is set by DRAM bandwidth divided by bytes read per token, and published analyses state this directly. On-chip memory (8 MiB TCM on Hexagon) is about 1.6% of a 500 MB model and cannot hold weights. It works as a streaming buffer, not a cache.

### Cited Findings
- **First-order bound [ARITH in the source]:** 8B INT4 at about 77 GB/s LPDDR5X gives about 19 t/s, weights only, "squarely in the range observed" on flagship phones. The paper says LPDDR bandwidth rather than NPU TOPS sets the rate, and measured rates fall below this by KV reads and overhead — [Patel & Singh 2026](https://arxiv.org/pdf/2608.28048). Survey-grade preprint, not a measurement.
- **LIMINAL [MODEL, 7.6% MAPE validated on 8xH100]:** decode time = max(compute, memory) + exposed time. Traffic scales with **active** parameters, so GPT-OSS-120B (about 5B active) gets the highest UTPS and dense Llama3-405B the lowest. An 8x bandwidth rise (4 to 30 TB/s) yields only about 1.1-4.2x, and above 10,000 t/s requires algorithmic advances too — [LIMINAL](https://arxiv.org/html/2507.14397). Server-scale; no phone results.
- **Hexagon specifics [paper-reported]:** 1 MiB L2, 8 MiB TCM. HMX and vector scatter/gather can only access TCM. 32×32 FP16 tile (2 KiB). HMX FP16 GEMM about 12,032 GFLOPS vs about 33 GFLOPS for one HVX thread. **DMA reads exceed 60 GB/s while HVX reads stay below 30 GB/s (26 GB/s in table)**. 32-bit address space, older NPUs limited to 2 GiB. 1.5B stays under 5 W on OnePlus 12. CPU-side logits take about 50% of compute time at batch 16 — [EuroSys '26](https://arxiv.org/html/2509.23324).
- NPU address space was a real blocker: 8 Gen 2 cannot host 3B+ in that study, and llada.cpp hit remapping overhead — [EuroSys '26](https://arxiv.org/html/2509.23324), [llada.cpp](https://arxiv.org/html/2606.13740).
- TCM round trips to DRAM are a named bottleneck that the compiler stack targets — [Hexagon-MLIR](https://www.arxiv.org/pdf/2602.19762) [SNIPPET].
- sd.npu reports 75% of NPU matrix cycles underutilized in baseline decode — [sd.npu](https://arxiv.org/html/2510.15312v4).
- **Bandwidth:** 8 Elite Gen 5 spec peak 84.8 GB/s (LPDDR5X-5300 MHz, 4×16-bit) — [Notebookcheck](https://www.notebookcheck.net/Qualcomm-Snapdragon-8-Elite-Gen-5-Processor-Benchmarks-and-Specs.1123169.0.html). A spec, not a measurement.
- **LPDDR6:** JEDEC starts at 10.67 Gb/s, scaling to 14.4 Gb/s. SK hynix: 38.4 GB/s per 24-bit chip at 14.4 Gb/s, not the naive 43.2 — [Guru3D](https://www.guru3d.com/story/sk-hynix-previews-144-gb-s-lpddr6-samsung-targets-128/). No confirmed phone SoC number found. [Search summary]
- **KV:** 144 KiB per token for Qwen3-8B per the speculation study (storage format not stated) — [Performance or Illusion](https://arxiv.org/html/2601.11580).
- **PIM:** LPDDR5 PIM-AI claims 25-45% more QPS vs mobile SoCs in mobile scenarios — [PIM-AI](https://arxiv.org/pdf/2411.17309v1) [SNIPPET]. LP-Spec 13.21x performance, 7.56x energy; abstract does not say whether results are simulated (architecture proposal, so almost certainly not silicon) — [LP-Spec](https://arxiv.org/abs/2508.07227).

### Inferences
- **Caching weights on-chip gives under 2%.** Forward passes read every weight once in sequence, so an LRU cache of a sequential scan has near-zero hit rate unless pinned. 8 MiB pinned out of 500 MB saves 1.6% of traffic [ARITH]. TCM value is in prefetch and streaming, so that DMA stays at full bandwidth while HMX runs. That is why tile layout and DMA scheduling matter, even though they do not raise the ceiling.
- **Your 40 GB/s looks close to real usable DMA.** Llama 3.2 1B at 67 t/s × about 0.5 GB ≈ 34 GB/s of weight traffic [ARITH, assuming ≈1B params at 4 bits; part of the model is 8-bit, so this is a floor]. That suggests GENIE w4a16 is already near the wall. In contrast, sd.npu's baseline 1.5B at 57 ms and about 0.84 GB uses about 15 GB/s on 8 Gen 2, so there is headroom in that stack. HVX-path reads (<30 GB/s) can't reach 40 GB/s, so a DMA path is required.
- **KV can consume the budget.** If a 2B model has something like 72 KiB per token (half the 8B figure, a guess), 1000 tokens of context costs about 72 MB per step, versus an 80 MB budget at 500 t/s. KV traffic reduces all ceilings at any context beyond short prompts, and low-bit KV is not optional. Hedged: per-token KV size at 2B is not sourced.
- **Compute looks sufficient at peak, not sustained.** 2B × about 2 FLOP/param ≈ 4 GFLOP per token, so 500 t/s ≈ 2 TFLOP/s FP16 against the reported 12 TFLOP/s peak [ARITH, rough]. However, thermal limits, per-call FastRPC and CPU-side logits may bite first. At batch 16, logits were about 50% of compute time, so logits must move onto the NPU.
- LPDDR6 likely buys well under one order of magnitude over LPDDR5X-class (my estimate; no phone SoC figure is confirmed), so it helps but does not close a 6-12x gap.

### Gaps
- No published direct measurement of phone DRAM t/s vs bandwidth curves, such as sweeping DDR frequency while decoding. Your own lab could produce this cheaply.
- No measured sustained NPU DMA bandwidth under thermal limits, or LPDDR energy per GB at 40 GB/s. Sustained power at 40 GB/s is possibly the actual limit.
- Vendor documents on VTCM allocation per generation and hardware weight decompression not found. Qualcomm QMX CPU blog page returned only a title, so its t/s was not retrieved — [Qualcomm](https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx).

## Most credible route to 500 t/s single-stream and 1000 t/s across 4 streams at about 40 GB/s with 2B-class models

### Takeaway
**No paper I found demonstrates either target on a phone or claims it for a dense 2B model.** The best real-model phone numbers found are 139 t/s (0.5B) and 66 t/s (1.5B) on a retrieval-friendly summary task. A dense 2B at 2 bpw has a best-case ceiling of about 240 t/s single-stream on general text. **500 t/s single-stream on a true dense 2B is unachievable with published methods on general text. This is my derivation from the ceilings below; no paper states it for this exact configuration.** 1000 t/s across 4 streams is closer, because weights are read once per batch, and is plausible only at about 1B dense or with 1 bpw.

### Cited Findings
Inputs are all cited above:

- 40 GB/s from the assignment.
- S ≈ 3 tokens per step on general text — [Performance or Illusion](https://arxiv.org/html/2601.11580).
- Batch-4 speculation 1.68-1.82x — [EAGLE-3](https://arxiv.org/html/2503.01840).
- 2-bit quality — [GLVQ](https://arxiv.org/html/2510.20984), [EVA](https://arxiv.org/html/2605.24144).
- Best phone baselines — [sd.npu](https://arxiv.org/html/2510.15312v4), [Qualcomm](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct).
- Active-parameter traffic — [LIMINAL](https://arxiv.org/html/2507.14397).

### Inferences
**[ARITH] Ceilings at 40 GB/s, weights only, excluding KV, drafting and logits. Assumes S=3 single-stream (optimistic) and S=1.75 for 4 streams. Real numbers will be lower, possibly by 30-50% (my estimate, from the measured-vs-bound gap, not from a paper).**

| Model × bpw | Weight bytes | Steps/s | Single-stream, S=3 | 4 streams, no spec | 4 streams, S=1.75 | Meets 500 / 1000? |
|---|---|---|---|---|---|---|
| 2B × 4.5 (today) | 1.125 GB | 36 | 107 | 142 | 249 | No / No |
| 2B × 2 | 0.5 GB | 80 | 240 | 320 | 560 | No / No |
| 2B × 1.58 (entropy-coded, hypothetical) | 0.395 GB | 101 | 304 | 405 | 709 | No / No |
| 2B × 1 (no quality data) | 0.25 GB | 160 | 480 | 640 | 1120 | Marginal / Ceiling only |
| 1B × 2 | 0.25 GB | 160 | 480 | 640 | 1120 | Marginal / Ceiling only |
| 0.5B × 2 | 0.125 GB | 320 | 960 | 1280 | 2240 | Yes / Yes at ceiling |

**Verdict by approach**

| Approach | Verdict for 500 t/s, 2B, 40 GB/s | Quality cost |
|---|---|---|
| Better kernels, HMX, DMA | Unachievable alone: capped at 36-80 t/s | None |
| Quantization alone | Unachievable: requires 0.32 bpw | No data at that rate; 2-bit costs PPL +0.3-1.5 on 7B-13B |
| On-chip caching | Unachievable: about 1.6% | None |
| Speculation alone (4.5 bpw) | Unachievable: about 107 t/s; ≥4x only on rewriting | None if verified |
| 2-bit + speculation | Reaches about 240 t/s at ceiling on chat, more on rewriting and summary | 2-bit loss |
| Diffusion | Not shown on phones: about 8 t/s at 8B | Uneven; DiffusionGemma lost 19 AIME points to Gemma 4 |
| RWKV | No benefit on weights; only removes KV | Unknown |
| LPDDR6 or PIM | Outside the 40 GB/s premise; PIM is unproven in silicon in what I read | None |
| 0.5B-1B model | Ceiling reachable | Large: for example 1B AWQ GSM8K 32.6 |

**Recommended route, in priority order (my judgement, not a published recipe)**

1. Stop treating "2B" as the fixed quantity and fix **bytes per step**: about 160 MB for 4-stream at 250 steps/s, or about 80-160 MB for single-stream with S=3 down to S=1.5.
2. Use **2-bit VQ or a natively trained low-bit model on about 1B of active weights**. Investigate a small MoE of about 2B total and about 0.5-1B active, on the LIMINAL finding that traffic follows active parameters. This is the only route where "2B-class capacity" and about 250 MB per step coexist. It is unproven at this size: I found no phone MoE at this scale, and per-stream routing could widen the batch read set and hurt speculation.
3. **Prioritize multi-stream first.** Batching shares weight reads, Vec-LUT shows up to 4.2x for that pattern on edge devices, and batching raises arithmetic intensity. 1000 t/s over 4 streams needs 250 steps/s, while 500 t/s single needs 500 steps/s without speculation.
4. **Adaptive speculation:** n-gram or SAM when context overlaps, a learned head otherwise. Budget S≈2-3 for chat and report S per workload.
5. **Engineering prerequisites:** DMA-driven weight streaming, logits and LM head on NPU, low-bit KV. The LM head and embedding of a small model with large vocabulary could be a large share of traffic if kept at 8-bit (the Qualcomm card notes part of the model is 8-bit; EuroSys '26 uses Q8 down-projection). Audit this first, since it is likely a cheap win. I did not obtain a verified parameter breakdown because the Llama 3.2 config returned 401.

**Realistic expectations (my estimate):** about 150-300 t/s single-stream general text and about 400-700 t/s across 4 streams on about 1-2B at 2 bpw. 500+ single-stream appears on rewriting and summary workloads, or with 0.5B models. Headline numbers built from repeated text or from multiplying S, sparsity and quantization are not supported by any end-to-end study.

### Gaps
- **No end-to-end paper combines** 2-bit VQ, speculation and batching on a phone. All multiplications above are assumptions and may not compose: VQ decode overhead raises compute, sparsity conflicts with batching, and draft heads for 2-bit targets are untested.
- Sustained thermal behaviour at these rates is unmeasured in what I read.
- Quality of 0.5-1B 2-bit models on real workloads is not published in what I found. This is the key missing measurement, and I would run it first.
