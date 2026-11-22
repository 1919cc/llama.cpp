# hui_infer：推理引擎优化与模型部署报告

本报告记录 llama.cpp_hui `hui_infer` 分支相对上游 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` 做了哪些优化：每一项的速度收益、内存收益和详细技术方案，以及模型在手机上的部署方式与实测结果。`hui_infer` 是唯一的推理引擎代码库，App（super_ai_com）只集成它的 Android 产物做模型部署。

> 维护约定：每一项优化只要实测有效，就在同一次提交里更新本报告（总览表加一行，正文加一节）。测速数据必须来自下面的「测速规则」。

## 目录

1. 设备与测速规则
2. 总览：每项优化的收益
3. 与上游 master 的对比基线
4. 引擎优化的技术方案
5. 部署层调优（使用上游已有能力）
6. 模型推理部署与实测
7. 工具链：NDK r30 A/B
8. 上游同步记录
9. 2 倍速度目标：现状与计划

## 1. 设备与测速规则

| 项目 | 值 |
|---|---|
| 主测设备 | OnePlus PLR110 · Snapdragon 8 Gen 5（SM8845）· Hexagon v81 NPU（HVX + HMX，8 MB VTCM）· 约 11 GB 可用内存 · Android 16 · UFS 顺序读约 3.9 GB/s |
| 对照设备 | SM8850（QDC Canoe 开发机，同为 HTP v81），仅部分实验 |
| 构建 | `scripts/snapdragon/build.py --target adb`：Docker 镜像 `ghcr.io/snapdragon-toolchain/arm64-android:v0.8`（NDK r29、Hexagon SDK 6.7.0.0 / Hexagon Tools 21.0.04）；NDK r30 已评估，无收益（第 7 节） |

测速规则（`tools/hui-lab/bench_gate.py` 自动执行）：

- 每次测量前确认 CPU、GPU、NPU 都在最高频率：打开固定性能模式（`cmd power set-fixed-performance-mode-enabled true`）、关闭省电；系统温控状态为 0；所有温控降频设备（CPU 各簇、`kgsl`/`gpu`、`cdsp`/`cdsp_hw`）的降频等级为 0；CPU 各簇 `scaling_max_freq` 等于 `cpuinfo_max_freq`。NPU 使用后端默认的 burst DCVS 档。
- 一次只测一个模型、一个进程；每次运行前清掉后台缓存应用（`am kill-all`），让每次运行的可用内存一致。
- 如果被降频，等回到满频后再测；A/B 两个构建逐轮交替执行，抵消温度漂移。
- 2026-11-22 之前的数据没有这套门控，表中注明「未门控」，可比性较弱（厂商温控可使速度相差 4 倍）。

## 2. 总览：每项优化的收益

收益均为本机、同一模型同一配置下的前后对比。

| # | 优化 | 适用 | 速度收益 | 内存收益 | 状态 |
|---|---|---|---|---|---|
| 1 | IQ 码本量化转码上 NPU（LUT3 / LUT4 tile） | IQ1_S/M、IQ2_XS/S、IQ3_XXS/S、IQ4_XS | 这些格式此前无法上 NPU；现在全部可以 | — | 已上线；IQ2_XXS 自 2026-11-22 改用上游原生内核 |
| 2 | CPU 层的 prefill 卸载到 NPU（offload_op + 写入即 repack） | 内存放不下、部分层留在 CPU 的模型 | 27B 1,813 token prefill 1.54 → 16.46 tok/s（10.7×） | NPU 计算缓冲只需 187 MiB | 已上线 |
| 3 | O_DIRECT 逐层权重流式 | 比内存大的稠密模型 | 27B decode 0.193 → 0.614 tok/s（3.2×）；12B decode 4.66 → 4.97、prefill 2.2 → 9.4 tok/s | CPU 层不再占页缓存 | 已上线 |
| 4 | MoE 路由感知并行预读 | 比内存大的 MoE | 125B decode 0.06 → 0.10 tok/s | 一次运行的闪存读取 1.2 TB → 60 GB（−95%） | 已上线 |
| 5 | decode / prefill 线程分离（2 / 6） | 分页类方案 | 125B decode 0.10 → 1.12 tok/s（11×，与 #4 叠加）；27B 0.19 → 0.48（限频时） | — | 已上线（部署参数） |
| 6 | rpcmem 预算保护 + 大模型跳过整文件预读 | 所有 NPU 部署 | — | 加载不再导致整机重启（此前一次加载占 8 GB 匿名内存）；125B 加载时 126 GB 预读取消 | 已上线 |
| 7 | 系统提示缓存（状态存闪存） | 固定系统提示的对话 | 27B 新对话 TTFT 840 s → 恢复 0.09 s | 每个缓存约 157 MB 闪存，保留最近 4 个 | 已上线（JNI 引擎） |
| 8 | 丢弃 llama.cpp DEBUG 日志（JNI） | App 内推理 | Gemma 4 12B decode 0.14–0.8 → 4.8 tok/s | — | 已上线 |
| 9 | 上下文复用 + 后缀裁剪（JNI） | 多轮对话 | 多轮只算新增 token（实测第 3 轮复用 40 token） | — | 已上线 |
| 10 | 算子融合：softcap（scale-tanh-scale）、softplus-mul | Gemma 系 softcap、Qwen3.5 DeltaNet | 未单独测量 | 少一次 DDR 往返 | 已上线 |
| 11 | DSP 工作队列卡死修复 + 响应等待上限 | 所有 NPU 部署 | 稳定性：不再无限期挂起 | — | 已上线 |
| 12 | server：跨 slot 借用公共前缀 | llama-server 多 slot | 共享前缀不再重算（kvdiff 验证无损） | — | 已上线 |
| 13 | 块稀疏 FFN（方案 B） | 27B | 0.079 tok/s，比方案 A 慢（未减少闪存读取） | — | 实验性，不推荐 |
| 14 | NDK r30（工具链） | 全部 | 无可测收益：E2B pp512 −0.6%、tg128 −0.9%，均在波动内（第 7 节） | — | 不采用，继续 r29 |
| 15 | 同步上游 611 个提交（Hexagon HMX / repack 重写）+ 适配 hui 改动 | 常驻 NPU 的模型 | Gemma 4 E2B pp512 1,607 → 3,980 tok/s（**2.48×**）；tg128 27.68 → 28.19（+1.8%） | IQ2_XXS tile 512 → 264 B（−48%） | 已上线（2026-11-22，门控实测） |
| 18 | 内存拟合（no_alloc）时不对专家张量做 madvise | App 内专家分页（26B-A4B 等，开启专家预读时） | App 加载 26B-A4B 从「GC 崩溃」到可用：decode 4.1 tok/s | — | 已上线（2026-11-22，smaps 复现 + 真机验证） |
| 17 | `load_mode=auto` 下比可用内存大的模型保留 mmap | 不能整个放进内存、又用 NPU 的模型（26B-A4B 等）的命令行运行 | 26B-A4B 从「加载即重启」到 pp128 30.7 / tg32 6.1 tok/s | 专家权重留在可回收的文件页，不再读进约 12 GB 匿名内存 | 已上线（2026-11-22，看门狗复现 + 实测） |
| 16 | Hexagon 主机缓冲默认关闭（上游默认开） | 所有 NPU 部署，尤其 Gemma 4 E 系列 | E2B decode +4.7%（27.91 → 29.22），prefill −11%（仍为合并前 2.29×） | 不再钉住约 1.8–2.2 GB 嵌入表；E4B 恢复可加载 | 已上线（2026-11-22，门控实测） |

## 3. 与上游 master 的对比基线

hui_infer 的优化面向「比手机内存大的模型」（Qwen3.8 27B 稠密、125B MoE）和 IQ 低比特格式。内存放得下、本来就全部在 NPU 上的小模型，大多不经过这些代码路径：

- **Gemma 4 E2B，hui_infer @ 4b0773c7d vs 上游 aa588a346（2026-10-31，hui-lab 报告）**：SM8850 上 pp512 −0.2%、tg64 +0.0%，内存相同，相同参数下输出文本一致。即小模型上**没有可测提升**，也没有退化。
- **NPU vs CPU（同一报告）**：SM8850 上 E2B pp512 2,229 tok/s，是 CPU 的 9.6 倍；生成 36.5 tok/s，与 6 线程 CPU 的 38.0 相当，但不占用 CPU 核心。矩阵乘占 NPU 时间约 89%，其中输出层（Q6_K，词表 262,144）单项约 19%。
- **2026-11-22 同步后**：hui_infer 已包含上游到 `cbb1c1c50` 的全部提交，常驻小模型的速度与上游同期一致，并比合并前的 hui_infer 快（E2B prefill 2.48×，第 6.3 节）。
- **大模型**：上游 master 在本机无法运行 27B（加载即超出内存）和 125B，没有可对比的基线；第 2 节的倍数都是相对 hui_infer 改动前的同一配置。

## 4. 引擎优化的技术方案

### 4.1 IQ 码本量化转码上 NPU（#1）

**问题**：IQ1/IQ2/IQ3 每 8 个权重存一个码本索引，Hexagon 没有对应的解码指令，这些格式原本只能在 CPU 上算；而 27B、125B 能装进 12 GB 手机，靠的正是 IQ2_XS / IQ1_S。

**做法**（`ggml/src/ggml-hexagon/htp-repack.cpp`、`htp/hvx-mm-kernels-tiled.h`、`htp/hmx-mm-kernels-tiled.h`）：

- 主机在 repack 时把每个权重解成 3 位码（IQ2 / IQ1 族）或 4 位码（IQ3 族），连同每 16 个权重一个的 FP16 scale，打进 32×32 的 tile：3 位码复用 Q3_K 的 tile 布局（512 B），4 位码复用 IQ4_NL 的布局（576 B）；IQ4_XS 无损转成 IQ4_NL。
- DSP 侧每类一张 int8 查找表：一条 `vlut32` 把码还原成 int8，然后直接复用 Q3_K / IQ4_NL 现成的 HVX 矩阵向量乘和 HMX 矩阵乘内核。
- tile 支持精确回读（码本反查），`get_tensor` 能还原原始数据。

**收益**：8 种 IQ 格式全部能上 NPU，`test-backend-ops` 全部通过。代价是 tile 比原格式大（IQ2_XS 约 2.3 bpw → 4 bpw），占用更多 NPU 内存。

**2026-11-22 调整**：上游新增了 IQ2_XXS 的原生内核（#31954，264 B 紧凑 tile、DSP 上 gather 解码，约 2.06 bpw）。同步上游时 IQ2_XXS 改用上游实现：NPU 内存占用比转码 tile 少 48%（512 → 264 B / tile）。IQ2_XS / IQ2_S 无法无损转成上游布局，继续用 hui 的 3 位码 tile，类型号改为独立的 `HTP_TYPE_IQ2_XS`（17）。

### 4.2 CPU 层的 prefill 卸载到 NPU（#2）

**问题**：27B 只有 14 层能常驻 NPU（tile 膨胀后约 200 MB/层，NPU 预算约 2.9 GB），其余 50 层的长提示 prefill 全在 CPU 上做矩阵乘。

**做法**（`ggml-hexagon.cpp`：`ggml_backend_hexagon_device_offload_op`）：

1. 调度器对「权重在主机缓冲」的算子先询问更高优先级后端的 `offload_op()`。
2. Hexagon 接手的条件：MUL_MAT；权重是可 repack 的量化类型、非视图、2D；token 数 ≥ `GGML_HEXAGON_OFFLOAD`（默认 32）；tile 不超过动态缓冲的一半。
3. 调度器在 NPU 计算缓冲里建副本，副本写入时即时 repack 成 tile 并打上 tile 标志（上游原本禁用 `offload_op`，因为副本按原始字节写入，而 NPU 内核只读 tile，会算错）。
4. repack 多线程：tile 布局里每 32 行是一段连续 tile，按 32 行切块、6 线程并行。
5. decode（1–4 token）不卸载：repack 一个约 100 MB 的权重比 CPU 直接读一遍更贵。

**收益**：27B 1,813 token 提示 prefill 1,175 s → 110 s（1.54 → 16.46 tok/s，10.7×），首 token 与 CPU 路径一致；NPU 计算缓冲只需 187 MiB（每个外来权重单独成段，缓冲复用）。开关：`GGML_HEXAGON_OFFLOAD=N`（0 关闭）。

### 4.3 O_DIRECT 逐层权重流式（#3）

**问题**：27B 有约 6.8 GB 的 CPU 层，页缓存只剩约 2.3 GB，装不下；每次前向都要靠 mmap 缺页从闪存重读约 7 GB，缺页读速只有约 0.66 GB/s（UFS 大块顺序读可达 3.9 GB/s）。依赖页缓存的 WILLNEED 预取在工作集大于缓存时无效且有害（读速降到 0.40 GB/s）。

**做法**（`src/llama-weight-stream.{h,cpp}`，`LLAMA_WEIGHT_STREAM=1`）：

- 加载后通过 `/proc/self/maps` 把每个 CPU 层的权重张量映射回（文件，偏移），合并成 4 KiB 对齐的连续读区间。
- 两块轮换缓冲（每块 = 最大层的字节数），4 个 I/O 线程以 4 MiB 块 `pread(O_DIRECT)`。
- 每层开头插入一个 CPU 自定义算子：恢复上一层张量的 mmap 指针、等本层缓冲就绪、把本层张量的 `data` 指向缓冲、再向另一块缓冲提交下一流式层的读取。最后一个 CPU 层之后提交下一次前向的第 0 层，读取与 NPU 层的计算重叠。

**收益**：27B（NPU 14 层 + MTP）decode 0.193 → 0.614 tok/s（3.2×），有效读速 1.46 GB/s，输出与 mmap 版本逐字一致；Gemma 4 12B（24 层上 NPU）decode 4.66 → 4.97、prefill 2.2 → 9.4 tok/s。

### 4.4 MoE 路由感知并行预读（#4）

**问题**：125B（Qwen3.8-Flash-Next，IQ1_S 约 70 GB）的 24,576 个路由专家远超内存；mmap 缺页时内核的顺序预读会顺带读入相邻、用不到的专家。

**做法**（`src/llama-graph.cpp`：`build_expert_prefetch`，`LLAMA_EXPERT_PREFETCH=0..3`）：

- 专家切片的地址映射回分片文件，按 4 MiB 块分给常驻 I/O 线程池并行 `pread`。
- 本层：路由选出 top-10 后同步等待其读入；下一层：用 `router(l+1) · x(l)` 预测下一层的专家，后台异步提交。
- 专家张量设 `MADV_RANDOM`，关闭缺页顺序预读。
- 读线程放在异步预取器中，madvise 和 pread 不阻塞计算关键路径。
- 与上游的 MoE cache 共存：使用 MoE cache 的层由 cache 自己上传专家，不再发预读。

**收益**：125B 一次运行的闪存读取 1.2 TB → 60 GB（−95%），decode 0.06 → 0.10 tok/s；再叠加 #5 达到 1.12 tok/s。

### 4.5 线程分离（#5）

125B 每次前向有 8,555 个图节点、238 个 NPU / CPU 分段。CPU 上每个算子结束都要在全部计算线程间做屏障，加上预读线程和 FastRPC 线程挤在限频的核上，线程越多越慢（`--poll 0` 无改善，说明代价在屏障本身）。prefill 是少数大矩阵乘，屏障成本被摊薄。因此 decode 用 2 线程、prefill 用 6 线程（`-t 2 -tb 6`）：125B 0.10 → 1.12 tok/s，27B 0.19 → 0.48 tok/s（限频时）。常驻小模型例外：Gemma 4 12B 4 线程 3.26、2 线程 2.85、6 线程 0.88 tok/s，取 4。

### 4.6 内存安全（#6）

- **rpcmem 预算保护**（`ggml-hexagon.cpp`）：rpcmem 不能换出、按需提交，曾因一次加载占用 8 GB 匿名内存导致整机重启。现在记录本进程已分配的 rpcmem 和首次分配时的可用内存，当「已分配 + 本次 + 保留量（`GGML_HEXAGON_MEM_RESERVE`，默认 2048 MiB）」超过基线时直接拒绝分配并给出原因。`ggml_backend_hexagon_rpcmem_bytes` 供 App 显示 NPU 占用。
- **跳过整文件预读**：超过内存一半的模型加载时不再把整个文件预读一遍（125B 的 126 GB 预读取消）。

### 4.7 Android JNI 引擎（#7、#8、#9，`tools/hui-jni`）

App 内嵌的引擎 `libhui_engine.so` + Kotlin 绑定 `com.hui.infer.HuiEngine`，由原 EdgeAgent 和 super_ai_com 两份引擎合并而来：

- **系统提示缓存**：固定前缀（系统回合）的推理状态用 `llama_state_seq_save_file` 存到闪存（目标上下文与 MTP 草稿上下文各一份，先写 `.tmp` 再 rename），冷上下文时恢复并校验 token。27B 新对话的 TTFT 从 840 s 降到恢复 0.09 s，输出与重算逐字一致；2026-11-22 在 App 内实测 891 token 恢复用时 5 ms。
- **上下文复用与后缀裁剪**：取新提示与已缓存 token 的最长公共前缀；能删后缀的上下文裁掉不匹配部分（并检查滑动窗口未被驱逐），递归（混合）模型只复用完全一致的前缀；投机草拟器随之重启，避免草拟已丢弃的回复（此前 Gemma 4 12B 草拟 64 token 只接受 7 个）。
- **prefill 可中断**：长提示在每个 batch 之间检查停止标志，已算的部分保留并记录。
- **丢弃 DEBUG 日志**：`llama_graph_n_input_tensors` 每次建图为每个输入张量写一行 DEBUG，每个 token 数千行，在 App 内把 12B decode 拖到 0.14–0.8 tok/s；过滤后 4.8 tok/s。
- **NPU 轮询按模型设置**：后端只在注册时读一次 `GGML_HEXAGON_OPPOLL`，引擎在每次加载时把环境变量写进后端的 `opt_oppoll`，常驻模型轮询、分页模型休眠。
- **模板化对话**：用 GGUF 内置 jinja 模板渲染多轮对话和工具定义，并解析工具调用；用户输入里的特殊 token 不被解析。
- **llama-bench 式测速与 GGUF 元数据读取**，供 EdgeAgent 使用。

### 4.8 算子融合（#10）

`SCALE → TANH [→ SCALE]`（最终 logit softcap）合成一个 `UNARY_SCALE_TANH`，`SOFTPLUS → MUL` 合成 `GLU_SOFTPLUS`，张量只过一次 DDR。融合位 `HTP_FUSE_SCALE_TANH`（1<<14）、`HTP_FUSE_SOFTPLUS_MUL`（1<<15）（2026-11-22 上游占用了 1<<13，hui 的两位顺延），`GGML_HEXAGON_OPFUSION` 可单独开关。尚未单独测速。

### 4.9 稳定性与可调参数（#11）

- **DSP 工作队列**：worker 只执行最新的任务（生产者逐个等待任务完成），复用槽位前重新检查序号，消除一处卡死；主机等待 DSP 响应超过 120 s 直接中止，不再无限自旋。2026-11-22 合并上游时适配了新的回调签名（带线程句柄）。
- **HMX 行数阈值可配置**：`GGML_HEXAGON_MM_HMX_MIN_NROWS`（默认 4），不超过阈值的矩阵乘留在 HVX。

### 4.10 llama-server 前缀借用（#12）

统一 KV cache 下，新提示从其他 slot 拷贝最长公共前缀（至少比自己的缓存多 32 token），不再重新解码；递归模型从对方的检查点开始；对方仍在解码该前缀时本 slot 等待。用 `llama-hui-kvdiff` 验证与精确 logits 一致。

### 4.11 块稀疏 FFN（#13，实验性）

`gguf-py/gguf/scripts/gguf_ffn_blockify.py` 离线把 27B 的 FFN 拆成 68 个 256 神经元块（无损，只重排 ffn_down 的量化块），每个 token 只读激活的 m 块（`LLAMA_FFN_SPARSE_BLOCKS`）。实测 PPL：68 块 5.94、60 块 6.35、51 块 7.31、34 块 13.5；但 decode 只有 0.079 tok/s（m = 60 + MTP），没有减少闪存读取，不推荐使用。

### 4.12 数值一致性工具

`tools/hui-debug`：`llama-hui-logit-cmp`（逐个 vs 批量解码的 logits 对比）、`llama-hui-kvdiff`（KV 复用 vs 精确 logits）、`llama-hui-quantize-f32`（全 F32 基准模型）。

### 4.13 主机缓冲默认关闭（#16）

**问题**：上游 #31744 把 `GGML_HEXAGON_HOSTBUF` 默认改成开启：留在 CPU 的权重（`token_embd`、Gemma 4 的 `per_layer_token_embd`）放进 FastRPC 共享内存（rpcmem），省去主机与 NPU 之间的拷贝。rpcmem 是钉住的、不可换页的内存，计入 rpcmem 预算（#6）。Gemma 4 E 系列的逐层嵌入表很大：E2B 约 1.8 GB、E4B 约 2.2 GB。结果在 12 GB 手机上：E4B 可用内存约 5.3 GB 时加载失败；后台开着几个应用（可用约 4.9 GB）时连 E2B 也加载失败。

**测量**（门控，Gemma 4 E2B，pp512 / tg128，原始数据 `tools/hui-lab/results/hostbuf-e2b`、`hostbuf-vs-nohost-e2b`）：

| 配置 | pp512 tok/s | tg128 tok/s | 钉住的主机内存 |
|---|---|---|---|
| 主机缓冲开（上游默认） | 4,131 ± 4 | 27.91 ± 0.15 | E2B 约 1.8 GB；E4B 约 2.2 GB（加载失败） |
| 主机缓冲关 | 3,674 ± 219（−11%） | 29.22 ± 0.81（+4.7%） | 0（嵌入表留在 mmap，可换页） |
| 主机缓冲开 + `--no-host` | 与「关」相同（3,666 vs 3,717、30.97 vs 30.85，差异在波动内） | | 0 |

**做法**：hui_infer 把默认值改回关闭（`opt_hostbuf = false`，与合并前一致），需要时用 `GGML_HEXAGON_HOSTBUF=1` 打开。`--no-host` 的结果说明 prefill 的那 11% 完全来自把嵌入权重放进共享内存，不值得用 2 GB 不可换页内存去换；decode 反而更快。

**收益**：Gemma 4 E4B 重新可以在手机上加载；E2B 少钉住约 1.8 GB；decode +4.7%。代价：E2B prefill −11%（仍是合并前的 2.29 倍）。

## 5. 部署层调优（使用上游已有能力）

以下能力来自上游，hui 的贡献是在本机测出的部署参数，写在 super_ai_com 的 `InferencePlan` 里：

| 调优 | 实测 |
|---|---|
| NPU 轮询（常驻模型开、分页模型关） | MiniCPM-V 1B 57 → 74 tok/s，Gemma 4 E2B 23 → 33，Gemma 4 12B 5.6 → 6.2；部分卸载时 NPU 与 CPU 线程池同时轮询，6 线程 decode 跌到约 1 tok/s，因此分页方案关闭 |
| `--lazy-mode on`（Gemma 4 的逐层嵌入按 token 读行） | E2B 峰值 RSS 3.3 → 1.5 GB，速度不变 |
| KV cache 用 f16（常驻方案） | E2B prefill 1,480 → 1,628 tok/s，decode 27.9 → 28.9；分页方案保留 q8_0，省下的 NPU 内存换常驻层 |
| n-gram 投机（ngram-mod，稠密模型） | 普通问答不草拟，零开销；改写、引用类回复 Gemma 4 12B 6.0 → 18.4 tok/s（接受率 89%）；MoE 不开（多 token 前向会调入更多专家） |
| MTP 投机（Qwen3.8 27B，草拟 3） | 接受率 39.7%，每次前向 2.2 token，1.93× |
| MoE 专家预取等级 | MiniCPM-V 4.7 35B-A3B（专家约一半放得进页缓存）2 级 6.6 tok/s，3 级 4.7；125B 用 3 级 |
| DSpark 草拟（MiniCPM-V 4.7） | 本机更慢（1B 23 vs 59 tok/s），不使用 |

## 6. 模型推理部署与实测

### 6.1 部署方案

| 方案 | 适用 | 做法 |
|---|---|---|
| 常驻（S / RESIDENT） | 内存放得下（≤ 12B） | NPU 预算内的层全部上 NPU，其余层 mmap；NPU 轮询 |
| 层级流式（A / STREAM） | 比内存大的稠密模型 | 最后若干层常驻 NPU，其余层 O_DIRECT 流式（#3）；MTP |
| 块稀疏（B） | 27B 分块版 | #13，不推荐 |
| 专家分页（C / EXPERTS） | 比内存大的 MoE | 稠密部分上 NPU，路由专家留在闪存，路由感知预读（#4） |

NPU 层数 = 从最后一层往前，能放进「当前可用内存 − 保留量（2 GiB + 1.5 GiB KV 与计算缓冲）」的层数。

### 6.2 本机实测（未门控，2026-10）

| 模型 | 方案 | NPU 层 | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| Qwen3.8 27B IQ2_XS + MTP | A | 14 | 2.14（长提示卸载后 16.46） | 0.614 |
| Qwen3.8-Flash-Next 125B IQ1_S | C | 40 | 1.45 | 1.12 |
| Qwen3.5 2B Q4_0 + MTP | S | 25 | 855.1 | 24.23 |
| Qwen3.5 4B Q4_0 + MTP | S | 28 | 340.5 | 10.42 |
| Qwen3.5 9B Q4_0 + MTP | S | 18 | 139.5 | 5.95 |
| Gemma 4 E2B QAT Q4_0 | S | 35 | 1,183.5 | 28.35 |
| Gemma 4 E4B QAT Q4_0 | S | 42 | 373.8 | 14.77 |

### 6.3 本机实测（门控，2026-11-22 起）

`bench_gate.py`，每次运行前确认满频并清掉后台缓存应用；`llama-bench -p 512 -n 128 -ngl 99 -fa 1 -t 4 --device HTP0`，5 轮交替，均值 ± 标准差。原始数据：`tools/hui-lab/results/ab3-e2b/runs.jsonl`。

| 模型 | 构建 | pp512 tok/s | tg128 tok/s | TTFT@512（推算） | TPOT |
|---|---|---|---|---|---|
| Gemma 4 E2B QAT Q4_0 | 合并前 hui_infer（d05071d11，NDK r29） | 1,607 ± 90 | 27.68 ± 0.12 | 319 ms | 36.1 ms |
| Gemma 4 E2B QAT Q4_0 | 合并后（NDK r29） | **3,980 ± 227** | **28.19 ± 0.50** | **129 ms** | **35.5 ms** |
| Gemma 4 E2B QAT Q4_0 | 合并后（NDK r30） | 3,957 ± 283 | 27.94 ± 0.23 | 129 ms | 35.8 ms |

TTFT@512 = 512 / pp512；TPOT = 1000 / tg128。

prefill 的倍数来自上游同期对 Hexagon 后端的重写（HMX 矩阵乘、repack 与缓冲分配，第 8 节），hui 的工作是把它与 hui 的 IQ tile、融合、工作队列修复和内存保护合在一起并验证正确（相同输入下贪心输出逐字一致）。decode 受内存带宽限制，基本不变。

**新模型逐个实测**（同一规则，一次一个模型，主机缓冲关闭 = 现在的默认；`-p 512 -n 128 -ngl 99 -fa 1 -t 4`，3 轮；原始数据 `tools/hui-lab/results/models-2026-11-22/`）：

| 模型 | NPU 层 | pp512 tok/s | tg128 tok/s | 贪心输出 |
|---|---|---|---|---|
| MiniCPM5 1B Q4_K_M | 全部 | 3,707 ± 19 | 66.13 ± 3.39 | 正确 |
| MiniCPM-V 4.6 1.3B Q4_0（语言部分） | 全部 | 2,975 ± 37 | 59.32 ± 0.14 | 正确 |
| Gemma 4 E2B QAT Q4_0 | 全部 | 3,674 ± 219 | 29.22 ± 0.81 | 与合并前逐字一致 |
| Qwen3.5 2B Q4_0（无 MTP） | 全部 | 2,065 ± 57 | 32.72 ± 0.65 | 正确 |
| MiniCPM5 2B Q4_K_M | 全部 | 1,738 ± 74 | 28.70 ± 0.40 | 正确 |
| Gemma 4 E4B QAT Q4_0 | 全部 | 1,420 ± 124 | 17.02 ± 0.66 | 正确 |
| Gemma 4 12B QAT Q4_0 | 22 / 48，其余 mmap（`--repack 0`） | 14.05 ± 0.22 | 4.69 ± 0.29（tg64，2 轮） | 与纯 CPU 逐字一致（无聊天模板的原始提示下复读问题，属模型行为） |
| Gemma 4 26B-A4B QAT Q4_0 | 稠密上 NPU、专家 mmap（`-ncmoe 30 --repack 0 -lm mmap`，`LLAMA_EXPERT_PREFETCH=2`） | 30.71 ± 1.06（pp128） | 6.08 ± 0.98（tg32，2 轮） | 正确（修复 #17 之后） |
| MiniCPM-V 4.7 35B-A3B Q4_0（App 方案 C 参数，未门控单次） | 稠密上 NPU、专家 mmap | — | — | 正确，最低可用内存 6.3 GB |
| Qwen3.8-Flash-Next 125B IQ1_S（App 方案 C 参数，未门控单次） | 稠密上 NPU、专家 mmap，2 / 6 线程 | 3.15（11 token） | 1.97 | 正确，最低可用内存 3.9 GB |

E4B 全部层上 NPU 需要约 5.5 GB 可用内存，后台应用多时其中一次运行被 rpcmem 预算拒绝、重试成功。

**Gemma 4 26B-A4B 加载导致整机重启：原因与修复（#17）**。2026-11-22 本机重启 4 次。用内存看门狗（每 100 ms 记录 `/proc/meminfo` 与进程内存，可用内存低于 1.5 GB 就杀进程，`tools/hui-lab/device/mem_guard.sh`）复现：进程匿名内存涨到 3.3 GB、再换出 2.4 GB 仍在涨，ION（rpcmem）稳定在 1.8 GB，文件页只有 7 MB——CPU 侧的专家权重被读进了匿名内存，而不是 mmap。原因是上游 #26081 的 `load_mode = auto`：只要有设备声明不支持 mmap（Hexagon 设备就是 `mmap_support = false`），整个模型都不用 mmap，留在 CPU 的权重全部读进匿名内存；14 GB 的模型放进 12 GB 手机只能耗尽内存。App 一直显式传 `--load-mode mmap`，不受影响；命令行工具（llama-bench、llama-completion 等）用默认值就会触发。

修复：`load_mode = auto` 时，如果模型比当前可用内存（`/proc/meminfo` 的 MemAvailable）还大，即使有设备不支持 mmap 也保留 mmap 并打印警告（`src/llama-model.cpp`）——不 mmap 时这种模型只能耗尽内存；放得下的模型仍按上游规则。验证：同一命令、默认 load mode，日志「HTP0 has no mmap support, but the model (13755 MiB) is larger than the available memory (7112 MiB): keeping mmap」，正常输出，过程中最低可用内存 5.5 GB，未重启。App 的专家分页模型（MiniCPM-V 35B-A3B、Qwen3.8-Flash-Next 125B）在合并后的引擎上复测正常（见上表）。

**App 内加载 26B-A4B 时进程崩溃：原因与修复（#18）**。App（显式 mmap）加载 26B-A4B 约 4 秒后，ART 的垃圾回收 abort：「mremap to move pages failed: Bad address」，每次复现。采样 `/proc/<pid>/smaps` 看到 Java 堆 `[anon:dalvik-main space]`（0x02000000–0x22000000）被拆成两段，前段多了 `rr`（`MADV_RANDOM`）标志，终点 0x08820000 恰好等于一个 `ffn_down_exps` 张量的字节数（704 × 2816 × 128 × 0.5625 = 142,737,408）。原因：参数拟合（memory-fit）用 `no_alloc` 加载模型、在未分配的缓冲上建图，张量的 `data` 只是偏移量；hui 的专家预读（#4）在建图时对专家张量 `posix_madvise(RANDOM)`，把从 0 开始的这段地址标成随机访问，正好落在 ART 的 Java 堆上。命令行进程那段地址没有映射，所以只在 App 里出事。修复（`src/llama-graph.cpp`）：`no_alloc` 模型不建专家预读，且只对确实落在已分配缓冲内的张量地址做 madvise。验证：同一 App 操作，Java 堆不再被拆分、没有崩溃，26B-A4B 正常回答（671 + 4 输入 / 92 输出，prefill 25 tok/s，decode 4.1 tok/s，NPU 1.9 GB，可用内存 5.6 GB）。

## 7. 工具链：NDK r30 A/B

**结论：NDK r30 没有可测收益，默认构建继续用镜像自带的 NDK r29。**

| 项目 | r29（默认） | r30 |
|---|---|---|
| 编译器 | 镜像 `arm64-android:v0.8` 自带 | Android clang 21.0.0（r574158c，+pgo +bolt +lto +mlgo） |
| Hexagon SDK | 6.7.0.0 / Tools 21.0.04（当前最新） | 同左（NPU 内核不受 NDK 影响） |
| `test-backend-ops -b HTP0` | 全部通过 | 全部通过（MUL_MAT 2198 / 2198 等，与 r29 相同） |
| Gemma 4 E2B pp512 | 3,980 ± 227 tok/s | 3,957 ± 283（−0.6%） |
| Gemma 4 E2B tg128 | 28.19 ± 0.50 tok/s | 27.94 ± 0.23（−0.9%） |

原因：NDK 只编译 CPU 侧代码（调度、采样、CPU 算子），而 NPU 常驻模型的时间几乎都花在 DSP 内核上，DSP 内核由 Hexagon SDK 编译，两次构建相同。r30 对 CPU 层占比高的方案（流式、专家分页）可能有差别，待这些方案的门控实测时再比。

r30 构建方法（未进入默认流程）：在同一 Docker 镜像里解压 NDK r30（macOS 文件系统不区分大小写，必须在容器内解压），把镜像放在 r29 sysroot 里的 OpenCL 头文件和 `libOpenCL.so` 复制进 r30 sysroot，然后用相同 preset 构建。

## 8. 上游同步记录

### 2026-11-22：合并上游 611 个提交

hui_infer 落后上游 611 个提交（上游 `cbb1c1c50`）。12 个文件有冲突，集中在 Hexagon 后端：上游同期重写了张量标志（`t->extra` 直接存标志位，去掉 `htp_tensor_extra` 与 shadow buffer）、缓冲分配（mmap 叠加，#32144）和 repack（#31744），并新增了 IQ2_XXS / NVFP4 的 NPU 支持（#31954）。合并决定：

| 范围 | 处理 |
|---|---|
| 张量标志 / tiled 判断 | 全部改用上游的 `ggml_hexagon_tensor_{is,can,will}_tile`；hui 的 `htp_tiled_type()`（IQ 类型 → tile 类型）叠加在上面 |
| IQ2_XXS | 改用上游原生内核与 264 B 紧凑 tile（NPU 内存 −48%）；hui 的 LUT3 路径不再处理 IQ2_XXS |
| IQ2_XS / IQ2_S | 保留 hui 的 3 位码 tile，类型号改为独立的 `HTP_TYPE_IQ2_XS`（17），tile 宏改为 `HTP_TENSOR_TILE_SIZE_IQ2_XS` |
| IQ1_S/M、IQ3_XXS/S、IQ4_XS | 保留 hui 实现（上游没有） |
| NVFP4、TIMESTEP_EMBEDDING 等新算子 | 采用上游 |
| 融合标志 | 上游占用 1<<13（RMS_NORM_MUL_ADD），hui 的 SCALE_TANH / SOFTPLUS_MUL 顺延到 1<<14 / 1<<15 |
| DSP 工作队列 | 保留 hui 的卡死修复，适配上游新的回调签名（带线程句柄） |
| MoE cache | 上游把 cache 的建图移进 `llama_moe_cache::build_lookup`（#32157）；删除 hui 重复的 `build_moe_cache_slots`，专家预读在 `cache.slots == nullptr` 时发出 |
| llama-server | 上游把选 slot 移到 `server-slot-sched`，检查点改为 `shared_ptr`；hui 的前缀借用保留并适配 |
| 权重流式、专家预读、offload_op、rpcmem 预算 | 自动合并，无冲突 |

**合并后的问题与修复**：上游把 `opt_hostbuf`（主机侧 rpcmem 缓冲，CPU 层权重可以放进去）默认改成了开启。hui 原来用「`!opt_hostbuf`」这个全局开关判断 IQ 权重能否 tile，于是合并后 7 种 hui IQ 格式在 NPU 上全部报「not supported」。改为按权重实际所在的缓冲判断（`htp_iq_weight_can_tile()` = 上游的 `ggml_hexagon_tensor_will_tile()`）：权重在设备缓冲里就 tile；放在主机缓冲里的（例如留在 CPU 的层）不 tile，相应的算子交给 CPU。

**验证**（本机 SM8845，`test-backend-ops -b HTP0`，与 CPU 结果逐元素比对）：

| 算子 | 结果 |
|---|---|
| MUL_MAT | 2198 / 2198 通过；其中 IQ2_XS、IQ2_S、IQ1_S、IQ1_M、IQ3_XXS 各 42 / 42，IQ3_S 48 / 48，IQ4_XS 64 / 64（合并前同为 42 / 42）；IQ2_XXS、NVFP4 走上游内核 |
| MUL_MAT_ID | 1169 / 1169 |
| GET_ROWS | 96 / 96 |
| 融合：SOFTCAP（scale-tanh-scale）、SOFTPLUS_MUL、ADD_SOFTPLUS_MUL | 1 / 1、13 / 13、9 / 9 |
| TANH、SOFTPLUS、SWIGLU、GEGLU、SOFT_MAX、RMS_NORM | 4 / 4、4 / 4、24 / 24、24 / 24、105 / 105、61 / 61 |

端到端：Gemma 4 E2B QAT Q4_0 在 NPU 上贪心生成 60 token，合并前后输出逐字相同。

注意：合并后 `llama-completion` 默认按模型训练上下文分配 KV，E2B 在可用内存约 6 GB 时会被 rpcmem 预算保护（#6）拒绝；端侧部署一律显式传 `-c`（App 内 JNI 引擎本来就按方案设置上下文）。


## 9. 2 倍速度目标：现状与计划

目标：在本机（SM8845）上把端侧推理速度提升到 hui_infer 当前基线的 2 倍，按第 1 节的规则逐模型验证。

**判断依据**：

- 常驻小模型的 decode 受内存带宽限制：Gemma 4 E2B 每个 token 约读 1.38 GB 权重，28 tok/s 相当于约 39 GB/s，约为 LPDDR5X 理论带宽（76.8 GB/s）的一半；prefill 受 NPU 算力限制，矩阵乘占约 89%，输出层单项约 19%。
- 部分卸载的模型（12B）和分页模型（27B、MoE）的瓶颈在 CPU 层和闪存读取，倍数空间更大，但受闪存带宽（约 3.9 GB/s）约束。
- 小模型上 hui_infer 与上游没有差距（第 3 节），2 倍必须来自新的优化，而不是已有改动。

**候选优化（按预期收益和把握排序，逐项实测后记入第 2 节）**：

| 方向 | 做法 | 主要受益 | 预期 |
|---|---|---|---|
| HVX 矩阵向量乘的带宽利用率 | 权重 DMA 双缓冲 / 预取深度调优，提高 decode 时的 DDR 利用率 | 所有常驻模型的 decode | 1.3–1.6×（带宽从约 50% 提到 70–80%） |
| 输出层 | 输出层改 Q4_0（tile 原生）或 HMX 计算；decode 时只算采样需要的部分 | Gemma（词表 262k）、Qwen | decode +10–20% |
| 投机解码扩展 | Gemma 4 用同词表小模型草拟；Qwen3.5 / 3.8 已有 MTP，调草拟长度 | 小模型 decode | 1.3–2×（视接受率） |
| 流式层上 NPU 计算 | 流式层的 O_DIRECT 缓冲换成 rpcmem，读入后在 NPU 上算（或预存 tile 格式文件） | 27B decode | 1.3–1.5×，上限为闪存带宽 |
| 减少 NPU / CPU 分段 | 小算子融合或整体留在 NPU，每层只切换一次后端 | 125B、MoE decode | +10–20% |
| CPU 码本量化批量矩阵乘 | IQ2_XS / IQ1_S 先解码成 int8 再用 i8mm / SDOT 做 GEMM | CPU 层 prefill | 3–5×（估计） |
| 工具链 | NDK r30、编译选项 | 全部 | 见第 7 节 |

**进度（2026-11-22，门控实测）**：

| 模型 | 指标 | 基线（合并前 hui_infer） | 当前 | 倍数 | 2× 目标 |
|---|---|---|---|---|---|
| Gemma 4 E2B Q4_0 | prefill（pp512） | 1,607 tok/s | 3,674 tok/s（主机缓冲关闭；开启时 3,980–4,131） | 2.29× | **已达成** |
| Gemma 4 E2B Q4_0 | decode（tg128） | 27.68 tok/s | 29.22 tok/s（主机缓冲关闭，#16） | 1.06× | 未达成 |

下一步：decode 是剩下的主要差距，按上表从「HVX 矩阵向量乘的带宽利用率」和「输出层」开始，每项单独门控实测；其余模型（E4B、12B、27B、MoE）的门控基线随后逐个补齐。
