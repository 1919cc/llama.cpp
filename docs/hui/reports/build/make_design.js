// Design deck for the engine / app changes of 2026-10-13..14 (16:9). Small-model numbers come from ../small_bench.json.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const benchPath = path.join(__dirname, "..", "small_bench.json");
const BENCH = fs.existsSync(benchPath) ? JSON.parse(fs.readFileSync(benchPath, "utf8")) : {};

const THEME = {
  name: "EdgeTeal",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "16202B", lt1: "FFFFFF", dk2: "0B4F4F", lt2: "EEF3F6",
    accent1: "0B6E6E", accent2: "13A3A3", accent3: "B5541C", accent4: "5B6676", accent5: "1F8A4C", accent6: "B42318",
    hlink: "0B6E6E", folHlink: "5B6676",
  },
};
const HEX = THEME.colors;

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "EdgeAgent 推理引擎改动设计方案";
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "Cover",
    background: { color: HEX.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.2, w: 11.9, h: 1.1, fontSize: 36, bold: true, color: C.background1, margin: 0, align: "left" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.4, w: 11.9, h: 1.6, fontSize: 16, color: C.background2, margin: 0, align: "left", valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "Content",
    background: { color: "FFFFFF" },
    margin: [0.4, 0.45, 0.5, 0.45],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.28, w: 12.4, h: 0.62, fontSize: 24, bold: true, color: C.text1, margin: 0, align: "left" }, text: "" } },
      { line: { x: 0.45, y: 0.98, w: 12.43, h: 0, line: { color: HEX.lt2, width: 1.5 } } },
      { text: { text: "EdgeAgent · 推理引擎改动设计方案 · 2026-10-14", options: { x: 0.45, y: 7.08, w: 8, h: 0.25, fontSize: 9, color: C.accent4, margin: 0 } } },
    ],
    slideNumber: { x: 12.4, y: 7.06, w: 0.5, h: 0.28, fontSize: 9, color: HEX.accent4, align: "right" },
  });

  // ---- helpers (fresh option objects every call: pptxgenjs mutates them)
  const box = (s, x, y, w, h, text, o = {}) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w, h, rectRadius: 0.06,
      fill: { color: o.fill || HEX.lt2 }, line: { color: o.line || o.fill || HEX.lt2, width: o.lw || 1 },
      objectName: o.name,
    });
    s.addText(text, {
      x: x + 0.06, y, w: w - 0.12, h, fontSize: o.fs || 11, bold: !!o.bold, color: o.color || C.text1,
      align: o.align || "center", valign: o.valign || "middle", margin: 0, isTextBox: true, paraSpaceAfter: 2,
    });
  };
  const arrow = (s, x, y, w, h, o = {}) => {
    s.addShape(pres.shapes.LINE, {
      x, y, w, h, flipV: !!o.flipV, flipH: !!o.flipH,
      line: { color: o.color || HEX.accent4, width: o.width || 1.5, endArrowType: "triangle", dashType: o.dash || "solid" },
    });
  };
  const label = (s, text, x, y, w, h, o = {}) => s.addText(text, {
    x, y, w, h, fontSize: o.fs || 10, color: o.color || C.accent4, bold: !!o.bold, italic: !!o.italic,
    align: o.align || "left", valign: o.valign || "top", margin: 0, isTextBox: true, paraSpaceAfter: o.psa || 3,
  });
  const bullets = (s, items, x, y, w, h, fs = 12) => s.addText(
    items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontSize: fs, color: C.text1, valign: "top", margin: 0, paraSpaceAfter: 5, isTextBox: true },
  );
  const content = (title, section) => {
    const s = pres.addSlide({ masterName: "Content", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };

  // ================= 1. cover
  pres.addSection({ title: "概览" });
  {
    const s = pres.addSlide({ masterName: "Cover", sectionTitle: "概览" });
    s.addText("EdgeAgent 推理引擎改动设计方案", { placeholder: "title" });
    s.addText([
      { text: "NPU prefill 卸载 · 系统提示缓存 · 线程分离 · 前台保活 · W4 小模型（方案 S）· 模板化对话层", options: { breakLine: true } },
      { text: "骁龙 8 Gen 5（Hexagon v81 NPU）· 11.4 GB RAM · llama.cpp 分支 hexagon-27b-streaming + EdgeAgent App", options: { fontSize: 13 } },
    ], { placeholder: "body" });
    s.addNotes("本文档说明 10 月 13–14 日对推理引擎与 App 的改动：每项解决什么问题、数据怎么流动、代码在哪里、实测效果。");
  }

  // ================= 2. change list
  {
    const s = content("改动总览：问题 → 做法 → 实测效果", "概览");
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } });
    const rows = [
      [hdr("改动"), hdr("解决的问题"), hdr("代码位置"), hdr("实测效果"), hdr("状态")],
      ["① 线程分离 -t 2 / -tb 6", "6 个计算线程在限频 CPU 上每个算子都做屏障同步，decode 被同步开销拖垮", "InferenceMode.kt threadsFor", "125B decode 0.10 → 1.12 tok/s；27B 0.19 → 0.48（限频时）", "已上线"],
      ["② NPU prefill 卸载（算子优化 1）", "NPU 只放得下 14 层，其余 50 层的长提示 prefill 全在 CPU 上做矩阵乘", "ggml-hexagon.cpp offload_op、repack_tensor_tiled_mt", "27B 1813 token prefill 1.54 → 16.46 tok/s（10.7×），首 token 相同", "已实现，精度对比中"],
      ["③ 系统提示缓存", "每个新对话重算约 740 token 系统提示，App 内 TTFT 840 s", "engine.cpp cache_restore/build；EngineController.setPromptCache", "恢复 0.09 s，输出与重算逐字一致", "已实现，待 App 内验证"],
      ["④ 前台服务保活", "锁屏后被厂商后台管理杀进程（UserAwareMgr ... flags='bg'）", "InferenceService.kt；AppViewModel keepAlive", "加载 / 生成 / 进化期间常驻", "已实现，待 App 内验证"],
      ["⑤ 方案 S：W4 小模型", "新增 Qwen3.5 2B/4B/9B、Gemma 4 E2B/E4B/12B（Q4_0）", "InferenceMode SMALL_MODELS；ModeScreen", "6 个模型已装入并校验；E2B decode 28.4、prefill 1184 tok/s", "已部署；12B 待测"],
      ["⑥ 模板化对话层", "Gemma 等非 ChatML 模型的对话模板与工具调用格式不同", "ChatFormat.kt；engine.cpp nativeChatRender/Parse", "单元测试 9/9 通过", "已实现"],
      ["⑦ 模型 / Benchmark 两个 tab", "看不到本地有哪些模型、在本机各自多快", "ModelsScreen、BenchmarkScreen；engine.cpp nativeModelInfo/nativeBench；BenchStore", "App 内已测 5 个小模型", "已实现"],
      ["⑧ 改名 · 设置页 · 中英文", "需要英文界面与集中设置", "I18n.kt、Settings.kt、SettingsScreen.kt；values / values-zh", "全部界面与 Agent 内容可切换", "已实现"],
    ];
    s.addTable(rows, {
      x: 0.45, y: 1.2, w: 12.43, colW: [2.35, 3.4, 2.7, 2.83, 1.15],
      fontSize: 10.5, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5DDE5" },
      fill: { color: "FFFFFF" }, valign: "middle", margin: 0.06, rowH: 0.5,
    });
    label(s, "所有改动未提交 git；llama.cpp 侧改动在 hexagon-27b-streaming 分支工作区，App 在 EdgeAgent 工程。", 0.45, 6.62, 12.4, 0.3, { fs: 10 });
  }

  // ================= 3. architecture
  pres.addSection({ title: "架构" });
  {
    const s = content("总体架构：从对话界面到 NPU（绿色 = 本次新增或修改）", "架构");
    const NEW = { fill: "DDF1E4", line: HEX.accent5, lw: 1.5 };
    const rowsY = [1.2, 1.95, 2.7, 3.5, 4.3, 5.1, 5.9];
    const tags = ["界面", "视图模型", "Agent", "引擎控制", "JNI 引擎", "llama.cpp", "后端 / 存储"];
    tags.forEach((t, i) => label(s, t, 0.45, rowsY[i] + 0.18, 1.15, 0.35, { fs: 11, bold: true, color: C.accent1 }));
    const X = 1.65, W = 11.2;
    const row = (i, parts) => {
      let x = X;
      const total = parts.reduce((a, p) => a + p[1], 0);
      parts.forEach(([t, frac, o]) => {
        const w = (W - 0.12 * (parts.length - 1)) * frac / total;
        box(s, x, rowsY[i], w, 0.62, t, Object.assign({ fs: 10.5 }, o || {}));
        x += w + 0.12;
      });
    };
    row(0, [["对话 ChatScreen", 1], ["模式 ModeScreen（新增方案 S 模型选择）", 1.6, NEW], ["进化 EvolutionScreen", 1], ["Bench 栏（提示缓存命中）", 1.2, NEW]]);
    row(1, [["AppViewModel：会话 · 发送 · 自进化 · 前台服务启停 · 首轮设置提示缓存", 1, NEW]]);
    row(2, [["Agent（ReAct 工具循环）", 1], ["ChatFormat：ChatMl / TemplateChat", 1.2, NEW], ["Tools（计算/记忆/笔记/提醒…）", 1.2], ["Evolution（RSI 评测门控）", 1]]);
    row(3, [["EngineController：专用线程 · 自动 NPU 层数 · 采样 · 提示缓存键", 2, NEW], ["InferenceService（前台 + 唤醒锁）", 1, NEW]]);
    row(4, [["generate（plain / MTP 投机）", 1], ["prepare_prompt + 提示缓存恢复/保存", 1.3, NEW], ["chat render / parse（common_chat）", 1.3, NEW]]);
    row(5, [["common：参数 · chat 模板 · speculative", 1.2], ["llama：qwen35 / qwen4exp / gemma4 图 · 流式读 · 专家预取", 1.6], ["ggml-backend sched：分段 · op offload", 1.3, NEW]]);
    row(6, [["CPU：decode 2 线程 / batch 6 线程", 1.1, NEW], ["Hexagon：HVX decode · HMX prefill · tile repack · 卸载副本", 1.6, NEW], ["UFS：模型 mmap / O_DIRECT · promptcache/*.state", 1.4, NEW]]);
    s.addNotes("自上而下是一次请求经过的层次。绿色框是本次新增或修改的部分。");
  }

  // ================= 4. data path decode vs prefill
  {
    const s = content("数据链路：27B 的 decode 与 prefill 走不同的路", "架构");
    const lane = (x, title, sub) => {
      box(s, x, 1.2, 6.0, 0.55, title, { fill: HEX.accent1, color: "FFFFFF", bold: true, fs: 13 });
      label(s, sub, x, 1.82, 6.0, 0.3, { fs: 10, align: "center" });
    };
    lane(0.45, "Decode（每步 1–4 token：普通生成 / MTP 验证）", "每次前向只读一遍权重，读闪存是瓶颈");
    lane(6.88, "Prefill（≥ 32 token：长提示 / 首轮对话）", "每次前向要算 512 token，矩阵乘是瓶颈");

    // decode lane
    box(s, 0.6, 2.3, 2.6, 0.75, "后 14 层（rpcmem 常驻 tile）", { fill: "E6F2F2" });
    box(s, 3.55, 2.3, 2.75, 0.75, "Hexagon HVX 矩阵向量乘", { fill: "E6F2F2" });
    arrow(s, 3.2, 2.67, 0.35, 0);
    box(s, 0.6, 3.5, 2.6, 0.75, "前 50 层权重（UFS 文件）", { fill: "F3F6F9" });
    box(s, 3.55, 3.5, 2.75, 0.75, "O_DIRECT 双缓冲（读下一层时算本层）", { fill: "F3F6F9" });
    arrow(s, 3.2, 3.87, 0.35, 0);
    box(s, 3.55, 4.65, 2.75, 0.75, "CPU 矩阵乘（2 线程）", { fill: "F3F6F9" });
    arrow(s, 4.92, 4.25, 0, 0.4);
    label(s, "实测：0.614 tok/s（6 线程，频率上限高时）；限频时 2 线程 0.48 tok/s。改到 NPU 计算（后续第 4 项）预计 1.3–1.5×，受闪存带宽约束。", 0.6, 5.65, 5.7, 0.9, { fs: 11, color: C.text1 });

    // prefill lane
    box(s, 7.0, 2.3, 2.5, 0.75, "前 50 层权重（UFS 文件）", { fill: "F3F6F9" });
    box(s, 9.85, 2.3, 2.9, 0.75, "O_DIRECT 缓冲 / mmap（主机内存）", { fill: "F3F6F9" });
    arrow(s, 9.5, 2.67, 0.35, 0);
    box(s, 9.85, 3.35, 2.9, 0.75, "sched：offload_op 判定 → 建副本", { fill: "DDF1E4", line: HEX.accent5, lw: 1.5 });
    arrow(s, 11.3, 3.05, 0, 0.3);
    box(s, 9.85, 4.4, 2.9, 0.75, "主机 6 线程 repack 成 32×32 tile", { fill: "DDF1E4", line: HEX.accent5, lw: 1.5 });
    arrow(s, 11.3, 4.1, 0, 0.3);
    box(s, 7.0, 4.4, 2.5, 0.75, "rpcmem 计算缓冲 → HMX 矩阵乘", { fill: "DDF1E4", line: HEX.accent5, lw: 1.5 });
    arrow(s, 9.5, 4.77, 0.35, 0, { flipH: true });
    label(s, "实测：1813 token 的提示 prefill 1175 s → 110 s（1.54 → 16.46 tok/s，10.7×）；HTP 计算缓冲只需 187 MiB（每个权重单独成段，缓冲复用）。", 7.0, 5.65, 5.8, 0.9, { fs: 11, color: C.text1 });
    s.addNotes("decode 每步读一遍权重、计算量小，在 CPU 上算比 repack 更便宜；prefill 每次要算 512 个 token，repack 的成本被摊薄，NPU 的 HMX 矩阵单元比 CPU 快一个数量级。");
  }

  // ================= 5. offload implementation
  pres.addSection({ title: "实现" });
  {
    const s = content("实现：NPU prefill 卸载（ggml-hexagon + ggml 调度器）", "实现");
    const steps = [
      ["1 调度器询问", "ggml_backend_sched 第 1 遍：权重在 CPU 主机缓冲的算子，先问更高优先级后端 offload_op() 要不要接手"],
      ["2 接手条件", "MUL_MAT；权重是可 repack 类型（IQ1/2/3/4、Q2_K–Q6_K、Q4_0…）、非视图、2D；token 数 ≥ GGML_HEXAGON_OFFLOAD（默认 32）；tile 大小 ≤ 动态缓冲的一半。接手时登记权重名"],
      ["3 建副本", "调度器在 HTP 计算缓冲里建副本 \"HTP0:0#blk.N.ffn_up.weight#0\"；每个外来权重单独起一个分段，用完即被下一个复用"],
      ["4 写入即 repack", "set_tensor：副本名中间段命中登记表 → 标 REPACK（不标 WEIGHT，DSP 把它当普通计算张量做缓存一致性）→ 转成 NPU tile"],
      ["5 多线程 repack", "repack_tensor_tiled_mt：tile 布局里每 32 行是一段连续 tile，按 32 行切块、6 线程并行；模型加载时的 repack 也一起变快"],
      ["6 decode 不卸载", "1–4 token 时 repack 一个 ~100 MB 权重比 CPU 直接读一遍更贵；阈值保证 decode / MTP 验证仍走原路径"],
    ];
    steps.forEach(([h, b], i) => {
      const y = 1.2 + i * 0.86;
      s.addShape(pres.shapes.OVAL, { x: 0.45, y: y + 0.08, w: 0.5, h: 0.5, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 } });
      s.addText(String(i + 1), { x: 0.45, y: y + 0.08, w: 0.5, h: 0.5, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      label(s, h.slice(2), 1.1, y + 0.02, 2.0, 0.6, { fs: 13, bold: true, color: C.text1, valign: "middle" });
      label(s, b, 3.05, y + 0.02, 6.0, 0.72, { fs: 11, color: C.text1, valign: "middle" });
    });
    box(s, 9.35, 1.2, 3.5, 5.05, "", { fill: HEX.lt2 });
    label(s, "为什么之前不能直接开", 9.55, 1.35, 3.1, 0.35, { fs: 13, bold: true, color: C.accent1 });
    label(s, "Hexagon 的 offload_op 原本被注释掉：调度器的副本按原始字节 memcpy 进 NPU 缓冲，而 NPU 内核只读 tile 布局，结果会错。本次的关键是让“写入副本”这一步做 repack，并给副本打上 tile 标志。", 9.55, 1.75, 3.1, 1.9, { fs: 11, color: C.text1 });
    label(s, "开关", 9.55, 3.75, 3.1, 0.35, { fs: 13, bold: true, color: C.accent1 });
    label(s, "GGML_HEXAGON_OFFLOAD=N（0 关闭）\nGGML_HEXAGON_REPACK_THREADS=N\n--no-op-offload（llama.cpp 通用开关）", 9.55, 4.12, 3.1, 1.0, { fs: 11, color: C.text1 });
    label(s, "验证", 9.55, 5.15, 3.1, 0.35, { fs: 13, bold: true, color: C.accent1 });
    label(s, "首 token 与 CPU 路径一致；困惑度对比进行中。", 9.55, 5.5, 3.1, 0.6, { fs: 11, color: C.text1 });
  }

  // ================= 6. prompt cache
  {
    const s = content("实现：系统提示缓存（只算一次，存到闪存）", "实现");
    box(s, 0.45, 1.25, 2.7, 0.8, "首轮对话：Agent.systemTurn()\n渲染固定的系统回合", { fs: 11 });
    arrow(s, 3.15, 1.65, 0.4, 0);
    box(s, 3.55, 1.25, 3.2, 0.8, "EngineController.setPromptCache\n键 = SHA-256(模型路径|大小|mtime|启动参数|前缀)", { fs: 10.5, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 6.75, 1.65, 0.4, 0);
    box(s, 7.15, 1.25, 2.6, 0.8, "nativeSetPromptCache\n分词得到前缀 token", { fs: 11, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 9.75, 1.65, 0.4, 0);
    box(s, 10.15, 1.25, 2.7, 0.8, "冷上下文时\nprepare_prompt 检查前缀", { fs: 11 });

    box(s, 0.9, 2.75, 5.4, 1.35, "文件不存在（第一次）\n① 正常 prefill 前缀\n② llama_state_seq_save_file：目标上下文 + MTP 草稿上下文\n③ 先写 .tmp 再 rename，文件存在即完整", { fs: 11, align: "left", fill: "FBEFE7", line: HEX.accent3 });
    box(s, 7.0, 2.75, 5.4, 1.35, "文件存在（之后每次新对话 / App 重启）\n① llama_state_seq_load_file 恢复，校验 token 一致\n② 只 prefill 用户消息\n③ 任一步失败 → 清空上下文、按第一次处理", { fs: 11, align: "left", fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 9.0, 2.05, 0, 0.7);
    arrow(s, 4.0, 2.05, 0, 0.7);

    bullets(s, [
      "让系统提示稳定：按查询检索的记忆原来拼进系统提示，导致每次都不同；现在放进用户回合，系统回合只取决于基因组 + 工具定义。",
      "混合模型（DeltaNet + 注意力）的状态以递归状态为主，27B 文件约 157 MB，与前缀长短关系不大；目录只保留最近 4 个。",
      "命令行验证：同一提示第 1 次计算并保存，第 2、3 次恢复（约 86 ms），三次生成文本逐字一致。",
      "进化出新基因组 → 系统回合变化 → 键变化 → 自动生成新缓存；旧缓存按 LRU 删除。",
    ], 0.45, 4.45, 12.4, 2.4, 12);
  }

  // ================= 7. threads
  {
    const s = content("实现：线程分离——decode 用 2 线程，prefill 用 6 线程", "实现");
    const chartOpts = (title, colors) => ({
      x: 0, y: 0, w: 0, h: 0, showTitle: true, title, titleFontSize: 13, titleColor: HEX.dk1,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.00",
      chartColors: colors, showLegend: false,
      catAxisLabelColor: HEX.accent4, valAxisLabelColor: HEX.accent4,
      valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
    });
    s.addChart(pres.charts.BAR, [{ name: "125B decode tok/s", labels: ["1 线程", "2 线程", "3 线程", "4 线程", "6 线程"], values: [0.69, 1.06, 0.21, 0.15, 0.10] }],
      Object.assign(chartOpts("方案 C（125B MoE）decode tok/s", [HEX.accent5]), { x: 0.45, y: 1.2, w: 6.0, h: 3.6 }));
    s.addChart(pres.charts.BAR, [{ name: "27B decode tok/s", labels: ["2 线程", "4 线程", "6 线程"], values: [0.48, 0.17, 0.19] }],
      Object.assign(chartOpts("方案 A（27B + MTP）decode tok/s（限频时）", [HEX.accent1]), { x: 6.85, y: 1.2, w: 6.0, h: 3.6 }));
    bullets(s, [
      "原因：125B 每次前向有 8,555 个图节点、238 个 NPU/CPU 分段；CPU 上每个算子结束都要在全部计算线程间做屏障，加上 6 个预读线程与 FastRPC 线程挤在被限频（上限 1.9 / 1.65 GHz）的核上，线程越多同步越慢。--poll 0（不自旋）无改善，说明代价在屏障本身。",
      "但 prefill 是少数大矩阵乘、计算密集，屏障成本被摊薄：-t 2 -tb 6 让两者各取所需（llama.cpp 原生支持 decode / batch 线程分开设置）。",
      "App：CPU 线程滑块新增“自动”档（默认），按方案取最优值；手动设置仍可覆盖。",
    ], 0.45, 5.0, 12.4, 1.9, 11.5);
  }

  // ================= 8. keep alive
  {
    const s = content("实现：前台服务保活（锁屏后不再被杀）", "实现");
    const xs = [0.45, 3.0, 5.85, 8.7];
    const items = [
      ["用户操作\n加载模型 / 发送 / 自进化", HEX.lt2],
      ["keepAlive()\nstartForegroundService", "DDF1E4"],
      ["InferenceService\nstartForeground(specialUse)\n+ PARTIAL_WAKE_LOCK（≤ 2 h）", "DDF1E4"],
      ["引擎忙：LOADING / GENERATING\n或 Agent 多步任务进行中", HEX.lt2],
    ];
    items.forEach(([t, f], i) => {
      box(s, xs[i], 1.35, i === 3 ? 2.15 : 2.45, 1.15, t, { fill: f, line: f === "DDF1E4" ? HEX.accent5 : f, fs: 11 });
      if (i < 3) arrow(s, xs[i] + 2.45, 1.92, xs[i + 1] - xs[i] - 2.45, 0);
    });
    box(s, 11.15, 1.35, 1.7, 1.15, "全部空闲\nstopService\n释放唤醒锁", { fill: "FBEFE7", line: HEX.accent3, fs: 11 });
    arrow(s, 10.85, 1.92, 0.3, 0);
    bullets(s, [
      "问题现场：用户在 App 里用 27B 对话，prefill 14 分钟后锁屏，logcat：UserAwareMgr: process killed ... flags='bg'，回复中断。",
      "做法：用户操作发生时（App 在前台）启动 specialUse 类型前台服务 + 常驻通知；AppViewModel 用 combine(引擎状态, busy) 监听，回到空闲立即停止。",
      "清单：FOREGROUND_SERVICE、FOREGROUND_SERVICE_SPECIAL_USE、WAKE_LOCK；PROPERTY_SPECIAL_USE_FGS_SUBTYPE 写明“端侧 LLM 推理”。",
      "唤醒锁设 2 小时上限：卡死的生成不会无限耗电。",
    ], 0.45, 3.0, 12.4, 3.6, 12);
  }

  // ================= 9. scheme S
  pres.addSection({ title: "小模型与对话层" });
  {
    const s = content("方案 S：W4 小模型（Qwen3.5 / Gemma 4，全部 Q4_0）", "小模型与对话层");
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } });
    const models = [
      ["Qwen3.5 2B", "qwen35（DeltaNet 混合）", "Qwen3.5-2B-MTP-Q4_0", "1.26 GB", "MTP"],
      ["Qwen3.5 4B", "qwen35", "Qwen3.5-4B-MTP-Q4_0", "2.67 GB", "MTP"],
      ["Qwen3.5 9B", "qwen35", "Qwen3.5-9B-MTP-Q4_0", "5.55 GB", "MTP"],
      ["Gemma 4 E2B", "gemma4（PLE）", "gemma-4-E2B-it-qat-Q4_0", "3.35 GB", "—"],
      ["Gemma 4 E4B", "gemma4", "gemma-4-E4B-it-qat-Q4_0", "5.15 GB", "—"],
      ["Gemma 4 12B", "gemma4", "gemma-4-12B-it-qat-Q4_0", "6.98 GB", "—"],
    ];
    const rows = [[hdr("模型"), hdr("架构"), hdr("文件"), hdr("大小"), hdr("投机"), hdr("NPU 层"), hdr("pp512 tok/s"), hdr("tg64 tok/s")]];
    models.forEach((m) => {
      const b = BENCH[m[2]] || {};
      rows.push([...m, b.ngl != null ? String(b.ngl) : "待测", b.pp != null ? String(b.pp) : "—", b.tg != null ? String(b.tg) : "—"]);
    });
    s.addTable(rows, {
      x: 0.45, y: 1.2, w: 12.43, colW: [1.55, 2.05, 2.85, 0.95, 0.8, 1.15, 1.5, 1.58],
      fontSize: 10.5, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, fill: { color: "FFFFFF" }, margin: 0.06, rowH: 0.42,
    });
    bullets(s, [
      "为什么选 Q4_0：它是 Hexagon 内核的原生 tile 格式（无需转码，HVX/HMX 路径最快）；Gemma 4 用 Google 官方 QAT Q4_0（量化感知训练，精度最接近原模型）；Qwen3.5 用 Unsloth 带 MTP 层的版本，App 自动开 MTP 投机解码。",
      "放置：NPU 预算约 3 GB（32 位地址窗口 + rpcmem 不可换出）。2B/4B/E2B 全部上 NPU；9B/12B 按每层 tile 字节自动选层数，其余层 mmap 留在页缓存，长提示由 NPU prefill 卸载加速。",
      "部署：ModelScope 下载（约 20 MB/s，比 HF 代理快 20 倍以上），逐个用 HF 官方 SHA-256 校验；推送到手机后先用 llama-bench 测速，再拷进 App 私有目录、删除中转文件。",
    ], 0.45, 4.35, 12.4, 2.6, 11.5);
  }

  // ================= 10. chat format
  {
    const s = content("实现：模板化对话层（让 Gemma 也能调用工具）", "小模型与对话层");
    box(s, 0.45, 1.3, 2.4, 1.0, "Agent\nReAct 工具循环", { fs: 12, bold: true });
    arrow(s, 2.85, 1.8, 0.45, 0);
    box(s, 3.3, 1.3, 2.6, 1.0, "ChatFormat 接口\nsystemText · render\nparse · assistantMsg", { fs: 11, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 5.9, 1.67, 0.6, 0);
    arrow(s, 5.9, 2.1, 0.6, 0.9);
    box(s, 6.5, 0.95 + 0.25, 6.35, 0.95, "ChatMl（Qwen 系：27B / 125B / Qwen3.5）\nKotlin 渲染 ChatML，助手回合按原样保存 → 下一步提示严格延续上一步，KV / 递归状态可复用", { fs: 10.5, align: "left" });
    box(s, 6.5, 2.35, 6.35, 1.3, "TemplateChat（Gemma 4 等）\nmessages / tools JSON → nativeChatRender → common_chat_session（GGUF 内置 jinja 模板）→ 提示\n输出 → nativeChatParse → 格式解析器（如 peg-gemma4）→ content + tool_calls", { fs: 10.5, align: "left", fill: "DDF1E4", line: HEX.accent5 });
    bullets(s, [
      "选择依据：加载后 nativeChatInfo 检查模板源码是否含 <|im_start|>；是 → ChatMl，否 → TemplateChat。对模型零配置。",
      "工具调用配对：助手回合的每个 tool_call 分配 id，随后的工具结果按顺序带上 tool_call_id 与 name，满足各家模板的要求。",
      "提示缓存同样适用：TemplateChat.systemTurn 用“只有 system、不加生成提示”渲染出前缀；引擎按 token 前缀校验，不匹配就不用缓存，不会出错。",
      "Reflector（自进化反思）也改用当前模型的格式渲染，Gemma 上也能进化。",
      "测试：新增 TemplateChat 单元测试（工具调用 id 配对、参数解析、工具不进系统文本），JVM 测试 9/9 通过。",
    ], 0.45, 3.95, 12.4, 3.0, 11.5);
  }

  // ================= 11. models + benchmark tabs
  {
    const s = content("App：模型 tab 与 Benchmark tab", "小模型与对话层");
    // flow: models dir -> scan -> Models tab ; bench run -> store -> Benchmark tab
    box(s, 0.45, 1.3, 2.3, 0.9, "files/models\n所有 .gguf（分片取第一片）", { fs: 11 });
    arrow(s, 2.75, 1.75, 0.4, 0);
    box(s, 3.15, 1.3, 3.0, 0.9, "LocalModels.scan\nnativeModelInfo 读 GGUF 头：架构、参数量、量化、层数、MTP", { fs: 10.5, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 6.15, 1.75, 0.4, 0);
    box(s, 6.55, 1.3, 2.7, 0.9, "模型 tab\n卡片：大小 · 架构 · 方案 · 最快速度\n加载 / 测速", { fs: 10.5, fill: "DDF1E4", line: HEX.accent5 });

    box(s, 0.45, 2.85, 2.3, 1.0, "测速（单个 / 全部小模型 / 全部）", { fs: 11 });
    arrow(s, 2.75, 3.35, 0.4, 0);
    box(s, 3.15, 2.85, 3.0, 1.0, "按模型的最快默认配置加载\n（方案、自动 NPU 层数、-tb 6）", { fs: 10.5 });
    arrow(s, 6.15, 3.35, 0.4, 0);
    box(s, 6.55, 2.85, 3.0, 1.0, "nativeBench：随机 token\nprefill 一次 + decode 线程扫描 1/2/4/6", { fs: 10.5, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 9.55, 3.35, 0.4, 0);
    box(s, 9.95, 2.85, 2.9, 1.0, "BenchStore\nbench/results.json（每次测量都留存）", { fs: 10.5, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 11.4, 2.85, 0, -0.6, { flipV: true });
    box(s, 9.95, 1.3, 2.9, 0.9, "Benchmark tab\n每个模型：最快 prefill + 最快 decode", { fs: 10.5, fill: "DDF1E4", line: HEX.accent5 });
    bullets(s, [
      "测法与 llama-bench 一致：清空上下文，随机 token 计时；prefill 用 batch 线程，decode 逐 token；llama_set_n_threads 换线程数无需重新加载。",
      "大模型（> 8 GB）每 token 需数秒，只测 128 / 16 token、2 线程（已知最快）；小模型测 512 / 64 token、1/2/4/6 线程。",
      "每条结果记录方案、NPU 层数、线程、token 数、时间与当时的 CPU 频率上限（厂商温控可使速度相差 4 倍），卡片上一并显示，便于判断数字是否可比。",
      "27B / 125B 早先的命令行实测作为“命令行实测”条目预置，与 App 内实测分开标注。",
      "测速期间同样启动前台服务；可随时“停止”（nativeStop 中断当前测量）。",
    ], 0.45, 4.25, 12.4, 2.7, 11.5);
  }

  // ================= 12. settings + i18n
  {
    const s = content("App：改名、设置页与中英文切换", "小模型与对话层");
    box(s, 0.45, 1.3, 2.6, 1.0, "设置页：语言\n跟随系统 / 简体中文 / English", { fs: 11, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 3.05, 1.8, 0.4, 0);
    box(s, 3.45, 1.3, 3.1, 1.0, "Settings.applyLanguage\nLocaleManager 设置应用语言（Android 13+）", { fs: 10.5, fill: "DDF1E4", line: HEX.accent5 });
    arrow(s, 6.55, 1.55, 0.4, -0.25, { flipV: true });
    arrow(s, 6.55, 2.05, 0.4, 0.25);
    box(s, 6.95, 1.05, 5.9, 0.7, "系统资源：App 名称（桌面图标）、通知 → values / values-zh", { fs: 10.5, align: "left" });
    box(s, 6.95, 1.95, 5.9, 0.7, "I18n.lang（StateFlow）→ tr(中文, English) → 界面以 key(lang) 整体重建", { fs: 10.5, align: "left" });
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } });
    s.addTable([
      [hdr("范围"), hdr("中文"), hdr("English")],
      ["App 名称", "模型部署推理引擎优化", "Model Deployment & Inference Engine Optimization"],
      ["界面", "6 个 tab、Bench 栏、全部按钮与说明", "all six tabs, the bench bar, every button and description"],
      ["Agent 内容", "种子系统指令、工具描述与结果、反思指令", "seed system prompt, tool descriptions / results, reflection prompt; reply-in-English instruction"],
      ["不变的部分", "自进化评测集的题目（保证各版本得分可比）", "evaluation tasks keep their wording so scores stay comparable"],
    ], { x: 0.45, y: 2.95, w: 12.43, colW: [1.8, 4.6, 6.03], fontSize: 10.5, color: HEX.dk1,
      border: { type: "solid", pt: 0.5, color: "D5DDE5" }, fill: { color: "FFFFFF" }, margin: 0.06, rowH: 0.4 });
    bullets(s, [
      "设置项：语言、主题（跟随系统 / 浅色 / 深色）、NPU prefill 卸载（重启生效）、系统提示缓存（含占用与清除）、锁屏保活、默认深度思考、自动自进化与间隔、清除测速记录、关于。",
      "术语对照：prefill / decode / TTFT、MTP speculative decoding、NPU-resident layers、O_DIRECT weight streaming、MoE expert paging、block-sparse FFN、prompt cache、foreground service。",
      "测试：新增英文模式单元测试（工具描述、种子指令、工具结果），JVM 测试 10/10 通过。",
    ], 0.45, 5.2, 12.4, 1.75, 11);
  }

  // ================= 13. roadmap
  pres.addSection({ title: "后续" });
  {
    const s = content("后续：算子优化 2–4（按次序推进）", "后续");
    const cards = [
      ["2", "CPU 码本量化批量矩阵乘", "IQ2_XS / IQ1_S 等先按块解码成 int8，再用 ARM i8mm / SDOT 做 GEMM，替代逐行逐 token 点积", "CPU 侧 prefill 3–5×（估计）；125B 专家 prefill 同样受益"],
      ["3", "125B 减少分段", "超连接、PLE 混合、路由 top-k 等小算子融合或整体留在 NPU，每层只切换一次后端；专家 MUL_MAT_ID 的 prefill 也走卸载（选择性复制专家）", "decode +10–20%（估计）"],
      ["4", "27B decode 走 NPU 流式计算", "流式层的 O_DIRECT 缓冲换成 rpcmem，读入后直接 repack / 或预存 tile 格式文件，由 HVX 计算", "decode 1.3–1.5×（估计），上限为闪存带宽"],
    ];
    cards.forEach(([n, t, how, gain], i) => {
      const x = 0.45 + i * 4.2;
      box(s, x, 1.25, 4.0, 4.8, "", { fill: HEX.lt2 });
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.45, w: 0.6, h: 0.6, fill: { color: HEX.accent3 }, line: { color: HEX.accent3 } });
      s.addText(n, { x: x + 0.2, y: 1.45, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      label(s, t, x + 0.95, 1.45, 2.9, 0.6, { fs: 14, bold: true, color: C.text1, valign: "middle" });
      label(s, "做法", x + 0.2, 2.3, 3.6, 0.3, { fs: 11, bold: true, color: C.accent1 });
      label(s, how, x + 0.2, 2.62, 3.6, 1.9, { fs: 11, color: C.text1 });
      label(s, "预期", x + 0.2, 4.55, 3.6, 0.3, { fs: 11, bold: true, color: C.accent1 });
      label(s, gain, x + 0.2, 4.87, 3.6, 1.0, { fs: 11, color: C.text1 });
    });
    label(s, "第 1 项（NPU prefill 卸载）已完成并实测 10.7×；表中“估计”均需实测确认。", 0.45, 6.3, 12.4, 0.4, { fs: 11 });
  }

  const out = path.join(__dirname, "..", "EdgeAgent-改动设计方案.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(out);
}

main().catch((e) => { console.error(e); process.exit(1); });
