// One-page technical overview slide (16:9) — numbers come from ../metrics.json
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const M = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "metrics.json"), "utf8"));

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

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "端侧大模型推理技术方案（一页总览）";
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "OnePager",
    background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.28, w: 12.4, h: 0.6, fontSize: 26, bold: true, color: C.text1, margin: 0, align: "left" }, text: "" } },
    ],
  });

  const s = pres.addSlide({ masterName: "OnePager" });
  s.addText("在 12 GB 手机上运行 27B 稠密与 125B MoE：三种闪存分页推理方案", { placeholder: "title" });
  s.addText("Snapdragon 8 Gen 5 · Hexagon v81 NPU（6×HVX + HMX, 8 MB VTCM）· 11.4 GB RAM · UFS 实测 2.6–3.9 GB/s · llama.cpp + ggml-hexagon", {
    x: 0.45, y: 0.88, w: 12.4, h: 0.32, fontSize: 12, color: C.text2, margin: 0, isTextBox: true,
  });

  // ---- three scheme columns
  const cols = [
    { k: "A", t: "层级分层 + MTP", m: "Qwen3.8-27B 稠密 · IQ2_XS 8.8 GB", fill: "E6F2F2", acc: C.accent1,
      pts: ["按每层 tile 字节与内存预算自动选 NPU 常驻层", "其余层 O_DIRECT 逐层流式（两块轮换缓冲，读下一层时算本层）：比 mmap 缺页快 3.2×", "MTP 草拟 3 token、一次前向验证：再快 1.93×"],
      stat: M.A_DECODE, statLabel: "decode tok/s", sub: M.A_NOTE },
    { k: "B", t: "FFN 块稀疏分页", m: "同模型 · ffn_down 无损块重排", fill: "FBEFE7", acc: C.accent3,
      pts: ["17408 神经元 → 68 块 × 256，块即“专家”，down 无损块重排", "Σ|SiLU(gate)| 选 top-m，up/down 走 MUL_MAT_ID 只读选中块", "m=60 PPL +6.9%，但闪存读取未减少；与 MTP 冲突"],
      stat: M.B_DECODE, statLabel: "decode tok/s", sub: M.B_NOTE },
    { k: "C", t: "MoE 专家分页", m: "Qwen3.8-Flash-Next 125B · UD-IQ1_S 72.5 GB", fill: "E9F4EC", acc: C.accent5,
      pts: ["稠密 40 层上 NPU；24,576 路由专家 + 29 GB PLE 表留闪存 mmap", "topk 后 6 线程并行 pread 本层 10 专家；router(l+1)·x(l) 预测下一层后台预读", "8,555 算子 / 238 分段：2 线程去屏障开销，0.06 → 1.12 tok/s"],
      stat: M.C_DECODE, statLabel: "decode tok/s", sub: M.C_NOTE },
  ];
  const x0 = 0.45, colW = 4.0, gap = 0.2, y0 = 1.38, colH = 3.25;
  cols.forEach((c, i) => {
    const x = x0 + i * (colW + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: colW, h: colH, fill: { color: c.fill }, line: { color: c.fill }, rectRadius: 0.08, objectName: `scheme-${c.k}` });
    s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: y0 + 0.18, w: 0.5, h: 0.5, fill: { color: c.acc }, line: { color: c.acc } });
    s.addText(c.k, { x: x + 0.18, y: y0 + 0.18, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(c.t, { x: x + 0.8, y: y0 + 0.14, w: colW - 0.95, h: 0.34, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
    s.addText(c.m, { x: x + 0.8, y: y0 + 0.47, w: colW - 0.95, h: 0.26, fontSize: 10.5, color: C.accent4, margin: 0, isTextBox: true });
    s.addText(c.pts.map((p, j) => ({ text: p, options: { bullet: true, breakLine: j < c.pts.length - 1 } })), {
      x: x + 0.18, y: y0 + 0.86, w: colW - 0.36, h: 1.45, fontSize: 11, color: C.text1, paraSpaceAfter: 3, valign: "top", margin: 0, isTextBox: true,
    });
    s.addText(String(c.stat), { x: x + 0.18, y: y0 + 2.36, w: 1.5, h: 0.62, fontSize: 30, bold: true, color: c.acc, margin: 0, isTextBox: true });
    s.addText([{ text: c.statLabel, options: { bold: true, breakLine: true } }, { text: c.sub }], {
      x: x + 1.72, y: y0 + 2.38, w: colW - 1.9, h: 0.7, fontSize: 10, color: C.accent4, valign: "top", margin: 0, isTextBox: true,
    });
  });

  // ---- operator / scheduling strip
  const y1 = y0 + colH + 0.2;
  const strips = [
    { h: "算子层", b: "权重 repack 为 32×32 tile；HVX vrmpy（32 行×4 MAC/指令）做解码，HMX fp16 tile 做 prefill；新增 8 种 IQ：主机把码本索引转成 3/4 位码，DSP 一条 vlut32 映射 int8，复用 Q3_K/IQ4_NL 内核，test-backend-ops 全通过" },
    { h: "调度层", b: "ggml_backend_sched 按权重位置与 supports_op 切图；用 ngl / override-tensor 放置层与专家；op batching 一次 FastRPC 送多算子；VTCM 预取深度 2–16 求解；CPU 回退 BF16 与词嵌入" },
    { h: "内存层", b: "AUTO 加载模式在 Hexagon 下关闭 mmap → 8 GB 匿名内存 → 整机重启；修复：--load-mode mmap + rpcmem 累计预算（基线 MemAvailable − 2 GiB 保留）+ 看门狗" },
  ];
  const sw = (12.4 - 2 * gap) / 3;
  strips.forEach((st, i) => {
    const x = x0 + i * (sw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y1, w: sw, h: 1.42, fill: { color: THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, rectRadius: 0.06 });
    s.addText(st.h, { x: x + 0.15, y: y1 + 0.1, w: sw - 0.3, h: 0.28, fontSize: 12.5, bold: true, color: C.accent1, margin: 0, isTextBox: true });
    s.addText(st.b, { x: x + 0.15, y: y1 + 0.4, w: sw - 0.3, h: 0.98, fontSize: 10, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  });

  // ---- bottom line
  s.addText([
    { text: "EdgeAgent App：", options: { bold: true } },
    { text: "顶部常驻 Bench（prefill/decode/TTFT/MTP 接受率/RSS/NPU/可用内存/闪存读速）· 模式页切换 A/B/C · Qwen 工具调用 Agent（计算/记忆/笔记/提醒/网页，副作用需确认）· RSI 自进化：反思→候选基因组→8 项沙箱评测→更优才采纳，可回滚" },
  ], { x: x0, y: 6.62, w: 12.4, h: 0.6, fontSize: 10.5, color: C.text2, margin: 0, valign: "top", isTextBox: true });

  s.addNotes("一页总览。三列为三种部署方案及其实测解码速度；下方为算子、调度、内存三层的关键实现；底部为 App 与自进化。详细数据见同名 PDF 技术报告。");

  const out = path.join(__dirname, "..", "EdgeAgent-一页总览.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(out);
}

main().catch((e) => { console.error(e); process.exit(1); });
