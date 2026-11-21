// Two-page briefing: (1) measured bench of every deployed model vs this phone's theoretical ceiling,
// (2) how 12B / 27B / 125B were made to run on it. 16:9.
const path = require("path");
const pptxgen = require("pptxgenjs");
const NOTES = require("./notes_zh");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "EdgeBrief",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "14202E", lt1: "FFFFFF", dk2: "0B4F4F", lt2: "F1F4F7",
    accent1: "0B6E6E", accent2: "13A3A3", accent3: "B5541C", accent4: "5B6676", accent5: "1F8A4C", accent6: "B42318",
    hlink: "0B6E6E", folHlink: "5B6676",
  },
};
const HEX = THEME.colors;

// ---- data: in-app llama-bench runs (plain decode) and CLI runs, ceilings from GGUF per-token bytes
const BW_DRAM = 76.8; // GB/s, LPDDR5X-9600 (4800 MHz), 4x16-bit
const BW_UFS = 4.2;   // GB/s, UFS 4.1 nominal sequential read (measured O_DIRECT 3.9)
const rows = [
  // name, quant · size, deployment, ngl, pp, ppNote, tg, ceiling, ceilNote, bottleneck, source
  { m: "Gemma 4 E2B", q: "Q4_0 QAT · 3.35 GB", how: "全部 35 层上 NPU", pp: 1183.5, act: 2.28, tg: 28.35, cap: BW_DRAM / 1.407, bn: "内存带宽", src: "App" , co: 1313.8 / 1.407 },
  { m: "Qwen3.5 2B", q: "Q4_0 · 1.26 GB", how: "全部 24 层上 NPU", pp: 855.1, act: 1.88, tg: 24.23, cap: BW_DRAM / 1.207, bn: "内存带宽", src: "App" , co: 1313.8 / 1.207 },
  { m: "Gemma 4 E4B", q: "Q4_0 QAT · 5.15 GB", how: "全部 42 层上 NPU", pp: 373.8, act: 4.64, tg: 14.77, cap: BW_DRAM / 2.827, bn: "内存带宽", src: "App" , co: 1313.8 / 2.827 },
  { m: "Qwen3.5 4B", q: "Q4_0 · 2.67 GB", how: "NPU 28 层 + CPU", pp: 340.5, act: 4.21, tg: 10.42, cap: BW_DRAM / 2.584, bn: "4 层在 CPU", src: "App" , co: 1313.8 / 2.584 },
  { m: "Qwen3.5 9B", q: "Q4_0 · 5.55 GB", how: "NPU 18 层 + CPU 页缓存", pp: 139.5, act: 7.94, tg: 5.95, cap: BW_DRAM / 4.815, bn: "14 层在 CPU", src: "App" , co: 1313.8 / 4.815 },
  { m: "Gemma 4 12B", q: "Q4_0 QAT · 6.98 GB", how: "NPU 24 层 + CPU 24 层（页缓存）", pp: 95.4, act: 11.91, tg: 3.33, cap: BW_DRAM / 6.960, bn: "24 层在 CPU", src: "CLI" , co: 1313.8 / 6.960 },
  { m: "Qwen3.8 125B MoE", q: "UD-IQ1_S · 72.5 GB", how: "稠密 40 层 NPU · 专家闪存分页", pp: 1.45, act: 6.67, tg: 1.12, cap: BW_UFS / 0.778, bn: "闪存随机读 + 同步", src: "CLI" , co: 1 / Math.max(3.532 / 1313.8, 0.778 / 10.8) },
  { m: "Qwen3.8 27B", q: "IQ2_XS · 8.76 GB", how: "NPU 14 层 · 50 层 O_DIRECT 流式 · MTP", pp: 16.46, act: 25.6, tg: 0.614, cap: (BW_UFS / 5.46) * 2.2, bn: "闪存顺序读", src: "CLI" , co: (1313.8 / 8.134) * 2.2 },
];

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "端侧大模型部署汇报";
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "Brief",
    background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.3, w: 12.4, h: 0.6, fontSize: 24, bold: true, color: C.text1, margin: 0, align: "left" }, text: "" } },
      { text: { text: "OnePlus Ace 6T · Snapdragon 8 Gen 5 (SM8845) · llama.cpp + ggml-hexagon · 2026-10-15", options: { x: 0.45, y: 7.08, w: 9, h: 0.25, fontSize: 9, color: C.accent4, margin: 0 } } },
    ],
    slideNumber: { x: 12.4, y: 7.06, w: 0.5, h: 0.28, fontSize: 9, color: HEX.accent4, align: "right" },
  });

  const text = (s, t, x, y, w, h, o = {}) => s.addText(t, {
    x, y, w, h, fontSize: o.fs || 10.5, bold: !!o.bold, color: o.color || C.text1, align: o.align || "left",
    valign: o.valign || "middle", margin: 0, isTextBox: true, paraSpaceAfter: o.psa || 0, fontFace: o.mono ? "Courier New" : undefined,
  });
  const rect = (s, x, y, w, h, fill, o = {}) => s.addShape(o.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: { color: o.line || fill, width: o.lw || 0.75 }, rectRadius: o.round ? 0.06 : undefined, objectName: o.name,
  });

  // ================= page 1: bench vs ceiling
  pres.addSection({ title: "实测与上限" });
  {
    const s = pres.addSlide({ masterName: "Brief", sectionTitle: "实测与上限" });
    s.addText("本机实测 vs 理论上限：已部署的 8 个模型", { placeholder: "title" });
    text(s, "理论上限（decode）= 带宽 ÷ 每个 token 必须读取的权重字节。本机：LPDDR5X 76.8 GB/s，闪存 UFS 4.1 标称 4.2 GB/s。最后一列为骁龙 8 Elite Extreme Gen 6（SM8975，LPDDR6 113.8 GB/s，UFS 5.0 10.8 GB/s）+ 光羽芯辰 TC1000（10 GB 3D DRAM，1200 GB/s）协同推理的理论值：权重按带宽比例拆到两颗芯片并行读取。",
      0.45, 0.92, 12.4, 0.5, { fs: 10.5, color: C.accent4, valign: "top" });

    // table grid
    const cols = [
      ["模型", 1.7], ["量化 · 文件", 1.42], ["部署方式", 2.08], ["Prefill tok/s", 1.0], ["有效算力", 0.8],
      ["Decode 实测", 0.92], ["理论上限", 0.78], ["达成率", 1.6], ["主要瓶颈", 1.13], ["8975 + 光羽\n协同上限", 1.0],
    ];
    const x0 = 0.45, y0 = 1.5, rh = 0.47, hh = 0.48;
    let x = x0;
    rect(s, x0, y0, 12.43, hh, HEX.dk2);
    const colX = cols.map(([h, w], k) => { const cx = x; text(s, h, cx + 0.06, y0, w - 0.1, hh, { fs: 10, bold: true, color: "FFFFFF", align: k === 9 ? "right" : "left" }); x += w; return cx; });
    rows.forEach((r, i) => {
      const y = y0 + hh + i * rh;
      if (i % 2 === 1) rect(s, x0, y, 12.43, rh, HEX.lt2);
      const big = r.m.includes("125B") || r.m.includes("27B");
      const v = (n, f) => (n == null ? "—" : n.toFixed(f));
      const cell = (k, t, o = {}) => text(s, t, colX[k] + 0.06, y, cols[k][1] - 0.1, rh, o);
      cell(0, r.m, { bold: true, fs: 10.5 });
      cell(1, r.q, { fs: 9.5, color: C.accent4 });
      cell(2, r.how, { fs: 9.5 });
      cell(3, v(r.pp, r.pp != null && r.pp < 10 ? 2 : 1), { fs: 11, bold: true, align: "right" });
      cell(4, r.pp == null ? "—" : `${(2 * r.act * r.pp / 1000).toFixed(r.pp < 20 ? 2 : 1)} TF`, { fs: 9.5, color: C.accent4, align: "right" });
      cell(5, v(r.tg, r.tg != null && r.tg < 1 ? 3 : 2), { fs: 11, bold: true, align: "right", color: big ? C.accent3 : C.text1 });
      cell(6, r.cap.toFixed(r.cap < 10 ? 2 : 1), { fs: 10, align: "right", color: C.accent4 });
      // achievement bar
      const bx = colX[7] + 0.12, bw = cols[7][1] - 0.75, by = y + rh / 2 - 0.08;
      rect(s, bx, by, bw, 0.16, "E2E8EE");
      if (r.tg != null) {
        const pct = Math.min(1, r.tg / r.cap);
        rect(s, bx, by, Math.max(0.02, bw * pct), 0.16, pct >= 0.45 ? HEX.accent5 : pct >= 0.3 ? HEX.accent1 : HEX.accent3);
        cell(7, `${Math.round(pct * 100)}%`, { fs: 10, bold: true, align: "right" });
      } else {
        cell(7, "待测", { fs: 10, align: "right", color: C.accent4 });
      }
      cell(8, r.bn, { fs: 9.5, color: C.accent4 });
      cell(9, r.co >= 100 ? Math.round(r.co).toLocaleString("en-US") : r.co.toFixed(1), { fs: 10.5, bold: true, align: "right", color: C.accent1 });
    });
    const yEnd = y0 + hh + rows.length * rh;
    s.addShape(pres.shapes.LINE, { x: x0, y: yEnd, w: 12.43, h: 0, line: { color: "C9D3DD", width: 0.75 } });

    // takeaways
    const ty = yEnd + 0.12;
    const notes = [
      ["小模型", "达到带宽上限的 30–54%，全部层上 NPU 的 Gemma 4 E2B / E4B 最高；E2B 最快：decode 28.4 tok/s、prefill 1184 tok/s（5.4 TFLOPS 有效算力）。"],
      ["超内存模型", "125B 每 token 只需从闪存读约 0.78 GB 专家，上限 5.4 tok/s，实测 1.12；27B 每次前向要读 5.5 GB，MTP 摊到 2.2 token 后上限 1.69，实测 0.61。"],
      ["口径", "App 内为纯 decode（无投机）；12B 为命令行按 App 配置实测，27B / 125B 为命令行实测（27B 含 MTP，上限同乘 2.2）。上限只计权重读取，是严格上界；温控限频可使同一配置相差 4 倍。"],
    ];
    notes.forEach(([h, b], i) => {
      text(s, h, 0.45, ty + i * 0.33, 1.2, 0.3, { fs: 10.5, bold: true, color: C.accent1, valign: "top" });
      text(s, b, 1.6, ty + i * 0.33, 11.25, 0.3, { fs: 10, valign: "top" });
    });
    s.addNotes(NOTES.p1);
  }

  // ================= page 2: how 12B / 27B / 125B run
  pres.addSection({ title: "部署方法" });
  {
    const s = pres.addSlide({ masterName: "Brief", sectionTitle: "部署方法" });
    s.addText("12B / 27B / 125B 如何在这台 12 GB 手机上跑起来", { placeholder: "title" });
    text(s, "约束：NPU 可用内存约 3 GB（32 位地址窗口，rpcmem 不可换出）；系统可用内存约 6 GB；闪存顺序读 3.9 GB/s。模型越大，越要回答“每个 token 读多少字节、从哪里读、谁来算”。",
      0.45, 0.92, 12.4, 0.5, { fs: 10.5, color: C.accent4, valign: "top" });

    const cards = [
      {
        k: "12B", t: "Gemma 4 12B · Q4_0 QAT · 6.98 GB", c: "问题：NPU 预算的 2.3 倍，CPU 那一半还要抢内存", acc: HEX.accent1,
        steps: [
          "按每层 tile 字节自动放置：24 层上 NPU，把约 3 GB 预算用满（HVX decode / HMX prefill），其余 24 层留 CPU",
          "CPU 层保持 mmap 文件页（不做 CPU repack）：约 3.1 GB 都是可回收页缓存。若按 CPU 默认 repack 再复制一份，实测可用内存跌到 554 MB，速度崩溃",
          "prefill 卸载：CPU 层的大批量矩阵乘转成 32×32 tile 交给 NPU 的 HMX，有一半层在 CPU 时 prefill 仍有 95 tok/s",
          "decode 用 2 线程（4 / 6 线程因同步开销降到 0.46 / 0.34 tok/s）；对话走 Gemma 4 自带模板与工具调用解析",
        ],
        res: [["decode", "3.33 tok/s"], ["prefill", "95.4 tok/s"]],
      },
      {
        k: "27B", t: "Qwen3.8 27B · IQ2_XS · 8.76 GB", c: "问题：权重超过可用内存，每次前向都得读闪存", acc: HEX.accent3,
        steps: [
          "IQ 码本转码：IQ1/2/3 解成 3/4 位码 + vlut32 查表，复用 NPU 的 Q3_K / IQ4_NL 内核，8 种 IQ 全部上 NPU",
          "后 14 层常驻 NPU；前 50 层（5.5 GB）用 O_DIRECT 逐层流式读入双缓冲，读下一层时算本层（比 mmap 缺页快 3.2×）",
          "MTP 投机解码：草拟 3 个 token、一次前向验证，接受率约 40%，每次前向 2.2 token",
          "长提示 prefill 卸载到 NPU（1.54 → 16.46 tok/s）；系统提示状态存闪存，新对话 0.09 s 恢复",
        ],
        res: [["decode", "0.61 tok/s"], ["长提示 prefill", "16.5 tok/s"]],
      },
      {
        k: "125B", t: "Qwen3.8-Flash-Next 125B MoE · UD-IQ1_S · 72.5 GB", c: "问题：模型是内存的 6 倍，只能放在闪存", acc: HEX.accent5,
        steps: [
          "稠密部分（注意力 / DeltaNet / 共享专家 / 路由器）40 层上 NPU；24,576 个路由专家与 29 GB PLE 表留在闪存（mmap，MADV_RANDOM）",
          "路由感知预读：本层 top-10 专家用 6 个 I/O 线程并行 pread 入页缓存；用 router(l+1)·x(l) 预测下一层专家，后台提前读",
          "读取量从 1.2 TB 压到 60 GB（缺页顺序预读会把相邻专家一起读进来）",
          "计算图每次前向 8,555 节点 / 238 分段：decode 只用 2 个线程，去掉屏障同步开销",
        ],
        res: [["decode", "1.12 tok/s"], ["较最初", "18×"]],
      },
    ];
    const cw = 4.05, gap = 0.14, cy = 1.5, ch = 4.95;
    cards.forEach((c, i) => {
      const cx = 0.45 + i * (cw + gap);
      rect(s, cx, cy, cw, ch, HEX.lt2, { round: true, name: `card-${c.k}` });
      rect(s, cx, cy, cw, 0.08, c.acc);
      text(s, c.k, cx + 0.2, cy + 0.2, 1.2, 0.55, { fs: 26, bold: true, color: c.acc.replace(/^/, "") });
      text(s, c.t, cx + 1.3, cy + 0.18, cw - 1.45, 0.6, { fs: 10, bold: true, valign: "middle" });
      text(s, c.c, cx + 0.2, cy + 0.85, cw - 0.4, 0.32, { fs: 10, color: C.accent4, valign: "top" });
      c.steps.forEach((st, j) => {
        const sy = cy + 1.25 + j * 0.76;
        s.addShape(pres.shapes.OVAL, { x: cx + 0.2, y: sy + 0.03, w: 0.28, h: 0.28, fill: { color: c.acc }, line: { color: c.acc } });
        text(s, String(j + 1), cx + 0.2, sy + 0.03, 0.28, 0.28, { fs: 9.5, bold: true, color: "FFFFFF", align: "center" });
        text(s, st, cx + 0.58, sy, cw - 0.75, 0.72, { fs: 9.5, valign: "top" });
      });
      // result strip
      const ry = cy + ch - 0.62;
      s.addShape(pres.shapes.LINE, { x: cx + 0.2, y: ry - 0.05, w: cw - 0.4, h: 0, line: { color: "D5DDE5", width: 0.75 } });
      c.res.forEach(([lab, val], j) => {
        const rx = cx + 0.2 + j * ((cw - 0.4) / 2);
        text(s, val, rx, ry, (cw - 0.4) / 2, 0.36, { fs: 15, bold: true, color: c.acc });
        text(s, lab, rx, ry + 0.34, (cw - 0.4) / 2, 0.22, { fs: 9, color: C.accent4 });
      });
    });

    // shared foundation
    const fy = cy + ch + 0.12;
    text(s, "共同基础", 0.45, fy, 1.1, 0.42, { fs: 10.5, bold: true, color: C.accent1 });
    text(s, "--load-mode mmap（非 NPU 权重是可回收的文件页）· rpcmem 预算保护（保留 2 GiB，避免整机重启）· 权重 repack 成 32×32 tile · 多线程 repack · decode 2 线程 / prefill 6 线程 · 前台服务保活",
      1.55, fy, 11.3, 0.42, { fs: 10 });
    s.addNotes(NOTES.p2);
  }

  const out = path.join(__dirname, "..", "EdgeAgent-汇报（两页）.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(out);
}

main().catch((e) => { console.error(e); process.exit(1); });
