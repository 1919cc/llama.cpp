// English edition of the two-page briefing (make_report2.js): same data, layout re-fitted for English text
// (longer words: narrower type, two-line cells, shorter phrasing). 16:9.
const path = require("path");
const pptxgen = require("pptxgenjs");
const NOTES = require("./notes_en");
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

const BW_DRAM = 76.8; // GB/s, LPDDR5X-9600 (4800 MHz), 4x16-bit
const BW_UFS = 4.2;   // GB/s, UFS 4.1 nominal sequential read (measured O_DIRECT 3.9)
const rows = [
  { m: "Gemma 4 E2B", q: "Q4_0 QAT · 3.35 GB", how: "All 35 layers on the NPU", pp: 1183.5, act: 2.28, tg: 28.35, cap: BW_DRAM / 1.407, bn: "Memory bandwidth" , co: 1313.8 / 1.407 },
  { m: "Qwen3.5 2B", q: "Q4_0 · 1.26 GB", how: "All 24 layers on the NPU", pp: 855.1, act: 1.88, tg: 24.23, cap: BW_DRAM / 1.207, bn: "Memory bandwidth" , co: 1313.8 / 1.207 },
  { m: "Gemma 4 E4B", q: "Q4_0 QAT · 5.15 GB", how: "All 42 layers on the NPU", pp: 373.8, act: 4.64, tg: 14.77, cap: BW_DRAM / 2.827, bn: "Memory bandwidth" , co: 1313.8 / 2.827 },
  { m: "Qwen3.5 4B", q: "Q4_0 · 2.67 GB", how: "28 layers NPU + 4 CPU", pp: 340.5, act: 4.21, tg: 10.42, cap: BW_DRAM / 2.584, bn: "4 layers on CPU" , co: 1313.8 / 2.584 },
  { m: "Qwen3.5 9B", q: "Q4_0 · 5.55 GB", how: "18 layers NPU + 14 CPU (page cache)", pp: 139.5, act: 7.94, tg: 5.95, cap: BW_DRAM / 4.815, bn: "14 layers on CPU" , co: 1313.8 / 4.815 },
  { m: "Gemma 4 12B", q: "Q4_0 QAT · 6.98 GB", how: "24 layers NPU + 24 CPU (page cache)", pp: 95.4, act: 11.91, tg: 3.33, cap: BW_DRAM / 6.960, bn: "24 layers on CPU" , co: 1313.8 / 6.960 },
  { m: "Qwen3.8 125B MoE", q: "UD-IQ1_S · 72.5 GB", how: "Dense part of 40 layers on NPU · experts paged from flash", pp: 1.45, act: 6.67, tg: 1.12, cap: BW_UFS / 0.778, bn: "Random flash reads, sync" , co: 1 / Math.max(3.532 / 1313.8, 0.778 / 10.8) },
  { m: "Qwen3.8 27B", q: "IQ2_XS · 8.76 GB", how: "14 layers NPU · 50 streamed from flash (O_DIRECT) · MTP", pp: 16.46, act: 25.6, tg: 0.614, cap: (BW_UFS / 5.46) * 2.2, bn: "Sequential flash reads" , co: (1313.8 / 8.134) * 2.2 },
];

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "On-device LLM deployment briefing";
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
    valign: o.valign || "middle", margin: 0, isTextBox: true, paraSpaceAfter: o.psa || 0,
  });
  const rect = (s, x, y, w, h, fill, o = {}) => s.addShape(o.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: { color: o.line || fill, width: o.lw || 0.75 }, rectRadius: o.round ? 0.06 : undefined, objectName: o.name,
  });

  // ================= page 1: bench vs ceiling
  pres.addSection({ title: "Measured vs ceiling" });
  {
    const s = pres.addSlide({ masterName: "Brief", sectionTitle: "Measured vs ceiling" });
    s.addText("Measured on this phone vs. theoretical ceiling: 8 deployed models", { placeholder: "title" });
    text(s, "Decode ceiling = bandwidth ÷ weight bytes read per token. This phone: LPDDR5X 76.8 GB/s; flash UFS 4.1, 4.2 GB/s nominal. " +
      "Last column: co-inference ceiling of a Snapdragon 8 Elite Extreme Gen 6 (SM8975: LPDDR6 113.8 GB/s, UFS 5.0 10.8 GB/s) + Guangyu TC1000 " +
      "(10 GB 3D DRAM, 1,200 GB/s), weights split by bandwidth and read in parallel.",
      0.45, 0.92, 12.4, 0.52, { fs: 9.5, color: C.accent4, valign: "top" });

    const cols = [
      ["Model", 1.85], ["Quant · file", 1.4], ["Deployment", 2.18], ["Prefill tok/s", 0.92], ["Compute", 0.7],
      ["Decode tok/s", 0.92], ["Ceiling", 0.68], ["Attained", 1.42], ["Bottleneck", 1.3], ["8975+TC1000\nceiling", 1.06],
    ];
    const x0 = 0.45, y0 = 1.5, rh = 0.46, hh = 0.48;
    let x = x0;
    rect(s, x0, y0, 12.43, hh, HEX.dk2);
    const colX = cols.map(([h, w], k) => {
      const cx = x;
      text(s, h, cx + 0.06, y0, w - 0.1, hh, { fs: 9.5, bold: true, color: "FFFFFF", align: (k >= 3 && k <= 6) || k === 9 ? "right" : "left" });
      x += w;
      return cx;
    });
    rows.forEach((r, i) => {
      const y = y0 + hh + i * rh;
      if (i % 2 === 1) rect(s, x0, y, 12.43, rh, HEX.lt2);
      const big = r.m.includes("125B") || r.m.includes("27B");
      const v = (n, f) => (n == null ? "—" : n.toFixed(f));
      const cell = (k, t, o = {}) => text(s, t, colX[k] + 0.06, y, cols[k][1] - 0.1, rh, o);
      cell(0, r.m, { bold: true, fs: 10 });
      cell(1, r.q, { fs: 9, color: C.accent4 });
      cell(2, r.how, { fs: 9 });
      cell(3, v(r.pp, r.pp != null && r.pp < 10 ? 2 : 1), { fs: 10.5, bold: true, align: "right" });
      cell(4, r.pp == null ? "—" : `${(2 * r.act * r.pp / 1000).toFixed(r.pp < 20 ? 2 : 1)} TF`, { fs: 9, color: C.accent4, align: "right" });
      cell(5, v(r.tg, r.tg != null && r.tg < 1 ? 3 : 2), { fs: 10.5, bold: true, align: "right", color: big ? C.accent3 : C.text1 });
      cell(6, r.cap.toFixed(r.cap < 10 ? 2 : 1), { fs: 9.5, align: "right", color: C.accent4 });
      const bx = colX[7] + 0.1, bw = cols[7][1] - 0.7, by = y + rh / 2 - 0.08;
      rect(s, bx, by, bw, 0.16, "E2E8EE");
      if (r.tg != null) {
        const pct = Math.min(1, r.tg / r.cap);
        rect(s, bx, by, Math.max(0.02, bw * pct), 0.16, pct >= 0.45 ? HEX.accent5 : pct >= 0.3 ? HEX.accent1 : HEX.accent3);
        cell(7, `${Math.round(pct * 100)}%`, { fs: 9.5, bold: true, align: "right" });
      } else {
        cell(7, "n/a", { fs: 9.5, align: "right", color: C.accent4 });
      }
      cell(8, r.bn, { fs: 9, color: C.accent4 });
      cell(9, r.co >= 100 ? Math.round(r.co).toLocaleString("en-US") : r.co.toFixed(1), { fs: 10.5, bold: true, align: "right", color: C.accent1 });
    });
    const yEnd = y0 + hh + rows.length * rh;
    s.addShape(pres.shapes.LINE, { x: x0, y: yEnd, w: 12.43, h: 0, line: { color: "C9D3DD", width: 0.75 } });

    const ty = yEnd + 0.1;
    const notes = [
      ["Small models", "Reach 30–54% of the bandwidth ceiling; the all-NPU Gemma 4 E2B / E4B rank highest. E2B is fastest: 28.4 tok/s decode, 1,184 tok/s prefill (5.4 TFLOPS effective)."],
      ["Beyond RAM", "125B reads only ~0.78 GB of experts per token from flash: ceiling 5.4 tok/s, measured 1.12. 27B reads 5.5 GB per forward pass; MTP spreads it over 2.2 tokens: ceiling 1.69, measured 0.61."],
      ["Method", "In-app runs are plain decode (no speculation); 12B was measured with the CLI using the app's settings, 27B / 125B with the CLI (27B with MTP, ceiling ×2.2). Ceilings count weight reads only, so they are strict upper bounds; thermal CPU caps can change a result 4×."],
    ];
    notes.forEach(([h, b], i) => {
      text(s, h, 0.45, ty + i * 0.4, 1.3, 0.38, { fs: 10, bold: true, color: C.accent1, valign: "top" });
      text(s, b, 1.75, ty + i * 0.4, 11.1, 0.38, { fs: 9.5, valign: "top" });
    });
    s.addNotes(NOTES.p1);
  }

  // ================= page 2: how 12B / 27B / 125B run
  pres.addSection({ title: "Deployment methods" });
  {
    const s = pres.addSlide({ masterName: "Brief", sectionTitle: "Deployment methods" });
    s.addText("How the 12B, 27B and 125B models run on a 12 GB phone", { placeholder: "title" });
    text(s, "Constraints: ~3 GB usable by the NPU (32-bit address window; rpcmem cannot be swapped out), ~6 GB free RAM, 3.9 GB/s sequential flash reads. " +
      "The larger the model, the more it comes down to: how many bytes per token, read from where, computed by what.",
      0.45, 0.92, 12.4, 0.52, { fs: 10, color: C.accent4, valign: "top" });

    const cards = [
      {
        k: "12B", t: "Gemma 4 12B · Q4_0 QAT · 6.98 GB", c: "Problem: 2.3× the NPU budget; the CPU half competes for RAM", acc: HEX.accent1,
        steps: [
          "Placement from per-layer tile bytes: 24 layers on the NPU, filling its ~3 GB budget (HVX decode, HMX prefill); the other 24 layers on the CPU",
          "CPU layers stay mmap-ed file pages (no CPU repack): ~3.1 GB of reclaimable page cache. With the CPU's default repack copy, free RAM fell to 554 MB and speed collapsed",
          "Prefill offload: large-batch matmuls of CPU layers become 32×32 tiles on the NPU's HMX, so prefill still reaches 95 tok/s with half the layers on the CPU",
          "Decode on 2 threads (4 / 6 threads drop to 0.46 / 0.34 tok/s from sync overhead); chats use Gemma 4's own template and tool-call parser",
        ],
        res: [["decode", "3.33 tok/s"], ["prefill", "95.4 tok/s"]],
      },
      {
        k: "27B", t: "Qwen3.8 27B · IQ2_XS · 8.76 GB", c: "Problem: weights exceed free RAM, so every forward pass reads flash", acc: HEX.accent3,
        steps: [
          "IQ codebook transcoding: IQ1/2/3 become 3/4-bit codes + vlut32 lookups that reuse the NPU's Q3_K / IQ4_NL kernels; all 8 IQ types run on the NPU",
          "Last 14 layers resident on the NPU; first 50 layers (5.5 GB) streamed with O_DIRECT into a double buffer, reading the next layer while computing this one (3.2× faster than mmap page faults)",
          "MTP speculative decoding: 3 drafted tokens verified per pass, ~40% accepted, 2.2 tokens per forward pass",
          "Long-prompt prefill offloaded to the NPU (1.54 → 16.46 tok/s); system-prompt state saved to flash, restored in 0.09 s",
        ],
        res: [["decode", "0.61 tok/s"], ["long-prompt prefill", "16.5 tok/s"]],
      },
      {
        k: "125B", t: "Qwen3.8-Flash-Next 125B MoE · UD-IQ1_S · 72.5 GB", c: "Problem: 6× the RAM, so it can only live on flash", acc: HEX.accent5,
        steps: [
          "Dense part (attention, DeltaNet, shared experts, routers) of 40 layers on the NPU; 24,576 routed experts and the 29 GB PLE table stay on flash (mmap, MADV_RANDOM)",
          "Routing-aware prefetch: the layer's top-10 experts are read into the page cache by 6 parallel pread threads; router(l+1)·x(l) predicts the next layer's experts, read in the background",
          "Flash reads cut from 1.2 TB to 60 GB per run (page-fault readahead dragged in neighbouring experts)",
          "8,555 graph nodes / 238 NPU–CPU splits per pass: decode uses 2 threads to avoid barrier overhead",
        ],
        res: [["decode", "1.12 tok/s"], ["vs. first run", "18×"]],
      },
    ];
    const cw = 4.05, gap = 0.14, cy = 1.52, ch = 5.0;
    cards.forEach((c, i) => {
      const cx = 0.45 + i * (cw + gap);
      rect(s, cx, cy, cw, ch, HEX.lt2, { round: true, name: `card-${c.k}` });
      rect(s, cx, cy, cw, 0.08, c.acc);
      text(s, c.k, cx + 0.2, cy + 0.2, 1.15, 0.55, { fs: 26, bold: true, color: c.acc });
      text(s, c.t, cx + 1.3, cy + 0.18, cw - 1.45, 0.6, { fs: 9.5, bold: true });
      text(s, c.c, cx + 0.2, cy + 0.82, cw - 0.4, 0.36, { fs: 9.5, color: C.accent4, valign: "top" });
      c.steps.forEach((st, j) => {
        const sy = cy + 1.22 + j * 0.8;
        s.addShape(pres.shapes.OVAL, { x: cx + 0.2, y: sy + 0.02, w: 0.26, h: 0.26, fill: { color: c.acc }, line: { color: c.acc } });
        text(s, String(j + 1), cx + 0.2, sy + 0.02, 0.26, 0.26, { fs: 9, bold: true, color: "FFFFFF", align: "center" });
        text(s, st, cx + 0.55, sy, cw - 0.72, 0.78, { fs: 9, valign: "top" });
      });
      const ry = cy + ch - 0.62;
      s.addShape(pres.shapes.LINE, { x: cx + 0.2, y: ry - 0.05, w: cw - 0.4, h: 0, line: { color: "D5DDE5", width: 0.75 } });
      c.res.forEach(([lab, val], j) => {
        const rx = cx + 0.2 + j * ((cw - 0.4) / 2);
        text(s, val, rx, ry, (cw - 0.4) / 2, 0.36, { fs: 15, bold: true, color: c.acc });
        text(s, lab, rx, ry + 0.34, (cw - 0.4) / 2, 0.22, { fs: 9, color: C.accent4 });
      });
    });

    const fy = cy + ch + 0.1;
    text(s, "Shared base", 0.45, fy, 1.2, 0.42, { fs: 10, bold: true, color: C.accent1 });
    text(s, "--load-mode mmap (non-NPU weights stay reclaimable file pages) · rpcmem budget guard (keeps 2 GiB free; no more device reboots) · " +
      "weights repacked into 32×32 tiles on multiple threads · 2 decode / 6 prefill threads · foreground service keeps work alive when locked",
      1.65, fy, 11.2, 0.42, { fs: 9.5 });
    s.addNotes(NOTES.p2);
  }

  const out = path.join(__dirname, "..", "EdgeAgent-Briefing-2pages-EN.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(out);
}

main().catch((e) => { console.error(e); process.exit(1); });
