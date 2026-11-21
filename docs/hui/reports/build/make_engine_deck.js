// Two-page deck on our llama.cpp changes (architecture + optimizations), Chinese and English editions from one
// layout. Usage: node make_engine_deck.js zh|en
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");
const T = require(process.argv[2] === "en" ? "./engine_text_en" : "./engine_text_zh");

const THEME = {
  name: "EdgeEngine",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "14202E", lt1: "FFFFFF", dk2: "0B4F4F", lt2: "F1F4F7",
    accent1: "0B6E6E", accent2: "13A3A3", accent3: "B5541C", accent4: "5B6676", accent5: "1F8A4C", accent6: "B42318",
    hlink: "0B6E6E", folHlink: "5B6676",
  },
};
const HEX = THEME.colors;
// module status colors (fill, border)
const ST = {
  new: { fill: "DDF1E4", line: "1F8A4C" },
  mod: { fill: "FBEBDD", line: "B5541C" },
  old: { fill: "F1F4F7", line: "C9D3DD" },
  app: { fill: "FFFFFF", line: "9AA5B1" },
};
// stage colors
const STAGE = { L: "2F3C7E", P: "7A3E9D", D: "0B6E6E" };

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = T.deckTitle;
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "Engine",
    background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.3, w: 12.4, h: 0.6, fontSize: 24, bold: true, color: C.text1, margin: 0, align: "left" }, text: "" } },
      { text: { text: T.footer, options: { x: 0.45, y: 7.08, w: 10, h: 0.25, fontSize: 9, color: C.accent4, margin: 0 } } },
    ],
    slideNumber: { x: 12.4, y: 7.06, w: 0.5, h: 0.28, fontSize: 9, color: HEX.accent4, align: "right" },
  });

  const text = (s, t, x, y, w, h, o = {}) => s.addText(t, {
    x, y, w, h, fontSize: o.fs || 10, bold: !!o.bold, color: o.color || C.text1, align: o.align || "left",
    valign: o.valign || "middle", margin: o.margin == null ? 0 : o.margin, isTextBox: true, paraSpaceAfter: o.psa || 0,
  });
  const box = (s, x, y, w, h, st, o = {}) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.05, fill: { color: st.fill }, line: { color: st.line, width: o.lw || 1, dashType: o.dash || "solid" },
    objectName: o.name,
  });
  const rect = (s, x, y, w, h, fill, line) => s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line || fill, width: 0.75 } });
  const arrow = (s, x, y, w, h, color, o = {}) => s.addShape(pres.shapes.LINE, {
    x, y, w, h, flipH: !!o.flipH, flipV: !!o.flipV, line: { color: color || HEX.accent4, width: o.width || 1.25, endArrowType: "triangle" },
  });
  const badge = (s, x, y, stage, label) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.24, h: 0.24, fill: { color: STAGE[stage] }, line: { color: STAGE[stage] } });
    text(s, label, x, y, 0.24, 0.24, { fs: 8.5, bold: true, color: "FFFFFF", align: "center" });
  };
  // a module: status-colored box, title line + detail line, stage badges in the top-right corner
  const module = (s, m) => {
    box(s, m.x, m.y, m.w, m.h, ST[m.st], { dash: m.st === "app" ? "dash" : "solid", name: m.name });
    // tall boxes carry their stage badges in the bottom-right corner, so the title gets the full width
    const tall = m.h >= 1.0, nb = m.stages ? m.stages.length : 0;
    text(s, m.t, m.x + 0.1, m.y + 0.04, m.w - 0.2 - (tall ? 0 : nb * 0.27), 0.26, { fs: 10, bold: true, valign: "top" });
    if (m.d) text(s, m.d, m.x + 0.1, m.y + 0.3, m.w - 0.2, m.h - 0.34 - (tall && nb ? 0.26 : 0), { fs: 8.5, color: C.accent4, valign: "top" });
    (m.stages || []).forEach((st, i) => badge(s, m.x + m.w - 0.3 - (nb - 1 - i) * 0.27, tall ? m.y + m.h - 0.3 : m.y + 0.06, st, T.stageMark[st]));
  };

  // ================= page 1: architecture
  pres.addSection({ title: T.sec1 });
  {
    const s = pres.addSlide({ masterName: "Engine", sectionTitle: T.sec1 });
    s.addText(T.p1Title, { placeholder: "title" });
    text(s, T.p1Sub, 0.45, 0.9, 8.4, 0.4, { fs: 10, color: C.accent4, valign: "top" });
    // legend
    const lg = [["new", T.legNew], ["mod", T.legMod], ["old", T.legOld]];
    lg.forEach(([k, l], i) => {
      const lx = 8.95 + i * 1.33;
      box(s, lx, 0.95, 0.26, 0.2, ST[k]);
      text(s, l, lx + 0.32, 0.92, 1.0, 0.26, { fs: 9 });
    });

    // layer labels on the left
    const X0 = 0.45, LW = 0.95, X1 = X0 + LW + 0.08, W = 8.85 - X1;
    const layers = T.layers; // [label, y, h]
    layers.forEach(([l, y, h]) => {
      rect(s, X0, y, LW, h, HEX.dk2);
      text(s, l, X0 + 0.06, y, LW - 0.12, h, { fs: 9.5, bold: true, color: "FFFFFF", align: "center" });
    });
    T.modules(X1, W).forEach((m) => module(s, m));

    // vertical flow arrows between layers (requests go down, data comes up)
    [[1.95, 2.08], [2.62, 2.75], [4.0, 4.13], [4.67, 4.8], [5.98, 6.11]].forEach(([y1, y2]) => arrow(s, 4.9, y1, 0, y2 - y1, "9AA5B1"));

    // stage data paths on the right
    const PX = 9.05, PW = 3.83;
    T.paths.forEach((p, i) => {
      const py = 1.42 + i * 1.86, ph = 1.74;
      rect(s, PX, py, PW, ph, "FFFFFF", "D5DDE5");
      rect(s, PX, py, PW, 0.34, STAGE[p.k]);
      text(s, p.title, PX + 0.12, py, PW - 0.24, 0.34, { fs: 10.5, bold: true, color: "FFFFFF" });
      p.steps.forEach((st, j) => {
        const sy = py + 0.42 + j * (1.26 / p.steps.length);
        s.addShape(pres.shapes.OVAL, { x: PX + 0.14, y: sy + 0.07, w: 0.09, h: 0.09, fill: { color: STAGE[p.k] }, line: { color: STAGE[p.k] } });
        if (j < p.steps.length - 1) s.addShape(pres.shapes.LINE, { x: PX + 0.185, y: sy + 0.16, w: 0, h: 1.26 / p.steps.length - 0.09, line: { color: STAGE[p.k], width: 1 } });
        text(s, st, PX + 0.32, sy, PW - 0.42, 1.26 / p.steps.length, { fs: 8.5, valign: "top" });
      });
    });
    s.addNotes(T.notes1);
  }

  // ================= page 2: optimizations
  pres.addSection({ title: T.sec2 });
  {
    const s = pres.addSlide({ masterName: "Engine", sectionTitle: T.sec2 });
    s.addText(T.p2Title, { placeholder: "title" });
    text(s, T.p2Sub, 0.45, 0.9, 12.4, 0.36, { fs: 10, color: C.accent4, valign: "top" });

    const TW = 4.05, TH = 2.72, G = 0.14, TX = 0.45, TY = 1.32;
    T.tiles.forEach((t, i) => {
      const x = TX + (i % 3) * (TW + G), y = TY + Math.floor(i / 3) * (TH + 0.12);
      rect(s, x, y, TW, TH, HEX.lt2, HEX.lt2);
      rect(s, x, y, 0.06, TH, t.color || HEX.accent1);
      text(s, t.n, x + 0.18, y + 0.1, 0.35, 0.3, { fs: 15, bold: true, color: t.color || HEX.accent1 });
      text(s, t.title, x + 0.52, y + 0.1, TW - 0.65, 0.3, { fs: 11.5, bold: true });
      // mini diagram area
      const dx = x + 0.18, dy = y + 0.5, dw = TW - 0.36, dh = 0.82;
      drawDiagram(s, t.diag, dx, dy, dw, dh, t.color || HEX.accent1);
      text(s, t.body, x + 0.18, y + 1.38, TW - 0.36, 0.78, { fs: 8.8, valign: "top" });
      // result
      s.addShape(pres.shapes.LINE, { x: x + 0.18, y: y + 2.18, w: TW - 0.36, h: 0, line: { color: "D5DDE5", width: 0.75 } });
      text(s, t.big, x + 0.18, y + 2.22, 1.7, 0.42, { fs: 17, bold: true, color: t.color || HEX.accent1 });
      text(s, t.small, x + 1.9, y + 2.22, TW - 2.05, 0.42, { fs: 8.5, color: C.accent4 });
    });
    s.addNotes(T.notes2);
  }

  // mini diagrams: chain (boxes with arrows) or timeline (double buffer)
  function drawDiagram(s, d, x, y, w, h, color) {
    if (d.type === "chain") {
      const n = d.items.length, gap = 0.22, bw = (w - gap * (n - 1)) / n;
      d.items.forEach((it, k) => {
        const bx = x + k * (bw + gap);
        const hl = d.hl && d.hl.includes(k);
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.08, w: bw, h: h - 0.16, rectRadius: 0.04,
          fill: { color: hl ? "FFFFFF" : "FFFFFF" }, line: { color: hl ? color : "B8C2CC", width: hl ? 1.5 : 0.75 } });
        text(s, it, bx + 0.03, y + 0.08, bw - 0.06, h - 0.16, { fs: 7.8, align: "center" });
        if (k < n - 1) arrow(s, bx + bw + 0.02, y + h / 2, gap - 0.04, 0, color, { width: 1.1 });
      });
    } else if (d.type === "timeline") {
      // two lanes: read (flash) and compute; read of layer k+1 overlaps compute of layer k
      const lanes = d.lanes, lh = (h - 0.08) / 2, cols = 4, cw = (w - 0.8) / (cols + 1);
      lanes.forEach((ln, r) => {
        const ly = y + 0.04 + r * lh;
        text(s, ln, x, ly, 0.78, lh, { fs: 7.8, color: HEX.accent4 });
        for (let k = 0; k < cols; k++) {
          const bx = x + 0.8 + (k + r) * cw;
          s.addShape(pres.shapes.RECTANGLE, { x: bx + 0.02, y: ly + 0.05, w: cw - 0.04, h: lh - 0.1,
            fill: { color: r === 0 ? "FFFFFF" : color }, line: { color, width: 1 } });
          text(s, d.cell(k), bx + 0.02, ly + 0.05, cw - 0.04, lh - 0.1, { fs: 7.5, align: "center", color: r === 0 ? HEX.dk1 : "FFFFFF", bold: r === 1 });
        }
      });
    } else if (d.type === "chips") {
      const n = d.items.length, gap = 0.1, bw = (w - gap * (n - 1)) / n;
      d.items.forEach((it, k) => {
        const bx = x + k * (bw + gap);
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.06, w: bw, h: h - 0.12, rectRadius: 0.04,
          fill: { color: "FFFFFF" }, line: { color, width: 1 } });
        text(s, it, bx + 0.05, y + 0.06, bw - 0.1, h - 0.12, { fs: 7.8, align: "center" });
      });
    }
  }

  const out = path.join(__dirname, "..", T.file);
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(out);
}

main().catch((e) => { console.error(e); process.exit(1); });
