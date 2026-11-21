// Build report/Gemma4-E2B-Hexagon-验证报告.docx from report/report.json (see report_to_docx.py).
const fs = require('fs'), path = require('path');
const d = require('docx');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, HeadingLevel, AlignmentType, LevelFormat, Footer, PageNumber, TableLayoutType } = d;

const R = path.join(__dirname, 'report');
const blocks = JSON.parse(fs.readFileSync(path.join(R, 'report.json')));
const W = 9638; // A4 with 0.75 inch side margins
const FONT = { ascii: 'Calibri', hAnsi: 'Calibri', eastAsia: 'Microsoft YaHei', cs: 'Calibri' };
const MONO = { ascii: 'Consolas', hAnsi: 'Consolas', eastAsia: 'Microsoft YaHei', cs: 'Consolas' };
const C = { accent: '0B6B66', muted: '56635F', line: 'D5DDDA', head: 'E8EFED', ok: 'E0F0ED', warn: 'F9E9E4', npu: '0B6B66', gpu: '7C6BB0', cpu: '9A7A3C', red: 'B4472F' };

const run = (r, size = 21) => new TextRun({ text: r.t, bold: r.b, font: r.code ? MONO : FONT, size: r.small ? size - 3 : (r.code ? size - 2 : size), color: r.small ? C.muted : undefined });
const line = { style: BorderStyle.SINGLE, size: 4, color: C.line };
const borders = { top: line, bottom: line, left: line, right: line };
const cellPad = { top: 50, bottom: 50, left: 80, right: 80 };

function widths(t) {
  const cols = t.rows[0].length;
  if (t.text) return cols === 4 ? [1500, 4000, 1800, 2338] : Array(cols).fill(Math.floor(W / cols));
  const first = cols >= 5 ? 2440 : 3238;
  const rest = Math.floor((W - first) / (cols - 1));
  const out = [first, ...Array(cols - 1).fill(rest)];
  out[cols - 1] += W - out.reduce((a, b) => a + b, 0);
  return out;
}

function table(t) {
  const ws = widths(t);
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: ws, layout: TableLayoutType.FIXED,
    rows: t.rows.map((row, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: row.map((cell, j) => new TableCell({
        width: { size: ws[j], type: WidthType.DXA }, borders, margins: cellPad,
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: C.head, color: 'auto' } : undefined,
        children: [new Paragraph({
          alignment: !t.text && j > 0 ? AlignmentType.RIGHT : AlignmentType.LEFT,
          children: cell.map(r => run(i === 0 ? { ...r, b: true } : r, 18)),
        })],
      })),
    })),
  });
}

function chart(c) {
  const ws = [2700, 5738, 1200];
  const color = cl => ({ 'c-npu': C.npu, 'c-gpu': C.gpu, 'c-cpu': C.cpu, 'c-warn': C.red }[cl] || C.npu);
  const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
  const nb = { top: none, bottom: none, left: none, right: none };
  const rows = c.rows.map(b => new TableRow({ children: [
    new TableCell({ width: { size: ws[0], type: WidthType.DXA }, borders: nb, children: [new Paragraph({ children: [new TextRun({ text: b.bl, font: FONT, size: 18, color: C.muted })] })] }),
    new TableCell({ width: { size: ws[1], type: WidthType.DXA }, borders: nb, children: [new Paragraph({ children: [new TextRun({ text: '█'.repeat(Math.max(1, Math.round(b.w / 100 * 30))), font: 'Arial', size: 16, color: color(b.cl) })] })] }),
    new TableCell({ width: { size: ws[2], type: WidthType.DXA }, borders: nb, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: b.bv, font: FONT, size: 18 })] })] }),
  ] }));
  return [
    new Paragraph({ keepNext: true, spacing: { before: 160, after: 40 }, children: [new TextRun({ text: c.title, bold: true, font: FONT, size: 20 }), new TextRun({ text: '  ' + c.unit, font: FONT, size: 17, color: C.muted })] }),
    new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: ws, layout: TableLayoutType.FIXED, rows }),
  ];
}

const lists = new Map();
const numbering = [];
const children = [];
let fill = null;
for (const b of blocks) {
  if (b.k === 'callout_start') { fill = b.kind === 'ok' ? C.ok : C.warn; continue; }
  if (b.k === 'callout_end') { fill = null; continue; }
  if (b.k === 'table') { children.push(table(b), new Paragraph({ spacing: { after: 100 }, children: [] })); continue; }
  if (b.k === 'chart') { children.push(...chart(b), new Paragraph({ spacing: { after: 60 }, children: [] })); continue; }
  if (b.k === 'h1') { children.push(new Paragraph({ heading: HeadingLevel.TITLE, spacing: { after: 120 }, children: b.runs.map(r => new TextRun({ text: r.t, bold: true, font: FONT, size: 40 })) })); continue; }
  if (b.k === 'h2' || b.k === 'h3') {
    children.push(new Paragraph({ heading: b.k === 'h2' ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2, keepNext: true, children: b.runs.map(r => new TextRun({ text: r.t, bold: true, font: FONT })) }));
    continue;
  }
  if (b.k === 'li') {
    const [tag, id] = b.list;
    if (!lists.has(id)) {
      const ref = `l${id}`;
      lists.set(id, ref);
      numbering.push({ reference: ref, levels: [{ level: 0, format: tag === 'ol' ? LevelFormat.DECIMAL : LevelFormat.BULLET, text: tag === 'ol' ? '%1.' : '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 280 } } } }] });
    }
    children.push(new Paragraph({ numbering: { reference: lists.get(id), level: 0 }, spacing: { after: 70, line: 300 }, children: b.runs.map(r => run(r)) }));
    continue;
  }
  children.push(new Paragraph({
    spacing: { after: 100, line: 300 },
    shading: fill ? { type: ShadingType.CLEAR, fill, color: 'auto' } : undefined,
    children: b.runs.map(r => b.note ? new TextRun({ text: r.t, font: r.code ? MONO : FONT, size: 18, color: C.muted, bold: r.b }) : run(r)),
  }));
}

const doc = new Document({
  creator: 'Claude', title: 'Gemma 4 E2B 在 Hexagon NPU 上的推理验证',
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Title', name: 'Title', basedOn: 'Normal', next: 'Normal', run: { font: FONT, size: 40, bold: true, color: '17211F' } },
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 30, bold: true, color: C.accent }, paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 24, bold: true, color: '17211F' }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: numbering },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT, ' / ', PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: C.muted })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => {
  const out = path.join(R, 'Gemma4-E2B-Hexagon-验证报告.docx');
  fs.writeFileSync(out, buf);
  console.log(out, buf.length);
});
