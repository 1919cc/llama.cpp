#!/usr/bin/env python3
"""Flatten report/report.html into report/report.json for report_to_docx.js."""
import json
from html.parser import HTMLParser
from pathlib import Path

R = Path(__file__).resolve().parent / "report"
BLOCK = {"h1", "h2", "h3", "p", "li"}


class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.stack, self.fmt = [], [], {"b": 0, "code": 0, "small": 0}
        self.block = self.table = self.row = self.cell = self.fig = self.bar = None
        self.in_main = False
        self.list_id = 0

    def attr(self, a, k):
        return dict(a).get(k) or ""

    def run(self, t):
        t = " ".join(t.split()) if not t.strip() else t.replace("\n", " ")
        if not t:
            return
        r = {"t": t, "b": bool(self.fmt["b"]), "code": bool(self.fmt["code"]), "small": bool(self.fmt["small"])}
        if self.cell is not None:
            self.cell.append(r)
        elif self.block is not None:
            self.block["runs"].append(r)

    def handle_starttag(self, tag, a):
        if tag == "main":
            self.in_main = True
        if not self.in_main:
            return
        cls = self.attr(a, "class")
        if tag in ("b", "strong"):
            self.fmt["b"] += 1
            if tag == "strong" and self.block is None and self.cell is None:
                self.block = {"k": "p", "runs": [], "own": True}
        elif tag in ("code", "small"):
            self.fmt[tag] += 1
        elif tag == "br":
            self.run(" / ")
        elif tag in ("ol", "ul"):
            self.list_id += 1
            self.stack.append((tag, self.list_id))
        elif tag == "section" or (tag == "div" and cls == "warn"):
            self.callout = "ok" if tag == "section" else "warn"
            self.out.append({"k": "callout_start", "kind": self.callout})
        elif tag == "strong" or tag in BLOCK:
            self.block = {"k": tag, "runs": []}
            if tag == "li":
                self.block["list"] = self.stack[-1]
            if tag == "p" and "note" in cls or tag == "p" and "sub" in cls:
                self.block["note"] = True
        elif tag == "table":
            self.table = {"k": "table", "text": "text" in cls, "rows": []}
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "figure":
            self.fig = {"k": "chart", "title": "", "unit": "", "rows": []}
        elif tag == "figcaption":
            self.cap = True
            self.block = {"k": "cap", "runs": []}
        elif tag == "i" and self.bar is not None:
            self.bar["cl"] = cls
            self.bar["w"] = float(self.attr(a, "style").split(":")[1].rstrip("%"))
        elif tag == "div" and cls == "bar":
            self.bar = {}
        elif tag == "span" and self.bar is not None and cls in ("bl", "bv"):
            self.barfield = cls
            self.block = {"k": "x", "runs": []}

    def handle_data(self, d):
        if self.in_main:
            self.run(d)

    def handle_endtag(self, tag):
        if tag == "main":
            self.in_main = False
        if not self.in_main:
            return
        if tag in ("b", "strong"):
            self.fmt["b"] -= 1
            if tag == "strong" and self.block is not None and self.block.get("own"):
                self.out.append(self.block)
                self.block = None
        elif tag in ("code", "small"):
            self.fmt[tag] -= 1
        elif tag in ("ol", "ul"):
            self.stack.pop()
        elif tag == "section" or (tag == "div" and getattr(self, "callout", None) and self.table is None and self.fig is None and self.bar is None and self.block is None and self.stack == [] and self._closes_warn()):
            self.out.append({"k": "callout_end"})
            self.callout = None
        elif tag in BLOCK and self.block is not None:
            self.out.append(self.block)
            self.block = None
        elif tag in ("td", "th"):
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr":
            self.table["rows"].append(self.row)
        elif tag == "table":
            self.out.append(self.table)
            self.table = None
        elif tag == "figcaption":
            runs = self.block["runs"]
            self.fig["title"], self.fig["unit"] = runs[0]["t"], runs[1]["t"] if len(runs) > 1 else ""
            self.block = None
        elif tag == "span" and self.bar is not None and self.block is not None and self.block["k"] == "x":
            self.bar[self.barfield] = "".join(r["t"] for r in self.block["runs"])
            self.block = None
        elif tag == "div" and self.bar is not None and "bv" in self.bar:
            self.fig["rows"].append(self.bar)
            self.bar = None
        elif tag == "figure":
            self.out.append(self.fig)
            self.fig = None

    def _closes_warn(self):
        # the warn box only holds paragraphs, so the first div end after a p belongs to it
        return self.callout == "warn" and self.out and self.out[-1]["k"] == "p"


p = P()
p.feed((R / "report.html").read_text())
(R / "report.json").write_text(json.dumps(p.out, ensure_ascii=False))
kinds = {}
for b in p.out:
    kinds[b["k"]] = kinds.get(b["k"], 0) + 1
print(kinds)
