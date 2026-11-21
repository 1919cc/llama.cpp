#!/usr/bin/env python3
# Fill report.html from metrics.json and print it to PDF with headless Chrome.
import json
import re
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

metrics = json.loads((DOCS / "metrics.json").read_text(encoding="utf-8"))
html = (DOCS / "report.html").read_text(encoding="utf-8")

missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", html)) - set(metrics))
if missing:
    sys.exit(f"metrics.json is missing: {', '.join(missing)}")

for k, v in metrics.items():
    html = html.replace("{{" + k + "}}", str(v))

filled = DOCS / "build" / "report.filled.html"
filled.write_text(html, encoding="utf-8")

out = DOCS / "EdgeAgent-端侧大模型推理技术方案.pdf"
subprocess.run([
    CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
    f"--print-to-pdf={out}", filled.as_uri(),
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(out)
