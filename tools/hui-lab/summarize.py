#!/usr/bin/env python3
"""Aggregate results/full/*.jsonl into results/full/summary.json and print a table."""
import json
import statistics as st
from pathlib import Path

D = Path(__file__).resolve().parent / "results" / "full"


def stat(v):
    return {"n": len(v), "median": st.median(v), "min": min(v), "max": max(v)}


summary = {}
for f in sorted(D.glob("*.jsonl")):
    dev = f.stem
    recs = [json.loads(l) for l in f.read_text().splitlines()]
    cases = {}
    for r in recs:
        c = cases.setdefault(r["case"], {"metrics": {}, "rc": [], "hwm": [], "minav": [], "peak": [], "start": [], "cap": [], "wait": [], "texts": set(), "ts": [], "wake": [], "cap0": [], "cap6": []})
        m = r["meta"]
        c["rc"].append(m["rc"])
        c["hwm"].append(m["vm_hwm_kb"] / 1024)
        c["minav"].append(m["min_avail_kb"] / 1024)
        c["peak"].append(m["peak_mc"] / 1000)
        c["start"].append(m["start_mc"] / 1000)
        c["cap"].append(m["min_big_cap_khz"] / 1000)
        c["wait"].append(m["cool_wait_s"])
        c["ts"].append(m.get("thermal_status"))
        c["wake"].append(m.get("wakefulness"))
        c["cap0"].append(m.get("cap0_khz", 0) / 1000)
        c["cap6"].append(m.get("cap6_khz", 0) / 1000)
        for t in r.get("tests", []):
            key = f"pp{t['n_prompt']}" if t["n_prompt"] else f"tg{t['n_gen']}"
            c["metrics"].setdefault(key, []).append(t["avg_ts"])
        p = r.get("perf")
        if p:
            c["metrics"].setdefault("load_ms", []).append(p["load time"]["ms"])
            c["metrics"].setdefault("prompt_ms", []).append(p["prompt eval time"]["ms"])
            c["metrics"].setdefault("eval_ts", []).append(p["eval time"]["n"] * 1000 / p["eval time"]["ms"])
            c["metrics"].setdefault("eval_n", []).append(p["eval time"]["n"])
            c["texts"].add(r["text"].strip())
    out = {}
    for name, c in cases.items():
        out[name] = {
            "metrics": {k: stat(v) for k, v in c["metrics"].items()},
            "failures": sum(1 for x in c["rc"] if x),
            "rss_mb": max(c["hwm"]), "min_avail_mb": min(c["minav"]), "peak_c": max(c["peak"]), "start_c": st.median(c["start"]),
            "min_cap_mhz": min(c["cap"]), "cool_wait_s": sum(c["wait"]), "texts": sorted(c["texts"]),
            "thermal": sorted({str(x) for x in c["ts"]}), "wake": sorted({str(x) for x in c["wake"]}), "cap0_mhz": [min(c["cap0"]), max(c["cap0"])], "cap6_mhz": [min(c["cap6"]), max(c["cap6"])],
        }
    summary[dev] = out
    print(f"== {dev} ({len(recs)} runs)")
    for name, c in out.items():
        ms = " ".join(f"{k}={v['median']:.1f}[{v['min']:.1f},{v['max']:.1f}]" for k, v in c["metrics"].items())
        print(f"{name:15} fail={c['failures']} rss={c['rss_mb']:.0f}M minav={c['min_avail_mb']:.0f}M start={c['start_c']:.0f}C peak={c['peak_c']:.0f}C cap={c['min_cap_mhz']:.0f} wait={c['cool_wait_s']}s ts={c['thermal']} wake={c['wake']} cap0={c['cap0_mhz']} cap6={c['cap6_mhz']} {ms}")
(D / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
