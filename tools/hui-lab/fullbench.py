#!/usr/bin/env python3
"""Full benchmark matrix for one device. Rounds interleave the cases so slow drift does not favor one of them.

Each process run is one sample and is saved at once, so an interrupted run keeps its data and can resume.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
OPT = "/data/local/tmp/llama.cpp"
BASE = "/data/local/tmp/hl/base"
OUT = "/data/local/tmp/hl/out"
PROMPT = "Explain in three sentences why the sky is blue."
PERF = re.compile(r"(load time|prompt eval time|eval time|total time) =\s+([\d.]+) ms(?: /\s+(\d+))?")


def cases(model):
    htp = ["GGML_HEXAGON_DEVICES=HTP0", "GGML_HEXAGON_POWER_MODE=high_performance"]
    bench = ["-m", model, "-r", "1", "-o", "json"]
    pp_tg = ["-p", "128,512", "-n", "64"]
    npu = ["-dev", "HTP0", "-ngl", "99", "-fa", "on"]
    gen = ["-m", model, "-c", "2048", "-n", "96", "--temp", "0", "--seed", "1", "-no-cnv", "-p", PROMPT]
    c = {
        # optimized build against the upstream merge-base
        "opt_htp": (OPT, "llama-bench", htp, bench + pp_tg + npu + ["-t", "2"]),
        "base_htp": (BASE, "llama-bench", htp, bench + pp_tg + npu + ["-t", "2"]),
        "opt_cpu": (OPT, "llama-bench", [], bench + pp_tg + ["-dev", "none", "-ngl", "0", "-t", "6"]),
        "base_cpu": (BASE, "llama-bench", [], bench + pp_tg + ["-dev", "none", "-ngl", "0", "-t", "6"]),
        "opt_gpu": (OPT, "llama-bench", [], bench + pp_tg + ["-dev", "GPUOpenCL", "-ngl", "99", "-t", "2"]),
        # end to end, with load time and the generated text
        "opt_gen": (OPT, "llama-completion", htp, gen + ["--device", "HTP0", "-ngl", "99", "-fa", "on", "-t", "2"]),
        "base_gen": (BASE, "llama-completion", htp, gen + ["--device", "HTP0", "-ngl", "99", "-fa", "on", "-t", "2"]),
        "opt_gen_cpu": (OPT, "llama-completion", [], gen + ["--device", "none", "-ngl", "0", "-t", "6"]),
        # host threads
        "t1": (OPT, "llama-bench", htp, bench + pp_tg + npu + ["-t", "1"]),
        "t4": (OPT, "llama-bench", htp, bench + pp_tg + npu + ["-t", "4"]),
        "t6": (OPT, "llama-bench", htp, bench + pp_tg + npu + ["-t", "6"]),
        # power modes
        "power_balanced": (OPT, "llama-bench", ["GGML_HEXAGON_DEVICES=HTP0", "GGML_HEXAGON_POWER_MODE=balanced"], bench + pp_tg + npu + ["-t", "2"]),
        # partial offload: the fork's host-weight paths only act on layers that stay on the CPU
        "ngl16": (OPT, "llama-bench", htp, bench + pp_tg + npu[:2] + ["-ngl", "16", "-fa", "on", "-t", "6"]),
        "ngl16_base": (BASE, "llama-bench", htp, bench + pp_tg + npu[:2] + ["-ngl", "16", "-fa", "on", "-t", "6"]),
        "ngl16_stream": (OPT, "llama-bench", htp + ["LLAMA_WEIGHT_STREAM=1"], bench + pp_tg + npu[:2] + ["-ngl", "16", "-fa", "on", "-t", "6"]),
        "ngl16_layerpf": (OPT, "llama-bench", htp + ["LLAMA_LAYER_PREFETCH=1"], bench + pp_tg + npu[:2] + ["-ngl", "16", "-fa", "on", "-t", "6"]),
        "ngl16_nooffload": (OPT, "llama-bench", htp + ["GGML_HEXAGON_OFFLOAD=0"], bench + pp_tg + npu[:2] + ["-ngl", "16", "-fa", "on", "-t", "6"]),
        # host thread sensitivity of the CPU paths
        "cpu_t2": (OPT, "llama-bench", [], bench + pp_tg + ["-dev", "none", "-ngl", "0", "-t", "2"]),
        "cpu_t4": (OPT, "llama-bench", [], bench + pp_tg + ["-dev", "none", "-ngl", "0", "-t", "4"]),
        "ngl16_t2": (OPT, "llama-bench", htp, bench + pp_tg + npu[:2] + ["-ngl", "16", "-fa", "on", "-t", "2"]),
        "ngl16_t4": (OPT, "llama-bench", htp, bench + pp_tg + npu[:2] + ["-ngl", "16", "-fa", "on", "-t", "4"]),
    }
    return c


def run_case(env, serial, name, spec, idx, cool, wake):
    pkg, tool, evars, args = spec
    prefix = f"{OUT}/{name}_{idx}"
    quoted = " ".join("'" + a.replace("'", "'\\''") + "'" for a in args)
    shell = f"{'WAKE_THERMAL=1 ' if wake else ''}COOL_MC={cool} sh /data/local/tmp/hl/run_case.sh {pkg} {prefix} {tool} {' '.join(evars)} -- {quoted}"
    r = subprocess.run(["adb", "-s", serial, "shell", shell], env=env, text=True, capture_output=True, timeout=1800)
    if r.returncode:
        raise RuntimeError(f"adb failed: {r.stderr[-300:]}")
    read = lambda ext: subprocess.run(["adb", "-s", serial, "shell", f"cat {prefix}.{ext}"], env=env, text=True, errors="replace", capture_output=True, timeout=120).stdout
    meta = json.loads(read("meta"))
    out, err = read("out"), read("err")
    rec = {"case": name, "idx": idx, "pkg": pkg, "tool": tool, "env": evars, "args": args, "meta": meta, "time": time.time()}
    if tool == "llama-bench":
        try:
            rec["tests"] = [{k: t.get(k) for k in ("n_prompt", "n_gen", "avg_ts", "stddev_ts", "avg_ns", "n_threads", "n_gpu_layers", "backends")} for t in json.loads(out[out.index("["):out.rindex("]") + 1])]
        except ValueError:
            rec["error"] = err[-1500:]
    else:
        rec["perf"] = {m[0]: {"ms": float(m[1]), "n": int(m[2]) if m[2] else None} for m in PERF.findall(err)}
        rec["text"] = out
        if not rec["perf"]:
            rec["error"] = err[-1500:]
    rec["err_tail"] = err[-600:] if meta["rc"] else ""
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", required=True)
    p.add_argument("--adb-port", type=int)
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--cool", type=int, default=50000)
    p.add_argument("--wake", action="store_true", help="wake the screen and wait for thermal status 0 before each run")
    p.add_argument("--only", help="comma separated cases")
    a = p.parse_args()
    env = dict(os.environ)
    if a.adb_port:
        env["ANDROID_ADB_SERVER_PORT"] = str(a.adb_port)
    out = LAB / "results" / "full" / f"{a.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["case"], r["idx"]))
    matrix = cases(a.model)
    if a.only:
        matrix = {k: matrix[k] for k in a.only.split(",")}
    # the main A/B cases get every round, the rest three
    main_cases = {"opt_htp", "base_htp", "opt_gen", "base_gen"}
    for i in range(a.rounds):
        for name, spec in matrix.items():
            if (name, i) in done or (name not in main_cases and i >= 3):
                continue
            for attempt in range(3):
                try:
                    rec = run_case(env, a.serial, name, spec, i, a.cool, a.wake)
                    break
                except Exception as e:
                    print(f"{name}#{i} attempt {attempt}: {e}", flush=True)
                    time.sleep(30)
                    if not a.adb_port:
                        subprocess.run(["adb", "connect", a.serial], env=env, capture_output=True)
            else:
                print(f"{name}#{i} gave up", flush=True)
                continue
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            brief = rec.get("tests") or rec.get("perf") or rec.get("error", "")[-200:]
            if isinstance(brief, list):
                brief = " ".join(f"p{t['n_prompt']}n{t['n_gen']}={t['avg_ts']:.1f}" for t in brief)
            print(f"{name}#{i} rc={rec['meta']['rc']} {brief}", flush=True)
    print("MATRIX_DONE", flush=True)


if __name__ == "__main__":
    main()
