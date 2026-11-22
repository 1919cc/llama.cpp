#!/usr/bin/env python3
"""A/B speed comparison of engine builds on the phone, measured only at full clocks.

Rules (docs/hui: every speed claim of hui_infer comes from runs like these):
  - CPU, GPU and NPU must be at their highest clocks: fixed performance mode on, low power off,
    thermal status 0, every thermal cooling device (cpu clusters, kgsl/gpu, cdsp/cdsp_hw) at 0,
    and every CPU cluster's scaling_max_freq equal to its cpuinfo_max_freq. The NPU itself runs
    in the backend's burst DCVS mode (GGML_HEXAGON_POWER_MODE default).
  - One model at a time, one llama-bench at a time; cached background apps are killed before each
    run (am kill-all) so every run starts with the same free memory.
  - Throttled: wait until the phone is back at full clocks before the next run.
  - Builds alternate run by run (A B, B A, ...) so slow drift hits both alike.

usage:
  bench_gate.py --model /data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf \\
      --build r29=/data/local/tmp/hui-ndk29 --build r30=/data/local/tmp/hui-ndk30 \\
      --rounds 3 --label ndk-ab -- -p 512 -n 64 -ngl 99 -fa 1 -t 4 --device HTP0
Results: results/<label>/runs.jsonl and a summary on stdout.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ADB = (os.environ.get("ADB") or shutil.which("adb")
       or str(Path.home() / "Library/Android/sdk/platform-tools/adb"))


def adb(args, cmd, check=True):
    base = [ADB] + (["-s", args.serial] if args.serial else [])
    r = subprocess.run(base + ["shell", cmd], text=True, capture_output=True)
    if check and r.returncode != 0:
        sys.exit(f"adb shell failed ({r.returncode}): {cmd}\n{r.stdout}{r.stderr}")
    return r.stdout


def clocks(args):
    """(full, detail): whether every clock is at its maximum and what was read."""
    caps = adb(args, "for p in /sys/devices/system/cpu/cpufreq/policy*; do "
                     "echo $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq); done").split("\n")
    pairs = [tuple(map(int, l.split())) for l in caps if l.strip()]
    thermal = adb(args, "dumpsys thermalservice")
    status = int(re.search(r"Thermal Status: (\d+)", thermal).group(1))
    hal = thermal.split("Current cooling devices from HAL:")[-1]
    cooling = {m.group(2): int(m.group(1))
               for m in re.finditer(r"CoolingDevice\{mValue=(\d+), mType=\d+, mName=([^}]+)\}", hal)}
    skin = re.search(r"mValue=([\d.]+), mType=3, mName=skin", thermal)
    full = status == 0 and all(v == 0 for v in cooling.values()) and all(a == b for a, b in pairs)
    detail = {
        "cpu_caps_khz": [a for a, _ in pairs],
        "cpu_max_khz": [b for _, b in pairs],
        "thermal_status": status,
        "cooling_nonzero": {k: v for k, v in cooling.items() if v},
        "skin_c": float(skin.group(1)) if skin else None,
    }
    return full, detail


def wait_full_clocks(args):
    start = time.time()
    while True:
        full, detail = clocks(args)
        if full:
            return detail, time.time() - start
        if time.time() - start > args.max_wait:
            sys.exit(f"still throttled after {args.max_wait} s: {detail}")
        print(f"  throttled, waiting: {detail}", flush=True)
        time.sleep(args.poll)


def run_bench(args, build_dir, bench_args):
    env = f"LD_LIBRARY_PATH={build_dir}/lib ADSP_LIBRARY_PATH={build_dir}/lib"
    cmd = (f"cd {build_dir} && {env} ./bin/llama-bench -m {args.model} -r 1 -o jsonl "
           + " ".join(bench_args))
    for attempt in range(2):  # a busy adb or DSP session can drop one run; retry it once
        base = [ADB] + (["-s", args.serial] if args.serial else [])
        r = subprocess.run(base + ["shell", cmd + " 2>&1"], text=True, capture_output=True)
        rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
        if rows:
            return rows
        print(f"  llama-bench printed no results (attempt {attempt + 1}, exit {r.returncode}):\n"
              f"{(r.stdout + r.stderr)[-1500:]}", flush=True)
        time.sleep(args.cooldown)
    sys.exit("llama-bench failed twice")


def test_name(row):
    if row.get("n_prompt", 0) and not row.get("n_gen", 0):
        return f"pp{row['n_prompt']}"
    if row.get("n_gen", 0) and not row.get("n_prompt", 0):
        return f"tg{row['n_gen']}"
    return f"pp{row.get('n_prompt', 0)}+tg{row.get('n_gen', 0)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model path on the phone")
    ap.add_argument("--build", action="append", required=True, help="NAME=device dir with bin/ and lib/")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--label", default="ab")
    ap.add_argument("--serial")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between clock checks")
    ap.add_argument("--max-wait", type=float, default=1800.0, help="give up after this many seconds throttled")
    ap.add_argument("--cooldown", type=float, default=20.0, help="pause after each run")
    ap.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    bench_args = [a for a in args.bench_args if a != "--"]
    builds = [b.split("=", 1) for b in args.build]

    adb(args, "cmd power set-fixed-performance-mode-enabled true; settings put global low_power 0", check=False)
    out_dir = LAB / "results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    try:
        for rnd in range(args.rounds):
            order = builds if rnd % 2 == 0 else builds[::-1]
            for name, path in order:
                # one model, one process: drop cached background apps so every run sees the same free memory
                adb(args, "am kill-all", check=False)
                detail, waited = wait_full_clocks(args)
                detail["mem_avail_mb"] = int(adb(args, "grep MemAvailable /proc/meminfo").split()[1]) // 1024
                rows = run_bench(args, path, bench_args)
                for row in rows:
                    rec = {"build": name, "round": rnd, "test": test_name(row), "tps": row["avg_ts"],
                           "waited_s": round(waited, 1), "clocks": detail, "model": args.model,
                           "bench_args": bench_args, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                    runs.append(rec)
                    print(f"round {rnd} {name:>6} {rec['test']:>8}: {rec['tps']:8.2f} t/s "
                          f"(skin {detail['skin_c']} C, avail {detail['mem_avail_mb']} MB, waited {waited:.0f} s)", flush=True)
                time.sleep(args.cooldown)
    finally:
        adb(args, "cmd power set-fixed-performance-mode-enabled false", check=False)
        with open(out_dir / "runs.jsonl", "a") as f:
            for r in runs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsummary ({args.rounds} rounds, full clocks every run), {args.model}")
    tests = sorted({r["test"] for r in runs})
    base = builds[0][0]
    for t in tests:
        means = {}
        for name, _ in builds:
            v = [r["tps"] for r in runs if r["build"] == name and r["test"] == t]
            means[name] = statistics.mean(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            delta = (means[name] / means[base] - 1) * 100 if name != base else 0.0
            print(f"  {t:>8} {name:>6}: {means[name]:8.2f} ± {sd:6.2f} t/s  ({delta:+.1f}% vs {base})")


if __name__ == "__main__":
    main()
