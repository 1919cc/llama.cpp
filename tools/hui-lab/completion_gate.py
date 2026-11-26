#!/usr/bin/env python3
"""A/B speed comparison of engine builds on real prompts (llama-completion), measured only at full clocks.

Same rules as bench_gate.py (full clocks, thermal status 0, launcher in front, cached apps killed, builds
alternating run by run), but every run generates from a real chat prompt instead of llama-bench's random
tokens: what a MoE expert cache sees in use depends on the text. Reports prompt (prefill) and generation
(decode) tokens/s from llama-completion's perf lines.

usage:
  completion_gate.py --model /data/local/tmp/gguf/Inkling-2-Mini-Q4_0.gguf --prompts prompts.txt \\
      --build 'base=/data/local/tmp/hui-base?LLAMA_EXPERT_PREFETCH=2+-t 2' \\
      --build 'opt=/data/local/tmp/hui-opt?LLAMA_HOT_EXPERTS_MB=2048+-t 4 -C 0x3C --cpu-strict 1' \\
      --rounds 3 --label ink-final -- -n 128 -ngl 99 -fa on --device HTP0
One line per prompt in prompts.txt (UTF-8). Results: results/<label>/runs.jsonl and a summary on stdout.
"""
import argparse
import json
import re
import shlex
import statistics
import time

from bench_gate import LAB, adb, wait_full_clocks

PERF = re.compile(r"(prompt eval|eval) time =\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs)")


def run_completion(args, build, prompt, extra):
    """(prefill t/s, decode t/s) of one llama-completion run, None if it printed no perf lines."""
    path, _, rest = build.partition("+")
    build_dir, _, env = path.partition("?")
    env = env.replace("&", " ")
    cmd = (f"cd {build_dir} && LD_LIBRARY_PATH={build_dir}/lib ADSP_LIBRARY_PATH={build_dir}/lib {env} "
           f"./bin/llama-completion -m {args.model} {rest} {extra} --temp 0 --jinja -cnv -st "
           f"-p {shlex.quote(prompt)} </dev/null 2>&1 | grep -E 'eval time ='")
    out = adb(args, cmd, check=False)
    res = {}
    for kind, ms, n in PERF.findall(out):
        res[kind] = int(n) / (float(ms) / 1000.0) if float(ms) > 0 else 0.0
    if "eval" not in res:
        print(f"  no perf lines: {out[-400:]}", flush=True)
        return None
    return res.get("prompt eval", 0.0), res["eval"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="text file, one prompt per line")
    ap.add_argument("--build", action="append", required=True,
                    help="NAME=device dir, optionally ?VAR=VAL&VAR2=VAL (env) and +ARGS (args for this build)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--label", default="completion-ab")
    ap.add_argument("--serial")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--max-wait", type=float, default=3600.0)
    ap.add_argument("--cooldown", type=float, default=20.0)
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="llama-completion args for every build")
    args = ap.parse_args()
    extra = " ".join(shlex.quote(a) for a in args.extra if a != "--")
    builds = [b.split("=", 1) for b in args.build]
    with open(args.prompts, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    adb(args, "cmd power set-fixed-performance-mode-enabled true; settings put global low_power 0", check=False)
    out_dir = LAB / "results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    try:
        for rnd in range(args.rounds):
            order = builds if rnd % 2 == 0 else builds[::-1]
            for name, build in order:
                for pi, prompt in enumerate(prompts):
                    adb(args, "am kill-all", check=False)
                    detail, waited = wait_full_clocks(args)
                    detail["mem_avail_mb"] = int(adb(args, "grep MemAvailable /proc/meminfo").split()[1]) // 1024
                    r = run_completion(args, build, prompt, extra)
                    if r is None:
                        continue
                    rec = {"build": name, "round": rnd, "prompt": pi, "prefill_tps": r[0], "decode_tps": r[1],
                           "waited_s": round(waited, 1), "clocks": detail, "model": args.model, "args": build,
                           "extra": extra, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                    runs.append(rec)
                    print(f"round {rnd} {name:>6} p{pi}: prefill {r[0]:7.2f} decode {r[1]:6.2f} t/s "
                          f"(skin {detail['skin_c']} C, avail {detail['mem_avail_mb']} MB, waited {waited:.0f} s)",
                          flush=True)
                    time.sleep(args.cooldown)
    finally:
        adb(args, "cmd power set-fixed-performance-mode-enabled false", check=False)
        with open(out_dir / "runs.jsonl", "a", encoding="utf-8") as f:
            for r in runs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsummary ({args.rounds} rounds x {len(prompts)} prompts, full clocks every run), {args.model}")
    base = builds[0][0]
    for metric in ("prefill_tps", "decode_tps"):
        means = {}
        for name, _ in builds:
            v = [r[metric] for r in runs if r["build"] == name]
            if not v:
                continue
            means[name] = statistics.mean(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            delta = (means[name] / means[base] - 1) * 100 if name != base and base in means else 0.0
            print(f"  {metric:>11} {name:>6}: {means[name]:7.2f} ± {sd:5.2f} t/s  ({delta:+.1f}% vs {base})")


if __name__ == "__main__":
    main()
