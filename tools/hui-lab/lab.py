#!/usr/bin/env python3
"""Run the same llama.cpp Hexagon benchmark on the local phone and on QDC sessions.

Reuses scripts/snapdragon/run.py and run_qdc_jobs.py from the llama.cpp tree. See README.md.
"""
import argparse
import datetime
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = Path(os.environ.get("LLAMA_REPO", LAB.parent.parent))  # tools/hui-lab -> repo root
PKG = "pkg-adb"
IMAGE = "ghcr.io/snapdragon-toolchain/arm64-android:v0.8"
DEVICE_DIR = "/data/local/tmp/llama.cpp"
MODEL_DIR = "/data/local/tmp/gguf"
VENV_PY = LAB / ".venv" / "bin" / "python"


def adb_env(args):
    env = dict(os.environ)
    if getattr(args, "adb_port", None):
        env["ANDROID_ADB_SERVER_PORT"] = str(args.adb_port)
    return env


def adb_base(args):
    return ["adb"] + (["-s", args.serial] if getattr(args, "serial", None) else [])


def adb_shell(args, cmd, check=False):
    r = subprocess.run(adb_base(args) + ["shell", cmd], env=adb_env(args), text=True, capture_output=True)
    if check and r.returncode != 0:
        sys.exit(f"adb shell failed: {cmd}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], text=True, capture_output=True).stdout.strip()


def cmd_doctor(args):
    ok = True

    def row(name, good, detail, required=True):
        nonlocal ok
        ok = ok and (good or not required)
        print(f"[{'ok' if good else ('MISSING' if required else 'todo')}] {name}: {detail}")

    row("repo", REPO.is_dir(), f"{REPO} @ {git('rev-parse', '--short', 'HEAD')}")
    docker = shutil.which("docker") is not None and subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    row("docker", docker, "running" if docker else "run: colima start")
    if docker:
        image = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
        row("toolchain image", image, IMAGE)
    pkg = REPO / PKG / "llama.cpp"
    bench = pkg / "bin" / "llama-bench"
    row("package", bench.exists(), str(pkg))
    if bench.exists():
        built = datetime.datetime.fromtimestamp(bench.stat().st_mtime)
        commit = datetime.datetime.fromtimestamp(int(git("log", "-1", "--format=%ct")))
        row("package is newer than HEAD", built >= commit, f"package {built:%F %T}, HEAD {commit:%F %T}", required=False)
    row("adb", shutil.which("adb") is not None, shutil.which("adb") or "not found")
    out = subprocess.run(adb_base(args) + ["devices"], env=adb_env(args), text=True, capture_output=True).stdout
    devs = [l.split("\t")[0] for l in out.splitlines()[1:] if l.endswith("\tdevice")]
    row("adb device", bool(devs), ", ".join(devs) or "none connected", required=False)
    if devs:
        print(f"       soc={adb_shell(args, 'getprop ro.soc.model')} model={adb_shell(args, 'getprop ro.product.model')}")
    sdk = VENV_PY.exists() and subprocess.run([str(VENV_PY), "-c", "import qualcomm_device_cloud_sdk"], capture_output=True).returncode == 0
    row("QDC SDK", sdk, str(VENV_PY))
    row("QDC_API_KEY", bool(os.environ.get("QDC_API_KEY")), "set" if os.environ.get("QDC_API_KEY") else "not set", required=False)
    return 0 if ok else 1


def cmd_build(args):
    uid = f"{os.getuid()}:{os.getgid()}"
    sh = f"cmake --preset arm64-android-snapdragon-release -B build-snapdragon && cmake --build build-snapdragon -j {args.jobs} && cmake --install build-snapdragon --prefix {PKG}/llama.cpp"
    return subprocess.run(["docker", "run", "--rm", "-u", uid, "--volume", f"{REPO}:/workspace", "--workdir", "/workspace", "--platform", "linux/amd64", IMAGE, "bash", "-c", sh]).returncode


def cmd_push(args):
    adb_shell(args, f"mkdir -p {DEVICE_DIR} {MODEL_DIR}", check=True)
    src = REPO / PKG / "llama.cpp"
    for sub in ("bin", "lib"):
        if subprocess.run(adb_base(args) + ["push", str(src / sub), f"{DEVICE_DIR}/"], env=adb_env(args)).returncode:
            return 1
    adb_shell(args, f"chmod -R 755 {DEVICE_DIR}/bin", check=True)
    if args.model:
        return subprocess.run(adb_base(args) + ["push", args.model, f"{MODEL_DIR}/"], env=adb_env(args)).returncode
    return 0


def parse_bench_json(text):
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("no JSON in llama-bench output")
    return json.loads(text[start:end + 1])


def cmd_bench(args):
    target = "adb" + (f":{args.serial}" if args.serial else "")
    model = f"{MODEL_DIR}/{args.model}"
    run = [sys.executable, str(REPO / "scripts/snapdragon/run.py"), "--target", target, "--install-dir", PKG, "--devices", args.backend, "--hex-power-mode", args.power_mode]
    run += ["--", "llama-bench", "-m", model, "-ngl", "99", "-p", str(args.n_prompt), "-n", str(args.n_gen), "-r", "1", "-o", "json"]

    meta = {
        "label": args.label,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src", "ggml", "common", "tools", "include")),
        "soc": adb_shell(args, "getprop ro.soc.model"),
        "product": adb_shell(args, "getprop ro.product.model"),
        "fingerprint": adb_shell(args, "getprop ro.build.fingerprint"),
        "model": args.model,
        "backend": args.backend,
        "power_mode": args.power_mode,
        "n_prompt": args.n_prompt,
        "n_gen": args.n_gen,
    }
    if not meta["soc"]:
        sys.exit("no device answers on this adb server (check --adb-port and --serial)")

    samples = {"prompt_ts": [], "gen_ts": []}
    logs = []
    for i in range(args.runs):
        if i and args.cooldown:
            subprocess.run(["sleep", str(args.cooldown)])
        r = subprocess.run(run, cwd=REPO, env=adb_env(args), text=True, capture_output=True)
        logs.append(r.stderr)
        if r.returncode != 0:
            sys.stderr.write(r.stdout + r.stderr)
            sys.exit(f"run {i + 1} failed with exit code {r.returncode}")
        for t in parse_bench_json(r.stdout):
            samples["gen_ts" if t.get("n_gen", 0) > 0 else "prompt_ts"].append(t["avg_ts"])
        print(f"run {i + 1}/{args.runs}: prompt {samples['prompt_ts'][-1]:.2f} t/s, gen {samples['gen_ts'][-1]:.2f} t/s", flush=True)

    out_dir = LAB / "results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{meta['commit'][:9]}.json"
    out.write_text(json.dumps({"meta": meta, "samples": samples, "stderr_last": logs[-1][-4000:]}, indent=2))
    print(f"saved {out}")
    return 0


def summarize(values):
    return statistics.median(values), min(values), max(values)


def cmd_compare(args):
    runs = [json.loads(Path(p).read_text()) for p in args.results]
    for key in ("model", "backend", "power_mode", "n_prompt", "n_gen"):
        if len({str(r["meta"][key]) for r in runs}) > 1:
            print(f"WARNING: {key} differs between results: not a fair comparison")
    if len({r["meta"]["commit"] for r in runs}) > 1:
        print("note: commits differ")
    if any(r["meta"]["dirty"] for r in runs):
        print("note: at least one run has uncommitted source changes")
    base = {}
    print(f"{'label':<14}{'soc':<10}{'commit':<11}{'metric':<11}{'median':>9}{'min':>9}{'max':>9}{'spread':>8}{'vs first':>10}")
    for r in runs:
        m = r["meta"]
        for metric in ("prompt_ts", "gen_ts"):
            med, lo, hi = summarize(r["samples"][metric])
            base.setdefault(metric, med)
            print(f"{m['label']:<14}{m['soc']:<10}{m['commit'][:9]:<11}{metric:<11}{med:>9.2f}{lo:>9.2f}{hi:>9.2f}{(hi - lo) / med:>8.1%}{med / base[metric] - 1:>+10.1%}")
    return 0


def cmd_qdc_targets(args):
    if not os.environ.get("QDC_API_KEY"):
        sys.exit("QDC_API_KEY is not set")
    code = """
import os
from qualcomm_device_cloud_sdk.api import qdc_api as q
c = q.get_public_api_client_using_api_key(api_key_header=os.environ['QDC_API_KEY'], app_name_header='hexagon-lab', on_behalf_of_header='hexagon-lab', client_type_header='Python')
r = q.targets_get.sync_detailed(client=q.get_client_with_headers_added(c), page_size=200)
print('status', r.status_code)
for t in r.parsed.data:
    print(t.to_dict())
"""
    return subprocess.run([str(VENV_PY), "-c", code]).returncode


def cmd_qdc_job(args):
    if not os.environ.get("QDC_API_KEY"):
        sys.exit("QDC_API_KEY is not set")
    cmd = [str(VENV_PY), str(REPO / "scripts/snapdragon/qdc/run_qdc_jobs.py"), "--pkg-dir", str(REPO / PKG / "llama.cpp"), "--device", args.device, "--test", args.test, "--model-url", args.model_url]
    return subprocess.run(cmd, cwd=REPO).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def adb_opts(s):
        s.add_argument("--adb-port", type=int, help="adb server port (the local port of a QDC session tunnel)")
        s.add_argument("--serial", help="adb device serial")

    s = sub.add_parser("doctor", help="check the environment")
    adb_opts(s)
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("build", help="build and package in the toolchain container")
    s.add_argument("--jobs", type=int, default=8)
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("push", help="push the package, and optionally a model, to a device")
    adb_opts(s)
    s.add_argument("--model", help="local .gguf file")
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("bench", help="run llama-bench several times and save the results")
    adb_opts(s)
    s.add_argument("--label", required=True, help="result name, for example sm8845-local")
    s.add_argument("--model", required=True, help=f"file name in {MODEL_DIR}")
    s.add_argument("--runs", type=int, default=5)
    s.add_argument("--cooldown", type=int, default=60, help="seconds between runs")
    s.add_argument("--backend", default="HTP0")
    s.add_argument("--power-mode", default="high_performance")
    s.add_argument("--n-prompt", type=int, default=128)
    s.add_argument("--n-gen", type=int, default=32)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("compare", help="compare result files, the first is the baseline")
    s.add_argument("results", nargs="+")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("qdc-targets", help="list QDC targets (needs QDC_API_KEY)")
    s.set_defaults(fn=cmd_qdc_targets)

    s = sub.add_parser("qdc-job", help="submit an automated job with run_qdc_jobs.py (needs QDC_API_KEY)")
    s.add_argument("--device", required=True, help="QDC chipset name")
    s.add_argument("--model-url", required=True)
    s.add_argument("--test", default="bench", choices=["bench", "backend-ops", "all"])
    s.set_defaults(fn=cmd_qdc_job)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
