#!/usr/bin/env python3
"""Compare top logprobs token by token up to the first text divergence: own vs own, own vs borrow."""
import json, urllib.request
from server_load import text
URL = "http://127.0.0.1:18080/v1/chat/completions"


def run(system, user, slot, n=64, cache=True):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "id_slot": slot, "n_predict": n,
            "temperature": 0, "seed": 1, "cache_prompt": cache, "ignore_eos": True, "logprobs": True, "top_logprobs": 5}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return r["choices"][0]["logprobs"]["content"]


def diff(a, b):
    worst = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        px = {t["token"]: t["logprob"] for t in x["top_logprobs"]}
        py = {t["token"]: t["logprob"] for t in y["top_logprobs"]}
        for k in set(px) & set(py):
            worst = max(worst, abs(px[k] - py[k]))
        if x["token"] != y["token"]:
            top = sorted(x["top_logprobs"], key=lambda t: -t["logprob"])[:2]
            return i, worst, abs(top[0]["logprob"] - top[1]["logprob"])
    return -1, worst, None


import sys
prefix = text(800, 42)
base = int(sys.argv[1])
for rep in range(2):
    q = f"{base} {rep}: " + text(12, 500 + base * 10 + rep)
    c0 = run(prefix, q, 0, cache=False)
    c0b = run(prefix, q, 0, cache=False)
    c1 = run(prefix, q, 1, cache=False)
    c2 = run(prefix, q, 2, cache=False)
    run(prefix, "warm", 0, 1)
    a = run(prefix, q, 0)
    run(prefix, "warm", 0, 1)
    b = run(prefix, q, 3)
    print(f"q={base}.{rep} cold0-cold0={diff(c0, c0b)} cold0-cold1={diff(c0, c1)} cold0-cold2={diff(c0, c2)} own-cold0={diff(a, c0)} own-borrow={diff(a, b)}", flush=True)
