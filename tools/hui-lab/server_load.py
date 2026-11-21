#!/usr/bin/env python3
"""Real request-flow load for llama-server on the phone, driven over adb forward.

Modes:
  prefix  one slot: cold vs. prefix-hit time to first token for several prefix sizes
  cross   N slots: a prefix warmed on one slot, then requested on the others
  ways    N closed-loop clients: per-request TTFT, stream t/s, total t/s

Nothing here is tuned: all numbers come from the client clock plus the server's
own "timings" block, so queueing shows up in TTFT.
"""
import argparse, json, random, statistics as st, threading, time, urllib.request

WORDS = ("the phone assistant reads notes messages calendar photos and answers short "
         "questions about travel weather meetings family work music food health").split()


def text(n_words, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n_words))


CHAT = False


def stream(url, prompt, n_predict, cache, temp=0.7, seed=1, timeout=300, slot=-1):
    body = {"id_slot": slot, "n_predict": n_predict, "cache_prompt": cache, "stream": True,
            "temperature": temp, "seed": seed, "ignore_eos": True, "timings_per_token": False}
    path = "/completion"
    if CHAT:
        # the first sentence mark splits system and user, so the server can checkpoint at the user start
        system, user = prompt.rsplit(" question ", 1) if " question " in prompt else prompt.rsplit(" user ", 1)
        body["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        path = "/v1/chat/completions"
    else:
        body["prompt"] = prompt
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft, last, n, timings = None, None, 0, {}
    collected = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            if raw.startswith(b"data: [DONE]"):
                continue
            d = json.loads(raw[6:])
            now = time.time()
            text_piece = d.get("content")
            if d.get("choices"):
                delta = d["choices"][0].get("delta", {})
                text_piece = delta.get("content") or delta.get("reasoning_content")
                collected.append(text_piece or "")
            if text_piece and ttft is None:
                ttft = now - t0
            if text_piece:
                last, n = now, n + 1
            if d.get("timings"):
                timings = d["timings"]
    return {"ttft": ttft, "total": time.time() - t0, "chunks": n, "t": timings, "text": "".join(collected)}


def show(tag, r):
    t = r["t"]
    print(f"{tag} ttft={r['ttft']*1000:.0f}ms prompt_n={t.get('prompt_n')} "
          f"prompt_tps={t.get('prompt_per_second', 0):.0f} "
          f"cache_n={t.get('cache_n')} decode_tps={t.get('predicted_per_second', 0):.1f}", flush=True)


def mode_prefix(a):
    for words in a.prefix_words:
        prefix = text(words, 42)
        for rep in range(a.reps):
            q = f" question {rep}: " + text(12, 100 + rep)
            show(f"prefix={words}w cold#{rep}", stream(a.url, prefix + q, a.n, False, seed=rep))
            # warm the slot with the same prefix, then hit it with a different suffix
            stream(a.url, prefix + " question warm", 1, True)
            q = f" question {rep}x: " + text(12, 200 + rep)
            show(f"prefix={words}w hit#{rep}", stream(a.url, prefix + q, a.n, True, seed=rep))


def mode_cross(a):
    """Occupy all slots at once so requests are forced onto slots that lack the prefix."""
    prefix = text(a.prefix_words[0], 42)
    stream(a.url, prefix + " question warm", 1, True)
    for rep in range(a.reps):
        res, threads = [None] * a.clients, []

        def one(i):
            q = f" user {i} round {rep}: " + text(12, 300 + 10 * rep + i)
            res[i] = stream(a.url, prefix + q, a.n, True, seed=i)
        for i in range(a.clients):
            threads.append(threading.Thread(target=one, args=(i,)))
            threads[-1].start()
        for th in threads:
            th.join()
        for i, r in enumerate(res):
            show(f"cross round={rep} client={i}", r)


def mode_verify(a):
    """Greedy output must not depend on how the prefix got into the slot."""
    global CHAT
    CHAT = True
    prefix = text(a.prefix_words[0], 42)
    bad = 0
    for rep in range(a.reps):
        qs = [f" question {rep} {i}: " + text(12, 400 + 10 * rep + i) for i in range(a.clients)]
        ref = [stream(a.url, prefix + q, a.n, False, temp=0, seed=1)["text"] for q in qs]
        stream(a.url, prefix + " question warm", 1, True, temp=0)
        res, threads = [None] * a.clients, []

        def one(i):
            res[i] = stream(a.url, prefix + qs[i], a.n, True, temp=0, seed=1)
        for i in range(a.clients):
            threads.append(threading.Thread(target=one, args=(i,)))
            threads[-1].start()
        [th.join() for th in threads]
        for i in range(a.clients):
            same = res[i]["text"] == ref[i]
            bad += 0 if same else 1
            print(f"verify rep={rep} client={i} same={same} cache_n={res[i]['t'].get('cache_n')} len={len(ref[i])}", flush=True)
            if not same:
                print("  cold:", repr(ref[i][:120]))
                print("  warm:", repr(res[i]["text"][:120]))
    print(f"verify mismatches={bad}")


def mode_verify2(a):
    """Serial, same batch shape: slot 0 hits its own checkpoint, an empty slot borrows it. Output must match exactly."""
    global CHAT
    CHAT = True
    prefix = text(a.prefix_words[0], 42)
    bad = fired = 0
    stream(a.url, prefix + " question warm", 1, True, temp=0, slot=0)
    for rep in range(3):
        q = f" question {a.reps} {rep}: " + text(12, 500 + a.reps * 10 + rep)
        own = stream(a.url, prefix + q, a.n, True, temp=0, slot=0)
        # slot 0 goes back to the prefix, so the borrow starts at the same token as the own hit
        stream(a.url, prefix + " question warm", 1, True, temp=0, slot=0)
        bor = stream(a.url, prefix + q, a.n, True, temp=0, slot=0 if a.think < 0 else rep + 1)
        same = own["text"] == bor["text"]
        bad += 0 if same else 1
        fired += 1 if (bor["t"].get("cache_n") or 0) > 0 else 0
        print(f"verify2 q={rep} own==borrow {same} own_cache={own['t'].get('cache_n')} borrow_cache={bor['t'].get('cache_n')} "
              f"ttft own={own['ttft']*1000:.0f} borrow={bor['ttft']*1000:.0f}ms chars={len(own['text'])}", flush=True)
        if not same:
            print("  own:   ", repr(own["text"][:200]))
            print("  borrow:", repr(bor["text"][:200]))
    print(f"verify2 fired={fired} mismatches={bad}")


def mode_ways(a):
    stop_at = time.time() + a.seconds
    out, lock = [], threading.Lock()
    system = text(a.prefix_words[0], 42)

    def client(i):
        k = 0
        rng = random.Random(i)
        while time.time() < stop_at:
            q = f" user {i} turn {k}: " + text(rng.randint(10, 40), i * 1000 + k)
            try:
                r = stream(a.url, system + q, a.n, a.cache, temp=a.temp, seed=i * 1000 + k)
                r["start"] = time.time() - r["total"]
                with lock:
                    out.append(r)
            except Exception as e:
                print(f"client {i} failed: {e}", flush=True)
                return
            k += 1
            if a.think > 0:
                time.sleep(rng.uniform(0, 2 * a.think))
    threads = [threading.Thread(target=client, args=(i,)) for i in range(a.clients)]
    t0 = time.time()
    [th.start() for th in threads]
    [th.join() for th in threads]
    wall = time.time() - t0
    ttft = sorted(r["ttft"] for r in out if r["ttft"])
    dec = [r["t"].get("predicted_per_second", 0) for r in out]
    toks = sum(r["t"].get("predicted_n", 0) for r in out)
    if not out:
        print("no completed requests")
        return
    print(f"ways={a.clients} cache={a.cache} n={a.n} think={a.think}s reqs={len(out)} wall={wall:.0f}s "
          f"total_tps={toks/wall:.1f} stream_tps_med={st.median(dec):.1f} "
          f"ttft_med={st.median(ttft)*1000:.0f}ms ttft_max={ttft[-1]*1000:.0f}ms", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prefix", "cross", "ways", "verify", "verify2"])
    p.add_argument("--url", default="http://127.0.0.1:18080")
    p.add_argument("--prefix-words", type=int, nargs="+", default=[200])
    p.add_argument("--clients", type=int, default=1)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--think", type=float, default=0)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--cache", type=int, default=1)
    p.add_argument("--chat", action="store_true")
    a = p.parse_args()
    a.cache = bool(a.cache)
    CHAT = a.chat
    {"prefix": mode_prefix, "cross": mode_cross, "ways": mode_ways, "verify": mode_verify, "verify2": mode_verify2}[a.mode](a)
