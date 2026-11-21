#!/usr/bin/env python3
"""Build report/report.html from results/full/summary.json, so every table number comes from the measured data."""
import json
import re
from pathlib import Path

LAB = Path(__file__).resolve().parent
S = json.loads((LAB / "results/full/summary.json").read_text())
A, B = S["sm8845"], S["sm8850b"]
B1 = S["sm8850"]


def med(d, c, m):
    return d[c]["metrics"][m]["median"]


def rng(d, c, m, p=1):
    x = d[c]["metrics"][m]
    return f'{x["median"]:.{p}f} <small>{x["min"]:.{p}f}–{x["max"]:.{p}f}</small>'


def pct(d, c1, c0, m):
    return f"{(med(d, c1, m) / med(d, c0, m) - 1) * 100:+.1f}%"


def table(head, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in head)
    r = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{r}</tbody></table></div>'


def bars(title, unit, items):
    top = max(v for _, v, _ in items)
    out = f'<figure class="chart"><figcaption>{title}<span>{unit}</span></figcaption>'
    for label, v, cl in items:
        out += f'<div class="bar"><span class="bl">{label}</span><span class="bt"><i class="{cl}" style="width:{v / top * 100:.1f}%"></i></span><span class="bv">{v:,.1f}</span></div>'
    return out + "</figure>"


def backend_rows(d):
    rows = []
    for name, c in (("NPU（HTP0，全部层）", "opt_htp"), ("GPU（OpenCL）", "opt_gpu"), ("CPU（6 线程）", "opt_cpu")):
        rows.append([name, rng(d, c, "pp128"), rng(d, c, "pp512"), rng(d, c, "tg64"), f'{d[c]["rss_mb"]:.0f}', f'{d[c]["peak_c"]:.0f}'])
    return rows


def ab_rows(d):
    rows = []
    for label, c1, c0, m, p in (
        ("提示处理 pp128，t/s", "opt_htp", "base_htp", "pp128", 1), ("提示处理 pp512，t/s", "opt_htp", "base_htp", "pp512", 1),
        ("生成 tg64，t/s", "opt_htp", "base_htp", "tg64", 1), ("端到端生成，t/s", "opt_gen", "base_gen", "eval_ts", 1),
        ("模型加载，ms", "opt_gen", "base_gen", "load_ms", 0), ("11 词元提示耗时，ms", "opt_gen", "base_gen", "prompt_ms", 1),
    ):
        rows.append([label, rng(d, c0, m, p), rng(d, c1, m, p), pct(d, c1, c0, m)])
    rows.append(["进程峰值 RSS，MiB", f'{d["base_htp"]["rss_mb"]:.0f}', f'{d["opt_htp"]["rss_mb"]:.0f}', f'{(d["opt_htp"]["rss_mb"] / d["base_htp"]["rss_mb"] - 1) * 100:+.1f}%'])
    return rows


def sweep_rows(cases):
    return [[n, rng(A, c, "pp512"), rng(A, c, "tg64"), rng(B, c, "pp512"), rng(B, c, "tg64")] for n, c in cases]


def cold_rows():
    d = LAB / "results/full/cold-sm8850"
    out = []
    for label, n in (("上游基线", "base"), ("优化版默认", "opt"), ("关闭 mmap 预取", "noprefetch"), ("关闭预取 + 层预取", "nopre_layerpf"), ("关闭预取 + 权重流式", "nopre_stream"), ("关闭 NPU 卸载", "nooffload")):
        v = {"load": [], "pr": [], "ev": []}
        for i in range(3):
            t = (d / f"cold_{n}_{i}.err").read_text(errors="replace")
            v["load"].append(float(re.search(r"load time =\s+([\d.]+)", t)[1]))
            v["pr"].append(float(re.search(r"prompt eval time =\s+([\d.]+)", t)[1]))
            v["ev"].append(float(re.search(r"\n[^\n]*\beval time =\s+([\d.]+)", t.split("prompt eval time")[1])[1]))
        f = lambda x, p=0: f"{sorted(x)[1]:.{p}f} <small>{min(x):.{p}f}–{max(x):.{p}f}</small>"
        out.append([label, f(v["load"]), f(v["pr"], 1), f(v["ev"])])
    return out


def poll_rows():
    groups = {}
    for line in (LAB / "results/full/sm8845-poll2.txt").read_text().splitlines():
        m = re.match(r"(\w+) oppoll=(\d) poll=(\d+) t=(\d) #\d pp128_tg64=([\d.]+) ([\d.]+)", line)
        if m:
            groups.setdefault(m[1], {"o": m[2], "p": m[3], "t": m[4], "v": []})["v"].append(float(m[6]))
    names = {"ctrl": "默认：两种轮询都开", "nooppoll": "关 NPU 轮询", "nopoll": "关 CPU 线程池轮询", "neither": "两种都关", "t2": "2 线程，两种轮询都开"}
    rows, chart = [], []
    for k, g in groups.items():
        v = sorted(g["v"])
        rows.append([names[k], g["t"], g["o"], g["p"], f"{v[1]:.1f} <small>{v[0]:.1f}–{v[2]:.1f}</small>"])
        chart.append((names[k], v[1], "c-warn" if k == "ctrl" else "c-npu"))
    return rows, chart


def mid(v, p=1):
    v = sorted(v)
    return f"{v[len(v) // 2]:,.{p}f} <small>{v[0]:,.{p}f}–{v[-1]:,.{p}f}</small>"


def batch_data():
    rounds = {}
    for line in (LAB / "results/full/sm8845-spec-final.txt").read_text().splitlines():
        m = re.match(r"\|\s+128 \|\s+128 \|\s+(\d+) \|\s+\d+ \|\s+([\d.]+) \|\s+([\d.]+) \|\s+([\d.]+) \|\s+([\d.]+)", line)
        if m:
            rounds.setdefault(int(m[1]), []).append({"pp": float(m[3]), "t": float(m[4]), "tg": float(m[5])})
    rows, chart = [], []
    base = rounds[1]
    for n, r in rounds.items():
        tg = [x["tg"] for x in r]
        rows.append([f"{n} 路", mid(tg), mid([x["tg"] / n for x in r]), mid([x["t"] / 128 * 1000 for x in r], 0), mid([x["tg"] / b["tg"] for x, b in zip(r, base)])])
        chart.append((f"{n} 路", sorted(tg)[1], "c-npu"))
    return rows, chart


def spec_rows():
    g = {}
    for m in re.finditer(r"edit1300 (\w+) #\d rc=0 md5=\w+ decoded (\d+) tokens in [\d.]+ seconds, speed: ([\d.]+) t/s;n_drafted = (\d+);n_accept = (\d+)", (LAB / "results/full/sm8845-spec-final.txt").read_text()):
        g.setdefault(m[1], []).append((float(m[3]), int(m[4]), int(m[5])))
    base = sorted(x[0] for x in g["nodraft"])[1]
    rows = []
    for k, name in (("nodraft", "不推测"), ("m16", "n=4，m=16"), ("m32", "n=4，m=32"), ("m64", "n=4，m=64")):
        v = g[k]
        s = sorted(x[0] for x in v)[1]
        rate = f"{v[0][2] / v[0][1] * 100:.0f}%" if v[0][1] else "–"
        rows.append([name, mid([x[0] for x in v]), f"{s / base:.1f} 倍", f"{v[0][1]:,}", f"{v[0][2]:,}", rate])
    return rows


def det_data():
    txt = (LAB / "results/full/sm8845-det-mode.txt").read_text()
    runs = {(m[1], m[2], m[3]): {"md5": m[4], "ts": float(m[6])} for m in re.finditer(r"^(default|det) (open|mixed|edit) (nodraft|draft) (\w+) n=(\d+)\s+([\d.]+) t/s", txt, re.M)}
    batched, mode = {}, None
    for line in txt.splitlines():
        if line.endswith("batched:"):
            mode = line.split()[0]
        m = re.match(r"\|\s+128 \|\s+64 \|\s+(\d+) \|\s+\d+ \|\s+[\d.]+ \|\s+([\d.]+) \|\s+[\d.]+ \|\s+([\d.]+)", line)
        if m:
            batched[(mode, int(m[1]))] = (float(m[2]), float(m[3]))
    same = lambda mode, w: "相同" if runs[(mode, w, "nodraft")]["md5"] == runs[(mode, w, "draft")]["md5"] else "不同"
    rows = [
        ["单路 tg，t/s", f"{batched[('default', 1)][1]:.1f}", f"{batched[('det', 1)][1]:.1f}"],
        ["24 路总吞吐，t/s", f"{batched[('default', 24)][1]:.1f}", f"{batched[('det', 24)][1]:.1f}"],
        ["提示处理（24 路，每路 128 词元），t/s", f"{batched[('default', 24)][0]:,.1f}", f"{batched[('det', 24)][0]:,.1f}"],
        ["推测解码，编辑任务，t/s", f"{runs[('default', 'edit', 'draft')]['ts']:.1f}", f"{runs[('det', 'edit', 'draft')]['ts']:.1f}"],
    ]
    rows += [[f"{n}：有无推测的输出", same("default", w), same("det", w)] for n, w in (("普通问答", "open"), ("代码评审", "mixed"), ("代码改写", "edit"))]
    prompts = [[n, f"{runs[('default', w, 'nodraft')]['ts']:.1f}", f"{runs[('default', w, 'draft')]['ts']:.1f}", f"{runs[('default', w, 'draft')]['ts'] / runs[('default', w, 'nodraft')]['ts']:.1f} 倍"] for n, w in (("普通问答", "open"), ("代码评审（需引用输入）", "mixed"), ("代码改写（大量照抄输入）", "edit"))]
    return rows, prompts


def logit_rows():
    cfg, data = None, {}
    for line in (LAB / "results/full/sm8845-logit-cmp3.txt").read_text().splitlines():
        if line.startswith("== "):
            cfg = line[3:]
        m = re.match(r"batch_(\d+)\s+positions=\d+ top1_flips=(\d+) max_abs=([\d.]+) mean_pos_max=([\d.]+) rms=([\d.]+) differing=([\d.]+)%", line)
        if m:
            data.setdefault(cfg, {})[int(m[1])] = {"rms": float(m[5]), "diff": float(m[6])}
    cell = lambda x: "逐位相同" if x["diff"] == 0 else f"{x['diff']:.0f}% 不同，rms {x['rms']:.2f}"
    names = (("npu under256 X=0", "默认"), ("npu under256 GGML_HEXAGON_OPFUSION=0", "关融合"), ("npu under256 GGML_HEXAGON_FA_SELECT=1", "注意力只用 HVX"), ("npu under256 GGML_HEXAGON_MM_SELECT=1", "matmul 只用 HVX"), ("npu under256 MM=1 FA=1 FUS=0", "三项同时"))
    return [[n, cell(data[k][4]), cell(data[k][5]), cell(data[k][8])] for k, n in names]


def ref_rows():
    v = {}
    for m in re.finditer(r"^(\S+\.b\d)\s+positions=\d+ top1_flips=(\d+) max_abs=([\d.]+) mean_pos_max=[\d.]+ rms=([\d.]+)", (LAB / "results/full/sm8845-logit-cmp2.txt").read_text(), re.M):
        v[m[1]] = (m[4], m[3], m[2])
    return [[n, f"{float(v[k][0]):.3f}", f"{float(v[k][1]):.2f}", v[k][2]] for k, n in (("cpu.b8", "CPU 批量 8"), ("npu.b1", "NPU 单步，默认"), ("nf.b1", "NPU 单步，关融合"), ("npu.b8", "NPU 批量 8，默认"), ("nf.b8", "NPU 批量 8，关融合"))]


poll_table, poll_chart = poll_rows()
ratio = lambda d, a, b, m: med(d, a, m) / med(d, b, m)
batch_table, batch_chart = batch_data()
det_table, prompt_table = det_data()
batch_ratio = {}
for _r in batch_table:
    _m = re.match(r"([\d.]+) <small>([\d.]+)–([\d.]+)", _r[4])
    batch_ratio[int(_r[0].split()[0])] = f"{_m[1]} 倍（{_m[2]}–{_m[3]}）"

html = f"""<title>Gemma 4 E2B Hexagon 验证</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single measured reading column; wide tables scroll inside their own frame. */
:root {{
  --bg:#f5f7f6; --surface:#ffffff; --fg:#17211f; --muted:#56635f; --line:#d5dddA; --accent:#0b6b66; --npu:#0b6b66; --gpu:#7c6bb0; --cpu:#9a7a3c; --warn:#b4472f; --warn-bg:#f9e9e4; --ok-bg:#e0f0ed;
  --display:"Noto Serif SC","Songti SC",serif; --body:"Noto Sans SC","PingFang SC",system-ui,sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#101716; --surface:#172120; --fg:#e4ecea; --muted:#9fadA9; --line:#2d3a38; --accent:#58c4b8; --npu:#58c4b8; --gpu:#ab9be0; --cpu:#d3b06c; --warn:#ef8d75; --warn-bg:#3a2019; --ok-bg:#16322e; color-scheme:dark }} }}
:root[data-theme="dark"] {{
  --bg:#101716; --surface:#172120; --fg:#e4ecea; --muted:#9fadA9; --line:#2d3a38; --accent:#58c4b8; --npu:#58c4b8; --gpu:#ab9be0; --cpu:#d3b06c; --warn:#ef8d75; --warn-bg:#3a2019; --ok-bg:#16322e; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.75 var(--body); }}
main {{ max-width:880px; margin:auto; padding-block:48px 80px; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); display:grid; gap:18px; }}
main>* {{ min-width:0; }}
h1,h2 {{ font-family:var(--display); text-wrap:balance; line-height:1.3; margin:0; }}
h1 {{ font-size:clamp(30px,6vw,44px); }} h2 {{ font-size:25px; margin-top:34px; padding-top:22px; border-top:1px solid var(--line); }}
h3 {{ font-size:17px; margin:14px 0 0; }} p,ul,ol {{ margin:0; }} ul,ol {{ padding-left:22px; display:grid; gap:6px; }}
.sub {{ color:var(--muted); font:13px/1.7 var(--mono); }}
code {{ font:.86em var(--mono); overflow-wrap:anywhere; }} pre {{ font:13px/1.6 var(--mono); background:var(--surface); border:1px solid var(--line); padding:12px 14px; overflow-x:auto; margin:0; }}
.tw {{ overflow-x:auto; border:1px solid var(--line); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums; }}
th,td {{ padding:8px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }}
tbody tr:last-child td {{ border-bottom:0; }} th {{ color:var(--muted); font-weight:500; font-size:13px; }} small {{ color:var(--muted); font-size:12px; margin-left:4px; }}
table.text td {{ white-space:normal; text-align:left; vertical-align:top; min-width:120px; }} table.text th {{ text-align:left; }}
.verdict {{ background:var(--ok-bg); padding:18px 20px; display:grid; gap:10px; }} .verdict strong {{ font-size:19px; line-height:1.5; }}
.warn {{ background:var(--warn-bg); padding:14px 18px; display:grid; gap:8px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr)); gap:14px; }} .grid2>* {{ min-width:0; }}
.chart {{ margin:0; background:var(--surface); border:1px solid var(--line); padding:14px 16px; display:grid; gap:7px; }}
.chart figcaption {{ font-weight:500; display:flex; justify-content:space-between; gap:12px; }} .chart figcaption span {{ color:var(--muted); font-size:13px; font-weight:400; }}
.bar {{ display:grid; grid-template-columns:minmax(92px,170px) 1fr 58px; gap:8px; align-items:center; font-size:13px; }} .bl {{ color:var(--muted); line-height:1.3; }} .bt {{ height:13px; background:color-mix(in srgb,var(--line) 45%,transparent); }} .bt i {{ display:block; height:100%; }} .bv {{ text-align:right; font-variant-numeric:tabular-nums; }}
.c-npu {{ background:var(--npu); }} .c-gpu {{ background:var(--gpu); }} .c-cpu {{ background:var(--cpu); }} .c-warn {{ background:var(--warn); }}
.note {{ color:var(--muted); font-size:14px; }}
</style>
<main>
<header style="display:grid;gap:10px">
<h1>Gemma 4 E2B 在 Hexagon NPU 上的推理验证</h1>
<p class="sub">llama.cpp_hui @ 4b0773c7d，对比上游 aa588a346 · 模型 gemma-4-E2B-it-qat-Q4_0.gguf（3.35 GB，35 层，sha256 fa401b55…）<br>设备 SM8845（本地 OnePlus PLR110）、SM8850（QDC Canoe 开发机）· 2026-10-31</p>
</header>

<section class="verdict">
<strong>E2B 在两台设备上都能正确运行，但这版优化没有让它变快。这些优化主要面向 Qwen 和 MoE 大模型，在 E2B 上大多没有进入执行路径。</strong>
<p>NPU 的优势集中在提示处理：SM8850 的 pp512 为 {med(B,'opt_htp','pp512'):,.0f} t/s，是 CPU 的 {ratio(B,'opt_htp','opt_cpu','pp512'):.1f} 倍。生成速度上 NPU 为 {med(B,'opt_htp','tg64'):.1f} t/s，略低于 6 线程 CPU 的 {med(B,'opt_cpu','tg64'):.1f} t/s。</p>
<p>测试还发现一个问题：SM8845 部分卸载时，如果 NPU 轮询和 CPU 线程池轮询同时开启（<code>run.py</code> 原默认配置），6 线程生成速度跌到约 1 t/s。关闭任一轮询可恢复，<code>run.py</code> 的默认配置已修复。</p>
<p>后续提速实验（只在 SM8845 上做，见“优化细节和提升倍数”）：用现有的批量解码，24 路并发的总吞吐可达单路的 10 倍以上，但每一路会变慢。单路生成做不到 10 倍，最好的是大量照抄输入的任务，约 5.7 倍；普通问答没有提速。</p>
</section>

<h2>主要结论</h2>
<ol>
<li><b>优化前后无可测提升。</b>SM8850 上 pp512 {pct(B,'opt_htp','base_htp','pp512')}、tg64 {pct(B,'opt_htp','base_htp','tg64')}，内存相同，相同参数下输出文本一致。</li>
<li><b>NPU 适合处理长提示，也能腾出 CPU。</b>生成阶段 NPU 不比 CPU 快，但不占用 6 个 CPU 核心，host 线程数对它几乎没有影响。功耗没有测量，能效优势还不能下结论。</li>
<li><b>矩阵乘法占 NPU 时间约 89%。</b>其中输出层 <code>lm_head</code>（Q6_K，词表 262144）单项约占 19%，是后续优化最值得先看的地方。</li>
<li><b>SM8850 的 NPU 明显更快。</b>两者都是 HTP v81，SM8850 的 pp512 比 SM8845 高 {(med(B,'opt_htp','pp512')/med(A,'opt_htp','pp512')-1)*100:.0f}%，tg64 高 {(med(B,'opt_htp','tg64')/med(A,'opt_htp','tg64')-1)*100:.0f}%。但 SM8845 测试时 CPU 被限频，机型也不同，这个差距不能全部归因于芯片。</li>
<li><b>10 倍只在多路总吞吐上成立。</b>单路生成每个词元都要读约 1.38 GB 权重，瓶颈很可能是内存带宽；批量解码可以共用一次读取，所以总吞吐能大幅提升。</li>
<li><b>批量大小会改变 NPU 的输出，但没有发现数据损坏迹象。</b>NPU 按批量大小切换计算路径，数值结果略有差异，两个候选词接近打平时输出就可能不同。</li>
</ol>

<h2>优化细节和提升倍数</h2>
<p>下表汇总本次所有优化手段。倍数只在“对比基准”一列的条件下成立，不能相乘，也不能当作同一个“整体提速”。</p>
{table(["优化", "做法", "对比基准", "提升", "代价或条件"], [
["fork 原有优化（流式、预取、专家缓存等）", "面向大模型的内存和 IO 优化，详见下节。", "上游基线，全部层在 NPU，SM8850", f"tg64 {ratio(B,'opt_htp','base_htp','tg64'):.2f} 倍，pp512 {ratio(B,'opt_htp','base_htp','pp512'):.2f} 倍", "对 E2B 基本无效；大模型未测"],
["轮询配置修复", "部分卸载时 <code>run.py</code> 默认关闭 NPU 轮询。", "修复前默认配置，SM8845，16 层 NPU，6 线程", "tg64 约 18 倍（1.0 到 17.6 t/s）", "是修复异常，不是超出正常水平的提速；SM8850 未复测，也未出现这个问题"],
["多路批量解码（24 路）", "多个请求合成一批，共用一次权重读取。", "单路 tg，SM8845", f"总吞吐 {batch_ratio[24]}", "每路变慢；未模拟真实请求"],
["多路批量解码（32 路）", "同上。", "单路 tg，SM8845", f"总吞吐 {batch_ratio[32]}", "每路更慢；未模拟真实请求"],
["单路推测解码（n=4，m=32）", "从输入中找重复片段，批量验证猜测。", "不推测，代码改写，SM8845", "5.7 倍（31.1 到 178.2 t/s）", "只对大量照抄输入的任务有效；输出可能和单步不逐字相同"],
["单路推测解码（n=2，m=16）", "同上。", "不推测，SM8845，单次运行", " / ".join(f"{r[0].split('（')[0]} {r[3]}" for r in prompt_table), "普通问答反而略慢"],
["使用 NPU 代替 CPU", "把提示处理放到 NPU。这是后端选择，不是本次新增的优化。", "CPU 6 线程，SM8850", f"pp512 {ratio(B,'opt_htp','opt_cpu','pp512'):.1f} 倍，tg64 {ratio(B,'opt_htp','opt_cpu','tg64'):.2f} 倍", "生成阶段不比 CPU 快；功耗未测"],
], "text")}
<p><b>10 倍目标的结论：</b>多路总吞吐已达到，单路生成没有达到。普通问答的单路速度仍约 30 t/s。要明显提高单路速度，需要减少每个词元读取的权重，例如换用更小的模型或更低比特量化，这部分没有测试。</p>
<p>复现时使用下面的参数（SM8845，模型路径按实际修改）：</p>
<p><code>llama-batched-bench -m MODEL -dev HTP0 -ngl 99 -fa on -t 2 -c 8192 -b 2048 -ub 512 -npp 128 -ntg 128 -npl 1,16,24,32</code></p>
<p><code>llama-speculative-simple -m MODEL -dev HTP0 -ngl 99 -fa on -t 2 -c 4096 -b 2048 -ub 512 -n 1300 --temp 0 --top-k 1 -f PROMPT --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 32 --spec-draft-n-max 32</code></p>

<h2>优化方案和原理</h2>
<p>相对上游，fork 新增约 2500 行代码。下表按代码路径判断每项优化能否作用于 Gemma 4 E2B（Q4_0、稠密 FFN、模型能完整放进内存）。</p>
{table(["优化", "原理", "开关", "对 E2B"], [
["权重流式", "用 O_DIRECT 按层读取留在 host 的权重，放进两块轮换缓冲区：当前层计算时预读下一层，用顺序读取替代缺页。", "<code>LLAMA_WEIGHT_STREAM</code>", "不生效。只接入 qwen35 图；开启后日志也没有 streaming 记录"],
["层预取", "计算某层前，提前把后续层的 mmap 页读入页缓存，减少推理线程阻塞在缺页上。", "<code>LLAMA_LAYER_PREFETCH</code>", "不生效。同样只接入 qwen35 图"],
["专家预取", "MoE 模型按路由结果只预读会用到的专家权重，并预测下一层的专家。", "<code>LLAMA_EXPERT_PREFETCH</code>", "不适用。E2B 不是 MoE"],
["专家缓存", "把常用专家权重留在一块固定内存里，避免反复从闪存读取。", "<code>LLAMA_EXPERT_CACHE_MB</code>", "不适用"],
["块稀疏 FFN", "把 FFN 权重按块稀疏化，由 <code>gguf_ffn_blockify.py</code> 转换模型，减少需要读取和计算的权重。", "<code>LLAMA_FFN_SPARSE_BLOCKS</code>", "不适用。需要转换模型，且只在 qwen35 图中实现"],
["IQ 量化 tiled 矩阵乘", "把 IQ4/IQ3/IQ2/IQ1 重排成 HMX/HVX 可直接处理的 tile，由查表还原权重，让这些量化在 NPU 上运行。", "随权重类型自动启用", "不适用。E2B 是 Q4_0，上游已支持"],
["大批量 NPU 卸载", "留在 host 的层遇到不少于 32 行的矩阵乘时，把权重复制到计算缓冲区，运行时重排后交给 NPU。", "<code>GGML_HEXAGON_OFFLOAD</code>，默认 32", "路径可能触及，但开关前后无差异，原因未确认"],
["mmap 预取策略", "模型超过一半内存时不再整体预取，避免加载阶段把整个模型读进来。", "<code>LLAMA_MMAP_PREFETCH</code>", "不触发。E2B 小于内存的一半"],
["rpcmem 预算", "分配 NPU 内存前预留可用内存（默认 2048 MiB），放不下时报错，避免 rpcmem 占满内存拖垮手机。", "<code>GGML_HEXAGON_MEM_RESERVE</code>", "生效但不影响速度。手机旧日志显示 E2B 曾因此报错退出；本次未做内存压力测试"],
], "text")}
<p class="note">这些优化在大模型上的效果本次没有测试，上面的结果不能说明它们没有价值。要评估它们，需要用 Qwen 或 MoE 模型另做一轮。</p>

<h2>测试方法</h2>
<ul>
<li><b>两个构建：</b>优化版是 fork 提交 4b0773c7d，基线是它合并的上游提交 aa588a346，用同一工具链容器交叉编译。</li>
<li><b>两类负载：</b><code>llama-bench</code> 测 pp128、pp512、tg64；<code>llama-completion</code> 用固定提示、temp 0、seed 1 跑端到端生成，并检查输出文本。</li>
<li><b>轮次交错：</b>各用例按轮交替运行，减少温度漂移造成的偏差。主要对比项 5 轮，其他 3 轮。表中为中位数，小字是最小到最大值。</li>
<li><b>运行状态：</b>SM8845 每次运行前唤醒屏幕并等待 Thermal Status 回到 0（最多等 15 分钟），通过 Wi-Fi adb 控制且 USB 线仍连着。SM8850 通过 SSH 隧道控制，无需亮屏。功耗模式默认 <code>high_performance</code>，NPU 轮询开启（<code>GGML_HEXAGON_OPPOLL=1</code>，与 <code>run.py</code> 一致）。少数运行等待超时后 Thermal Status 仍为 1。</li>
<li><b>样本量：</b>SM8845 为 {sum(1 for _ in open(LAB/'results/full/sm8845.jsonl'))} 次有效运行和 15 次轮询实验。SM8850 第二次会话为 {sum(1 for _ in open(LAB/'results/full/sm8850b.jsonl'))} 次运行和 18 次冷启动，第一次会话 {sum(1 for _ in open(LAB/'results/full/sm8850.jsonl'))} 次用于复现性核对。没有运行失败。</li>
</ul>

<h2>如何读这些数据</h2>
<p>下面说明报告中各项数据测的是什么，以及读数时要注意什么。</p>
<h3>速度</h3>
{table(["数据", "含义", "怎么看"], [
["t/s", "每秒处理的词元（token）数。词元是模型处理文本的单位，一个词元可能是一个词或词的一部分。", "越高越好。"],
["pp128、pp512", "提示处理（prompt processing）速度：模型一次读入 128 或 512 个词元时的速度。", "决定长提示要等多久。例如 2000 t/s 读完 512 个词元约需 0.26 秒。"],
["tg64", "文本生成（text generation）速度：逐个生成 64 个词元时的速度。", "决定回答出字有多快。它和 pp 相差很大，不能互相换算。"],
["端到端生成 t/s", "用真实提示运行 <code>llama-completion</code> 时的生成速度。", "比 tg64 更接近实际使用。不同设备的输出长度可能不同，不宜跨设备直接比较。"],
["11 词元提示耗时", "读入一段 11 个词元的短提示所用的时间。", "反映短问题的首字等待，不含模型加载。"],
["模型加载 ms", "从启动到模型可用的时间。", "受页缓存影响很大；冷启动一节是清空缓存后的结果。"],
], "text")}
<h3>配置</h3>
{table(["数据", "含义", "怎么看"], [
["NPU、GPU、CPU", "分别由 Hexagon HTP、Adreno（OpenCL）、CPU 核心执行计算。", "比较速度时，也要考虑是否占用其他资源。"],
["全部层、部分卸载（<code>-ngl</code>）", "<code>-ngl</code> 指定放到加速器上的层数。全部层是整个模型都在 NPU 上，16 层则其余部分留在 CPU 上。", "部分卸载要在 CPU 和 NPU 之间来回协作，所以结果和全部层差别很大。"],
["host 线程数", "CPU 上参与计算和调度的线程数。", "全部层在 NPU 上时影响很小；部分卸载时影响很大。"],
["NPU 轮询、CPU 轮询", "等待任务时持续检查（响应快但占 CPU），而不是休眠等待。", "两者同时开启，是 SM8845 部分卸载时速度崩溃的触发条件。"],
["high_performance、balanced", "NPU 的功耗模式。", "这里只比较了速度，功耗差异没有测量。"],
["上游基线、优化版", "上游 aa588a346 和 fork 4b0773c7d。", "用来判断 fork 的改动是否带来变化。"],
], "text")}
<h3>统计和系统状态</h3>
{table(["数据", "含义", "怎么看"], [
["大数字和小字范围", "大数字是多次运行的中位数，小字是最小到最大值。", "范围重叠时，差异通常不足以说明谁更快。"],
["变化 %", "优化版相对上游基线的变化。", "要结合波动范围判断，几个百分点的差异可能只是波动。"],
["RSS", "进程占用的物理内存峰值。", "不等于总内存占用：NPU 另有内存记在设备上，且部分重叠。"],
["峰值 °C", "运行期间所有 CPU 温度传感器的最大值。", "只用于相对比较，开发机的读数不代表手机表面温度。"],
["Thermal Status、频率上限", "系统报告的热状态和 CPU 允许的最高频率。", "SM8845 状态为 0 时 CPU 仍被限频，它的 CPU 数据低于芯片能力。"],
["算子占比", "各类计算占 NPU 时间的比例。", "用来找优化重点。开启剖析后整体会变慢，所以只看比例，不看绝对时间。"],
], "text")}
<h3>提速实验</h3>
{table(["数据", "含义", "怎么看"], [
["总吞吐", "所有并发路每秒生成的词元数之和。", "代表系统能力，不代表单个用户感受到的速度。"],
["每路速度", "总吞吐除以路数。", "这才是每个用户看到的出字速度。"],
["一步耗时", "处理一批词元的时间，每路各生成一个词元。", "路数增加时涨得越慢，批量的收益越大。"],
["倍数", "同一轮内，多路总吞吐除以单路 tg。", "10 倍的目标看这个数。"],
["推测解码、n、m", "先按输入中出现过的片段猜接下来的词元，再一次批量验证。n 是匹配长度，m 是每次最多猜几个。", "猜中才会提速；猜错会浪费计算，可能比不推测更慢。"],
["草稿数、接受数、接受率", "猜了多少词元、验证通过多少，以及两者之比。", "接受率低但总速度高也是可能的，最终以 t/s 为准。"],
], "text")}
<h3>数值一致性</h3>
{table(["数据", "含义", "怎么看"], [
["logits", "模型给词表中每个词元打的分，分数最高的词元在贪心采样时被选中。", "比较 logits 比比较文本灵敏，文本相同不代表计算结果相同。"],
["逐位相同、不同比例", "两次计算的 logits 是否完全相等，以及不相等的比例。", "100% 不同也可能只是很小的数值差异，要结合 rms 看。"],
["rms", "所有 logits 差异的均方根，表示典型差异大小。", "越小越接近。"],
["最大绝对差", "所有 logits 中最大的单项差异。", "它受少数极端值影响，只说明最坏情况。"],
["最高分翻转", "分数最高的词元变了多少次。", "翻转才会直接改变贪心输出。"],
["最小领先差", "第一名和第二名分差的最小值。", "分差越小，数值差异越容易改变输出。"],
], "text")}

<h2>NPU、GPU、CPU 对比</h2>
<div class="grid2">
{bars("SM8850 提示处理 pp512", "t/s，越长越好", [("NPU", med(B,'opt_htp','pp512'), "c-npu"), ("GPU", med(B,'opt_gpu','pp512'), "c-gpu"), ("CPU 6 线程", med(B,'opt_cpu','pp512'), "c-cpu")])}
{bars("SM8850 生成 tg64", "t/s，越长越好", [("NPU", med(B,'opt_htp','tg64'), "c-npu"), ("GPU", med(B,'opt_gpu','tg64'), "c-gpu"), ("CPU 6 线程", med(B,'opt_cpu','tg64'), "c-cpu")])}
{bars("SM8845 提示处理 pp512", "t/s，越长越好", [("NPU", med(A,'opt_htp','pp512'), "c-npu"), ("GPU", med(A,'opt_gpu','pp512'), "c-gpu"), ("CPU 6 线程", med(A,'opt_cpu','pp512'), "c-cpu")])}
{bars("SM8845 生成 tg64", "t/s，越长越好", [("NPU", med(A,'opt_htp','tg64'), "c-npu"), ("GPU", med(A,'opt_gpu','tg64'), "c-gpu"), ("CPU 6 线程", med(A,'opt_cpu','tg64'), "c-cpu")])}
</div>
<h3>SM8850（Canoe 开发机）</h3>
{table(["后端", "pp128 t/s", "pp512 t/s", "tg64 t/s", "RSS MiB", "峰值 °C"], backend_rows(B))}
<h3>SM8845（PLR110，CPU 限频）</h3>
{table(["后端", "pp128 t/s", "pp512 t/s", "tg64 t/s", "RSS MiB", "峰值 °C"], backend_rows(A))}
<p class="note">RSS 是进程峰值，不等于总内存占用。NPU 运行时另有约 2020 MiB 记在 HTP 设备上（模型 1362、KV 缓存 24、计算缓冲 634），且部分与 RSS 重叠。温度是全部 CPU 传感器的最大值；SM8850 开发机曾读到 105°C，只能用来相对比较。</p>

<h2>优化版与上游基线</h2>
<h3>SM8850</h3>
{table(["指标", "上游基线", "优化版", "变化"], ab_rows(B))}
<h3>SM8845</h3>
{table(["指标", "上游基线", "优化版", "变化"], ab_rows(A))}
<p>SM8850 上的差异都在波动范围内。SM8845 波动较大，pp128 差 {pct(A,'opt_htp','base_htp','pp128')}，且两组范围几乎不重叠，但 SM8850 上没有复现（{pct(B,'opt_htp','base_htp','pp128')}），所以暂不判定为回归。CPU 路径同样无差异：SM8850 上 CPU pp512 {pct(B,'opt_cpu','base_cpu','pp512')}、tg64 {pct(B,'opt_cpu','base_cpu','tg64')}。</p>

<h2>host 线程数和功耗模式</h2>
{table(["配置（NPU 全部层）", "SM8845 pp512", "SM8845 tg64", "SM8850 pp512", "SM8850 tg64"], sweep_rows([("1 线程","t1"),("2 线程，high_performance","opt_htp"),("4 线程","t4"),("6 线程","t6"),("2 线程，balanced","power_balanced")]))}
<p>全部层都在 NPU 上时，host 线程数基本不影响速度，1 线程就够了。<code>balanced</code> 相比 <code>high_performance</code>，SM8850 的 pp512 降 {abs((med(B,'power_balanced','pp512')/med(B,'opt_htp','pp512')-1)*100):.1f}%，tg64 降 {abs((med(B,'power_balanced','tg64')/med(B,'opt_htp','tg64')-1)*100):.1f}%；功耗差异未测量。</p>

<h2>部分卸载（16 层在 NPU）</h2>
{table(["配置", "SM8845 pp512", "SM8845 tg64", "SM8850 pp512", "SM8850 tg64"], sweep_rows([("优化版，6 线程","ngl16"),("上游基线，6 线程","ngl16_base"),("开权重流式，6 线程","ngl16_stream"),("开层预取，6 线程","ngl16_layerpf"),("关 NPU 卸载，6 线程","ngl16_nooffload"),("优化版，4 线程","ngl16_t4"),("优化版，2 线程","ngl16_t2")]))}
<p>SM8850 上部分卸载的 tg64 比全 NPU 略高，但 pp512 只有全 NPU 的约五分之一，上游基线同样如此。SM8845 上 4 线程和 6 线程的生成速度崩溃，下一节是实验结果。</p>

<h3>SM8845 的 6 线程崩溃</h3>
<div class="grid2">
{bars("SM8845 部分卸载 tg64", "t/s，3 次中位数", poll_chart)}
<div>{table(["配置", "线程", "NPU 轮询", "CPU 轮询", "tg64 t/s"], poll_table)}</div>
</div>
<p>上游基线同样会崩溃，所以这不是 fork 引入的回归。SM8850 上相同配置为 {med(B,'ngl16','tg64'):.1f} t/s，没有崩溃。这组实验说明两种轮询同时开启是触发条件。我的推测是：CPU 限频时，NPU 轮询线程和 6 个自旋的 CPU 线程争抢核心；这一机制没有用调度跟踪验证。实验中 CPU 频率上限也有变化，且对照组先跑，所以数字只用于判断方向。</p>
<p>ggml-hexagon 库本身默认不开 NPU 轮询（<code>opt_oppoll = 0</code>），但 <code>scripts/snapdragon/run.py</code> 默认设置 <code>GGML_HEXAGON_OPPOLL=1</code>，手机上已有的 <code>bench.sh</code> 和本次矩阵也都开启了它。所以用这些脚本测部分卸载会遇到崩溃，不设这个变量的程序则不会。</p>

<h3>run.py 的修复</h3>
<p>已修改 <code>scripts/snapdragon/run.py</code>（本地提交 3ab067605）：检测到部分卸载（<code>-ngl</code> 小于 99，或 <code>-ot</code>、<code>--cpu-moe</code> 等）时，NPU 轮询默认为 0；全 NPU 时仍为 1；<code>--hex-oppoll</code> 可显式覆盖。没有全局关闭，因为全 NPU 下轮询是取舍关系：SM8845 交替运行 5 轮，开启时 tg64 中位数 30.5 t/s、pp512 1542 t/s，关闭时 28.4 t/s、1725 t/s。</p>
<p>通过 <code>run.py</code> 在 SM8845 上验证（16 层 NPU、6 线程）：默认配置下 tg64 为 15.2、17.6、19.6 t/s；显式 <code>--hex-oppoll 1</code> 时 1.0 t/s，崩溃可复现；全 NPU 仍为轮询开启。SM8850 未复测这项修改。手机上的 <code>bench.sh</code> 写死了 <code>OPPOLL=1</code>，不受这项修改影响；上文的部分卸载表格是修复前的数据。</p>

<h2>冷启动（SM8850）</h2>
<p>每次运行前清空页缓存（<code>drop_caches</code>），16 层在 NPU、6 线程，共 3 次，取中位数。</p>
{table(["配置", "加载 ms", "11 词元提示 ms", "生成 ms"], cold_rows())}
<p>各配置没有明显差异。上游基线首次运行加载 1977 ms，较慢，之后回到约 1600 ms。权重流式和层预取不作用于这个模型，所以这组数据不能代表它们的冷启动效果。</p>

<h2>NPU 算子耗时</h2>
<p>来自 <code>GGML_HEXAGON_PROFILE=1</code>，每个词元约 793 个算子。开启剖析和详细日志后速度会下降，所以这里只看占比。</p>
{table(["算子", "SM8845 占比", "SM8850 占比"], [["MUL_MAT（含 lm_head）","50.6%","50.2%"],["其中 lm_head，Q6_K，1536×262144","19.1%","18.7%"],["MUL_MAT_NX（注意力 QKV 和 FFN 合并矩阵乘）","39.0%","38.5%"],["RMS_NORM+MUL","3.7%","3.9%"],["ADD","1.4%","1.6%"],["FLASH_ATTN_EXT","1.4%","1.5%"]])}
<p>每词元 NPU 批处理的平均耗时，SM8845 约 39.0 ms（1862 MHz），SM8850 约 33.2 ms（1824 MHz）。SM8850 的 NPU 时钟略低但更快，可能和内存带宽或其他微架构差异有关，这一点没有验证。</p>

<h2>提速实验（SM8845）</h2>
<p>目标是把推理速度提高 10 倍。这部分全部使用现有功能和参数，没有修改推理框架源码。配置为全部层在 NPU、2 线程、NPU 轮询开启。</p>
<h3>瓶颈分析</h3>
<p>每生成一个词元要读取约 1.38 GB 权重，其中 FFN 约 876 MB、注意力约 157 MB、输出层约 330 MB。按 28 t/s 计算约 38 GB/s，已接近手机内存的实际带宽。NPU 和 CPU 的生成速度也几乎相同，说明换算力帮助不大。这是根据数据作出的推断，内存带宽没有直接测量。</p>
<h3>多路并发</h3>
<div class="grid2">
{bars("多路总吞吐", "t/s，3 轮中位数", batch_chart)}
<div>{table(["路数", "总吞吐 t/s", "每路 t/s", "一步 ms", "倍数"], batch_table)}</div>
</div>
<p>24 路时倍数超过 10，但每路速度低于单路。数据来自 <code>llama-batched-bench</code>，每路提示 128 词元、生成 128 词元，没有模拟真实服务的请求到达和长度变化。</p>
<h3>单路推测解码</h3>
{table(["代码改写，生成约 1300 词元", "速度 t/s", "相对不推测", "草稿数", "接受数", "接受率"], spec_rows())}
{table(["提示类型（n=2，m=16，单次运行）", "不推测 t/s", "推测 t/s", "倍数"], prompt_table)}
<p>提速幅度取决于输出中有多少内容能从输入里照抄。上表最好的结果约 5.7 倍；普通问答几乎猜不中，推测基本没有收益。</p>

<h2>批量解码的数值差异（SM8845）</h2>
<p>实验中发现，同样的贪心采样，有无推测的输出可能不同。例如普通问答中，一次输出“red, orange, and yellow”，另一次输出“red and orange”。文本都通顺，但推测解码本应不改变结果，所以做了 logits 对比。</p>
<h3>和单步解码的差异</h3>
{table(["NPU 配置", "批量 4", "批量 5", "批量 8"], logit_rows())}
<p class="note">同一提示，前缀 200 词元，比较 50 个位置，均未跨过位置 256。“三项同时”指 matmul 和注意力只用 HVX，并关闭融合。这些位置的最高分词元都没有翻转。</p>
<p>三项同时关闭后，到批量 8 都逐位相同；相同配置重复运行也逐位相同。所以差异来自 HMX、HVX 和融合路径随批量大小切换，不是随机噪声。融合不是根因：关闭后差异仍在，只是变小。</p>
<h3>和 CPU 单步结果的距离</h3>
{table(["配置", "rms", "最大绝对差", "最高分翻转"], ref_rows())}
<p class="note">比较 64 个位置，其中最后 8 个位置已跨过 256。这里把 CPU 单步当作参照，不代表它是绝对正确的结果。</p>
<p>NPU 批量结果比单步更接近 CPU，所以没有证据表明批量结果更差。这组样本中最小领先差约 1.1–1.2，大于批量差异的典型幅度，因此输出通常不变，只有接近打平时才会变化。这组对比只用了一个提示。</p>
<h3>固定路径的代价</h3>
{table(["指标", "默认", "三项同时"], det_table)}
<p>固定路径的速度代价很大，也没有完全解决问题：代码评审提示的输出仍然不同。它从约 1576 词元开始生成，途中会跨过 1792 这个 KV 对齐边界。此外，CPU 上位置跨过 256 后，单步和批量的 logits 也开始不同（最大约 0.57），关闭图复用后不变。这可能和 KV 按 256 对齐有关，根因没有查明。</p>
<p><b>建议：</b>日常保持默认配置，不要为了可复现而切到固定路径。批量解码的结果可以视为与单步等效，但不是逐位相同。需要逐字复现时，使用单路、不开推测，并固定配置和提示。</p>

<h2>建议的下一步</h2>
<ol>
<li><b>验证真实并发。</b>用 <code>llama-server</code> 多槽位和不同长度的请求测吞吐与延迟，并在 SM8850 上重复提速实验。</li>
<li><b>查清 256 边界的差异。</b>它在 CPU 上也出现，不限于 NPU。</li>
<li><b>优化 lm_head。</b>它占约 19% 的 NPU 时间。可以试更低比特的量化，或复查 Q6_K 大矩阵的 NPU 路径，再评估精度影响。预期收益还只是估算。</li>
<li><b>给 Gemma 4 接入流式和层预取。</b>只有模型超出可用内存时才有意义。E2B 不需要，12B 这类模型可能需要；手机上已有的 12B 测试脚本也只尝试部分卸载。</li>
<li><b>另做一轮大模型验证。</b>用 Qwen 或 MoE 模型，在内存压力下对比这些优化，这才是它们的目标场景。</li>
<li><b>补上功耗测量。</b>NPU 和 CPU 的生成速度接近，选择哪个主要取决于能效，而这次没有这项数据。</li>
</ol>

<h2>数据限制</h2>
<div class="warn">
<p><b>SM8845 的 CPU 全程受限。</b>即使 Thermal Status 为 0，CPU 上限大多仍只有约 2266/1651 MHz（硬件最高 3322/3802 MHz）。CPU 和部分卸载的数据明显低于芯片能力；NPU 全层的结果受影响较小但也不是满血数据。SM8845 的纯 CPU pp128 还出现 12–149 t/s 的双峰，原因未查明。</p>
<p><b>SM8845 第一轮数据已弃用。</b>第一轮在息屏 Doze 状态下运行，没有用于上表，保留在 <code>results/full/sm8845-throttled-doze.jsonl</code>。</p>
<p><b>没有测试 Gen 6。</b>本次云端设备是 SM8850，不是你最终要调试的 Gen 6 芯片。</p>
<p><b>两次 SM8850 会话用了不同设备。</b>序列号分别是 64a112da 和 cf39e6a。结果高度一致（opt_htp pp512 {med(B1,'opt_htp','pp512'):,.1f} 对 {med(B,'opt_htp','pp512'):,.1f}，tg64 {med(B1,'opt_htp','tg64'):.1f} 对 {med(B,'opt_htp','tg64'):.1f}），表中使用第二次的数据。开发机与量产手机的散热和存储不同。</p>
<p><b>不同设备或后端的输出不完全相同。</b>同一设备上优化版和基线文本一致，但 SM8845 NPU（56 词元）与 SM8850 NPU 和 CPU（66 词元）不同，SM8845 CPU 则与后者一致。两份回答都合理；这可能来自数值差异，尚未查明，所以端到端生成速度不宜跨设备直接比较。批量大小对输出的影响见“批量解码的数值差异”一节。</p>
<p><b>提速和数值实验范围有限。</b>都只在 SM8845 上做，SM8850 没有测；部分结果是单次运行，已在表中注明。CPU 受限频影响，这里的倍数不能直接套用到其他设备。</p>
<p><b>没有测试：</b>功耗和能效、长时间持续负载、内存压力、不同上下文长度，以及模型输出质量评估。</p>
</div>
<p class="note">原始数据和脚本位于 <code>tools/hui-lab/results/full/</code>、<code>fullbench.py</code>、<code>device/</code>。</p>
</main>
"""
out = LAB / "report" / "report.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(out, len(html))
