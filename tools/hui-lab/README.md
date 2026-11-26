# hui-lab：端侧测速与验证（本地 8845 + QDC）

原 `hexagon-lab`，后放在 EdgeAgent/validation，现归入引擎仓库 `llama.cpp_hui`（`hui_infer` 分支）的 `tools/hui-lab/`。数值一致性工具（logit-cmp、kvdiff）在 `tools/hui-debug/`，App 里的引擎 JNI 在 `tools/hui-jni/`，优化与实测报告是仓库根目录的 `HUI_REPORT.md`。

门控测速：`bench_gate.py` 用 llama-bench（随机 token），`completion_gate.py` 用 `prompts/` 里的真实提示跑 llama-completion（同样的满频门控，报告 prefill / decode）。MoE 专家缓存类优化要用后者：随机 token 的路由和真实文本不同。

`results/full/` 里两份 22 MB 的 `sm88xx-diag.err` 原始日志没有入库。

日常优化用本地 SM8845。需要目标芯片的数据时，用同一个包、同一条命令在 QDC 真机上运行，再用 `compare` 对比。

| 环节 | 位置 | 工具 |
|---|---|---|
| 构建和打包 | Mac 上的 Colima 容器 | `lab.py build` |
| 日常调试 | 本地手机，adb 默认端口 5037 | `lab.py push`、`lab.py bench`、`scripts/snapdragon/run.py` |
| 目标芯片测速 | QDC 交互会话，SSH 隧道转发 adb | 同样的命令，加 `--adb-port <端口>` |
| 批量回归 | QDC 自动任务，可排队 | `lab.py qdc-job`（封装 `run_qdc_jobs.py`） |

`LLAMA_REPO` 可以指定其他 llama.cpp 目录，默认是本仓库根目录。

第一次使用先建虚拟环境（只有 QDC 自动任务需要）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 每次开始

```bash
colima start
```

```bash
python3 lab.py doctor
```

## 本地 8845

无线调试的配对和连接方式见下面的“连接手机”。连接后：

```bash
python3 lab.py build
```

```bash
python3 lab.py push --model /path/to/model.gguf
```

```bash
python3 lab.py bench --label sm8845-local --model model.gguf
```

默认跑 5 次，每次间隔 60 秒，功耗模式为 `high_performance`。结果保存在 `results/<label>/`，文件内记录提交、SoC、模型和参数。

### 连接手机

开启「开发者选项 → 无线调试」后，用屏幕上显示的地址：

```bash
adb pair <手机IP>:<配对端口> <配对码>
```

```bash
adb connect <手机IP>:<连接端口>
```

如果开热点时无法打开无线调试，先用 USB 运行 `adb tcpip 5555`，再运行 `adb connect <手机IP>:5555`。

USB 和 Wi-Fi 同时连接时，adb 会看到两台设备，必须指定序列号。`lab.py` 加 `--serial`，或者在当前终端设置一次：

```bash
export ANDROID_SERIAL=192.168.1.33:5555
```

手机重启后要重新运行 `adb tcpip 5555`。手机 IP 变化后要重新 `adb connect`。

## QDC 准备（只需一次）

1. 用 Qualcomm ID 登录 https://qdc.qualcomm.com ，在设备目录查找目标芯片。
2. 如果分钟数不够，在设备页面提交 Request Minutes。
3. 如果要用自动任务，在 User Settings → API Keys 生成密钥，并只放在环境变量里。密钥 6 个月后过期，重新生成会替换旧密钥。

```bash
read -rs QDC_API_KEY && export QDC_API_KEY
```

```bash
python3 lab.py qdc-targets
```

## QDC 交互会话

1. 创建交互会话时提供 SSH 公钥。设备被占用时无法创建，需要稍后再试。
2. 会话启动后点 Connect，运行页面给出的端口转发命令，把本地端口从 5037 改成 15037，以免与本地 adb 冲突。每个会话的主机名不同：

```bash
ssh -i <QDC 私钥.pem> -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:15037:<会话主机>:5037 -N sshtunnel@ssh.qdc.qualcomm.com
```

私钥权限必须是 600。隧道必须先于 adb 命令启动，否则 adb 可能在 15037 上启动一个本地服务。
3. 使用同样的命令，只多加端口：

```bash
python3 lab.py doctor --adb-port 15037
```

```bash
python3 lab.py push --adb-port 15037 --model /path/to/model.gguf
```

```bash
python3 lab.py bench --adb-port 15037 --label gen6-qdc --model model.gguf
```

本地手机和 QDC 使用不同的 adb 端口，可以同时连接。

## QDC 自动任务

```bash
python3 lab.py qdc-job --device SM8850 --model-url <模型直链>
```

`run_qdc_jobs.py` 的 `DEVICE_PLATFORM` 目前只列出 SM8750、SM8650、SM8850 和 QCS9075M。其他芯片要先在这张表里加一项。

## 对比结果

第一个文件是基线：

```bash
python3 lab.py compare results/sm8845-local/<文件>.json results/gen6-qdc/<文件>.json
```

## 满频门控 A/B（速度结论以此为准）

`bench_gate.py` 在本地手机上对比两个或多个引擎构建，每次运行前都确认 CPU、GPU、NPU 处于最高频率（固定性能模式、温控状态 0、所有降频设备为 0、CPU 各簇 `scaling_max_freq == cpuinfo_max_freq`），被降频就等回到满频再测；一次只测一个模型，构建逐轮交替。结果写入 `results/<label>/runs.jsonl`，并打印均值 ± 标准差和相对第一个构建的变化。

```bash
python3 bench_gate.py --model /data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf \
    --build base=/data/local/tmp/hui-a --build new=/data/local/tmp/hui-b --rounds 5 --label my-ab \
    -- -p 512 -n 128 -ngl 99 -fa 1 -t 4 --device HTP0
```

每个构建目录要有 `bin/llama-bench` 和 `lib/`。adb 依次取 `$ADB`、PATH、`~/Library/Android/sdk/platform-tools/adb`。

## 对比规则

- 模型、功耗模式、batch 和 token 数必须相同，否则 `compare` 会警告。
- `spread` 是波动幅度。提升幅度小于波动时，不算有效提升。
- 只比较同类结论：同一芯片上比较提交前后，不同芯片上比较同一提交。
- QDC 设备可能是 QRD 或 MTP，散热和闪存可能与量产机不同。
- 新芯片的 HTP 版本如果高于 v85，需要新版 Hexagon SDK，并在 `ggml/src/ggml-hexagon/CMakeLists.txt` 增加构建项。

## 目录

| 路径 | 内容 |
|---|---|
| `lab.py` | 构建、推送、测速、QDC、对比 |
| `bench_gate.py` | 满频门控的多构建 A/B 测速（`HUI_REPORT.md` 的速度数据来源） |
| `fullbench.py`、`summarize.py` | 整套测速矩阵（交错轮次、可续跑）和汇总到 `results/full/summary.json` |
| `server_load.py`、`logit_cmp_borrow*.py` | llama-server 真实请求负载，以及前缀借用前后的 logprob 对比 |
| `device/` | 推到手机上执行的实验脚本；`*.sh`（根目录）是在 Mac 上串起这些实验的驱动 |
| `build_report.py`、`report_to_docx.*` | 从测量数据生成 `report/` 下的 HTML 和 Word 报告 |
| `results/` | 原始测量数据 |
