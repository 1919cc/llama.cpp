#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
G=/data/local/tmp/gguf/small; E=$G/gemma-4-E2B-it-qat-Q4_0.gguf; K=$G/Qwen3.5-2B-UD-Q2_K_XL.gguf
input keyevent KEYCODE_WAKEUP
for i in 1 2 3; do echo "Q2K $(./bin/llama-batched-bench -m $K -dev HTP0 -ngl 99 -fa on -t 2 -c 256 -b 2048 -ub 512 -npp 16 -ntg 32 -npl 1 2>/dev/null | grep -E '^\|  *[0-9]')"; done
echo "Q2K text: $(./bin/llama-completion -m $K -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -n 25 --temp 0 --seed 1 -no-cnv -p 'The capital of France is' 2>/dev/null | tail -c 180)"
input keyevent KEYCODE_WAKEUP
echo "== real parallel, greedy, gemma"
timeout 240 ./bin/llama-parallel -m $E -dev HTP0 -ngl 99 -fa on -t 6 -c 65536 -b 4096 -ub 512 -np 256 -ns 256 -n 32 --temp 0 --top-k 1 2>par.err >par.out; echo rc=$?; grep -E 'Total prompt|Total gen|Total speed|Cache|External|speed' par.err par.out | tail -6 | cut -c1-140; tail -2 par.err | cut -c1-140
input keyevent KEYCODE_WAKEUP
GGML_HEXAGON_PROFILE=1 ./bin/llama-batched-bench -m $G/qwen35-2b-Q1_0.gguf -dev HTP0 -ngl 99 -fa on -t 2 -c 256 -b 2048 -ub 512 -npp 16 -ntg 8 -npl 1 -v >/dev/null 2>q1.err; wc -l q1.err
