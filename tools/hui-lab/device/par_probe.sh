#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
E=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
run() { input keyevent KEYCODE_WAKEUP; echo "--- np=$1 mask=$2"; GGML_HEXAGON_OPFUSION=$2 timeout 200 ./bin/llama-parallel -m $E -dev HTP0 -ngl 99 -fa on -t 6 -c $((256*$1)) -b 4096 -ub 512 -np $1 -ns $1 -n 32 --temp 0 --top-k 1 -v >par.out 2>par.err; echo rc=$?; grep -hE 'Total|n_parallel|Run parameters|Sequences decoded|Input|Tokens predicted|Total (prompt|gen|speed)|speed:' par.err par.out | tail -6 | cut -c1-130; [ -s par.err ] && grep -vE ' D | I ' par.err | tail -2 | cut -c1-150; }
run 8 1; run 32 1; run 128 1; run 256 1; run 256 8190
