#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
E=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
for np in 32 128 255; do input keyevent KEYCODE_WAKEUP; echo "--- np=$np"; timeout 280 ./bin/llama-parallel -m $E -dev HTP0 -ngl 99 -fa on -t 6 -c $((1024*(np+1))) -b 4096 -ub 512 -np $np -ns $np -n 32 --temp 0 --top-k 1 -v >par.out 2>par.err; echo rc=$?; grep -hE 'Total|speed' par.err par.out | tail -5 | cut -c1-130; grep -E ' E | W decode' par.err | tail -1 | cut -c1-130; done
