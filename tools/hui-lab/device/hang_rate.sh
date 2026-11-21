#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
for m in 1 8190; do ok=0; bad=0; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do input keyevent KEYCODE_WAKEUP
 if GGML_HEXAGON_OPFUSION=$m timeout 45 ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -b 2048 -ub 512 -npp 64 -ntg 64 -npl 2 2>/dev/null | grep -qE '^\|  *[0-9]'; then ok=$((ok+1)); else bad=$((bad+1)); echo "mask=$m run $i hang/fail"; fi; done; echo "mask=$m ok=$ok bad=$bad"; done
