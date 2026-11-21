#!/system/bin/sh
# full NPU offload: effect of NPU op polling, 2 host threads
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
for i in 0 1 2 3 4; do for o in 1 0; do
    input keyevent KEYCODE_WAKEUP
    r=$(GGML_HEXAGON_OPPOLL=$o ./bin/llama-bench -m $M -r 1 -o json -p 512 -n 64 -dev HTP0 -ngl 99 -fa on -t 2 2>/dev/null | grep '"avg_ts"' | tr -dc '0-9.\n' | tr '\n' ' ')
    echo "oppoll=$o #$i pp512_tg64=$r"
done; done
