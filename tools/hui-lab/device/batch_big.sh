#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
input keyevent KEYCODE_WAKEUP
./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 12800 -b 4096 -ub 512 -npp 16 -ntg 32 -npl 1,32,64,128,256 2>big.err
echo rc=$?; tail -3 big.err | cut -c1-200
