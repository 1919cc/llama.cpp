#!/system/bin/sh
# full NPU offload: how the cost of one decode step grows with the batch size
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
ls bin/llama-batched-bench bin/llama-speculative bin/llama-speculative-simple 2>&1
input keyevent KEYCODE_WAKEUP
./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 8192 -b 2048 -ub 512 -npp 128 -ntg 32 -npl 1,2,4,8,16,32 2>batch.err
echo rc=$?; tail -3 batch.err | cut -c1-200
