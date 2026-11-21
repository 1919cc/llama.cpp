#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
input keyevent KEYCODE_WAKEUP
GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_VERBOSE=1 ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 12800 -b 4096 -ub 512 -npp 16 -ntg 4 -npl 256 -v >prof.out 2>prof.err
echo rc=$?; wc -l prof.out prof.err; grep -m3 -n -i 'profile\|usec\|cycles' prof.err | cut -c1-220; tail -6 prof.out | cut -c1-120
