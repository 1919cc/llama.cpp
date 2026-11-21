#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
G=/data/local/tmp/gguf/small
for q in Q4_0 Q2_0 Q1_0; do M=$G/qwen35-2b-$q.gguf
 input keyevent KEYCODE_WAKEUP; echo "=== $q"
 ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 12800 -b 4096 -ub 512 -npp 16 -ntg 32 -npl 1,1,1,32,128,256 2>$q.err | grep -E '^\|  *[0-9]'; tail -1 $q.err | cut -c1-140
 timeout 180 ./bin/llama-completion -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -n 60 --temp 0 --seed 1 -no-cnv -p "<|im_start|>user
What is the capital of France? Answer in one sentence.<|im_end|>
<|im_start|>assistant
<think>

</think>

" 2>/dev/null | tail -c 300; echo; sleep 30
done
