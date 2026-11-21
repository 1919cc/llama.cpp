#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
G=/data/local/tmp/gguf/small
./bin/llama-quantize --token-embedding-type q4_0 --output-tensor-type q4_0 $G/Qwen3.5-2B-BF16.gguf $G/qwen35-2b-Q4_0-h4.gguf Q4_0 6 2>&1 | grep -E 'quant size|error' | cut -c1-100
rm $G/Qwen3.5-2B-BF16.gguf
P="<|im_start|>user
Explain why the sky is blue in three sentences.<|im_end|>
<|im_start|>assistant
<think>

</think>

"
for n in qwen35-2b-Q4_0 qwen35-2b-Q4_0-h4; do M=$G/$n.gguf; echo "=== $n"
 for r in 1 2 3; do for npl in 1 2 4; do input keyevent KEYCODE_WAKEUP; echo "$(timeout 60 ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c $((512*npl)) -b 2048 -ub 512 -npp 64 -ntg 64 -npl $npl 2>/dev/null | grep -E '^\|  *[0-9]' || echo TIMEOUT)"; done; done
 timeout 90 ./bin/llama-completion -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -n 90 --temp 0 --seed 1 -no-cnv -p "$P" 2>/dev/null | tail -c 500; echo
done
