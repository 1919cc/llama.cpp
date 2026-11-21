#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
G=/data/local/tmp/gguf/small
for n in gemma-4-E2B-it-qat-Q4_0 qwen35-2b-Q4_0 Qwen3.5-2B-UD-Q2_K_XL; do M=$G/$n.gguf; echo "=== $n"
 for r in 1 2 3; do input keyevent KEYCODE_WAKEUP
  for npl in 1 2 3 4 8; do echo "#$r $(./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c $((512*npl)) -b 2048 -ub 512 -npp 64 -ntg 64 -npl $npl 2>/dev/null | grep -E '^\|  *[0-9]')"; done
 done
 for npl in 1 4; do GGML_HEXAGON_PROFILE=1 ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c $((512*npl)) -b 2048 -ub 512 -npp 64 -ntg 8 -npl $npl -v >/dev/null 2>/data/local/tmp/prof_${n}_$npl.err; done
done
