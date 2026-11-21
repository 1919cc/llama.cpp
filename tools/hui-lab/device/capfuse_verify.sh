#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
input keyevent KEYCODE_WAKEUP
P="Explain why the sky is blue, then list three uses of copper."
for m in 1 8190; do
 GGML_HEXAGON_OPFUSION=$m ./bin/llama-completion -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -n 160 --temp 0 --seed 1 -no-cnv -p "$P" >out_$m.txt 2>/dev/null; echo "greedy mask=$m rc=$? $(md5sum out_$m.txt | cut -c1-12) $(wc -c < out_$m.txt)b"
done
cmp out_1.txt out_8190.txt && echo IDENTICAL
for m in 1 8190; do for i in 1 2 3 4 5; do input keyevent KEYCODE_WAKEUP
 echo "mask=$m #$i $(GGML_HEXAGON_OPFUSION=$m ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 12800 -b 4096 -ub 512 -npp 16 -ntg 32 -npl 256 2>/dev/null | grep '| *256 |')  temp=$(cat /sys/class/thermal/thermal_zone0/temp)"; sleep 20; done; done
