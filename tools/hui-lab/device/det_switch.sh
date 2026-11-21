#!/system/bin/sh
# does greedy output still depend on drafting when one NPU code path is switched off?
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf; O=/data/local/tmp/spec
one() { timeout 150 ./bin/llama-speculative-simple -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 4096 -b 2048 -ub 512 -n 300 --temp 0 --top-k 1 --seed 42 -f $O/open.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n $1 --spec-ngram-simple-size-m $2 --spec-draft-n-max $2 > $O/D.out 2> $O/D.err; r=$?; echo "rc=$r $(md5sum < $O/D.out | cut -c1-8) $(grep -E 'decoded ' $O/D.err | sed 's/.*speed: //')"; }
for c in NONE=0 GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_FA_SPLIT=0 GGML_HEXAGON_OPFUSION=0 GGML_HEXAGON_RT_REPACK=0 GGML_HEXAGON_AR_SELECT=1; do
    input keyevent KEYCODE_WAKEUP
    (export $c; echo "$c nodraft $(one 1000 1)"; echo "$c draft $(one 2 16)")
done
