#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_VERBOSE=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf; O=/data/local/tmp/spec
./bin/llama-speculative-simple -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 4096 -b 2048 -ub 512 -n 12 --temp 0 --top-k 1 --seed 42 -f $O/mixed.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 2 --spec-ngram-simple-size-m 16 --spec-draft-n-max 16 > $O/V.out 2> $O/V.err
wc -l $O/V.err; grep -iE 'glu|gelu' $O/V.err | sed -E 's/^[0-9.]+ . //' | cut -c1-170 | sort | uniq -c | sort -rn | head -30
