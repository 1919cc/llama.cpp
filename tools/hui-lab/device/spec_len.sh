#!/system/bin/sh
# longer drafts on the prompt that repeats its input
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
O=/data/local/tmp/spec
for w in edit mixed open; do for d in 8 16 32 64; do for n in 2 4; do
    input keyevent KEYCODE_WAKEUP
    ./bin/llama-speculative-simple -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 4096 -b 2048 -ub 512 -n 300 --temp 0 --top-k 1 --seed 42 -f $O/$w.txt --spec-type ngram-simple --spec-ngram-simple-size-n $n --spec-ngram-simple-size-m $d --spec-ngram-simple-min-hits 1 --spec-draft-n-max $d > $O/L.$w.$d.$n.out 2> $O/L.$w.$d.$n.err
    echo "$w m=$d n=$n rc=$? md5=$(md5sum < $O/L.$w.$d.$n.out | cut -c1-8) $(grep -E 'decoded |n_predict|n_drafted|n_accept' $O/L.$w.$d.$n.err | sed 's/.* I //' | tr -s ' ' | tr '\n' ';')"
done; done; done
