#!/system/bin/sh
# single stream greedy generation on the NPU with and without speculation, on three kinds of prompt
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
O=/data/local/tmp/spec; 
for w in edit mixed open; do for s in none ngram-mod ngram-map-k ngram-simple; do for i in 0 1; do
    input keyevent KEYCODE_WAKEUP
    case $s in none) a="--spec-type none";; *) a="--spec-type $s";; esac
    ./bin/llama-speculative-simple -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 4096 -b 2048 -ub 512 -n 300 --temp 0 --top-k 1 --seed 42 -f $O/$w.txt $a > $O/$w.$s.$i.out 2> $O/$w.$s.$i.err
    echo "$w $s #$i rc=$? md5=$(md5sum < $O/$w.$s.$i.out | cut -c1-8) $(grep -E 'decoded |n_draft|n_predict|n_drafted|n_accept|accept  ' $O/$w.$s.$i.err | tr -s ' ' | cut -c1-70 | tr '\n' ';')"
done; done; done
