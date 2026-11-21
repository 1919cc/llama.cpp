#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
G=/data/local/tmp/gguf/small; O=/data/local/tmp/spec
R="./bin/llama-speculative-simple -fa on -t 2 -c 4096 -b 2048 -ub 512 --temp 0 --top-k 1 --seed 42 -dev HTP0 -ngl 99 --spec-type ngram-simple --spec-ngram-simple-min-hits 1"
s() { grep -E 'decoded |n_drafted|n_accept' $O/I.err | sed 's/.* I //' | tr -s ' ' | tr '\n' ';'; }
for c in "SmolLM2-135M-Instruct-Q4_0 chatml" "qwen2.5-0.5b-instruct-q4_0 chatml" "LFM2-350M-Q4_0 chatml" "gemma-3-270m-it-Q4_0 gemma3"; do set -- $c; M=$G/$1.gguf; echo "=== $1"
 for i in 0 1 2; do input keyevent KEYCODE_WAKEUP
  $R -m $M -n 200 -f $O/open.$2.txt --spec-ngram-simple-size-n 1000 --spec-ngram-simple-size-m 1 --spec-draft-n-max 1 > $O/I.$1.open0.out 2>$O/I.err; echo "open nodraft $(s)"
  $R -m $M -n 200 -f $O/open.$2.txt --spec-ngram-simple-size-n 2 --spec-ngram-simple-size-m 8 --spec-draft-n-max 8 > $O/I.$1.open8.out 2>$O/I.err; echo "open ngram8 $(s)"
  $R -m $M -n 1000 -f $O/edit.$2.txt --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 32 --spec-draft-n-max 32 > $O/I.$1.edit.out 2>$O/I.err; echo "edit ngram32 $(s)"
 done
 echo "--- open:"; tail -c 450 $O/I.$1.open0.out | tr '\n' ' '; echo; echo "--- edit head:"; sed -n '/```c/,+6p' $O/I.$1.edit.out | tail -7 | tr '\n' '~' | cut -c1-260; echo; echo "ctx=$(tail -c 3000 $O/I.$1.edit.out | grep -c 'ctx') state=$(tail -c 3000 $O/I.$1.edit.out | grep -c 'state')"
done
