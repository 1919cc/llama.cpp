#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
G=/data/local/tmp/gguf/small; O=/data/local/tmp/spec
R="./bin/llama-speculative-simple -fa on -t 2 -c 4096 -b 2048 -ub 512 --temp 0 --top-k 1 --seed 42"
show() { echo "$1 rc=$2 md5=$(md5sum < $O/H.out | cut -c1-8) $(grep -E 'decoded |n_drafted|n_accept' $O/H.err | sed 's/.* I //' | tr -s ' ' | tr '\n' ';')"; grep -E "vocab|terminating" $O/H.err | head -1 | cut -c1-150; }
for n in LFM2-350M-Q4_0 SmolLM2-135M-Instruct-Q4_0; do T=$G/$n.gguf; echo "=== $n"
 for i in 0 1 2; do input keyevent KEYCODE_WAKEUP
  $R -m $T -dev HTP0 -ngl 99 -n 1000 -f $O/edit.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 1000 --spec-ngram-simple-size-m 1 --spec-draft-n-max 1 > $O/H.out 2> $O/H.err; show "edit nodraft #$i" $?
  $R -m $T -dev HTP0 -ngl 99 -n 1000 -f $O/edit.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 32 --spec-draft-n-max 32 > $O/H.out 2> $O/H.err; show "edit ngram32 #$i" $?
  $R -m $T -dev HTP0 -ngl 99 -n 200 -f $O/open.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 2 --spec-ngram-simple-size-m 8 --spec-draft-n-max 8 > $O/H.out 2> $O/H.err; show "open ngram8 #$i" $?
 done
done
echo "=== draft model on CPU, gemma family"
T=$G/gemma-4-E2B-it-qat-Q4_0.gguf
for i in 0 1; do input keyevent KEYCODE_WAKEUP
 $R -m $T -dev HTP0 -ngl 99 -n 200 -f $O/open.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 1000 --spec-ngram-simple-size-m 1 --spec-draft-n-max 1 > $O/H.out 2> $O/H.err; show "E2B open nodraft #$i" $?
 for k in 2 4 8; do $R -m $T -dev HTP0 -ngl 99 -n 200 -f $O/open.txt -md $G/gemma-3-270m-it-Q4_0.gguf -devd none -ngld 0 -td 4 --spec-type draft-simple --spec-draft-n-max $k > $O/H.out 2> $O/H.err; show "E2B open draft270m-cpu k=$k #$i" $?; done
done
