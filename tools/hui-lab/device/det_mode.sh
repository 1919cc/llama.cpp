#!/system/bin/sh
# cost and effect of the configuration with one kernel path for all batch sizes
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf; O=/data/local/tmp/spec
one() { ./bin/llama-speculative-simple -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 4096 -b 2048 -ub 512 -n 300 --temp 0 --top-k 1 --seed 42 -f $O/$1.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n $2 --spec-ngram-simple-size-m $3 --spec-draft-n-max $3 > $O/D.out 2> $O/D.err; echo "$(head -c 100000 $O/D.out | md5sum | cut -c1-8) n=$(grep -E 'n_predict' $O/D.err | sed 's/.*= //') $(grep -E 'decoded ' $O/D.err | sed 's/.*speed: //')"; }
bb() { ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 8192 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,24 2>/dev/null | grep -E '^\|  +128'; }
for mode in default det; do
  if [ $mode = det ]; then export GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_OPFUSION=0; fi
  for w in open mixed edit; do input keyevent KEYCODE_WAKEUP; echo "$mode $w nodraft $(one $w 1000 1)"; echo "$mode $w draft $(one $w 2 16)"; done
  input keyevent KEYCODE_WAKEUP; echo "$mode batched:"; bb
done
