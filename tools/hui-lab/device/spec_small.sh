#!/system/bin/sh
# small models on one stream, then speculation: ngram on rewrite text, small draft model on open text
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
G=/data/local/tmp/gguf/small; O=/data/local/tmp/spec
Q="Explain why the sky is blue in two sentences."
for n in SmolLM2-135M-Instruct-Q4_0 gemma-3-270m-it-Q4_0 LFM2-350M-Q4_0 qwen2.5-0.5b-instruct-q4_0; do M=$G/$n.gguf; echo "=== $n"
  for r in 1 2 3; do input keyevent KEYCODE_WAKEUP
    ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 2048 -b 2048 -ub 512 -npp 64 -ntg 128 -npl 1,2,4 2>/dev/null | grep -E '^\|  *[0-9]'
  done
  ./bin/llama-completion -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -n 60 --temp 0 --top-k 1 --seed 42 -no-cnv -p "$Q" 2>/dev/null | tr '\n' ' ' | cut -c1-420; echo
done
R="./bin/llama-speculative-simple -fa on -t 2 -c 4096 -b 2048 -ub 512 --temp 0 --top-k 1 --seed 42"
show() { echo "$1 rc=$2 md5=$(md5sum < $O/G.out | cut -c1-8) $(grep -E 'decoded |n_drafted|n_accept' $O/G.err | sed 's/.* I //' | tr -s ' ' | tr '\n' ';')"; cp $O/G.out "$O/G.$3.out"; }
echo "=== speculation qwen35-2b target"
T=$G/qwen35-2b-Q4_0.gguf
for i in 0 1; do input keyevent KEYCODE_WAKEUP
  $R -m $T -dev HTP0 -ngl 99 -n 1000 -f $O/edit.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 1000 --spec-ngram-simple-size-m 1 --spec-draft-n-max 1 > $O/G.out 2> $O/G.err; show "edit nodraft #$i" $? edit.nodraft
  $R -m $T -dev HTP0 -ngl 99 -n 1000 -f $O/edit.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 32 --spec-draft-n-max 32 > $O/G.out 2> $O/G.err; show "edit ngram32 #$i" $? edit.m32
  $R -m $T -dev HTP0 -ngl 99 -n 300 -f $O/open.txt --spec-type ngram-simple --spec-ngram-simple-min-hits 1 --spec-ngram-simple-size-n 1000 --spec-ngram-simple-size-m 1 --spec-draft-n-max 1 > $O/G.out 2> $O/G.err; show "open nodraft #$i" $? open.nodraft
  $R -m $T -dev HTP0 -ngl 99 -n 300 -f $O/open.txt -md $G/qwen2.5-0.5b-instruct-q4_0.gguf -devd none -ngld 0 --spec-type draft-simple --spec-draft-n-max 4 > $O/G.out 2> $O/G.err; show "open draft0.5b-cpu n4 #$i" $? open.d4
  $R -m $T -dev HTP0 -ngl 99 -n 300 -f $O/open.txt -md $G/qwen2.5-0.5b-instruct-q4_0.gguf -devd none -ngld 0 --spec-type draft-simple --spec-draft-n-max 8 > $O/G.out 2> $O/G.err; show "open draft0.5b-cpu n8 #$i" $? open.d8
done
tail -3 $O/G.err | cut -c1-200
echo "=== fusion softplus mask=1 vs 16382"
for mask in 1 16382 1 16382; do input keyevent KEYCODE_WAKEUP
  GGML_HEXAGON_OPFUSION=$mask GGML_HEXAGON_PROFILE=1 ./bin/llama-batched-bench -m $T -dev HTP0 -ngl 99 -fa on -t 2 -c 2048 -b 2048 -ub 512 -npp 64 -ntg 16 -npl 1 -v >/dev/null 2>/data/local/tmp/prof_fuse_$mask.err
  echo "mask=$mask ops=$(grep -c SOFTPLUS /data/local/tmp/prof_fuse_$mask.err) last_batch=$(grep OPBATCH /data/local/tmp/prof_fuse_$mask.err | tail -5 | sed 's/.*usec \([0-9]*\).*/\1/' | tr '\n' ' ')"
done
GGML_HEXAGON_OPFUSION=1 ./bin/llama-completion -m $T -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -n 80 --temp 0 --top-k 1 --seed 42 -no-cnv -p "$Q" 2>/dev/null | md5sum
GGML_HEXAGON_OPFUSION=16382 ./bin/llama-completion -m $T -dev HTP0 -ngl 99 -fa on -t 2 -c 1024 -n 80 --temp 0 --top-k 1 --seed 42 -no-cnv -p "$Q" 2>/dev/null | md5sum
