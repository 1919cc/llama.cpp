#!/system/bin/sh
# repeated runs: real no-draft baseline, long copy-like output, batch throughput, and determinism of batched verification
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf; O=/data/local/tmp/spec
R="./bin/llama-speculative-simple -m $M -fa on -t 2 -c 4096 -b 2048 -ub 512 --temp 0 --top-k 1 --seed 42 --spec-type ngram-simple --spec-ngram-simple-min-hits 1"
show() { echo "$1 rc=$2 md5=$(md5sum < $O/F.out | cut -c1-8) $(grep -E 'decoded |n_drafted|n_accept' $O/F.err | sed 's/.* I //' | tr -s ' ' | tr '\n' ';')"; cp $O/F.out "$O/F.$3.out"; }
export GGML_HEXAGON_DEVICES=HTP0
for i in 0 1 2; do
  for c in "nodraft 1000 1" "m32 4 32" "m64 4 64" "m16 4 16"; do set -- $c
    input keyevent KEYCODE_WAKEUP
    $R -dev HTP0 -ngl 99 -n 1300 -f $O/edit.txt --spec-ngram-simple-size-n $2 --spec-ngram-simple-size-m $3 --spec-draft-n-max $3 > $O/F.out 2> $O/F.err
    show "edit1300 $1 #$i" $? "edit.$1"
  done
done
for c in "nodraft 1000 1" "n2 2 16" "n4 4 16"; do set -- $c
    $R -dev HTP0 -ngl 99 -n 300 -f $O/open.txt --spec-ngram-simple-size-n $2 --spec-ngram-simple-size-m $3 --spec-draft-n-max $3 > $O/F.out 2> $O/F.err; show "open-npu $1" $? "open.npu.$1"
done
unset GGML_HEXAGON_DEVICES
for c in "nodraft 1000 1" "n2 2 16"; do set -- $c
    $R -dev none -ngl 0 -n 300 -f $O/open.txt --spec-ngram-simple-size-n $2 --spec-ngram-simple-size-m $3 --spec-draft-n-max $3 > $O/F.out 2> $O/F.err; show "open-cpu $1" $? "open.cpu.$1"
done
export GGML_HEXAGON_DEVICES=HTP0
for i in 0 1 2; do input keyevent KEYCODE_WAKEUP
  ./bin/llama-batched-bench -m $M -dev HTP0 -ngl 99 -fa on -t 2 -c 8192 -b 2048 -ub 512 -npp 128 -ntg 128 -npl 1,16,24,32 2>/dev/null | grep -E '^\|  +128'
done
