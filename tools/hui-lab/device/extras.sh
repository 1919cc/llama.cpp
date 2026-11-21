#!/system/bin/sh
# Cold-start runs on a rooted device (page cache dropped before each run) and one profiled, verbose run.
M=$1; O=/data/local/tmp/hl/out; R=/data/local/tmp/hl/run_case.sh; OPT=/data/local/tmp/llama.cpp; BASE=/data/local/tmp/hl/base
H="GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance"
A="-m $M -c 2048 -n 96 --temp 0 --seed 1 -no-cnv --device HTP0 -ngl 16 -fa on -t 6"
P="Explain in three sentences why the sky is blue."
cold() { n=$1; pkg=$2; shift 2
  for i in 0 1 2; do
    sync; echo 3 > /proc/sys/vm/drop_caches
    COOL_MC=50000 sh $R $pkg $O/cold_${n}_$i llama-completion $H "$@" -- $A -p "$P"
  done
}
cold base $BASE
cold opt $OPT
cold noprefetch $OPT LLAMA_MMAP_PREFETCH=0
cold nopre_layerpf $OPT LLAMA_MMAP_PREFETCH=0 LLAMA_LAYER_PREFETCH=1
cold nopre_stream $OPT LLAMA_MMAP_PREFETCH=0 LLAMA_WEIGHT_STREAM=1
cold nooffload $OPT GGML_HEXAGON_OFFLOAD=0
COOL_MC=50000 sh $R $OPT $O/diag llama-completion $H GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 -- -v -m $M -c 2048 -n 32 --temp 0 --seed 1 -no-cnv --device HTP0 -ngl 99 -fa on -t 2 -p "$P"
echo EXTRAS_DONE
