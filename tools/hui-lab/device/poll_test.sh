#!/system/bin/sh
# partial-offload collapse at 6 host threads: NPU op polling and CPU threadpool polling
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
tg() { # label oppoll cpupoll threads
  for i in 0 1 2; do
    input keyevent KEYCODE_WAKEUP
    while [ "$(dumpsys thermalservice | grep -m1 'Thermal Status' | tr -dc 0-9)" != "0" ]; do sleep 15; done
    s=$(date +%s)
    r=$(GGML_HEXAGON_OPPOLL=$2 ./bin/llama-bench -m $M -r 1 -o csv -p 0 -n 64 -dev HTP0 -ngl 16 -fa on -t $4 --poll $3 2>/dev/null | tail -1 | tr ',' '\n' | grep -E '^"[0-9]+\.[0-9]+"$' | tr -d '"' | head -1)
    echo "$1 oppoll=$2 poll=$3 t=$4 #$i tg_ts=$r wall=$(($(date +%s)-s)) cap=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq)"
  done
}
tg ctrl 1 50 6
tg nooppoll 0 50 6
tg nopoll 1 0 6
tg neither 0 0 6
tg t2 1 50 2
echo POLL_DONE
