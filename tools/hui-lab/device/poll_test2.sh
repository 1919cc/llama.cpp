#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf
for cfg in "ctrl 1 50 6" "nooppoll 0 50 6" "nopoll 1 0 6" "neither 0 0 6" "t2 1 50 2"; do set -- $cfg
  for i in 0 1 2; do
    input keyevent KEYCODE_WAKEUP
    w=0; while [ "$(dumpsys thermalservice 2>/dev/null | grep -m1 'Thermal Status' | tr -dc 0-9)" != "0" ] && [ $w -lt 300 ]; do sleep 15; w=$((w+15)); done
    r=$(GGML_HEXAGON_OPPOLL=$2 ./bin/llama-bench -m $M -r 1 -o json -p 128 -n 64 -dev HTP0 -ngl 16 -fa on -t $4 --poll $3 2>/dev/null | grep '"avg_ts"' | tr -dc '0-9.\n' | tr '\n' ' ')
    echo "$1 oppoll=$2 poll=$3 t=$4 #$i pp128_tg64=$r cap=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq) ts=$(dumpsys thermalservice 2>/dev/null | grep -m1 'Thermal Status' | tr -dc 0-9)"
  done
done
