#!/system/bin/sh
# usage: server_start.sh <model path> <parallel> <ctx per slot> [extra server args...]
M=$1; NP=$2; CTX=$3; shift 3
cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
pkill -9 -x llama-server; sleep 2
input keyevent KEYCODE_WAKEUP
exec nohup ./bin/llama-server -m "$M" -dev HTP0 -ngl 99 -fa on -t 2 -c $((CTX*NP)) -np "$NP" -b 2048 -ub 512 --host 127.0.0.1 --port 18080 --no-webui "$@" > /data/local/tmp/server.log 2>&1
