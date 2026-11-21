#!/bin/bash
# usage: hang_server.sh <tag> <rounds> <seconds per round> [clients]
# Runs real 4-way server load, restart per round, reports survive or hang time.
TAG=$1; ROUNDS=$2; SECS=$3; CL=${4:-4}
S=3B666H0171500000
M=${MODEL:-/data/local/tmp/gguf/small/qwen35-2b-Q4_0.gguf}
export NO_PROXY='*' no_proxy='*'
cd "$(dirname "$0")"
for r in $(seq 1 "$ROUNDS"); do
  adb -s $S shell "nohup sh /data/local/tmp/server_start.sh $M 4 2048 >/dev/null 2>&1 &"
  for i in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:18080/health | grep -q ok && break; perl -e 'select(undef,undef,undef,3)'; done
  N=$(adb -s $S shell 'pidof llama-server | wc -w'); [ "$N" = "1" ] || echo "WARN procs=$N"
  T0=$(date +%s)
  OUT=$(timeout $((SECS+200)) python3 server_load.py ways --clients "$CL" --prefix-words 300 --n 64 --seconds "$SECS" 2>&1 | grep -E '^ways|no completed' | head -1)
  T1=$(date +%s)
  LOG=$(adb -s $S shell 'grep -E "no response from DSP|dsp-rsp|Remote|0x8000|error:|dspqueue_read failed|Fatal|fastrpc" /data/local/tmp/server.log | head -2 | cut -c1-160')
  ALIVE=$(adb -s $S shell pidof llama-server)
  TEMP=$(adb -s $S shell 'dumpsys battery | grep temperature' | tr -dc 0-9)
  echo "tag=$TAG batt=$TEMP round=$r elapsed=$((T1-T0))s alive=${ALIVE:+yes} err=[$LOG] $OUT"
  adb -s $S shell 'pkill -9 -x llama-server; sleep 1'
  perl -e 'select(undef,undef,undef,2)'
done
