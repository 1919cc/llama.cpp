#!/bin/bash
# chat flow: split KV without borrow vs. unified KV with borrow, alternated at the same temperature
S=3B666H0171500000; M=/data/local/tmp/gguf/small/qwen35-2b-Q4_0.gguf; export NO_PROXY='*' no_proxy='*'
cd "$(dirname "$0")"
start() { adb -s $S shell "nohup sh /data/local/tmp/server_start.sh $M 4 2048 $1 >/dev/null 2>&1 &"; perl -e 'select(undef,undef,undef,4)'; for i in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:18080/health | grep -q ok && break; perl -e 'select(undef,undef,undef,2)'; done; }
for rep in 1 2; do for mode in "--no-kv-unified" "--kv-unified"; do
  echo "=== rep=$rep mode=$mode batt=$(adb -s $S shell 'dumpsys battery | grep temperature' | tr -dc 0-9)"
  start "$mode"
  python3 server_load.py verify --prefix-words 800 --clients 4 --reps 2 --n 48
  python3 server_load.py cross --chat --prefix-words 800 --clients 4 --reps 3 --n 32
  for w in 1 2 4; do python3 server_load.py ways --chat --clients $w --prefix-words 800 --n 64 --seconds 40; done
  python3 server_load.py ways --chat --clients 4 --prefix-words 800 --n 64 --seconds 60 --think 5
  echo "borrowed=$(adb -s $S shell 'grep -c borrowed /data/local/tmp/server.log') procs=$(adb -s $S shell 'pidof llama-server | wc -w') err=$(adb -s $S shell 'grep -cE "GGML_ASSERT|no response from DSP" /data/local/tmp/server.log')"
  adb -s $S shell 'pkill -9 -x llama-server'
done; done
