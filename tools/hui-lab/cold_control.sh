#!/bin/bash
# hang rate from a cool phone: unfixed vs. fixed queue, 8 min idle before each round
S=3B666H0171500000; D=/private/tmp/claude-501/-Users-will-code-inference/378637e7-7142-4277-99bd-733108a3499b/scratchpad
cd "$(dirname "$0")"
for kind in origq fix origq fix origq fix; do
  adb -s $S shell 'pkill -9 -x llama-server'
  perl -e 'select(undef,undef,undef,480)'
  adb -s $S push $D/htp-v81-$kind.so /data/local/tmp/llama.cpp/lib/libggml-htp-v81.so >/dev/null
  bash hang_server.sh cold-$kind 1 240
done
adb -s $S push $D/htp-v81-fix.so /data/local/tmp/llama.cpp/lib/libggml-htp-v81.so >/dev/null
