#!/system/bin/sh
# mem_guard.sh <log> <cmd...>: run cmd, sample memory every 100 ms, kill it before the phone runs out.
LOG=$1; shift
LIMIT_KB=${LIMIT_KB:-1536000}
"$@" > $LOG.out 2>&1 &
PID=$!
echo "start pid=$PID limit=${LIMIT_KB}kB cmd=$*" > $LOG
while kill -0 $PID 2>/dev/null; do
  m=$(awk '/^MemAvailable|^MemFree|^AnonPages|^Mapped|^Cached|^IonTotalUsed|^SwapFree/{printf "%s=%d ", $1, $2/1024}' /proc/meminfo)
  r=$(awk '/^VmRSS|^RssAnon|^RssFile|^RssShmem|^VmSwap/{printf "%s=%d ", $1, $2/1024}' /proc/$PID/status 2>/dev/null)
  echo "$(date +%H:%M:%S.%N | cut -c1-12) $m| $r" >> $LOG
  avail=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt "$LIMIT_KB" ]; then
    kill -9 $PID; echo "KILLED: MemAvailable ${avail}kB < ${LIMIT_KB}kB" >> $LOG; sync; break
  fi
  sync
  sleep 0.1
done
wait $PID; echo "exit=$?" >> $LOG; sync
