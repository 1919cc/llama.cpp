#!/system/bin/sh
# usage: run_case.sh <pkg dir> <out prefix> <tool> [VAR=value ...] -- <tool args>
# Waits for the CPU to cool, runs the tool with a memory watchdog and samples memory, temperature and frequency.
PKG=$1; OUT=$2; TOOL=$3; shift 3
while [ "$1" != "--" ]; do export "$1"; shift; done; shift
export LD_LIBRARY_PATH=$PKG/lib ADSP_LIBRARY_PATH=$PKG/lib CDSP_LIBRARY_PATH=$PKG/lib GGML_HEXAGON_OPPOLL=1

maxt() {
    m=0
    for z in /sys/class/thermal/thermal_zone*; do
        case "$(cat $z/type 2>/dev/null)" in
            cpu*trip*) ;;
            cpu*) t=$(cat $z/temp 2>/dev/null); [ "${t:-0}" -gt "$m" ] && m=$t ;;
        esac
    done
    echo $m
}

COOL=${COOL_MC:-50000}
w=0
while [ "$(maxt)" -gt "$COOL" ] && [ $w -lt 300 ]; do sleep 10; w=$((w+10)); done

# WAKE_THERMAL=1: keep the screen awake and wait for thermal status 0, the phone caps the CPU otherwise
tstat() { dumpsys thermalservice | grep -m1 "Thermal Status" | tr -dc 0-9; }
if [ -n "$WAKE_THERMAL" ]; then
    input keyevent KEYCODE_WAKEUP
    while [ "$(tstat)" != "0" ] && [ $w -lt 900 ]; do sleep 15; w=$((w+15)); input keyevent KEYCODE_WAKEUP; done
fi
TS0=$(tstat)
WAKE0=$(dumpsys power | grep -m1 mWakefulness= | cut -d= -f2)
CAP0=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)
CAP6=$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq)

cd $PKG
START_T=$(maxt)
S0=$(date +%s)
./bin/$TOOL "$@" > "$OUT.out" 2> "$OUT.err" &
pid=$!
minav=99999999; hwm=0; peakt=0; killed=0
capmin=99999999
: > "$OUT.samples"
while kill -0 $pid 2>/dev/null; do
    av=$(grep MemAvailable /proc/meminfo | tr -s ' ' | cut -d' ' -f2)
    [ "$av" -lt "$minav" ] && minav=$av
    h=$(grep VmHWM /proc/$pid/status 2>/dev/null | tr -s ' \t' ' ' | cut -d' ' -f2)
    [ "${h:-0}" -gt "$hwm" ] && hwm=$h
    t=$(maxt); [ "$t" -gt "$peakt" ] && peakt=$t
    cap=$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq 2>/dev/null)
    [ "${cap:-99999999}" -lt "$capmin" ] && capmin=$cap
    echo "$(($(date +%s)-S0)) $av ${h:-0} $t ${cap:-0}" >> "$OUT.samples"
    if [ "$av" -lt 1228800 ]; then kill -9 $pid; killed=1; break; fi
    sleep 1
done
wait $pid
rc=$?
echo "{\"rc\":$rc,\"wall_s\":$(($(date +%s)-S0)),\"min_avail_kb\":$minav,\"vm_hwm_kb\":$hwm,\"start_mc\":$START_T,\"peak_mc\":$peakt,\"min_big_cap_khz\":$capmin,\"cool_wait_s\":$w,\"watchdog_killed\":$killed,\"thermal_status\":\"$TS0\",\"wakefulness\":\"$WAKE0\",\"cap0_khz\":$CAP0,\"cap6_khz\":$CAP6}" > "$OUT.meta"
