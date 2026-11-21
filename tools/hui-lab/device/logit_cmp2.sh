#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1 LC_POS=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf; O=/data/local/tmp/spec; T=/data/local/tmp/logit_cmp; L=/data/local/tmp/spec/lg
input keyevent KEYCODE_WAKEUP
echo "== npu default"; $T $M $O/mixed.txt HTP0 200 64 $L.npu 2>/dev/null
echo "== npu nofusion"; GGML_HEXAGON_OPFUSION=0 $T $M $O/mixed.txt HTP0 200 64 $L.nf 2>/dev/null | grep -v 'per-pos'
input keyevent KEYCODE_WAKEUP
echo "== npu FA hvx"; GGML_HEXAGON_FA_SELECT=1 $T $M $O/mixed.txt HTP0 200 64 2>/dev/null | grep -v 'per-pos'
echo "== cpu"; $T $M $O/mixed.txt none 200 64 $L.cpu 2>/dev/null
unset LC_POS
echo "== against cpu b1 (closest to reference)"
for x in cpu.b8 npu.b1 npu.b8 nf.b1 nf.b8; do $T cmp $L.cpu.b1 $L.$x $x; done
rm -f $L.*
