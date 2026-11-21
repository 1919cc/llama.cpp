#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf; O=/data/local/tmp/spec; T=/data/local/tmp/logit_cmp
f() { grep -E 'batch_(3|4|5|8) ' | cut -d' ' -f1,5-; }
# stays below position 256
for c in "X=0" "GGML_HEXAGON_MM_SELECT=1" "GGML_HEXAGON_FA_SELECT=1" "GGML_HEXAGON_OPFUSION=0" "GGML_HEXAGON_GDN_SELECT=1"; do
  input keyevent KEYCODE_WAKEUP; echo "== npu under256 $c"; (export $c; $T $M $O/mixed.txt HTP0 200 50 2>/dev/null | f)
done
input keyevent KEYCODE_WAKEUP; echo "== npu under256 MM=1 FA=1 FUS=0"; (export GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_OPFUSION=0; $T $M $O/mixed.txt HTP0 200 50 2>/dev/null | f)
# crosses position 256
echo "== cpu cross256"; LC_POS= ; $T $M $O/mixed.txt none 200 64 2>/dev/null | f
echo "== cpu cross256 no graph reuse"; LLAMA_GRAPH_REUSE_DISABLE=1 $T $M $O/mixed.txt none 200 64 2>/dev/null | f
