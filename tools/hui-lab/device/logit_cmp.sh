#!/system/bin/sh
cd /data/local/tmp/llama.cpp; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0 GGML_HEXAGON_POWER_MODE=high_performance GGML_HEXAGON_OPPOLL=1
M=/data/local/tmp/gguf/small/gemma-4-E2B-it-qat-Q4_0.gguf; O=/data/local/tmp/spec; T=/data/local/tmp/logit_cmp
input keyevent KEYCODE_WAKEUP
echo "== npu default"; $T $M $O/mixed.txt HTP0 200 64 2>/dev/null
echo "== npu no-glu-fusion"; GGML_HEXAGON_OPFUSION=7934 $T $M $O/mixed.txt HTP0 200 64 2>/dev/null
input keyevent KEYCODE_WAKEUP
echo "== npu no fusion"; GGML_HEXAGON_OPFUSION=0 $T $M $O/mixed.txt HTP0 200 64 2>/dev/null
echo "== npu hvx matmul"; GGML_HEXAGON_MM_SELECT=1 $T $M $O/mixed.txt HTP0 200 64 2>/dev/null
input keyevent KEYCODE_WAKEUP
echo "== cpu"; $T $M $O/mixed.txt none 200 64 2>/dev/null
