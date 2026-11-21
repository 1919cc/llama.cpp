# hui-debug：数值一致性工具

hui_infer 分支自带的调试工具，用来验证 Hexagon 优化没有改变计算结果。原来放在 hexagon-lab/tools，测试脚本和测量结果在 `EdgeAgent/validation/`。

| 目标 | 作用 |
|---|---|
| `llama-hui-logit-cmp` | 同一串 token 逐个解码与按 N 个一批解码，比较 logits（rms、最大差、top1 翻转）。`<model> <prompt 文件> <设备|none> <n_prompt> <n_test>` 采集；`cmp <a> <b> <tag>` 比较两份结果 |
| `llama-hui-kvdiff` | CPU 上比较 KV 缓存的不同用法（FA 开关、分批、unified 与否、前缀拷贝）对 logits 的影响，用来判断 server 前缀借用是否无损。`<model> [prompt 长度=200]` |
| `llama-hui-quantize-f32` | 把模型转成全 F32，作为上面两个工具的精确基准。`<in.gguf> <out.gguf>` |

构建（Android 包里会一起产出，再 `adb push` 到 `/data/local/tmp/` 运行）：

```bash
cmake --build build --target llama-hui-logit-cmp llama-hui-kvdiff llama-hui-quantize-f32
```
