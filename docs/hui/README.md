# hui_infer 分支说明

本分支是个人 fork 的主分支：所有自己的提交都放在这里，并定期从上游 `ggml-org/llama.cpp` 的 `master` 同步。

同步方式：

```bash
git fetch origin && git switch hui_infer && git merge origin/master
```

- `research/`：手机端推理引擎优化调研（端侧框架、服务引擎、论文）。
- `tools/hui-debug/`：数值一致性工具。
- 端侧测速、验证脚本和测量结果在 `EdgeAgent/validation/`（原 hexagon-lab）。
