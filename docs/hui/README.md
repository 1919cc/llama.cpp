# hui_infer 分支说明

本分支是个人 fork 的主分支，也是唯一的推理引擎代码库：所有推理引擎的优化都在这里做，并定期从上游 `ggml-org/llama.cpp` 的 `master` 同步。App（`super_ai_com`）只集成本分支产出的引擎做模型部署，不再保留引擎源码拷贝。

同步方式：

```bash
git fetch origin && git switch hui_infer && git merge origin/master
```

- `research/`：手机端推理引擎优化调研（端侧框架、服务引擎、论文）。
- `reports/`：端侧推理技术方案、llama.cpp 改动架构与优化、测速数据（`metrics.json`、`small_bench.json`）和生成这些文档的脚本（`reports/build/`，`npm install` 后运行）。
- `tools/hui-jni/`：App 内嵌的 Android 引擎（`libhui_engine.so` + Kotlin 绑定 `com.hui.infer.HuiEngine`）。
- `tools/hui-debug/`：数值一致性工具。
- `tools/hui-lab/`：端侧测速、验证脚本和测量结果（原 hexagon-lab / EdgeAgent/validation）。
