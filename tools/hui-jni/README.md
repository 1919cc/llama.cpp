# hui-jni：App 内嵌的 Android 推理引擎

`libhui_engine.so` + Kotlin 绑定 `com.hui.infer.HuiEngine`。App 只负责拼 llama.cpp 命令行（部署方案）、设置环境变量、渲染对话；推理本身全在这里。

原先有两份 App 侧的引擎拷贝，现在合并到这里，后续只在本分支维护：

| 来源 | 带进来的能力 |
|---|---|
| EdgeAgent `app/src/main/cpp/engine.cpp` | 系统提示缓存（状态存闪存，冷启动恢复）、用模型自带 chat 模板渲染/解析工具调用、llama-bench 式测速、GGUF 元数据 |
| super_ai_com `android/native/edge_jni.cpp` | 上下文复用时裁掉不匹配的后缀（`trim`）、prefill 中途可停、每次加载按环境变量设置 NPU 轮询、丢弃 DEBUG 日志（12B 解码 0.8 → 4.8 tok/s） |

## 接口

见 [java/com/hui/infer/HuiEngine.kt](java/com/hui/infer/HuiEngine.kt)。要点：

- `open()` 之前设置 `ADSP_LIBRARY_PATH`（FastRPC 找 `libggml-htp-v*.so` 的目录）和 `GGML_HEXAGON_*`：Hexagon 后端在库加载时打开 NPU 会话。
- `load(args)` 之前设置每个模型的引擎开关：`LLAMA_WEIGHT_STREAM`、`LLAMA_EXPERT_PREFETCH`、`LLAMA_FFN_SPARSE_BLOCKS`、`LLAMA_LAYER_PREFETCH`、`GGML_HEXAGON_OPPOLL`。
- `generate(prompt)`：App 自己渲染的完整提示（如 ChatML，逐字节一致以复用上下文）。
- `generateChat(messages)`：OpenAI 风格 messages，用模型自带模板渲染，用户输入里的特殊 token 不会被解析。
- `chatRender` / `chatParse`：带 tools 的模板渲染与工具调用解析。
- `stats()` 下标见 `HuiEngine.Stats`。
- `embedLoad(path, threads, pooling)` / `embed(texts)`：独立的句向量槽（如 bge-small-zh，CPU，自带锁），与对话模型同时加载、互不等待；返回 L2 归一化向量，余弦相似度即点积。

## 构建

随 Android 包一起产出（只在 `ANDROID` 下构建），安装到 `pkg-adb/llama.cpp/lib/`：

```bash
python3 scripts/snapdragon/build.py --target adb
```

App 同步引擎（super_ai_com）：

```bash
../super_ai_com/tools/native/sync_hui_engine.sh
```
