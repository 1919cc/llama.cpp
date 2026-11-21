// Kotlin binding of the hui_infer Android engine (tools/hui-jni/hui_engine.cpp in llama.cpp_hui,
// branch hui_infer). Apps copy this file unchanged next to libhui_engine.so (super_ai_com:
// tools/native/sync_hui_engine.sh); change it here, together with the native side.
package com.hui.infer

/**
 * One llama.cpp engine per process: one model loaded, one prompt evaluated at a time. A deployment
 * scheme is a llama.cpp command line ([load]); engine knobs read from the environment
 * (LLAMA_WEIGHT_STREAM, LLAMA_EXPERT_PREFETCH, LLAMA_FFN_SPARSE_BLOCKS, LLAMA_LAYER_PREFETCH,
 * GGML_HEXAGON_OPPOLL) must be set before [load]. The Hexagon backend opens its NPU sessions when
 * libggml-hexagon is loaded, so ADSP_LIBRARY_PATH (where FastRPC finds libggml-htp-v*.so) and the
 * other GGML_HEXAGON_* options must be set before [open].
 *
 * Every call except [stop], [stats], [describe] and [npuBytes] is serialized on the engine.
 */
// One Kotlin function per native entry point: the count follows the engine's JNI surface.
@Suppress("TooManyFunctions")
object HuiEngine {
    /** Receives the reply as whole UTF-8 characters; return false to stop it. */
    fun interface ByteListener {
        fun onBytes(bytes: ByteArray): Boolean
    }

    /** Returned by [generate] / [generateChat] when the engine failed. */
    const val FAILED = -1

    /** Returned by [generate] / [generateChat] when the prompt does not fit the context. */
    const val TOO_LONG = -2

    /** Indexes into [stats]. */
    object Stats {
        const val PREFILL_MS = 0
        const val PREFILL_TOKENS = 1
        const val DECODE_MS = 2
        const val DECODE_TOKENS = 3
        const val FIRST_TOKEN_MS = 4
        const val DRAFTED = 5
        const val ACCEPTED = 6

        /** target forward passes while decoding (tokens / passes = tokens per pass) */
        const val PASSES = 7

        /** prompt tokens served from the context instead of evaluated again */
        const val REUSED = 8
        const val CONTEXT_USED = 9
        const val CONTEXT_SIZE = 10
        const val LOAD_MS = 11

        /** 1 when speculative decoding is on */
        const val SPECULATIVE = 12

        /** of [REUSED], restored from the prompt cache file */
        const val CACHED = 13
        const val CACHE_MS = 14

        /** rpcmem the NPU holds now */
        const val NPU_BYTES = 15
        const val COUNT = 16
    }

    @Volatile var ready = false
        private set

    /** Why [open] failed, for logs. */
    @Volatile var openError: String? = null
        private set

    /**
     * Loads libhui_engine.so (and the llama / ggml libraries it links) once; false when it cannot.
     */
    @Synchronized
    fun open(): Boolean {
        if (ready) return true
        return runCatching {
            System.loadLibrary("hui_engine")
            init()
            ready = true
        }.onFailure { openError = it.message ?: it.javaClass.simpleName }.isSuccess
    }

    @JvmStatic private external fun init()

    /**
     * Loads the model the llama.cpp arguments (without argv[0]) describe; null on success, else why
     * not.
     */
    @JvmStatic external fun load(args: Array<String>): String?

    @JvmStatic external fun unload()

    /** Stops the running prefill or reply at its next batch or token. */
    @JvmStatic external fun stop()

    /** Forgets the conversation held in the context. */
    @JvmStatic external fun resetContext()

    /**
     * Continues a fully rendered [prompt] (the app's own chat format, e.g. ChatML kept byte-exact
     * so the next prompt extends the context). Returns the number of tokens written (0 when
     * stopped during the prefill), [FAILED] or [TOO_LONG].
     */
    @JvmStatic external fun generate(prompt: String, maxTokens: Int, listener: ByteListener): Int

    /**
     * Answers OpenAI-style [messages] (JSON array) rendered with the model's own chat template; see
     * [generate].
     */
    @JvmStatic external fun generateChat(
        messages: String,
        thinking: Boolean,
        maxTokens: Int,
        listener: ByteListener
    ): Int

    /** {"chatml": Boolean, "thinking": Boolean} of the loaded model's chat template. */
    @JvmStatic external fun chatInfo(): String

    /**
     * [messages] and [tools] (OpenAI-style JSON arrays, tools may be "") rendered by the model's
     * chat
     * template, with the assistant turn opened when [addGeneration]; null on error. Keeps the
     * parser
     * for [chatParse].
     */
    @JvmStatic external fun chatRender(
        messages: String,
        tools: String,
        thinking: Boolean,
        addGeneration: Boolean
    ): String?

    /**
     * The reply to the last [chatRender]ed prompt as an OpenAI-style message JSON (content,
     * tool_calls).
     */
    @JvmStatic external fun chatParse(text: String): String

    /**
     * Prompt prefix whose evaluated state is kept in [path] and restored into a cold context ("" =
     * off).
     */
    @JvmStatic external fun setPromptCache(path: String, prefix: String)

    /** The last reply's measurements; see [Stats]. */
    @JvmStatic external fun stats(): DoubleArray

    /** "gemma4 12B Q4_0" and the like. */
    @JvmStatic external fun describe(): String

    /** Tokens of [text] for the loaded model; -1 when none is loaded. */
    @JvmStatic external fun tokenCount(text: String): Int

    /** rpcmem the NPU holds now (0 without the Hexagon backend). */
    @JvmStatic external fun npuBytes(): Long

    /**
     * NPU bytes of each of the first [layers] layers after repack, then the output head, read from
     * every split; without the routed experts when [denseOnly]. Null when a file cannot be read.
     */
    @JvmStatic external fun layerNpuBytes(path: String, layers: Int, denseOnly: Boolean): LongArray?

    /** The model's transformer layer count (block_count); 0 when unreadable. */
    @JvmStatic external fun layerCount(path: String): Int

    /**
     * GGUF metadata of a model (all splits) as JSON: arch, name, size_label, n_layer, nextn,
     * params,
     * bytes, splits, quant, ctx_train; null when unreadable.
     */
    @JvmStatic external fun modelInfo(path: String): String?

    /**
     * llama-bench style run on the loaded model (clears the context): [pp tok/s, tg tok/s, pp ms,
     * tg ms, n_pp, n_tg]; [prefillTokens] = 0 measures decode only. Null when it failed.
     */
    @JvmStatic external fun bench(
        prefillTokens: Int,
        decodeTokens: Int,
        threads: Int,
        threadsBatch: Int
    ): DoubleArray?
}
