import Foundation
import llama

enum NextToken {
    case token(String)
    case endOfString
}

final actor Llama {
    private let model: LlamaModel
    let context: LlamaContext
    private var batch: LlamaBatch
    private var sampler: LlamaSampler!
    private var mtmd: LlamaMtmdContext?

    // Configuration

    private let config: LlamaConfig
    let maxTokenCount: UInt32
    /// Tracks the current position in the token sequence during decoding.
    var currentTokenPosition: Int32 = 0
    var processedTokens: [llama_token] = []
    /// On-disk prefix snapshots (nil when the config names no directory).
    private let promptCache: LlamaPromptCache?
    /// Sliding-window size of the model's SWA layers (0 when it has none).
    private let swaWindow: Int
    /// The warm-up decode has run (or a real decode has paged the weights in).
    private var weightsWarmed = false
    /// Bytes of a character split across tokens, waiting for the rest
    /// (see `LlamaUTF8`).
    private var pendingPieceBytes: [UInt8] = []

    init(modelPath: String, config: LlamaConfig) throws {
        self.config = config
        llama_backend_init()
        var model_params = llama_model_default_params()

        if !config.useGPU {
            model_params.n_gpu_layers = 0
        }

        #if targetEnvironment(simulator)
                model_params.n_gpu_layers = 0
                print("Running on simulator, force use n_gpu_layers = 0")
        #endif

        let forceCPUOnlyPath: Bool = {
            #if targetEnvironment(simulator)
            return true
            #else
            return false
            #endif
        }()

        // Memory-map the weights (llama.cpp's AUTO resolves to this on Apple
        // GPUs too, but say so): mapped pages stay clean and evictable, which
        // the app's iOS memory budget depends on.
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP

        let model = LlamaModel(path: modelPath, parameters: model_params)
        guard let model else {
            print("Could not load model at \(modelPath)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = ProcessInfo.processInfo.processorCount - 1
        print("Using \(n_threads) threads")

        var contextParam = llama_context_default_params()
        contextParam.n_ctx = config.maxTokenCount
        contextParam.n_threads       = 1 // UInt32(n_threads) its actually faster if less threads are doing work
        contextParam.n_threads_batch = 1 // UInt32(n_threads)
        contextParam.n_batch = config.batchSize
        contextParam.n_ubatch = config.batchSize
        contextParam.offload_kqv = true

        // On the iOS Simulator the Metal backend reports MTLGPUFamilyApple2,
        // which lacks simdgroup reduction / matmul. With Flash Attention on
        // AUTO, llama.cpp auto-enables it and the forward pass silently
        // produces zero logits (n_outputs = 0) — then the sampler aborts.
        // Force Flash Attention off and skip KV offload on simulator.
        if forceCPUOnlyPath {
            contextParam.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED
            contextParam.offload_kqv = false
            print("Running on simulator, force disable Flash Attention + KV offload")
        }

        let context = LlamaContext(model: model, parameters: contextParam)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }


        self.maxTokenCount = min(UInt32(model.trainedContextSize()), config.maxTokenCount)
        self.model = context.model
        self.context = context
        self.batch = .init(initialSize: Int32(config.batchSize))
        self.swaWindow = Int(max(0, llama_model_n_swa(model.modelPointer)))

        if let directory = config.promptCacheDirectory {
            // Everything that changes the KV layout or contents is part of
            // the identity, so a snapshot only ever restores into the model
            // and context it was made with.
            let attributes = try? FileManager.default.attributesOfItem(atPath: modelPath)
            let fileSize = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
            let identity = [
                URL(fileURLWithPath: modelPath).lastPathComponent,
                String(fileSize),
                "ctx=\(contextParam.n_ctx)",
                "batch=\(config.batchSize)",
                "gpu=\(config.useGPU && !forceCPUOnlyPath)"
            ].joined(separator: "|")
            // The file names also carry the snapshot format + the engine's
            // state-file version; files from any other version are deleted
            // here, so an older llama.cpp's snapshots are never offered.
            self.promptCache = LlamaPromptCache(directory: directory, maxFiles: config.promptCacheMaxFiles, identity: identity)
        } else {
            self.promptCache = nil
        }
    }

    deinit {
        llama_backend_free()
    }

    // Expose some backend/system utilities for convenience
    /// Return system info string from the backend.
    static func printSystemInfo() -> String {
        guard let c = llama_print_system_info() else { return "" }
        return String(cString: c)
    }

    /// Expose the underlying context to trusted callers (tests / advanced users).
    /// Access is actor-isolated; callers must `await`.
    func contextHandle() -> LlamaContext { context }

    // MARK: - Testing & Introspection helpers (actor-safe)

    func getLastLogits() -> [Float]? { context.lastLogits() }
    func getEmbeddings() -> [Float]? { context.embeddings(at: -1) }
    func enableEmbeddingsOutput(_ enabled: Bool) { context.setEmbeddingsOutput(enabled) }
    func saveStateData() -> Data { context.saveState() }
    func loadStateData(_ data: Data) -> Bool { context.loadState(data) }
    func setThreads(nThreads: Int32, nThreadsBatch: Int32) { context.setThreads(nThreads: nThreads, nThreadsBatch: nThreadsBatch) }
    func getThreads() -> (Int32, Int32) { (context.nThreads(), context.nThreadsBatch()) }
    func kvMinPosition() -> Int32 { context.memory.minPosition(for: 0) }
    func kvMaxPosition() -> Int32 { context.memory.maxPosition(for: 0) }
    func clearKV() { context.clearKVCache() }

    /// Return the full processed token id sequence (prompt + generated).
    func getProcessedTokenIds() -> [llama_token] { processedTokens }

    /// Load the multimodal projector for this model. Idempotent — if
    /// already loaded, returns immediately without reloading. Throws
    /// `LlamaMtmdError.initializationFailed` if the projector file is
    /// invalid for the loaded model.
    func loadMmproj(at url: URL) throws {
        if mtmd != nil { return }
        let path = url.path(percentEncoded: false)
        guard let ctx = LlamaMtmdContext(mmprojPath: path, model: model, useGPU: config.useGPU) else {
            throw LlamaMtmdError.initializationFailed
        }
        self.mtmd = ctx
    }

    var hasMtmd: Bool { mtmd != nil }

    /// Initialize a multimodal completion. The prompt should contain one
    /// `LlamaMtmdContext.defaultMarker` ("<__media__>") per image.
    /// Images are passed as encoded bytes (jpg/png/etc).
    ///
    /// After this returns, `currentTokenPosition` and the KV cache are
    /// positioned for sampling the next token. The token-prefix cache
    /// optimization used by `initializeCompletion(text:)` is bypassed —
    /// multimodal turns always re-process from position 0.
    func initializeMultimodalCompletion(prompt: String, images: [Data]) throws {
        guard let mtmd else { throw LlamaMtmdError.initializationFailed }
        pendingPieceBytes = []

        var bitmaps: [LlamaBitmap] = []
        bitmaps.reserveCapacity(images.count)
        for data in images {
            guard let bitmap = LlamaBitmap.decode(data: data, in: mtmd) else {
                throw LlamaMtmdError.bitmapDecodeFailed
            }
            bitmaps.append(bitmap)
        }

        let chunks = try mtmd.tokenize(
            text: prompt,
            addSpecial: model.shouldAddBos(),
            parseSpecial: true,
            bitmaps: bitmaps
        )

        // The KV cache may hold a prior turn; mtmd has no equivalent of
        // the text-only prefix-match optimization (image chunks aren't
        // comparable token-for-token across turns), so start fresh.
        clear()

        let newNPast = try mtmd.evalChunks(
            chunks,
            into: context,
            nPast: 0,
            seqId: 0,
            nBatch: Int32(config.batchSize),
            logitsLast: true
        )

        currentTokenPosition = newNPast
        // We don't track per-token ids for the multimodal path because
        // image chunks don't have llama-vocab tokens we could store. This
        // means a subsequent text-only `initializeCompletion(text:)` call
        // will fall through the prefix-match check and full-reprocess —
        // which is correct (it has no way to know what the image stood
        // for) and matches the conservative "clear on switch" policy.
        processedTokens = []
    }

    func initializeCompletion(messages: [LlamaChatMessage], addAssistant: Bool? = nil) throws {
        let formattedPrompt = model.applyChatTemplate(to: messages, addAssistant: addAssistant)
        try initializeCompletion(text: formattedPrompt)
    }

    /// Initialize a completion from an already-formatted prompt string, bypassing
    /// the built-in chat template. Useful for models whose GGUF template uses
    /// Jinja features (macros, etc.) that `llama_chat_apply_template` does not
    /// support — in that case the template helper silently returns "" and the
    /// caller should format the prompt itself.
    func initializeCompletion(prompt: String) throws {
        try initializeCompletion(text: prompt)
    }

    /// Like `initializeCompletion(prompt:)`, with `cacheablePrefix` naming the
    /// leading part of `prompt` that stays the same across turns (system
    /// prompt + tool declarations). When the KV cache can't simply be extended
    /// or trimmed, that prefix is restored from its on-disk snapshot (or
    /// processed once and saved) instead of being re-processed.
    func initializeCompletion(prompt: String, cacheablePrefix: String?) throws {
        try initializeCompletion(text: prompt, cacheablePrefix: cacheablePrefix)
    }

    /// Page the model in before the first real prompt: a one-token decode
    /// touches every layer's weights and builds the GPU pipelines, then
    /// the newest prefix snapshot (if any) is restored, so the next prompt
    /// that starts with it only processes what follows. Does nothing while
    /// the KV cache is in use. (A plain decode: llama.cpp's warm-up mode is
    /// deprecated upstream and only ever differed for MoE models.)
    func warmUp() throws {
        guard processedTokens.isEmpty else {
            weightsWarmed = true
            return
        }
        if !weightsWarmed {
            let start = CFAbsoluteTimeGetCurrent()
            batch.reset()
            batch.addToken(model.bosToken(), at: 0, logits: true)
            defer { clear() }
            do {
                try context.decode(batch: batch)
            } catch {
                throw LlamaError.decodingError
            }
            weightsWarmed = true
            print("### Warm-up decode: \(Int((CFAbsoluteTimeGetCurrent() - start) * 1000))ms")
        }
        if processedTokens.isEmpty, let url = promptCache?.newestSnapshotURL() {
            _ = restoreSnapshot(at: url, expecting: nil)
        }
    }

    private func initializeCompletion(text: String, cacheablePrefix: String? = nil) throws {
        pendingPieceBytes = []
        print("attempting to complete \"\(text)\"")

        let tokenList = model.tokenize(text: text, addBos: model.shouldAddBos(), special: true)
        guard tokenList.count < maxTokenCount - 4 else {
            throw LlamaError.contextSizeLimitExeeded
        }

        // The prefix only counts when the prompt really starts with it
        // token-for-token and something follows it.
        var prefixTokens: [llama_token]?
        if promptCache != nil, let cacheablePrefix, !cacheablePrefix.isEmpty {
            let candidate = model.tokenize(text: cacheablePrefix, addBos: model.shouldAddBos(), special: true)
            if candidate.count >= LlamaKVReuse.minimumSnapshotTokens,
               candidate.count < tokenList.count,
               tokenList.starts(with: candidate) {
                prefixTokens = candidate
            }
        }

        let plan = LlamaKVReuse.plan(
            prompt: tokenList,
            processed: processedTokens,
            prefixCount: prefixTokens?.count,
            swaWindow: swaWindow,
            minResidentPosition: Int(context.memory.minPosition(for: 0))
        )

        switch plan {
        case .extend:
            print("### Using cached processing")
            try processPrompt(tokens: Array(tokenList[processedTokens.count...]), startIndex: processedTokens.count)
        case .trim(let position):
            print("### Trimming cache back to position \(position)")
            if context.memory.remove(sequenceId: 0, from: Int32(position), to: -1) {
                processedTokens = Array(processedTokens[..<position])
                currentTokenPosition = Int32(position)
                try processPrompt(tokens: Array(tokenList[position...]), startIndex: position)
            } else {
                print("### Trim refused by the cache, full reprocessing")
                clear()
                try processPrompt(tokens: tokenList, startIndex: 0)
            }
        case .restorePrefix:
            if let prefixTokens {
                try loadOrBuildPrefix(prefixTokens)
                try processPrompt(tokens: Array(tokenList[prefixTokens.count...]), startIndex: prefixTokens.count)
            } else {
                clear()
                try processPrompt(tokens: tokenList, startIndex: 0)
            }
        case .full:
            print("### Full reprocessing required")
            clear()
            try processPrompt(tokens: tokenList, startIndex: 0)
        }
        weightsWarmed = true
    }

    /// Make the KV cache hold exactly `prefix`: restore its snapshot, or
    /// process it and save one for next time.
    private func loadOrBuildPrefix(_ prefix: [llama_token]) throws {
        guard let promptCache else {
            clear()
            try processPrompt(tokens: prefix, startIndex: 0)
            return
        }
        let url = promptCache.url(for: prefix)
        if restoreSnapshot(at: url, expecting: prefix) { return }

        clear()
        let start = CFAbsoluteTimeGetCurrent()
        try processPrompt(tokens: prefix, startIndex: 0)
        let processedMs = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)
        let saveStart = CFAbsoluteTimeGetCurrent()
        let bytes = promptCache.save(tokens: prefix, from: context, to: url)
        print("### Prefix processed: \(prefix.count) tokens in \(processedMs)ms; snapshot \(bytes / 1_000_000)MB saved in \(Int((CFAbsoluteTimeGetCurrent() - saveStart) * 1000))ms")
    }

    /// Restore a prefix snapshot into sequence 0. On any failure (missing,
    /// unreadable, or not the expected tokens) the cache is left empty, the
    /// bad file is removed, and false is returned.
    private func restoreSnapshot(at url: URL, expecting expected: [llama_token]?) -> Bool {
        guard let promptCache, FileManager.default.fileExists(atPath: url.path) else { return false }
        let start = CFAbsoluteTimeGetCurrent()
        clear()
        guard let loaded = context.loadSequenceState(from: url.path, destSeqId: 0, capacity: Int(maxTokenCount)),
              loaded.count > 0,
              loaded.count < Int(maxTokenCount) - 4,
              expected == nil || loaded.tokens == expected! else {
            print("### Prefix snapshot unusable, removing it")
            clear()
            promptCache.remove(url)
            return false
        }
        processedTokens = loaded.tokens
        currentTokenPosition = Int32(loaded.count)
        promptCache.touch(url)
        print("### Prefix restored: \(loaded.count) tokens in \(Int((CFAbsoluteTimeGetCurrent() - start) * 1000))ms")
        return true
    }

    func generateNextToken() throws -> NextToken {
        // Stop before sampling if we've reached the context limit to avoid mutating sampler state
        if currentTokenPosition >= Int32(maxTokenCount) {
            return .endOfString
        }
        let newTokenId = sampler.sample(context: context)

        if model.isEogToken(newTokenId) || currentTokenPosition >= Int32(maxTokenCount) {
            return .endOfString
        }

        batch.reset()
        batch.addToken(newTokenId, at: currentTokenPosition, logits: true)
        processedTokens.append(newTokenId)

        currentTokenPosition += 1
        do {
            try context.decode(batch: batch)
        } catch {
            // A failed decode can leave the cache half-written: start the
            // next prompt clean, and throw the error callers map (the
            // context's own error type is internal detail).
            clear()
            throw LlamaError.decodingError
        }

        // Emit only whole characters; carry a split one to the next token.
        let split = LlamaUTF8.splitIncompleteTail(pendingPieceBytes + model.pieceBytes(from: newTokenId))
        pendingPieceBytes = split.tail
        return .token(String(decoding: split.complete, as: UTF8.self))
    }

    func updateSamplingConfig(_ config: LlamaSamplingConfig) {
        self.sampler = .init(config: config, model: model)
    }

    private func clear() {
        context.clearKVCache()
        processedTokens = []
        currentTokenPosition = 0
        pendingPieceBytes = []
        batch = .init(initialSize: Int32(config.batchSize))
    }

    private func processBatch() throws {
        do {
            try context.decode(batch: batch)
        } catch {
            print("llama_decode() failed")
            throw LlamaError.decodingError
        }
    }

    private func processPrompt(tokens: [llama_token], startIndex: Int) throws {
        guard !tokens.isEmpty else { return }
        batch.reset()

        do {
            // Decode each full batch, and the final partial one with logits
            // on its last token. (The old loop decoded a full final batch,
            // reset it, then flagged logits on the EMPTY batch: index -1, and
            // an empty decode, whenever the prompt was an exact batch multiple.)
            for i in 0..<tokens.count {
                let isLast = i == tokens.count - 1
                batch.addToken(tokens[i], at: Int32(startIndex + i), logits: isLast)
                processedTokens.append(tokens[i])
                if isLast || batch.size == config.batchSize {
                    try processBatch()
                    batch.reset()
                }
            }
        } catch {
            // A failed decode leaves the cache partly written: start clean.
            clear()
            throw error
        }

        currentTokenPosition = Int32(processedTokens.count)
    }
}
