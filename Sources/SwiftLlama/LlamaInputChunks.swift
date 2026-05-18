import Foundation
import llama

/// RAII wrapper around `mtmd_input_chunks`. The container owns its
/// `mtmd_input_chunk`s, so freeing the container frees the chunks too.
///
/// You don't construct chunks yourself; `LlamaMtmdContext.tokenize(...)`
/// populates an empty container by calling `mtmd_tokenize`.
public final class LlamaInputChunks {
    let pointer: OpaquePointer

    public init() {
        // `mtmd_input_chunks_init` always returns a valid pointer in the
        // current implementation (it never returns nullptr), so the
        // force-unwrap is safe; if that ever changes upstream this will
        // crash deterministically rather than silently misbehave.
        self.pointer = mtmd_input_chunks_init()!
    }

    deinit {
        mtmd_input_chunks_free(pointer)
    }

    public var count: Int { mtmd_input_chunks_size(pointer) }

    /// Total number of tokens across all chunks (image-chunk tokens count
    /// as the encoder's output token count). Useful for KV-cache budgeting.
    public var totalTokens: Int { mtmd_helper_get_n_tokens(pointer) }
}
