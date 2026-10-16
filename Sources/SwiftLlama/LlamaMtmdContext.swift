import Foundation
import llama

public enum LlamaMtmdError: Error {
    case initializationFailed
    case bitmapDecodeFailed
    case markerCountMismatch
    case imagePreprocessingFailed
    case evaluationFailed(Int32)
}

/// Swift wrapper around `mtmd_context`. An mtmd context binds an
/// already-loaded `LlamaModel` to a vision/audio projector (mmproj) GGUF
/// file. It is required to tokenize and evaluate multimodal inputs.
///
/// Not thread-safe with respect to evaluation — `evalChunks` calls into
/// `llama_decode` under the hood and must be serialized with all other
/// access to the same `LlamaContext`. The `Llama` actor takes care of
/// that serialization for callers in this package.
public final class LlamaMtmdContext {
    let pointer: OpaquePointer

    /// Initialize an mtmd context from an mmproj GGUF file and an
    /// already-loaded text model. Returns `nil` if the underlying
    /// `mtmd_init_from_file` call fails (corrupt file, model mismatch,
    /// out of memory, etc.).
    public init?(mmprojPath: String, model: LlamaModel, useGPU: Bool = true, nThreads: Int32 = 1) {
        var params = mtmd_context_params_default()
        params.use_gpu = useGPU
        params.n_threads = nThreads
        params.print_timings = false
        params.warmup = false

        // Force CPU on simulator: the Metal backend's MTLGPUFamilyApple2
        // path lacks the simdgroup operations the vision encoder needs and
        // silently produces zero embeddings, which then crashes decode.
        #if targetEnvironment(simulator)
        params.use_gpu = false
        #endif

        let p = mmprojPath.withCString { cPath in
            mtmd_init_from_file(cPath, model.modelPointer, params)
        }
        guard let p else { return nil }
        self.pointer = p
    }

    deinit {
        mtmd_free(pointer)
    }

    /// Default text marker that mtmd replaces with image/audio token
    /// chunks during `tokenize`. Currently `<__media__>`; callers should
    /// use this constant rather than hard-coding it so the prompt format
    /// tracks upstream changes.
    public static var defaultMarker: String {
        guard let c = mtmd_default_marker() else { return "<__media__>" }
        return String(cString: c)
    }

    public var supportsVision: Bool { mtmd_support_vision(pointer) }
    public var supportsAudio: Bool { mtmd_support_audio(pointer) }

    /// Tokenize a prompt containing one `defaultMarker` per supplied
    /// bitmap. The returned chunks alternate between text-token chunks
    /// and image/audio chunks; pass them to `evalChunks(...)` to feed
    /// the model.
    public func tokenize(text: String, addSpecial: Bool, parseSpecial: Bool, bitmaps: [LlamaBitmap]) throws -> LlamaInputChunks {
        let chunks = LlamaInputChunks()

        // mtmd_tokenize takes the bitmap pointer array as
        // `UnsafePointer<OpaquePointer?>?` (the C signature is
        // `const mtmd_bitmap * const *`; const-on-pointee doesn't survive
        // the forward-struct import).
        //
        // The outer `withExtendedLifetime(bitmaps)` is load-bearing: it
        // pins the `[LlamaBitmap]` array across the C call so ARC can't
        // optimistically release the wrapper objects (and free their
        // underlying mtmd_bitmap pointers) before tokenize is done
        // reading their pixel data. A trailing `withExtendedLifetime`
        // after the C call would be a no-op — the C call has already
        // returned and the lifetime extension would not apply.
        let bitmapPointers: [OpaquePointer?] = bitmaps.map { $0.pointer }

        let rc: Int32 = withExtendedLifetime(bitmaps) {
            text.withCString { cText -> Int32 in
                // `text_len` is the UTF-8 byte count: since llama.cpp v0.7.0
                // mtmd reads exactly that many bytes (0 would be an empty
                // prompt), so it must match what withCString hands over.
                var input = mtmd_input_text(
                    text: cText,
                    text_len: text.utf8.count,
                    add_special: addSpecial,
                    parse_special: parseSpecial
                )
                return bitmapPointers.withUnsafeBufferPointer { buf in
                    mtmd_tokenize(pointer, chunks.pointer, &input, buf.baseAddress, buf.count)
                }
            }
        }

        switch rc {
        case 0: return chunks
        case 1: throw LlamaMtmdError.markerCountMismatch
        case 2: throw LlamaMtmdError.imagePreprocessingFailed
        default: throw LlamaMtmdError.evaluationFailed(rc)
        }
    }

    /// Evaluate a list of tokenized chunks into the given llama context,
    /// starting at KV position `nPast`. Returns the new position after
    /// the evaluation (caller should use this as the next decode position).
    ///
    /// Wraps `mtmd_helper_eval_chunks`, which internally:
    /// 1. Runs `llama_decode` on text-token chunks.
    /// 2. Runs `mtmd_encode` on image chunks, then feeds the embeddings
    ///    into `llama_decode`.
    ///
    /// NOT thread-safe; the caller must hold the same actor as the
    /// llama context.
    public func evalChunks(_ chunks: LlamaInputChunks, into context: LlamaContext, nPast: Int32, seqId: Int32 = 0, nBatch: Int32, logitsLast: Bool) throws -> Int32 {
        var newNPast: llama_pos = 0
        let rc = mtmd_helper_eval_chunks(
            pointer,
            context.contextPointer,
            chunks.pointer,
            nPast,
            seqId,
            nBatch,
            logitsLast,
            &newNPast
        )
        if rc != 0 {
            throw LlamaMtmdError.evaluationFailed(rc)
        }
        return newNPast
    }
}
