import Foundation
import llama

/// RAII wrapper around `mtmd_bitmap`. Holds an encoded-image byte buffer
/// (jpg/png/bmp/gif — anything stb_image handles; webp needs mtmd's
/// ffmpeg video path, which the Apple builds leave out) that mtmd will
/// decode internally when the bitmap is constructed.
///
/// The underlying mtmd_bitmap is heap-allocated in C and must be freed by
/// calling `mtmd_bitmap_free`. We do that in `deinit`.
public final class LlamaBitmap {
    let pointer: OpaquePointer

    /// Construct a bitmap from raw RGB pixels (24-bit, row-major,
    /// `width * height * 3` bytes). Use this if you already have decoded
    /// pixel data — for encoded image bytes prefer `init(data:in:)`.
    public init?(width: UInt32, height: UInt32, rgbData: [UInt8]) {
        let expected = Int(width) * Int(height) * 3
        guard rgbData.count == expected else { return nil }
        guard let p = rgbData.withUnsafeBufferPointer({ buf in
            mtmd_bitmap_init(width, height, buf.baseAddress)
        }) else { return nil }
        self.pointer = p
    }

    private init(pointer: OpaquePointer) {
        self.pointer = pointer
    }

    /// Construct a bitmap from encoded image bytes (jpg/png/etc).
    /// Requires an active `LlamaMtmdContext` because mtmd's helper needs
    /// to know which model is loaded to pick the right decoder path.
    /// Returns `nil` if the bytes can't be decoded as an image.
    public static func decode(data: Data, in mtmdContext: LlamaMtmdContext) -> LlamaBitmap? {
        let p: OpaquePointer? = data.withUnsafeBytes { raw -> OpaquePointer? in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return nil }
            // Since llama.cpp v0.7.0 the helper returns a bitmap + video
            // context pair. A video context only appears for video input
            // (never built into the iOS slices: MTMD_VIDEO=OFF); free one if
            // it ever does and refuse it, since this wrapper takes stills.
            let wrapper = mtmd_helper_bitmap_init_from_buf(
                mtmdContext.pointer, base, raw.count,
                false, mtmd_helper_init_opt_default())
            if let video = wrapper.video_ctx {
                if let bitmap = wrapper.bitmap { mtmd_bitmap_free(bitmap) }
                mtmd_helper_video_free(video)
                return nil
            }
            return wrapper.bitmap
        }
        guard let p else { return nil }
        return LlamaBitmap(pointer: p)
    }

    public var width: UInt32 { mtmd_bitmap_get_nx(pointer) }
    public var height: UInt32 { mtmd_bitmap_get_ny(pointer) }
    public var isAudio: Bool { mtmd_bitmap_is_audio(pointer) }

    deinit {
        mtmd_bitmap_free(pointer)
    }
}
