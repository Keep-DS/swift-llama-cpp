//
//  LlamaUTF8.swift
//  swift-llama-cpp
//
//  Streaming text out of token pieces. Byte-level BPE vocabularies can split
//  one character (an emoji, most CJK) across several tokens, so a token's
//  bytes are not always valid UTF-8 on their own. Generation keeps the
//  incomplete tail and prepends it to the next token's bytes.
//

enum LlamaUTF8 {

    /// Split `bytes` into the longest prefix that does not end inside a
    /// multi-byte UTF-8 character, and that unfinished tail (at most 3 bytes).
    /// Invalid bytes elsewhere stay in the prefix; decoding replaces them.
    static func splitIncompleteTail(_ bytes: [UInt8]) -> (complete: [UInt8], tail: [UInt8]) {
        var index = bytes.count - 1
        var continuationBytes = 0
        // Walk back over up to 3 continuation bytes (10xxxxxx) to the lead byte.
        while index >= 0, continuationBytes < 3, bytes[index] & 0xC0 == 0x80 {
            continuationBytes += 1
            index -= 1
        }
        guard index >= 0 else { return (bytes, []) }
        let needed: Int
        switch bytes[index] {
        case 0xC0...0xDF: needed = 1
        case 0xE0...0xEF: needed = 2
        case 0xF0...0xF7: needed = 3
        default: needed = 0   // ASCII, or a byte no character starts with
        }
        guard needed > continuationBytes else { return (bytes, []) }
        return (Array(bytes[..<index]), Array(bytes[index...]))
    }
}
