import Testing
import Foundation
import llama
@testable import SwiftLlama

/// The 1B test model, when it's been put in Resources (it isn't committed).
/// Model tests below are skipped without it instead of trapping.
private let testModelURL: URL? =
    Bundle.module.url(forResource: "Llama-3.2-1B-Instruct-Q4_K_M", withExtension: "gguf", subdirectory: "Resources")

/// Review fixes for the v0.7.0 upgrade: streaming UTF-8, buffer retries,
/// decode return codes, snapshot version tags.
struct LlamaHardeningTests {

    // MARK: Streaming UTF-8

    @Test("Whole characters pass straight through")
    func completeTextHasNoTail() {
        for text in ["", "plain ASCII", "naïve café", "你好", "😀🎉"] {
            let split = LlamaUTF8.splitIncompleteTail(Array(text.utf8))
            #expect(split.complete == Array(text.utf8))
            #expect(split.tail.isEmpty)
        }
    }

    @Test("A character cut short is held back until the rest arrives")
    func incompleteTailIsHeldBack() {
        let emoji = Array("😀".utf8)   // F0 9F 98 80
        for cut in 1..<emoji.count {
            let split = LlamaUTF8.splitIncompleteTail(Array("ok ".utf8) + emoji.prefix(cut))
            #expect(split.complete == Array("ok ".utf8))
            #expect(split.tail == Array(emoji.prefix(cut)))
        }
        let accent = LlamaUTF8.splitIncompleteTail([0x61, 0xC3])   // "a" + first byte of "é"
        #expect(accent.complete == [0x61])
        #expect(accent.tail == [0xC3])
    }

    @Test("Feeding pieces byte by byte rebuilds the text exactly")
    func bytewiseStreamRebuildsText() {
        let text = "Hi 😀, naïve 你好 🎉!"
        var pending: [UInt8] = []
        var out = ""
        for byte in Array(text.utf8) {
            let split = LlamaUTF8.splitIncompleteTail(pending + [byte])
            pending = split.tail
            out += String(decoding: split.complete, as: UTF8.self)
        }
        #expect(pending.isEmpty)
        #expect(out == text)
    }

    @Test("Stray continuation bytes are not held forever")
    func strayContinuationBytesFlush() {
        let split = LlamaUTF8.splitIncompleteTail([0x80, 0x80, 0x80, 0x80])
        #expect(split.tail.isEmpty)
        #expect(split.complete.count == 4)
    }

    // MARK: Buffer retries

    @Test("A negative count retries once at the size llama.cpp asked for")
    func fillRetriesAtRequestedSize() {
        var calls: [Int32] = []
        let result = LlamaModel.fillRetryingOnce(initialCapacity: 4, filler: Int32(0)) { buffer, capacity in
            calls.append(capacity)
            guard capacity >= 10 else { return -10 }
            for i in 0..<10 { buffer[i] = Int32(i) }
            return 10
        }
        #expect(calls == [4, 10])
        #expect(result == Array(0..<10))
    }

    @Test("Overflow and repeated refusals give an empty result, never a trap")
    func fillGivesUpCleanly() {
        let overflow = LlamaModel.fillRetryingOnce(initialCapacity: 8, filler: CChar(0)) { _, _ in Int32.min }
        #expect(overflow.isEmpty)
        var calls = 0
        let stubborn = LlamaModel.fillRetryingOnce(initialCapacity: 8, filler: CChar(0)) { _, capacity in
            calls += 1
            return -(capacity + 1)
        }
        #expect(stubborn.isEmpty)
        #expect(calls == 2)
        let overReport = LlamaModel.fillRetryingOnce(initialCapacity: 3, filler: UInt8(0)) { _, _ in 99 }
        #expect(overReport.count == 3)
    }

    // MARK: Decode return codes

    @Test("Only 0 is a successful decode; 1 (no KV slot) and 2 (aborted) fail")
    func decodeReturnCodes() {
        #expect(LlamaContext.succeeded(0))
        for code: Int32 in [1, 2, -1, -2, 3] {
            #expect(!LlamaContext.succeeded(code))
        }
    }

    // MARK: Snapshot version tags

    @Test("Snapshot names lead with the format and engine state version")
    func snapshotVersionTag() {
        let tag = LlamaPromptCache.currentVersionTag
        #expect(tag == "v2s\(LLAMA_STATE_SEQ_VERSION)")
        let name = LlamaKVReuse.snapshotName(identity: "m.gguf|1", tokens: [1, 2, 3])
        #expect(LlamaKVReuse.isCurrentVersion(fileName: name, versionTag: tag))
        #expect(!LlamaKVReuse.isCurrentVersion(fileName: name, versionTag: "v2s5"))
        // The untagged names written before v2 never count as current.
        #expect(!LlamaKVReuse.isCurrentVersion(fileName: "1eeddce243cbbe0f-3f97df29b7170ba9.kvprefix", versionTag: tag))
    }

    @Test("Opening the cache deletes snapshots from other versions only")
    func initDeletesOtherVersions() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("kvprefix-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let current = LlamaKVReuse.snapshotName(identity: "other-model", tokens: [7])
        let stale = ["1eeddce243cbbe0f-3f97df29b7170ba9.kvprefix", "v1s2-aaaa-bbbb.kvprefix", "v2s3-aaaa-bbbb.kvprefix"]
        for name in stale + [current, "notes.txt"] {
            try Data([1]).write(to: dir.appendingPathComponent(name))
        }
        _ = LlamaPromptCache(directory: dir, maxFiles: 2, identity: "id")
        let left = Set(try FileManager.default.contentsOfDirectory(atPath: dir.path))
        #expect(left == [current, "notes.txt"])
    }

    // MARK: With the test model

    @Test("Streaming emoji through token pieces loses nothing", .enabled(if: testModelURL != nil))
    func emojiSurviveTokenPieces() throws {
        let url = try #require(testModelURL)
        let model = try #require(LlamaModel(path: url.path))
        let text = "Party time 🎉🥳 naïve 你好 😀"
        let tokens = model.tokenize(text: text, addBos: false, special: false)
        #expect(!tokens.isEmpty)
        var pending: [UInt8] = []
        var streamed = ""
        var bytes: [UInt8] = []
        for token in tokens {
            let piece = model.pieceBytes(from: token)
            bytes += piece
            let split = LlamaUTF8.splitIncompleteTail(pending + piece)
            pending = split.tail
            streamed += String(decoding: split.complete, as: UTF8.self)
        }
        #expect(bytes == Array(text.utf8))
        #expect(pending.isEmpty)
        #expect(streamed == text)
    }

    @Test("Tokenizing a long, token-dense text never traps", .enabled(if: testModelURL != nil))
    func tokenizeLongText() throws {
        let url = try #require(testModelURL)
        let model = try #require(LlamaModel(path: url.path))
        let text = String(repeating: "🎉 a1 ", count: 2_000)
        let tokens = model.tokenize(text: text, addBos: true, special: false)
        #expect(tokens.count > 2_000)
        #expect(model.detokenize(tokens: tokens).contains("🎉 a1"))
    }
}
