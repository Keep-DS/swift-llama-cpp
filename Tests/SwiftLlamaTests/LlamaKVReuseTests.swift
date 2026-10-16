import Testing
import Foundation
@testable import SwiftLlama

/// The KV-reuse plan, which decides between extending, trimming, restoring a
/// prefix snapshot, or reprocessing. Pure, so no model is needed.
struct LlamaKVReuseTests {

    private func plan(_ prompt: [Int32], _ processed: [Int32], prefix: Int? = nil,
                      swa: Int = 0, minPos: Int = 0) -> LlamaKVReuse.Plan {
        LlamaKVReuse.plan(prompt: prompt, processed: processed, prefixCount: prefix,
                          swaWindow: swa, minResidentPosition: minPos)
    }

    @Test("A prompt that continues the cache only processes the rest")
    func extendsWhenCacheIsAPrefix() {
        #expect(plan([1, 2, 3, 4], [1, 2, 3]) == .extend)
    }

    @Test("An identical prompt still decodes its last token")
    func identicalPromptTrimsOneToken() {
        let tokens = Array(Int32(0)..<20)
        #expect(plan(tokens, tokens) == .trim(to: 19))
    }

    @Test("Without a snapshot, trim only when most of the cache survives")
    func trimHeuristicWithoutSnapshot() {
        let processed = Array(Int32(0)..<100)
        var late = processed; late[80] = -1
        var early = processed; early[20] = -1
        #expect(plan(late, processed) == .trim(to: 80))
        #expect(plan(early, processed) == .full)
        #expect(plan([9, 9, 9], []) == .full)
    }

    @Test("A sliding window that scrolled past the divergence forbids trimming")
    func swaWindowBlocksUnsafeTrim() {
        // Window 512; the cache only still holds positions 5000+ for those
        // layers, so re-decoding from 4100 would miss [3588, 4100).
        #expect(!LlamaKVReuse.canTrim(to: 4100, swaWindow: 512, minResidentPosition: 5000))
        // A divergence near the end (a replayed tool call) is fine.
        #expect(LlamaKVReuse.canTrim(to: 5900, swaWindow: 512, minResidentPosition: 5000))
        // Models without SWA layers can always trim.
        #expect(LlamaKVReuse.canTrim(to: 10, swaWindow: 0, minResidentPosition: 5000))
    }

    @Test("With a snapshot, an unsafe trim restores the prefix instead")
    func snapshotReplacesUnsafeTrim() {
        let processed = Array(Int32(0)..<6000)
        var prompt = Array(processed[0..<4500]); prompt.append(contentsOf: [-1, -2, -3])
        #expect(plan(prompt, processed, prefix: 4100, swa: 512, minPos: 5000) == .restorePrefix)
        // Safe trim that keeps more than the prefix wins over a disk read.
        #expect(plan(prompt, processed, prefix: 4100, swa: 512, minPos: 3000) == .trim(to: 4500))
    }

    @Test("A fresh cache with a snapshot restores it")
    func freshCacheRestores() {
        #expect(plan(Array(Int32(0)..<200), [], prefix: 100) == .restorePrefix)
    }

    @Test("A divergence inside the prefix restores the prefix")
    func divergenceInsidePrefixRestores() {
        let processed = Array(Int32(0)..<300)
        var prompt = processed; prompt[50] = -1
        #expect(plan(prompt, processed, prefix: 100) == .restorePrefix)
    }

    @Test("Snapshot names are stable and separate models and prompts")
    func snapshotNames() {
        let a = LlamaKVReuse.snapshotName(identity: "m.gguf|1|ctx=8192", tokens: [1, 2, 3])
        #expect(a == LlamaKVReuse.snapshotName(identity: "m.gguf|1|ctx=8192", tokens: [1, 2, 3]))
        #expect(a != LlamaKVReuse.snapshotName(identity: "m.gguf|1|ctx=4096", tokens: [1, 2, 3]))
        #expect(a != LlamaKVReuse.snapshotName(identity: "m.gguf|1|ctx=8192", tokens: [1, 2, 4]))
        #expect(a.hasPrefix(LlamaPromptCache.currentVersionTag + "-" + LlamaKVReuse.identityTag("m.gguf|1|ctx=8192") + "-"))
        #expect(a.hasSuffix(".kvprefix"))
        // FNV-1a reference value for "a".
        #expect(LlamaKVReuse.fnv1a(Array("a".utf8)) == 0xaf63_dc4c_8601_ec8c)
    }

    @Test("Pruning keeps the newest snapshots")
    func pruneKeepsNewest() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("kvprefix-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = LlamaPromptCache(directory: dir, maxFiles: 2, identity: "id")
        let now = Date()
        for i in 0..<4 {
            let url = cache.url(for: [Int32(i)])
            try Data([UInt8(i)]).write(to: url)
            try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(Double(i))], ofItemAtPath: url.path)
        }
        try Data().write(to: dir.appendingPathComponent("x.kvprefix.tmp"))
        cache.prune()
        let left = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()
        #expect(left.count == 2)
        #expect(Set(left) == Set([cache.url(for: [2]).lastPathComponent, cache.url(for: [3]).lastPathComponent]))
        #expect(cache.newestSnapshotURL()?.lastPathComponent == cache.url(for: [3]).lastPathComponent)
    }
}
