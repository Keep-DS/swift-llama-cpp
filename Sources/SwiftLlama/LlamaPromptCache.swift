//
//  LlamaPromptCache.swift
//  swift-llama-cpp
//
//  KV reuse across prompts: what to keep, what to trim, and on-disk
//  snapshots of a stable prompt prefix (system prompt + tool declarations)
//  so a fresh load restores it instead of re-processing thousands of tokens.
//

import Foundation
import llama

/// Pure KV-reuse decisions. No model needed, so they're unit-tested directly.
enum LlamaKVReuse {

    /// Below this a prefix isn't worth a snapshot file.
    static let minimumSnapshotTokens = 64

    enum Plan: Equatable {
        /// The KV cache already holds the start of this prompt: process the rest.
        case extend
        /// Drop the KV tail from this position, then process the rest.
        case trim(to: Int)
        /// Restore (or build and save) the prefix snapshot, then process the rest.
        case restorePrefix
        /// Clear and process the whole prompt.
        case full
    }

    /// Whether the KV tail can be dropped back to `position` without losing
    /// context. Sliding-window (SWA) layers only keep the newest positions,
    /// so re-decoding from `position` needs `[position - swaWindow, position)`
    /// still resident; once it has scrolled out, trimming would silently
    /// starve those layers of context (llama-server applies the same rule).
    static func canTrim(to position: Int, swaWindow: Int, minResidentPosition: Int) -> Bool {
        guard swaWindow > 0, minResidentPosition >= 0 else { return true }
        return minResidentPosition <= max(0, position - swaWindow)
    }

    /// How to bring the KV cache from `processed` to `prompt`.
    /// - Parameters:
    ///   - prefixCount: token count of a cacheable prefix `prompt` starts
    ///     with, or nil when there's no snapshot to fall back on.
    ///   - swaWindow / minResidentPosition: see `canTrim`.
    static func plan<T: Equatable>(
        prompt: [T],
        processed: [T],
        prefixCount: Int?,
        swaWindow: Int,
        minResidentPosition: Int
    ) -> Plan {
        if !processed.isEmpty, prompt.count > processed.count, prompt.starts(with: processed) {
            return .extend
        }
        // Common prefix, capped so at least one prompt token is decoded
        // (the sampler needs fresh logits for the last one).
        var shared = 0
        let limit = min(prompt.count - 1, processed.count)
        while shared < limit, prompt[shared] == processed[shared] { shared += 1 }

        let trimSafe = shared > 0 && canTrim(
            to: shared, swaWindow: swaWindow, minResidentPosition: minResidentPosition)

        if let prefixCount {
            // Trimming keeps at least the prefix: cheaper than a disk read.
            if trimSafe, shared >= prefixCount { return .trim(to: shared) }
            return .restorePrefix
        }
        // No snapshot: trim only when most of the cache survives.
        if trimSafe, processed.count >= 10, Double(shared) / Double(processed.count) >= 0.5 {
            return .trim(to: shared)
        }
        return .full
    }

    /// FNV-1a 64-bit. Stable across launches and platforms (unlike Hasher).
    static func fnv1a<S: Sequence>(_ bytes: S, seed: UInt64 = 0xcbf2_9ce4_8422_2325) -> UInt64 where S.Element == UInt8 {
        var hash = seed
        for byte in bytes {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    static func hex(_ value: UInt64) -> String {
        let s = String(value, radix: 16)
        return String(repeating: "0", count: max(0, 16 - s.count)) + s
    }

    /// Bump when the snapshot layout or its identity rules change.
    static let snapshotFormat = "v2"

    /// Leads every snapshot file name: the snapshot format plus the engine's
    /// state-file version (LLAMA_STATE_SEQ_VERSION), e.g. "v2s4".
    static func versionTag(stateVersion: Int) -> String {
        "\(snapshotFormat)s\(stateVersion)"
    }

    /// Whether a snapshot file was written under `versionTag`. Files from any
    /// other version (including the untagged names before v2) are stale.
    static func isCurrentVersion(fileName: String, versionTag: String) -> Bool {
        fileName.hasPrefix(versionTag + "-")
    }

    /// Snapshot file name: `<version>-<identity hash>-<token hash>.kvprefix`.
    /// The identity (model file, context size, batch, backend) is part of the
    /// name so a snapshot is never restored into a context it wasn't made for.
    static func snapshotName(identity: String, tokens: [Int32],
                             versionTag: String = LlamaPromptCache.currentVersionTag) -> String {
        "\(versionTag)-\(identityTag(identity))-\(hex(tokenHash(tokens))).\(LlamaPromptCache.fileExtension)"
    }

    static func identityTag(_ identity: String) -> String {
        hex(fnv1a(Array(identity.utf8)))
    }

    static func tokenHash(_ tokens: [Int32]) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in tokens {
            let v = UInt32(bitPattern: token)
            hash = fnv1a([UInt8(v & 0xff), UInt8((v >> 8) & 0xff), UInt8((v >> 16) & 0xff), UInt8(v >> 24)], seed: hash)
        }
        return hash
    }
}

/// On-disk prefix snapshots for one loaded model. Used only inside the
/// `Llama` actor.
final class LlamaPromptCache {
    static let fileExtension = "kvprefix"

    /// The version tag for snapshots this build of the engine writes.
    static let currentVersionTag = LlamaKVReuse.versionTag(stateVersion: Int(LLAMA_STATE_SEQ_VERSION))

    let directory: URL
    let maxFiles: Int
    let identity: String
    let versionTag: String

    /// Deletes snapshots written under any other version tag (an older
    /// engine's files can never be restored, so they're only dead weight).
    init(directory: URL, maxFiles: Int, identity: String, versionTag: String = LlamaPromptCache.currentVersionTag) {
        self.directory = directory
        self.maxFiles = maxFiles
        self.identity = identity
        self.versionTag = versionTag
        removeOtherVersions()
    }

    func url(for tokens: [llama_token]) -> URL {
        directory.appendingPathComponent(LlamaKVReuse.snapshotName(identity: identity, tokens: tokens, versionTag: versionTag))
    }

    /// The most recently used snapshot made for this model + context.
    func newestSnapshotURL() -> URL? {
        let tag = "\(versionTag)-\(LlamaKVReuse.identityTag(identity))-"
        return snapshots()
            .filter { $0.url.lastPathComponent.hasPrefix(tag) }
            .max { $0.date < $1.date }?
            .url
    }

    /// Save sequence 0 (which must hold exactly `tokens`) to `url`.
    /// Writes to a temp file first so a kill mid-write never leaves a
    /// truncated snapshot under the real name.
    @discardableResult
    func save(tokens: [llama_token], from context: LlamaContext, to url: URL) -> Int {
        let fm = FileManager.default
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let tmp = url.appendingPathExtension("tmp")
        try? fm.removeItem(at: tmp)
        let bytes = context.saveSequenceState(to: tmp.path, seqId: 0, tokens: tokens)
        guard bytes > 0 else {
            try? fm.removeItem(at: tmp)
            return 0
        }
        try? fm.removeItem(at: url)
        do {
            try fm.moveItem(at: tmp, to: url)
        } catch {
            try? fm.removeItem(at: tmp)
            return 0
        }
        prune()
        return bytes
    }

    func touch(_ url: URL) {
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: url.path)
    }

    func remove(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    /// Remove `.kvprefix` files whose name doesn't carry `versionTag`.
    func removeOtherVersions() {
        for snapshot in snapshots()
        where !LlamaKVReuse.isCurrentVersion(fileName: snapshot.url.lastPathComponent, versionTag: versionTag) {
            try? FileManager.default.removeItem(at: snapshot.url)
        }
    }

    /// Keep the newest `maxFiles` snapshots (any model), drop stray temp files.
    func prune() {
        let fm = FileManager.default
        let all = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for url in all where url.pathExtension == "tmp" { try? fm.removeItem(at: url) }
        let sorted = snapshots().sorted { $0.date > $1.date }
        for stale in sorted.dropFirst(maxFiles) { try? fm.removeItem(at: stale.url) }
    }

    private func snapshots() -> [(url: URL, date: Date)] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        return urls
            .filter { $0.pathExtension == Self.fileExtension }
            .map { url in
                let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                return (url, date)
            }
    }
}
