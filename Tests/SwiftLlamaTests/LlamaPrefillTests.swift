import Testing
import Foundation
@testable import SwiftLlama

/// Reading ahead and a chat's next turn, on a real model with sliding-window
/// layers (Gemma 3 270M is enough): set SWIFTLLAMA_SWA_MODEL to its .gguf
/// path. Skipped otherwise.
struct LlamaPrefillTests {
    static let modelPath = ProcessInfo.processInfo.environment["SWIFTLLAMA_SWA_MODEL"]

    private func words(_ stem: String, _ count: Int) -> String {
        (0..<count).map { "\(stem)\($0 % 89)" }.joined(separator: " ")
    }

    private func turn(_ role: String, _ text: String) -> String {
        "<start_of_turn>\(role)\n\(text)<end_of_turn>\n"
    }

    private func model() throws -> Llama {
        try Llama(modelPath: Self.modelPath!, config: LlamaConfig(batchSize: 256, maxTokenCount: 4096))
    }

    @Test("Reading ahead leaves only the new message to read", .enabled(if: modelPath != nil))
    func prefillThenExtend() async throws {
        let llama = try model()
        let history = turn("user", words("alpha", 500)) + turn("model", words("beta", 200))
        try await llama.prefill(prompt: history, cacheablePrefix: nil)
        let prompt = history + turn("user", "Note: it is 3:42 pm.") + turn("user", "And then?") + "<start_of_turn>model\n"
        try await llama.initializeCompletion(prompt: prompt, cacheablePrefix: nil)
        #expect(await llama.lastPlan == .extend)
    }

    /// The next turn drops the last turn's note and adds a new one before
    /// the new message: the cache goes back to where the history ended, far
    /// further than the sliding window, and reads as a fresh start would.
    @Test("A chat's next turn reads on from the end of its history", .enabled(if: modelPath != nil))
    func nextTurnTrimsToTheHistory() async throws {
        let llama = try model()
        let window = await llama.swaWindow
        #expect(window > 0, "the model needs sliding-window layers")

        let history = turn("user", words("alpha", 700)) + turn("model", words("beta", 300))
        let first = history + turn("user", "Note: it is 3:42 pm.") + turn("user", words("gamma", 150)) + "<start_of_turn>model\n"
        try await llama.initializeCompletion(prompt: first, cacheablePrefix: nil)
        await llama.updateSamplingConfig(LlamaSamplingConfig(temperature: 0.7, seed: 7))
        for _ in 0..<60 { _ = try await llama.generateNextToken() }

        let historyTokens = await llama.tokenCount(history)
        let second = history + turn("user", words("gamma", 150)) + turn("model", words("delta", 80))
            + turn("user", "Note: it is 3:44 pm.") + turn("user", words("epsilon", 60)) + "<start_of_turn>model\n"
        try await llama.initializeCompletion(prompt: second, cacheablePrefix: nil)
        let plan = try #require(await llama.lastPlan)
        guard case .trim(let position) = plan else {
            Issue.record("expected a trim, got \(plan)")
            return
        }
        #expect(position >= historyTokens)
        #expect(historyTokens - window > 0, "a window-sized trim couldn't have reached it")
        let resumed = try #require(await llama.lastLogits())

        let fresh = try model()
        try await fresh.initializeCompletion(prompt: second, cacheablePrefix: nil)
        #expect(await fresh.lastPlan == .full)
        let scratch = try #require(await fresh.lastLogits())
        let largest = zip(resumed, scratch).map { abs($0 - $1) }.max() ?? .infinity
        #expect(largest < 0.05, "logits differ by up to \(largest)")
    }
}
