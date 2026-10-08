//
//  LlamaConfig.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 05/11/2024.
//

import Foundation

public struct LlamaConfig: Equatable, Sendable {
    public let batchSize: UInt32
    public let maxTokenCount: UInt32
    public let useGPU: Bool
    /// Where on-disk KV snapshots of a stable prompt prefix live (see
    /// `LlamaService.streamCompletion(prompt:cacheablePrefix:images:samplingConfig:)`).
    /// nil disables prefix snapshots; in-memory prefix reuse still works.
    public let promptCacheDirectory: URL?
    /// How many snapshot files to keep in `promptCacheDirectory`, newest first.
    public let promptCacheMaxFiles: Int

    public init(
        batchSize: UInt32,
        maxTokenCount: UInt32,
        useGPU: Bool = true,
        promptCacheDirectory: URL? = nil,
        promptCacheMaxFiles: Int = 2
    ) {
        self.batchSize = batchSize
        self.maxTokenCount = maxTokenCount
        self.useGPU = useGPU
        self.promptCacheDirectory = promptCacheDirectory
        self.promptCacheMaxFiles = max(1, promptCacheMaxFiles)
    }
}
