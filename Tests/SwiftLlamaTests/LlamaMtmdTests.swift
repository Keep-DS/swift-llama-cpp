//
//  LlamaMtmdTests.swift
//  swift-llama-cpp
//
//  Smoke tests for the mtmd (multimodal) bindings. These tests verify
//  the C symbols are linked into the framework and the Swift wrappers
//  are callable. End-to-end image inference requires a model + mmproj
//  pair on disk and runs only when those resources are present.
//

import Testing
import Foundation
@testable import SwiftLlama

struct LlamaMtmdTests {

    @Test("Default media marker is non-empty (mtmd symbols linked)")
    func testDefaultMarker() {
        let marker = LlamaMtmdContext.defaultMarker
        #expect(!marker.isEmpty, "mtmd_default_marker returned empty string — mtmd symbols may not be linked")
        // Current default is "<__media__>"; we don't pin to it (might change
        // upstream) but it should be wrapped in angle brackets.
        #expect(marker.hasPrefix("<") && marker.hasSuffix(">"), "Expected angle-bracket marker, got \(marker)")
    }

    @Test("LlamaService with mmproj URL initializes without throwing")
    func testServiceInitWithMmproj() {
        let model = URL(fileURLWithPath: "/nonexistent/model.gguf")
        let mmproj = URL(fileURLWithPath: "/nonexistent/mmproj.gguf")
        let config = LlamaConfig(batchSize: 256, maxTokenCount: 2048, useGPU: false)

        // Init must not throw; the model is loaded lazily on first call.
        _ = LlamaService(modelUrl: model, mmprojUrl: mmproj, config: config)
        _ = LlamaService(modelUrl: model, mmprojUrl: nil, config: config)
    }

    @Test("streamCompletion(prompt:images:) without mmproj rejects image input")
    func testImageCallWithoutMmprojFails() async {
        let model = URL(fileURLWithPath: "/nonexistent/model.gguf")
        let config = LlamaConfig(batchSize: 256, maxTokenCount: 2048, useGPU: false)
        let service = LlamaService(modelUrl: model, mmprojUrl: nil, config: config)

        let sampling = LlamaSamplingConfig(temperature: 0.7, seed: 1)
        let imageData = Data([0xFF, 0xD8, 0xFF, 0xE0]) // arbitrary jpeg-shaped bytes

        do {
            _ = try await service.streamCompletion(prompt: "describe this", images: [imageData], samplingConfig: sampling)
            Issue.record("Expected LlamaMtmdError.initializationFailed when calling images path without mmprojUrl")
        } catch LlamaMtmdError.initializationFailed {
            // expected
        } catch {
            Issue.record("Wrong error type: \(error)")
        }
    }

    @Test("Empty images array delegates to text-only path (no mmproj required)")
    func testEmptyImagesArrayFallsThrough() async {
        // Without a real model on disk the call will fail at load time —
        // we just want to confirm it does NOT fail with
        // LlamaMtmdError.initializationFailed (which would mean it tried
        // the multimodal path).
        let model = URL(fileURLWithPath: "/nonexistent/model.gguf")
        let config = LlamaConfig(batchSize: 256, maxTokenCount: 2048, useGPU: false)
        let service = LlamaService(modelUrl: model, mmprojUrl: nil, config: config)

        let sampling = LlamaSamplingConfig(temperature: 0.7, seed: 1)
        do {
            _ = try await service.streamCompletion(prompt: "hello", images: [], samplingConfig: sampling)
            Issue.record("Expected model-load failure")
        } catch LlamaMtmdError.initializationFailed {
            Issue.record("Should not hit mtmd init path when images is empty")
        } catch {
            // Some other error (couldNotInitializeContext, file not found, etc.) — OK
        }
    }
}
