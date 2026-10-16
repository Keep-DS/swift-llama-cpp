// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// llama.cpp v0.7.0 (stable, 2026-10-15) is the same commit as build tag
// b11669 (620d222); the build tag names the release asset.
let llamaVersion = "b11669"
// Custom xcframework built by .github/workflows/build-xcframework.yml with
// llama.cpp's own build-xcframework.sh (iOS device + iOS Simulator + macOS).
// Upstream now builds mtmd (multimodal, for Gemma 4 vision) into the
// framework itself, so the old header/CMake patches are gone; we still host
// our own build because ggml-org's release ships only iOS device + macOS
// slices, and the app's unit tests run on the iOS Simulator.
let llamaChecksum = "ed953f4077b5fdf8bf84d306cd1509edba6b1fdfe816e376bae5c6242abe9c9d"

let package = Package(
    name: "swift-llama-cpp",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "SwiftLlama",
            targets: ["SwiftLlama"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "SwiftLlama",
            dependencies: [
                "llama"
            ]
        ),
        .binaryTarget(
            name: "llama",
            url: "https://github.com/Keep-DS/swift-llama-cpp/releases/download/\(llamaVersion)-mtmd/llama-\(llamaVersion)-mtmd-xcframework.zip",
            checksum: llamaChecksum
        ),
        .testTarget(
            name: "SwiftLlamaTests",
            dependencies: ["SwiftLlama"],
            resources: [.copy("Resources")]
        ),
    ]
)
