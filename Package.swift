// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// llama.cpp v0.8.0 (stable, 2026-10-28) is the same commit as build tag
// b11960 (8642541b); the build tag names the release asset. Rollback:
// b11669 (v0.7.0), release b11669-mtmd, checksum 4e481c2f...
let llamaVersion = "b11960"
// Custom xcframework built by .github/workflows/build-xcframework.yml with
// llama.cpp's own build-xcframework.sh (iOS device + iOS Simulator + macOS).
// Upstream now builds mtmd (multimodal, for Gemma 4 vision) into the
// framework itself, so the old header/CMake patches are gone; we still host
// our own build because ggml-org's release ships only iOS device + macOS
// slices, and the app's unit tests run on the iOS Simulator.
// Release b11960-mtmd: workflow run 42692207102 at 95a8b3d, Xcode 26.4.1.
let llamaChecksum = "a61165f7c9668a7c77c78c4960db6e9f3190f696080243088ff2b9b42dd3af80"

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
