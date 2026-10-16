# swift-llama-cpp

Run any LLM locally on iOS or MacOS. Powered by [llama.cpp](https://github.com/ggml-org/llama.cpp)

This is Keep-DS's fork of [pgorzelany/swift-llama-cpp](https://github.com/pgorzelany/swift-llama-cpp)
by Piotr Gorzelany. It pins llama.cpp v0.7.0 (build tag `b11669`) and adds
multimodal (mtmd) image input, raw-prompt completion, prompt-prefix KV
snapshots, a warm-up call and SWA-safe cache trimming. The llama.cpp binary
is an xcframework (iOS device, iOS Simulator, macOS) built by
`.github/workflows/build-xcframework.yml` and attached to this repo's
releases.

To browse upstream C/C++ source at the same revision as the pinned xcframework, see [Reference/README.md](Reference/README.md).

## Coverage

This wrapper covers:
- Model loading (single file and splits), save, metadata, size, params, encoder/decoder flags
- Vocab API (token text, score, attrs, special tokens), tokenize/detokenize
- Context creation/free, threads, embeddings and causal-attention toggles
  (llama.cpp's warm-up mode is deprecated upstream; `LlamaService.warmUp()`
  warms the model with a plain one-token decode)
- Memory API (sequence remove/copy/keep/add/div/min/max/canShift)
- Encode/decode, logits/embeddings getters, synchronize
- State/session save-load, per-sequence state
- Chat templates (apply and list built-ins)
- Sampler chain (grammar, top-k, top-p, temp, penalties, dist), sample/accept/reset/clone
- LoRA adapter load/apply/remove/clear, control vectors
- Backend init/free, capability queries, system info, logging hook

## Basic usage

Here is a quick example of how to use `SwiftLlama` to generate text from a model.

First, make sure you have a GGUF model file accessible in your project. You can download models from sources like [Hugging Face](https://huggingface.co/models?search=gguf).

```swift
import SwiftLlama
import Foundation

// 1. Get the model URL
// Make sure to add a GGUF model to your project and get its URL.
guard let modelUrl = Bundle.main.url(forResource: "your-model-name", withExtension: "gguf") else {
    print("Model file not found")
    return
}

// 2. Initialize the LlamaService
// This service manages the model and context.
let llamaService = LlamaService(modelUrl: modelUrl, config: .init(batchSize: 256, maxTokenCount: 4096, useGPU: true))

// 3. Prepare your messages
// The conversation history can be provided as an array of messages.
let messages = [
    LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
    LlamaChatMessage(role: .user, content: "Tell me a short story."),
]

// 4. Generate text
// The `streamCompletion` method returns an `AsyncThrowingStream` of tokens.
do {
    let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: .init(temperature: 0.8, seed: 42))
    var generatedText = ""
    for try await token in stream {
        generatedText += token
        print("Generated token: \(token)")
    }
    print("Generated text: \(generatedText)")
} catch {
    print("Error generating text: \(error.localizedDescription)")
}
``` 

## Credits and licences

- The Swift wrapper is MIT-licensed, copyright Piotr Gorzelany (see
  [LICENSE](LICENSE)); this fork's changes are under the same licence.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml, compiled into
  the xcframework, are MIT-licensed, copyright the ggml authors. Each
  xcframework release carries llama.cpp's licence as `llama.cpp-LICENSE.txt`.
