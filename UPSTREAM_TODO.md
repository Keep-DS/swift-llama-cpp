# Upstream TODO

Patches carried in this fork that should eventually be proposed to their
upstreams. Keep this list short and concrete; delete entries once merged.

## ggml-org/llama.cpp

Nothing carried as of v0.7.0 (b11669). Both earlier entries are resolved
upstream:

- The missing `mtmd_decoder_pos` typedef: `mtmd-helper.h` now spells it
  `struct mtmd_decoder_pos *`, so strict-C module builds are fine.
- `build-xcframework.sh` vs `LLAMA_BUILD_TOOLS=ON` on iOS: the script now
  builds mtmd on its own (`LLAMA_BUILD_MTMD=ON`) and copies `mtmd.h` /
  `mtmd-helper.h` into the framework, so no tools are needed.

Open nice-to-have: ggml-org's release workflow builds the xcframework with
`./build-xcframework.sh macos ios-device` only (no iOS Simulator slice), which
is why `.github/workflows/build-xcframework.yml` still builds and hosts our
own zip. If upstream ever ships the simulator slice again, Package.swift can
point straight at their release asset.
