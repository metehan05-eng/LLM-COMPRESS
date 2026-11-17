# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `compress --quality-file` to compare source and quantized-model perplexity on the same corpus.
- `info` command that prints GGUF metadata and the full tensor table, with
  per-tensor type and approximate size.
- `estimate` command that projects output sizes for every preset and whether
  they fit in free RAM.
- `--preset auto` selects the highest-quality preset that fits in RAM.
- `bench` command wrapping `llama-bench`, accepting multiple `--model` values.
- `perplexity` command wrapping `llama-perplexity`, with an optional `--baseline`
  model to report the PPL delta.
- `pull` command that downloads a GGUF from the Hugging Face Hub, with a progress
  bar, free-space check, atomic `.part` handling, GGUF validation, `--revision`,
  `HF_TOKEN` support, and the `HF_ENDPOINT` mirror override.
- `--json` output for `scan`, `info`, `estimate`, `perplexity`, `pull`, and
  `status` for scripting.
- Split GGUF support: `split.count`/`split.no` are parsed and shards are merged
  into a single entry during `scan`.
- Cross-platform hardware reporting through `sysinfo` (no longer Linux-only).
- Library crate (`src/lib.rs`) so the modules are testable and embeddable.
- Integration tests with a synthetic GGUF fixture under `tests/`.
- CI workflow covering formatting, clippy, tests, and the MSRV.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `rustfmt.toml`, and `.gitignore`.

### Changed

- `run` now links to llama.cpp's shared C library and performs GGUF model loading,
  tokenization, chat-template formatting, and generation in-process. Added
  `--context` and `--predict` controls; `llama-cli` is no longer needed for chat.
- The build locates `libllama` and `libggml` in standard library directories or
  the llama.cpp path set by `LLM_COMPRESS_LLAMA_LIB_DIR`; plugin backends load at
  startup.
- Quantization writes to a temporary sibling GGUF, validates its metadata and parameter count, then moves it into place. Failed or invalid outputs are removed, and generated imatrix files are cleaned up.
- Compression summaries report both size ratio and bytes saved (or growth).
- `compress` now validates inputs before launching llama.cpp: it rejects
  identical input/output paths, missing calibration/imatrix files, and existing
  outputs unless `--force` is passed, and it checks free disk space against a
  preset size estimate. It also prints the projected output size first.
- `IQ3_XXS` without an importance matrix now fails before any work starts.
- `--threads 0` is clamped to one worker.
- Removed the unused `tokio` dependency.
- Raised the minimum supported Rust version to 1.85.

### Fixed

- Hardened GGUF parsing with upper bounds on metadata entries, tensors, and
  array lengths.

## [0.1.0] - 2026-11-10

### Added

- Initial prototype: model discovery (Ollama, Hugging Face cache, custom paths),
  GGUF header parsing, `llama-quantize`/`llama-imatrix` wrappers, Q4_0 AVX2/FMA
  and scalar kernels, memory mapping, and a system status report.
