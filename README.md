# llm-compress

`llm-compress` is a Rust command-line utility for discovering, inspecting, downloading, compressing, and running GGUF language models on local hardware. It wraps llama.cpp for quantization, benchmarking, perplexity evaluation, and in-process inference.

The `run` command calls the llama.cpp C API in the same process. llama.cpp handles model loading, GGUF architecture support, tokenization, chat templates, and transformer execution. Rust also includes Q4_0/Q8_0 SIMD kernels and Transformer math primitives; these kernels are not currently part of the `run` inference path.

## Features

- **Model discovery (`scan`):** Finds models in Ollama, the Hugging Face cache, and custom directories. Split GGUF shards are grouped into one model entry.
- **GGUF inspection (`info`):** Displays metadata, architecture, parameter count, quantization type, and tensor details.
- **Model downloads (`pull`):** Downloads GGUF files from Hugging Face into a temporary `.part` file, checks the GGUF signature, then moves the file into place. Supports gated repositories with a Hugging Face token. Downloads do not resume.
- **Compression (`compress`):** Runs `llama-quantize` with `Q4_K_M`, `Q3_K_M`, `IQ3_XXS`, or `Q2_K`. The `auto` preset estimates a suitable size from available RAM. IQ3_XXS can use a precomputed importance matrix or generate one from a calibration corpus.
- **Compression quality comparison:** `compress --quality-file` evaluates the source and quantized models on the same text corpus and reports the perplexity (PPL) change.
- **Benchmarking (`bench`):** Runs `llama-bench` on one or more models.
- **Perplexity (`perplexity`):** Runs `llama-perplexity` and can compare a model against a baseline.
- **In-process chat (`run`):** Loads a GGUF model through the llama.cpp shared library and applies its chat template when available. Configure the context length and maximum reply length with `--context` and `--predict`.
- **Tokenizer inspection (`tokenize`):** Reads vocabulary metadata and can decode token IDs. Its text-to-token encoding is a basic approximation and may not match the model's tokenizer implementation.
- **System tools (`status`, `doctor`):** Reports hardware details and checks for external llama.cpp command-line tools.
- **Cleanup (`clean`):** Finds interrupted downloads and temporary files. It lists candidates by default and deletes only when `--yes` is supplied.
- **JSON output:** Available for `scan`, `info`, `estimate`, `perplexity`, `pull`, `status`, `doctor`, `tokenize`, and `clean`.

## Requirements

- Rust 1.85 or newer (`cargo` and `rustc`).
- A compatible llama.cpp build with the `libllama` and `libggml` shared libraries. These are required at build and runtime for `run`.
- llama.cpp command-line tools as needed:
  - `llama-quantize` for compression.
  - `llama-imatrix` for generating an importance matrix from calibration text.
  - `llama-bench` for benchmarking.
  - `llama-perplexity` for perplexity evaluation.

The build script searches standard library locations for llama.cpp. If its libraries are installed elsewhere, set `LLM_COMPRESS_LLAMA_LIB_DIR` to the directory containing `libllama`. The GGML library must also be discoverable in that directory, a nearby standard location, or a system library directory. The llama.cpp tools can be placed on `PATH` or configured with the environment variables listed below.

## Build

```bash
cargo build --release
```

The executable is written to `target/release/llm-compress`. During development, use `cargo run -- <command>`.

## Quick start

```bash
# Check the system and external llama.cpp tools
cargo run -- doctor

# Download an F16 model from Hugging Face
cargo run -- pull \
  --repo bartowski/SmolLM2-135M-Instruct-GGUF \
  --file SmolLM2-135M-Instruct-f16.gguf \
  --out ./models

# Inspect the downloaded model
cargo run -- info --model ./models/SmolLM2-135M-Instruct-f16.gguf

# Quantize it to Q4_K_M
cargo run -- compress \
  --model ./models/SmolLM2-135M-Instruct-f16.gguf \
  --preset Q4_K_M \
  --out ./models/SmolLM2-135M-Q4_K_M.gguf

# Start an in-process chat session
cargo run -- run \
  --model ./models/SmolLM2-135M-Q4_K_M.gguf \
  --threads 6 --context 2048 --predict 128
```

Type `/exit` or `/quit` to leave the chat session.

## Command reference

### Discover and inspect models

```bash
cargo run -- scan --path ./models
cargo run -- scan --path ./models --json
cargo run -- info --model ./models/model.gguf
cargo run -- estimate --model ./models/model.gguf
cargo run -- status
cargo run -- doctor --json
```

`scan` also searches `~/.ollama/models` and `~/.cache/huggingface/hub`. Supply `--path` more than once to include additional directories.

### Download a model

```bash
cargo run -- pull \
  --repo bartowski/Llama-3.2-1B-Instruct-GGUF \
  --file Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --out ./models
```

For gated repositories, set `HF_TOKEN` or pass `--token`. To use a Hugging Face-compatible mirror, set `HF_ENDPOINT`.

### Compress and evaluate quality

Quantization accepts unquantized F16, BF16, or F32 GGUF input. `--preset auto` chooses from Q4_K_M, Q3_K_M, and Q2_K based on the estimated output size and available RAM. IQ3_XXS requires an importance matrix or calibration text.

```bash
# Standard Q4_K_M quantization
cargo run -- compress \
  --model ./models/model-f16.gguf \
  --preset Q4_K_M \
  --out ./models/model-Q4_K_M.gguf

# Choose a preset automatically
cargo run -- compress --model ./models/model-f16.gguf --preset auto

# Build an importance matrix from calibration text for IQ3_XXS
cargo run -- compress \
  --model ./models/model-f16.gguf \
  --preset IQ3_XXS \
  --calibration ./calibration.txt

# Compare source and quantized perplexity on the same corpus
cargo run -- compress \
  --model ./models/model-f16.gguf \
  --preset Q4_K_M \
  --quality-file ./evaluation.txt
```

Use `--imatrix FILE` to provide an existing importance matrix instead of generating one. Use `--force` to replace an existing output file. Compression stages the output and validates it before moving it to the final path; incomplete staged files are cleaned up on failure.

### Benchmark and perplexity

```bash
# Compare multiple models
cargo run -- bench \
  --model ./models/model-f16.gguf \
  --model ./models/model-Q4_K_M.gguf

# Select llama-bench output format
cargo run -- bench --model ./models/model-Q4_K_M.gguf --output json

# Compare perplexity against a baseline model
cargo run -- perplexity \
  --model ./models/model-Q4_K_M.gguf \
  --baseline ./models/model-f16.gguf \
  --file ./evaluation.txt
```

### Run a model

```bash
cargo run -- run \
  --model ./models/model-Q4_K_M.gguf \
  --threads 6 \
  --context 2048 \
  --predict 128
```

`--context` defaults to the model's training context, and `--predict` defaults to 128 new tokens per reply. The selected context affects memory use. llama.cpp loads the model's embedded chat template when available; models without a template use plain role-prefixed text.

### Inspect tokenizer data

```bash
# Approximate text encoding
cargo run -- tokenize --model ./models/model.gguf --text "Hello, world"

# Decode token IDs
cargo run -- tokenize --model ./models/model.gguf --ids "1, 15043, 29889"

# Show vocabulary metadata
cargo run -- tokenize --model ./models/model.gguf --vocab-info
```

The built-in `tokenize --text` path uses a simple greedy vocabulary match with SentencePiece-style whitespace handling. It does not implement every tokenizer algorithm, including full BPE and byte fallback, and should not be used when exact token IDs are required. In-process chat uses llama.cpp's tokenizer instead.

### Clean temporary files

```bash
# List cleanup candidates without deleting them
cargo run -- clean --path ./models

# Delete matching .part, .tmp, and .temp files
cargo run -- clean --path ./models --yes
```

## Environment variables

| Variable | Description |
| --- | --- |
| `LLM_COMPRESS_PATHS` | Additional model search and cleanup directories, separated by `:`. |
| `LLM_COMPRESS_THREADS` | Default worker thread count for supported commands. |
| `LLM_COMPRESS_LLAMA_LIB_DIR` | Directory containing the llama.cpp shared library used at build time. |
| `LLM_COMPRESS_QUANTIZE_BIN` | Path to `llama-quantize`. |
| `LLM_COMPRESS_IMATRIX_BIN` | Path to `llama-imatrix`. |
| `LLM_COMPRESS_BENCH_BIN` | Path to `llama-bench`. |
| `LLM_COMPRESS_PPL_BIN` | Path to `llama-perplexity`. |
| `HF_TOKEN` | Hugging Face access token for gated repositories. |
| `HF_ENDPOINT` | Optional Hugging Face-compatible endpoint or mirror. |

## Project structure

```text
src/
├── main.rs          # Entry point and command dispatch
├── cli.rs           # Command-line argument definitions
├── detector.rs      # GGUF metadata parsing and model discovery
├── quantizer.rs     # llama.cpp quantization tool wrapper
├── llama_backend.rs # In-process llama.cpp C API bindings and chat loop
├── engine.rs        # SIMD kernels and Transformer math primitives
└── tokenizer.rs     # GGUF vocabulary inspection and basic token operations

build.rs             # Locate and link llama.cpp shared libraries
tests/               # Integration tests
```

## Development checks

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## License

MIT
# LLM-COMPRESS
