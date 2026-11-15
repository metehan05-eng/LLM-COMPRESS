# llm-compress

A command-line tool for compressing and running large language models on low-memory hardware, such as systems with 16 GB of RAM and integrated GPUs.

`llm-compress` scans your machine for existing models (Ollama, the Hugging Face cache, and custom directories), requantizes GGUF files to smaller presets (Q4_K_M, Q3_K_M, IQ3_XXS, Q2_K), and launches them through `llama-cli` for CPU inference. It also ships native Q4_0 & Q8_0 SIMD kernels (AVX2/FMA and ARM NEON) and memory-mapping plumbing for an upcoming in-process inference path.

> **Status: prototype.** Discovery, GGUF header parsing, the Q4_0 & Q8_0 AVX2/FMA, ARM NEON, and scalar kernels, and the memory-mapping and thread-pool plumbing are implemented. Quantization and token generation are delegated to llama.cpp. A native transformer forward pass is not implemented yet.

## Features

- **Model discovery:** finds models in `~/.ollama/models`, `~/.cache/huggingface/hub`, and any directories you specify. Ollama digests are resolved to their GGUF blobs, and split GGUF shards are merged into one entry.
- **Header inspection:** reads GGUF metadata (architecture, parameter count, quantization type) without loading tensor data.
- **Tensor detail:** the `info` command lists every tensor with its dimensions, ggml type, element count, and approximate size.
- **Compression:** wraps `llama-quantize` with progress reporting, and supports importance matrices (imatrix) for better low-bit quality.
- **Remote download:** `pull` fetches a GGUF from the Hugging Face Hub with a progress bar, free-space check, optional token for gated repositories, and an atomic `.part` file.
- **Preset planning:** `estimate` shows the projected size of every preset and whether it fits in RAM; `--preset auto` picks the best one that does.
- **Benchmarking:** `bench` wraps `llama-bench` and can compare several model files in one run.
- **Quality check:** `perplexity` runs `llama-perplexity` and reports the PPL delta against a baseline model, such as the F16 original.
- **Fast CPU kernels & Transformer Primitives:** AVX2/FMA (x86_64) and ARM NEON (aarch64 / Apple Silicon) dot products for Q4_0 and Q8_0, plus native RMSNorm, RoPE, Softmax, and SwiGLU building blocks parallelized with `rayon`.
- **Embedded Tokenizer:** `tokenize` reads vocabulary metadata directly from the GGUF header to encode text or decode token IDs without needing external tools.
- **Memory mapping:** models are mapped with `memmap2`, so pages load on demand.
- **Disk management:** `clean` scans for and safely removes interrupted downloads (`.part` files) and temporary files to reclaim disk space.
- **Environment diagnostics:** `doctor` audits your hardware, SIMD instructions, RAM, disk, and checks whether external `llama.cpp` tools are installed.
- **System report:** `status` shows CPU model, SIMD support, RAM, and free disk space.
- **Scripting:** `--json` produces machine-readable output for `scan`, `info`, `estimate`, `perplexity`, `pull`, `status`, `doctor`, `tokenize`, and `clean`.

## How it works

`llm-compress` is a thin orchestration layer around llama.cpp. It does not re-implement the quantizers or the transformer; it locates models, validates them, and invokes the reference tools with the right arguments.

1. **Download (`pull`).** Resolves `https://huggingface.co/<repo>/resolve/<revision>/<file>`, follows redirects to the CDN, streams the body to `<output>.part`, and renames it into place only after the GGUF magic validates. A `Content-Length` header drives the progress bar and a free-space check; `HF_TOKEN` supplies auth for gated repositories, and `HF_ENDPOINT` selects a mirror.
2. **Discovery (`scan`).** Walks `~/.ollama/models`, `~/.cache/huggingface/hub`, and any `--path` directories. For Ollama, each manifest is read and the `application/vnd.ollama.image.model` layer is resolved to its blob. For every GGUF only the header is parsed (magic, version, metadata, tensor descriptors) to report architecture, parameter count, and quantization; tensor data is never loaded. Shards of a split model are grouped via `split.count`/`split.no`.
3. **Inspection (`info`).** Prints the full GGUF metadata, split status, and a tensor table (name, dimensions, ggml type, elements, approximate size). `--json` is available for tooling.
4. **Planning (`estimate`).** Estimates the output size of every preset from the parameter count and bits per weight, shows the ratio against the current file, and marks which presets fit in free RAM.
5. **Compression (`compress`).** Resolves the model name or path, validates inputs, optionally builds an importance matrix with `llama-imatrix` from a calibration text, then runs `llama-quantize <input> <output> <type> <threads>` (prefixed with `--imatrix` when available). With `--preset auto` it picks the highest-quality preset that fits in RAM. Progress is parsed from the tool's `[ n/total ]` output.
6. **Benchmarking (`bench`).** Resolves one or more models and runs `llama-bench`, passing `-m`, `-t`, `-r`, and an optional `-o` format through.
7. **Quality check (`perplexity`).** Runs `llama-perplexity -m <model> -f <text>` and parses the final `PPL =` line. With `--baseline` it runs a reference model too and reports the percentage change.
8. **Run (`run`).** Opens and validates the GGUF header, then starts `llama-cli -m <model> -t <threads> -cnv`, so the full forward pass happens inside llama.cpp. The native Q4_0 kernels and `memmap2` mapping in `engine.rs` are the building blocks for moving inference in-process later.
9. **Status (`status`).** Reports architecture, logical CPU count, AVX2/FMA/AVX-512 support, free RAM, and free disk space through `sysinfo`, so it works on Linux, macOS, and Windows.

### Typical workflow

```bash
# 1. (Optional) Download a GGUF from Hugging Face
llm-compress pull --repo bartowski/Llama-3.2-1B-Instruct-GGUF --file Llama-3.2-1B-Instruct-Q4_K_M.gguf --out ./models

# 2. See what is installed
llm-compress scan --path ./models

# 3. Plan the size (and check RAM headroom)
llm-compress estimate --model llama3:8b

# 4. Shrink a model, letting the tool choose a preset that fits
llm-compress compress --model llama3:8b --preset auto

# 5. Compare throughput of the files you kept
llm-compress bench --model ./llama3-8b-Q4_K_M.gguf --model ./llama3-8b-Q2_K.gguf

# 6. Check quality loss against the original
llm-compress perplexity --model ./llama3-8b-Q2_K.gguf --baseline ./llama3-8b-f16.gguf --file wiki.test.txt

# 7. Run the smaller file
llm-compress run --model ./llama3-8b-Q2_K.gguf --threads 8

# 8. Inspect tokenizer and encode text directly
llm-compress tokenize --model ./llama3-8b-Q2_K.gguf --text "Hello world"

# 9. Check the hardware if you are unsure
llm-compress status

# 10. Verify your system and llama.cpp dependencies
llm-compress doctor

# 11. Clean interrupted downloads and reclaim disk space
llm-compress clean --yes
```

Everything runs locally. The only network access is the optional `pull` command, which downloads from the Hugging Face Hub.

## Requirements

- Rust 1.85 or newer
- [llama.cpp](https://github.com/ggerganov/llama.cpp) binaries on your `PATH`: `llama-quantize`, `llama-imatrix`, `llama-cli`, `llama-bench`, and `llama-perplexity`
- Linux is the primary target. Hardware reporting uses `sysinfo`, so macOS and Windows work too; the SIMD kernels are x86_64-only with a scalar fallback elsewhere.

## Installation

```bash
git clone https://github.com/your-username/llm-compress.git
cd llm-compress
cargo build --release
```

The binary is written to `target/release/llm-compress`.

If the llama.cpp tools are not on your `PATH`, set their locations explicitly:

```bash
export LLM_COMPRESS_QUANTIZE_BIN=/path/to/llama-quantize
export LLM_COMPRESS_IMATRIX_BIN=/path/to/llama-imatrix
export LLM_COMPRESS_CLI_BIN=/path/to/llama-cli
export LLM_COMPRESS_BENCH_BIN=/path/to/llama-bench
export LLM_COMPRESS_PPL_BIN=/path/to/llama-perplexity
```

## Usage

### Download a model

```bash
# Public repository, into ./models
llm-compress pull --repo bartowski/Llama-3.2-1B-Instruct-GGUF \
  --file Llama-3.2-1B-Instruct-Q4_K_M.gguf --out ./models

# Gated repository (reads HF_TOKEN when --token is omitted)
export HF_TOKEN=hf_...
llm-compress pull --repo meta-llama/Llama-3.2-1B-Instruct-GGUF \
  --file Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Pin a revision and rename the result
llm-compress pull --repo org/repo --file models/model.gguf \
  --revision v1.0 --name model-q4.gguf --out ./models --json

# Mirror (e.g. behind a firewall)
export HF_ENDPOINT=https://hf-mirror.com
```

`--file` is the path inside the repository; the output name defaults to the last path segment. Downloads stream to `<name>.part` and are renamed only after the GGUF magic validates, so an interrupted run never leaves a half-written model under the final name. The free-space check uses `Content-Length`, and `--force` overwrites an existing file. For split GGUFs, pass each shard as a separate `--file`; `scan` groups them afterwards.

### Scan for models

```bash
llm-compress scan
llm-compress scan --path /data/gguf --path /mnt/models
llm-compress scan --json
```

Extra directories can also be set with the `LLM_COMPRESS_PATHS` environment variable, separated by colons:

```bash
export LLM_COMPRESS_PATHS=/data/gguf:/mnt/models
```

### Inspect a model

```bash
llm-compress info --model llama3:8b
llm-compress info --model ./model.gguf --json
```

`info` prints GGUF version, architecture, quantization, split status, and a table of every tensor with its dimensions, ggml type, element count, and approximate size.

### Estimate output sizes

```bash
llm-compress estimate --model llama3:8b
llm-compress estimate --model llama3:8b --json
```

`estimate` projects the size of each preset from the parameter count and its typical bits per weight, compares it with the current file, and marks whether it fits in free RAM (with a 15% headroom).

### Compress a model

```bash
# Standard quantization
llm-compress compress --model llama3:8b --preset Q4_K_M

# Let the tool choose the best preset that fits in RAM (never IQ3_XXS)
llm-compress compress --model llama3:8b --preset auto

# Custom output path
llm-compress compress --model ./model-f16.gguf --preset Q3_K_M --out ./small.gguf

# Low-bit quantization with a calibration corpus
llm-compress compress --model llama3:8b --preset IQ3_XXS --calibration calib.txt

# Reuse a precomputed importance matrix
llm-compress compress --model llama3:8b --preset Q2_K --imatrix model.imatrix

# Overwrite an existing output file
llm-compress compress --model llama3:8b --preset Q4_K_M --out small.gguf --force
```

| Option       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `--model`    | File path, or a name shown by `scan` (e.g. `llama3:8b`)                     |
| `--preset`   | `Q4_K_M` (default), `Q3_K_M`, `IQ3_XXS`, `Q2_K`, or `auto`                  |
| `--out`      | Output file. Defaults to `<input>-<preset>.gguf` next to the input          |
| `--imatrix`  | Precomputed importance matrix file. Conflicts with `--calibration`           |
| `--calibration` | Plain-text corpus used to generate an importance matrix                  |
| `--force`    | Overwrite the output file if it already exists                               |
| `--threads`  | Worker threads. Defaults to the logical CPU count (`LLM_COMPRESS_THREADS`)  |

Notes:
- `IQ3_XXS` requires an importance matrix. The command fails early if you do not provide one with `--calibration` or `--imatrix`.
- `--preset auto` picks the highest-quality preset whose estimated size fits in 85% of free RAM. It skips `IQ3_XXS` because that preset needs an importance matrix.
- `compress` refuses to write over its input, missing calibration/imatrix files, and existing outputs without `--force`; it also checks free disk space against a preset size estimate before starting.
- Size estimates are approximate and ignore header and metadata overhead.
- Only GGUF input is accepted. Convert Hugging Face safetensors first with llama.cpp's `convert_hf_to_gguf.py`.

### Benchmark models

```bash
llm-compress bench --model ./llama3-8b-Q4_K_M.gguf
llm-compress bench --model ./llama3-8b-Q4_K_M.gguf --model ./llama3-8b-Q2_K.gguf --reps 5
```

`bench` resolves each `--model` and runs `llama-bench` with `-m`, `-t`, and `-r`. Pass `--output md|csv|json|sql` to forward llama-bench's own `-o` format.

### Check quality

```bash
# Perplexity of one file
llm-compress perplexity --model ./llama3-8b-Q2_K.gguf --file wiki.test.txt

# Compare against the F16 original and print the delta
llm-compress perplexity --model ./llama3-8b-Q2_K.gguf --baseline ./llama3-8b-f16.gguf --file wiki.test.txt

# Machine-readable result
llm-compress perplexity --model ./llama3-8b-Q2_K.gguf --baseline ./llama3-8b-f16.gguf --file wiki.test.txt --json
```

`perplexity` runs `llama-perplexity` (passing `-c` and `--chunks` through when set), parses the final `PPL =` line, and reports the percentage change versus `--baseline`. Lower is better; a large positive delta means the quantization hurt quality.

### Run a model

```bash
llm-compress run --model llama3:8b --threads 8
```

This memory-maps the model, reports its metadata, and starts an interactive chat session through `llama-cli`.

### System status

```bash
llm-compress status
llm-compress status --json
```

## Project structure

```
src/
├── main.rs        # Entry point and command dispatch
├── lib.rs         # Library surface (used by integration tests and embedders)
├── cli.rs         # Command-line definitions (clap)
├── detector.rs    # Model discovery and GGUF header parsing
├── quantizer.rs   # Wrappers around llama-quantize and llama-imatrix
└── engine.rs      # Memory mapping, Q4_0 kernels, thread pool, REPL

tests/
└── gguf_header.rs # Integration test with a synthetic GGUF fixture
```

## Testing

```bash
cargo test
```

The test suite covers AVX2 and scalar Q4_0 kernel agreement, f16 conversion, split GGUF suffix parsing and shard merging, tensor-size block layouts, `auto` preset selection, `PPL =` parsing, and an integration test that parses a synthetic GGUF header through the library API.

## Configuration

| Variable                       | Purpose                                        |
|--------------------------------|------------------------------------------------|
| `LLM_COMPRESS_PATHS`           | Colon-separated extra directories to scan      |
| `LLM_COMPRESS_THREADS`         | Default worker thread count                    |
| `LLM_COMPRESS_QUANTIZE_BIN`    | Path to `llama-quantize`                       |
| `LLM_COMPRESS_IMATRIX_BIN`     | Path to `llama-imatrix`                        |
| `LLM_COMPRESS_CLI_BIN`         | Path to `llama-cli`                            |
| `LLM_COMPRESS_BENCH_BIN`       | Path to `llama-bench`                          |
| `LLM_COMPRESS_PPL_BIN`         | Path to `llama-perplexity`                     |
| `HF_TOKEN`                     | Hugging Face token for `pull` (gated repos)    |
| `HF_ENDPOINT`                  | Hugging Face base URL for `pull` (mirror)      |

## Limitations

- **Inference is delegated.** `run` launches `llama-cli`, which performs the forward pass. The native kernels are not yet connected to a transformer implementation.
- **Thread count.** `--threads` uses logical CPUs, which include SMT siblings. Memory-bound matrix-vector products often run faster with fewer threads, so benchmark on your hardware.
- **Memory mapping.** Pages can be evicted under memory pressure, so models larger than free RAM still run, but may be slower because data is re-read from disk.
- **Platform support.** Hardware reporting now uses `sysinfo` and works on Linux, macOS, and Windows. The SIMD kernels are x86_64-only, with a scalar fallback on other architectures.

## Roadmap

- Validate tensor data offsets against the file size and apply `general.alignment`
- Connect the Q4_0 kernels to attention and feed-forward projections for native inference
- Add K-quant kernels (Q4_K, Q3_K) with 256-weight superblocks
- Physical core detection for thread defaults

## License

MIT