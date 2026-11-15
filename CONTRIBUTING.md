# Contributing

Thanks for helping improve `llm-compress`.

## Development setup

```bash
git clone https://github.com/your-username/llm-compress.git
cd llm-compress
cargo build
```

You also need [llama.cpp](https://github.com/ggerganov/llama.cpp) binaries on
your `PATH` (`llama-quantize`, `llama-imatrix`, `llama-cli`, `llama-bench`,
`llama-perplexity`) to exercise the commands end to end.

## Before sending a pull request

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo +1.85 check --all-targets
```

CI runs the same checks on stable and on the minimum supported Rust version.

## Guidelines

- Keep changes focused and include tests for new parsing or numeric code.
- Run `cargo fmt` before committing.
- Do not commit model files, importance matrices, or anything under `target/`.
- Describe user-visible changes in `CHANGELOG.md`.

## Minimum supported Rust version

The crate targets Rust 1.85. This is enforced by the `rust-version` field in
`Cargo.toml` and by the CI matrix. Bumping it is a breaking change and should be
called out in the changelog.

The lockfile pins a few transitive crates (`idna`, `url`, and the `icu_*`
family) because their latest releases require Rust 1.88. Do not run a blanket
`cargo update`; if you must update dependencies, re-run
`cargo +1.85 check --all-targets` and keep the lockfile changes minimal.
