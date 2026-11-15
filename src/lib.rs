//! Core library for `llm-compress`.
//!
//! The binary in `main.rs` is a thin CLI around these modules. Keeping the
//! parsing, discovery, quantization, and engine code in a library makes it
//! usable from integration tests and, later, from an embedding API.

pub mod cli;
pub mod detector;
pub mod engine;
pub mod quantizer;
