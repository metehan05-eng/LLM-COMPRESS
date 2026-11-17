//! Compression core.
//!
//! K-quant and IQ-quant encoders are delegated to llama.cpp's `llama-quantize`
//! and `llama-imatrix`, which are the reference implementations. This module
//! handles validation, tool discovery, importance-matrix generation, and
//! progress reporting.
//!
//! Attention and FFN-down tensors receive higher precision automatically under
//! the `_M` presets. llama.cpp applies that policy per layer, so no mixing logic
//! is duplicated here.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli::Preset;
use crate::detector::is_gguf;

/// Locate an external tool: explicit env override first (if valid), then `PATH`.
pub fn find_tool(bin: &str, env_var: &str) -> Result<PathBuf> {
    if let Ok(explicit) = std::env::var(env_var) {
        let p = PathBuf::from(&explicit);
        if p.is_file() {
            return Ok(p);
        }
    }
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!("`{bin}` not found on PATH. Install llama.cpp, or set {env_var}=/path/to/{bin}")
}

/// Generate an importance matrix from a calibration text file.
pub fn build_imatrix(
    model: &Path,
    calibration: &Path,
    out: &Path,
    threads: usize,
) -> Result<PathBuf> {
    if let Some(parent) = out.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating imatrix directory {}", parent.display()))?;
    }
    let tool = find_tool("llama-imatrix", "LLM_COMPRESS_IMATRIX_BIN")?;

    let spinner = ProgressBar::new_spinner();
    spinner.set_message("computing importance matrix (this can take a while)");
    spinner.enable_steady_tick(Duration::from_millis(120));

    let status = Command::new(&tool)
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(calibration)
        .arg("-o")
        .arg(out)
        .arg("-t")
        .arg(threads.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status();

    spinner.finish_and_clear();
    let status = match status {
        Ok(status) => status,
        Err(error) => {
            let _ = fs::remove_file(out);
            return Err(error).with_context(|| format!("launching {}", tool.display()));
        }
    };
    if !status.success() {
        let _ = fs::remove_file(out);
        bail!("llama-imatrix failed with {status}");
    }
    let metadata = match fs::metadata(out) {
        Ok(metadata) => metadata,
        Err(error) => {
            let _ = fs::remove_file(out);
            return Err(error).with_context(|| {
                format!(
                    "llama-imatrix reported success but {} is missing",
                    out.display()
                )
            });
        }
    };
    if metadata.len() == 0 {
        let _ = fs::remove_file(out);
        bail!("llama-imatrix produced an empty file at {}", out.display());
    }
    Ok(out.to_path_buf())
}

/// Requantize `input` into `output`. Returns (original_bytes, output_bytes).
pub fn compress(
    input: &Path,
    output: &Path,
    preset: Preset,
    imatrix: Option<&Path>,
    threads: usize,
) -> Result<(u64, u64)> {
    if !is_gguf(input)? {
        bail!(
            "{} is not a GGUF file. Convert Hugging Face safetensors first with \
             llama.cpp's convert_hf_to_gguf.py",
            input.display()
        );
    }
    let input_meta = crate::detector::read_gguf_meta(input)
        .with_context(|| format!("validating input GGUF {}", input.display()))?;
    if let Some(ft) = input_meta.file_type {
        // llama.cpp strictly requires unquantized float models: F32 (0), F16 (1), or BF16 (28)
        if ft != 0 && ft != 1 && ft != 28 {
            let current_quant = input_meta.quant_name();
            bail!(
                "cannot quantize already-quantized model '{}' ({current_quant}).\n\
                 llama.cpp requires an unquantized source model (f16, bf16, or f32) as the input for quantization \
                 because re-quantizing an already quantized model (like {current_quant}) severely degrades quality and is disabled by llama.cpp.",
                input.display()
            );
        }
    }
    if preset == Preset::Iq3Xxs && imatrix.is_none() {
        bail!(
            "{} requires an importance matrix. Pass --calibration <text> or --imatrix <file>",
            preset.llama_name()
        );
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    let staging = staging_path(output)?;
    let result = (|| -> Result<(u64, u64)> {
        let tool = find_tool("llama-quantize", "LLM_COMPRESS_QUANTIZE_BIN")?;
        let mut cmd = Command::new(&tool);
        if let Some(im) = imatrix {
            cmd.arg("--imatrix").arg(im);
        }
        cmd.arg(input)
            .arg(&staging)
            .arg(preset.llama_name())
            .arg(threads.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("launching {}", tool.display()))?;
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("could not capture llama-quantize output");
            }
        };

        let bar = ProgressBar::new(0);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} tensors  {elapsed_precise}",
            )?
            .progress_chars("=>-"),
        );

        // Keep the last lines so a failure can be explained.
        let mut tail: VecDeque<String> = VecDeque::with_capacity(20);
        let mut read_error = None;
        for line in BufReader::new(stdout).lines() {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    read_error = Some(error);
                    let _ = child.kill();
                    break;
                }
            };
            if let Some((done, total)) = parse_progress(&line) {
                bar.set_length(total);
                bar.set_position(done);
            }
            if tail.len() == 20 {
                tail.pop_front();
            }
            tail.push_back(line);
        }

        let status = child.wait().context("waiting for llama-quantize")?;
        bar.finish_and_clear();
        if let Some(error) = read_error {
            return Err(error).context("reading llama-quantize output");
        }
        if !status.success() {
            bail!(
                "llama-quantize failed with {status}. Last output:\n{}",
                tail.into_iter().collect::<Vec<_>>().join("\n")
            );
        }

        let output_meta = crate::detector::read_gguf_meta(&staging)
            .with_context(|| format!("validating quantized output {}", staging.display()))?;
        if output_meta.param_count != input_meta.param_count {
            bail!(
                "quantized output has {} parameters; expected {} from the input",
                output_meta.param_count,
                input_meta.param_count
            );
        }

        let before = fs::metadata(input)?.len();
        let after = fs::metadata(&staging)
            .with_context(|| format!("quantized output {} is missing", staging.display()))?
            .len();
        fs::rename(&staging, output).with_context(|| {
            format!("could not move validated output into {}", output.display())
        })?;
        Ok((before, after))
    })();

    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .context("output path has no file name")?
        .to_string_lossy();
    let pid = std::process::id();
    for attempt in 0..100u32 {
        let candidate = parent.join(format!(".{name}.{pid}.{attempt}.partial.gguf.tmp"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not allocate a temporary output path beside {}",
        output.display()
    )
}

/// Parses llama-quantize progress lines of the form `[  12/ 291] ...`.
fn parse_progress(line: &str) -> Option<(u64, u64)> {
    let rest = line.trim_start().strip_prefix('[')?;
    let (inside, _) = rest.split_once(']')?;
    let (done, total) = inside.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}
