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
        .status()
        .with_context(|| format!("launching {}", tool.display()))?;

    spinner.finish_and_clear();
    if !status.success() {
        bail!("llama-imatrix failed with {status}");
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
    if let Ok(meta) = crate::detector::read_gguf_meta(input) {
        if let Some(ft) = meta.file_type {
            // llama.cpp only allows quantizing from F32 (0), F16 (1), BF16 (28), or Q8_0 (7)
            if ft != 0 && ft != 1 && ft != 28 && ft != 7 {
                let current_quant = meta.quant_name();
                bail!(
                    "cannot requantize already-quantized model '{}' ({current_quant}).\n\
                     llama.cpp requires an unquantized source model (F16, BF16, or Q8_0) for quantization \
                     because re-quantizing an already lossy format (like {current_quant}) severely degrades quality.",
                    input.display()
                );
            }
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

    let tool = find_tool("llama-quantize", "LLM_COMPRESS_QUANTIZE_BIN")?;
    let mut cmd = Command::new(&tool);
    if let Some(im) = imatrix {
        cmd.arg("--imatrix").arg(im);
    }
    cmd.arg(input)
        .arg(output)
        .arg(preset.llama_name())
        .arg(threads.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    let mut child = cmd
        .spawn()
        .with_context(|| format!("launching {}", tool.display()))?;
    let stdout = child
        .stdout
        .take()
        .context("capturing llama-quantize output")?;

    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} tensors  {elapsed_precise}",
        )?
        .progress_chars("=>-"),
    );

    // Keep the last lines so a failure can be explained.
    let mut tail: VecDeque<String> = VecDeque::with_capacity(20);
    for line in BufReader::new(stdout).lines() {
        let line = line.context("reading llama-quantize output")?;
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

    if !status.success() {
        let _ = fs::remove_file(output);
        bail!(
            "llama-quantize failed with {status}. Last output:\n{}",
            tail.into_iter().collect::<Vec<_>>().join("\n")
        );
    }

    let before = fs::metadata(input)?.len();
    let after = fs::metadata(output)
        .with_context(|| {
            format!(
                "llama-quantize reported success but {} is missing",
                output.display()
            )
        })?
        .len();
    Ok((before, after))
}

/// Parses llama-quantize progress lines of the form `[  12/ 291] ...`.
fn parse_progress(line: &str) -> Option<(u64, u64)> {
    let rest = line.trim_start().strip_prefix('[')?;
    let (inside, _) = rest.split_once(']')?;
    let (done, total) = inside.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}
