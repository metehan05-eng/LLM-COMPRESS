use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use colored::Colorize;
use comfy_table::{presets::UTF8_FULL, Table};
use indicatif::{ProgressBar, ProgressStyle};
use llm_compress::{cli, detector, engine, quantizer, tokenizer};
use serde_json::json;
use sysinfo::{Disks, System};

mod llama_backend;

use cli::{Cli, Command, Preset, PresetArg};

fn main() {
    if let Err(err) = real_main() {
        // `{:#}` prints the full context chain.
        eprintln!("{} {err:#}", "error:".red().bold());
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Scan { paths, json } => cmd_scan(&paths, json),
        Command::Compress {
            model,
            preset,
            out,
            imatrix,
            calibration,
            quality_file,
            threads,
            force,
        } => cmd_compress(
            &model,
            preset,
            out,
            imatrix,
            calibration,
            quality_file,
            threads,
            force,
        ),
        Command::Info { model, json } => cmd_info(&model, json),
        Command::Estimate { model, json } => cmd_estimate(&model, json),
        Command::Bench {
            models,
            threads,
            reps,
            output,
        } => cmd_bench(&models, threads, reps, output.as_deref()),
        Command::Perplexity {
            model,
            file,
            baseline,
            threads,
            context,
            chunks,
            json,
        } => cmd_perplexity(
            &model,
            &file,
            baseline.as_deref(),
            threads,
            context,
            chunks,
            json,
        ),
        Command::Pull {
            repo,
            file,
            out,
            name,
            revision,
            token,
            force,
            json,
        } => cmd_pull(&repo, &file, out, name, revision, token, force, json),
        Command::Run {
            model,
            threads,
            context,
            predict,
        } => cmd_run(&model, threads, context, predict),
        Command::Status { json } => cmd_status(json),
        Command::Doctor { json } => cmd_doctor(json),
        Command::Tokenize {
            model,
            text,
            ids,
            vocab_info,
            json,
        } => cmd_tokenize(&model, text.as_deref(), ids.as_deref(), vocab_info, json),
        Command::Clean { paths, yes, json } => cmd_clean(&paths, yes, json),
    }
}

fn cmd_scan(extra: &[PathBuf], json: bool) -> Result<()> {
    let models = detector::scan(extra)?;
    if json {
        let entries: Vec<_> = models.iter().map(model_json).collect();
        println!("{}", serde_json::to_string_pretty(&entries)?);
        return Ok(());
    }
    if models.is_empty() {
        println!("{}", "No models found.".yellow());
        println!(
            "Searched ~/.ollama/models, ~/.cache/huggingface/hub, and any --path directories."
        );
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL).set_header(vec![
        "Name", "Source", "Format", "Arch", "Params", "Quant", "Size",
    ]);
    for m in &models {
        let (arch, params, quant) = match &m.meta {
            Some(meta) => (
                meta.architecture.clone().unwrap_or_else(|| "-".into()),
                format_params(meta.param_count),
                meta.quant_name(),
            ),
            None => ("-".into(), "-".into(), "-".into()),
        };
        table.add_row(vec![
            m.name.clone(),
            m.source.label().to_string(),
            format_label(m.format).to_string(),
            arch,
            params,
            quant,
            format_bytes(m.size_bytes),
        ]);
    }
    println!("{table}");
    println!("{} model(s) found.", models.len().to_string().bold());
    Ok(())
}

fn cmd_info(model: &str, json: bool) -> Result<()> {
    let path = resolve_model(model)?;
    let meta = detector::read_gguf_meta(&path)?;
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    if json {
        let tensors: Vec<_> = meta
            .tensors
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "dims": t.dims,
                    "type": detector::ggml_type_name(t.ggml_type),
                    "elements": t.elements,
                    "offset": t.offset,
                    "size_bytes": detector::tensor_size_bytes(t.ggml_type, t.elements),
                })
            })
            .collect();
        let out = json!({
            "path": path.display().to_string(),
            "size_bytes": size,
            "version": meta.version,
            "architecture": meta.architecture,
            "file_type": meta.file_type,
            "quant": meta.quant_name(),
            "tensor_count": meta.tensor_count,
            "param_count": meta.param_count,
            "context_length": meta.context_length(),
            "block_count": meta.block_count(),
            "embedding_length": meta.embedding_length(),
            "head_count": meta.head_count(),
            "split_count": meta.split_count,
            "split_no": meta.split_no,
            "tensors": tensors,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("{}", path.display().to_string().bold());
    println!("  Size         : {}", format_bytes(size));
    println!("  GGUF version : {}", meta.version);
    println!(
        "  Architecture : {}",
        meta.architecture
            .clone()
            .unwrap_or_else(|| "unknown".into())
    );
    println!("  Quantization : {}", meta.quant_name());
    println!("  Tensors      : {}", meta.tensor_count);
    println!("  Parameters   : {}", format_params(meta.param_count));
    if let Some(ctx) = meta.context_length() {
        println!("  Context      : {ctx} tokens");
    }
    if let Some(blocks) = meta.block_count() {
        println!("  Layers       : {blocks}");
    }
    if let Some(emb) = meta.embedding_length() {
        println!("  Embedding dim: {emb}");
    }
    if let Some(heads) = meta.head_count() {
        println!("  Attn heads   : {heads}");
    }
    if let Some(total) = meta.split_count.filter(|n| *n > 1) {
        println!(
            "  Split        : {}/{}",
            meta.split_no.unwrap_or(0) + 1,
            total
        );
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_header(vec!["Name", "Dimensions", "Type", "Elements", "Size"]);
    for t in &meta.tensors {
        let dims = t
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let size = detector::tensor_size_bytes(t.ggml_type, t.elements)
            .map(format_bytes)
            .unwrap_or_else(|| "-".into());
        table.add_row(vec![
            t.name.clone(),
            dims,
            detector::ggml_type_name(t.ggml_type).to_string(),
            t.elements.to_string(),
            size,
        ]);
    }
    println!("{table}");
    Ok(())
}

fn cmd_estimate(model: &str, json: bool) -> Result<()> {
    let path = resolve_model(model)?;
    let meta = detector::read_gguf_meta(&path)?;
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let ram = available_ram();

    if json {
        let presets: Vec<_> = Preset::all()
            .iter()
            .map(|preset| {
                json!({
                    "preset": preset.llama_name(),
                    "bits_per_weight": preset.bits_per_weight(),
                    "estimated_bytes": preset.estimate_bytes(meta.param_count),
                    "fits_in_ram": ram.map(|r| fits_in_ram(*preset, meta.param_count, r)),
                })
            })
            .collect();
        let out = json!({
            "path": path.display().to_string(),
            "size_bytes": size,
            "param_count": meta.param_count,
            "available_ram_bytes": ram,
            "presets": presets,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("{}", path.display().to_string().bold());
    println!("  Parameters : {}", format_params(meta.param_count));
    println!("  Current    : {}", format_bytes(size));
    match ram {
        Some(bytes) => println!("  Free RAM   : {}", format_bytes(bytes)),
        None => println!("  Free RAM   : unavailable"),
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL).set_header(vec![
        "Preset",
        "Bits/weight",
        "Est. size",
        "vs current",
        "Fits RAM",
    ]);
    for preset in Preset::all() {
        let estimate = preset.estimate_bytes(meta.param_count);
        let ratio = if size > 0 {
            format!("{:.0}%", estimate as f64 / size as f64 * 100.0)
        } else {
            "-".to_string()
        };
        let fits = match ram {
            Some(bytes) if fits_in_ram(preset, meta.param_count, bytes) => "yes",
            Some(_) => "no",
            None => "?",
        };
        table.add_row(vec![
            preset.llama_name().to_string(),
            format!("{:.2}", preset.bits_per_weight()),
            format_bytes(estimate),
            ratio,
            fits.to_string(),
        ]);
    }
    println!("{table}");
    Ok(())
}

fn cmd_bench(
    models: &[String],
    threads: Option<usize>,
    reps: u32,
    output: Option<&str>,
) -> Result<()> {
    let paths = models
        .iter()
        .map(|model| resolve_model(model))
        .collect::<Result<Vec<_>>>()?;
    let tool = quantizer::find_tool("llama-bench", "LLM_COMPRESS_BENCH_BIN")?;

    let mut cmd = std::process::Command::new(&tool);
    for path in &paths {
        cmd.arg("-m").arg(path);
    }
    if let Some(threads) = threads {
        cmd.arg("-t").arg(threads.max(1).to_string());
    }
    cmd.arg("-r").arg(reps.to_string());
    if let Some(format) = output {
        cmd.arg("-o").arg(format);
    }

    println!(
        "{} benchmarking {} model(s) with {}",
        "bench".cyan().bold(),
        paths.len(),
        tool.display()
    );
    let status = cmd
        .status()
        .with_context(|| format!("launching {}", tool.display()))?;
    if !status.success() {
        bail!("llama-bench exited with {status}");
    }
    Ok(())
}

fn cmd_perplexity(
    model: &str,
    file: &Path,
    baseline: Option<&str>,
    threads: Option<usize>,
    context: Option<u32>,
    chunks: Option<u32>,
    json: bool,
) -> Result<()> {
    let path = resolve_model(model)?;
    if !file.is_file() {
        bail!("perplexity text file {} does not exist", file.display());
    }
    let baseline_path = baseline.map(resolve_model).transpose()?;
    let tool = quantizer::find_tool("llama-perplexity", "LLM_COMPRESS_PPL_BIN")?;
    let echo = !json;

    if echo {
        println!(
            "{} measuring perplexity on {}",
            "ppl".cyan().bold(),
            file.display()
        );
    }
    let ppl = run_ppl(&tool, &path, file, threads, context, chunks, echo)?;
    let baseline_ppl = match &baseline_path {
        Some(base) => Some(run_ppl(&tool, base, file, threads, context, chunks, echo)?),
        None => None,
    };
    let delta = baseline_ppl.and_then(|base| {
        if base > 0.0 {
            Some((ppl - base) / base * 100.0)
        } else {
            None
        }
    });

    if json {
        let out = json!({
            "model": path.display().to_string(),
            "file": file.display().to_string(),
            "ppl": ppl,
            "baseline": baseline_path.as_ref().map(|p| p.display().to_string()),
            "baseline_ppl": baseline_ppl,
            "delta_percent": delta,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("{} {}", "model".bold(), path.display());
    println!("  PPL        : {ppl:.4}");
    if let Some(base) = &baseline_path {
        println!(
            "  Baseline   : {} ({:.4})",
            base.display(),
            baseline_ppl.unwrap_or(f64::NAN)
        );
        if let Some(delta) = delta {
            println!("  Delta      : {delta:+.2}%");
        }
    }
    Ok(())
}

fn run_ppl(
    tool: &Path,
    model: &Path,
    file: &Path,
    threads: Option<usize>,
    context: Option<u32>,
    chunks: Option<u32>,
    echo: bool,
) -> Result<f64> {
    let mut cmd = std::process::Command::new(tool);
    cmd.arg("-m").arg(model).arg("-f").arg(file);
    if let Some(threads) = threads {
        cmd.arg("-t").arg(threads.max(1).to_string());
    }
    if let Some(context) = context {
        cmd.arg("-c").arg(context.to_string());
    }
    if let Some(chunks) = chunks {
        cmd.arg("--chunks").arg(chunks.to_string());
    }
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .with_context(|| format!("launching {}", tool.display()))?;
    let stdout = child
        .stdout
        .take()
        .context("capturing llama-perplexity stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("capturing llama-perplexity stderr")?;

    let (tx, rx) = mpsc::channel::<(bool, String)>();
    let tx_err = tx.clone();
    let out_reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(|line| line.ok()) {
            if tx.send((false, line)).is_err() {
                break;
            }
        }
    });
    let err_reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(|line| line.ok()) {
            if tx_err.send((true, line)).is_err() {
                break;
            }
        }
    });

    let mut ppl = None;
    let mut tail = VecDeque::with_capacity(12);
    for (is_stderr, line) in rx {
        if let Some(value) = parse_ppl(&line) {
            ppl = Some(value);
        }
        if tail.len() == 12 {
            tail.pop_front();
        }
        tail.push_back(format!(
            "{}{}",
            if is_stderr { "stderr: " } else { "" },
            line
        ));
        if echo {
            if is_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
        }
    }
    let _ = out_reader.join();
    let _ = err_reader.join();

    let status = child.wait().context("waiting for llama-perplexity")?;
    if !status.success() {
        bail!(
            "llama-perplexity exited with {status}. Last output:\n{}",
            tail.into_iter().collect::<Vec<_>>().join("\n")
        );
    }
    ppl.with_context(|| {
        format!(
            "no 'PPL =' line in llama-perplexity output for {}. Last output:\n{}",
            model.display(),
            tail.into_iter().collect::<Vec<_>>().join("\n")
        )
    })
}

/// Extract the value after `PPL =` from a llama-perplexity result line.
fn parse_ppl(line: &str) -> Option<f64> {
    let (_, rest) = line.split_once("PPL =")?;
    rest.split_whitespace()
        .next()?
        .trim_end_matches(',')
        .parse()
        .ok()
}

fn fits_in_ram(preset: Preset, params: u64, available_ram: u64) -> bool {
    preset.estimate_bytes(params) <= (available_ram as f64 * 0.85) as u64
}

#[allow(clippy::too_many_arguments)]
fn cmd_compress(
    model: &str,
    preset: PresetArg,
    out: Option<PathBuf>,
    imatrix: Option<PathBuf>,
    calibration: Option<PathBuf>,
    quality_file: Option<PathBuf>,
    threads: Option<usize>,
    force: bool,
) -> Result<()> {
    let input = resolve_model(model)?;

    if let Some(cal) = calibration.as_ref().filter(|p| !p.is_file()) {
        bail!("calibration file {} does not exist", cal.display());
    }
    if let Some(im) = imatrix.as_ref().filter(|p| !p.is_file()) {
        bail!("importance matrix {} does not exist", im.display());
    }
    if let Some(file) = quality_file.as_ref().filter(|p| !p.is_file()) {
        bail!("quality evaluation file {} does not exist", file.display());
    }
    if !detector::is_gguf(&input)? {
        bail!(
            "{} is not a GGUF file. Convert Hugging Face safetensors first with \
             llama.cpp's convert_hf_to_gguf.py",
            input.display()
        );
    }

    let meta = detector::read_gguf_meta(&input)?;
    let ram = available_ram();
    let preset = preset.resolve(meta.param_count, ram);
    if preset == Preset::Iq3Xxs && imatrix.is_none() && calibration.is_none() {
        bail!(
            "{} requires an importance matrix. Pass --calibration <text> or --imatrix <file>",
            preset.llama_name()
        );
    }

    let output = out.unwrap_or_else(|| default_output(&input, preset));
    let threads = threads.unwrap_or_else(engine::default_threads).max(1);

    if same_file(&input, &output) {
        bail!(
            "output {} is the same file as the input; choose another --out",
            output.display()
        );
    }
    if output.exists() && !force {
        bail!(
            "{} already exists; pass --force to overwrite it",
            output.display()
        );
    }

    let ppl_tool = quality_file
        .as_ref()
        .map(|_| quantizer::find_tool("llama-perplexity", "LLM_COMPRESS_PPL_BIN"))
        .transpose()?;

    let estimate = preset.estimate_bytes(meta.param_count);
    if let Some(available) = disk_available(output_dir(&output)) {
        if estimate > available {
            bail!(
                "not enough free space for an estimated {} output ({} available on that volume)",
                format_bytes(estimate),
                format_bytes(available)
            );
        }
    }
    if let Some(ram) = ram {
        if estimate > ram {
            eprintln!(
                "warning: the estimated {} output may not fit in {} of free RAM",
                format_bytes(estimate),
                format_bytes(ram)
            );
        }
    }
    println!(
        "{} preset {} -> estimated {}",
        "plan".cyan().bold(),
        preset.llama_name(),
        format_bytes(estimate)
    );

    let generated_imatrix = imatrix.is_none() && calibration.is_some();
    let imatrix_path = match (imatrix, calibration) {
        (Some(p), _) => Some(p),
        (None, Some(text)) => {
            let p = temporary_imatrix_path(&output)?;
            println!(
                "{} building importance matrix -> {}",
                "step".cyan().bold(),
                p.display()
            );
            Some(quantizer::build_imatrix(&input, &text, &p, threads)?)
        }
        (None, None) => None,
    };

    println!(
        "{} {} -> {} ({})",
        "compress".cyan().bold(),
        input.display(),
        output.display(),
        preset.llama_name()
    );

    let compression =
        quantizer::compress(&input, &output, preset, imatrix_path.as_deref(), threads);
    if generated_imatrix {
        if let Some(path) = imatrix_path.as_ref() {
            if let Err(error) = fs::remove_file(path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    eprintln!(
                        "warning: could not remove temporary imatrix {}: {error}",
                        path.display()
                    );
                }
            }
        }
    }
    let (before, after) = compression?;
    let ratio = if before > 0 {
        after as f64 / before as f64 * 100.0
    } else {
        0.0
    };
    let size_change = if after <= before {
        format!("saved {}", format_bytes(before - after))
    } else {
        format!("grew by {}", format_bytes(after - before))
    };
    println!(
        "{} {} -> {} ({:.1}% of original, {})",
        "done".green().bold(),
        format_bytes(before),
        format_bytes(after),
        ratio,
        size_change
    );

    if let (Some(file), Some(tool)) = (quality_file.as_ref(), ppl_tool.as_ref()) {
        println!(
            "{} evaluating perplexity on {}",
            "quality".cyan().bold(),
            file.display()
        );
        let original_ppl = run_ppl(tool, &input, file, Some(threads), None, None, false)
            .with_context(|| {
                format!(
                    "measuring source-model perplexity; quantized output is saved at {}",
                    output.display()
                )
            })?;
        let quantized_ppl = run_ppl(tool, &output, file, Some(threads), None, None, false)
            .with_context(|| {
                format!(
                    "measuring quantized-model perplexity; quantized output is saved at {}",
                    output.display()
                )
            })?;
        let delta_pct = if original_ppl > 0.0 {
            (quantized_ppl - original_ppl) / original_ppl * 100.0
        } else {
            f64::NAN
        };
        println!("  Source PPL    : {original_ppl:.4}");
        println!("  Quantized PPL : {quantized_ppl:.4}");
        if delta_pct.is_finite() {
            println!("  PPL change    : {delta_pct:+.2}% (positive means higher perplexity)");
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_pull(
    repo: &str,
    file: &str,
    out: Option<PathBuf>,
    name: Option<String>,
    revision: Option<String>,
    token: Option<String>,
    force: bool,
    json: bool,
) -> Result<()> {
    let remote_name = Path::new(file)
        .file_name()
        .and_then(|name| name.to_str())
        .context("--file must end with a file name")?;
    let out_name = name.unwrap_or_else(|| remote_name.to_string());
    if Path::new(&out_name).file_name().and_then(|n| n.to_str()) != Some(out_name.as_str()) {
        bail!("--name must be a plain file name, got '{out_name}'");
    }

    let out_dir = out.unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let output = out_dir.join(&out_name);
    if output.exists() && !force {
        bail!(
            "{} already exists; pass --force to overwrite it",
            output.display()
        );
    }

    let revision = revision.unwrap_or_else(|| "main".to_string());
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let endpoint = endpoint.trim_end_matches('/');
    let url = format!(
        "{endpoint}/{repo}/resolve/{revision}/{}",
        file.trim_start_matches('/')
    );

    let mut request = ureq::get(&url);
    if let Some(token) = token.as_deref() {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }

    if !json {
        println!("{} {url}", "pull".cyan().bold());
    }

    let response = request.call().map_err(|error| match error {
        ureq::Error::Status(code, response) => anyhow!(
            "Hugging Face returned HTTP {code} {} for {url}",
            response.status_text()
        ),
        other => anyhow!("request to {url} failed: {other}"),
    })?;

    let total = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    if let Some(total) = total {
        if let Some(available) = disk_available(&out_dir) {
            if total > available {
                bail!(
                    "not enough free space: download is {} but only {} is available",
                    format_bytes(total),
                    format_bytes(available)
                );
            }
        }
        if !json {
            println!("  size       : {}", format_bytes(total));
        }
    }

    let part = output.with_file_name(format!("{out_name}.part"));
    let bar = if json {
        None
    } else {
        Some(progress_bar(total)?)
    };

    let download = (|| -> Result<u64> {
        let mut reader = response.into_reader();
        let mut writer =
            File::create(&part).with_context(|| format!("creating {}", part.display()))?;
        let mut buffer = [0u8; 64 * 1024];
        let mut written = 0u64;
        loop {
            let read = reader
                .read(&mut buffer)
                .with_context(|| format!("downloading {url}"))?;
            if read == 0 {
                break;
            }
            writer.write_all(&buffer[..read])?;
            written += read as u64;
            if let Some(bar) = &bar {
                bar.inc(read as u64);
            }
        }
        Ok(written)
    })();

    if let Some(bar) = &bar {
        bar.finish_and_clear();
    }

    let downloaded = match download {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = fs::remove_file(&part);
            return Err(error);
        }
    };

    match detector::is_gguf(&part) {
        Ok(true) => {}
        Ok(false) => {
            let _ = fs::remove_file(&part);
            bail!(
                "downloaded file is not a valid GGUF; check --repo, --file, --revision, and --token"
            );
        }
        Err(error) => {
            let _ = fs::remove_file(&part);
            return Err(error);
        }
    }

    fs::rename(&part, &output)
        .with_context(|| format!("moving {} into place", output.display()))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "repo": repo,
                "revision": revision,
                "file": file,
                "path": output.display().to_string(),
                "bytes": downloaded,
            }))?
        );
    } else {
        println!(
            "{} {} ({})",
            "downloaded".green().bold(),
            output.display(),
            format_bytes(downloaded)
        );
        println!(
            "  scan with: llm-compress scan --path {}",
            out_dir.display()
        );
    }
    Ok(())
}

fn progress_bar(total: Option<u64>) -> Result<ProgressBar> {
    let bar = match total {
        Some(total) => {
            let bar = ProgressBar::new(total);
            bar.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                )?
                .progress_chars("=>-"),
            );
            bar
        }
        None => {
            let bar = ProgressBar::new_spinner();
            bar.set_style(ProgressStyle::with_template(
                "{spinner:.green} {bytes} downloaded",
            )?);
            bar.enable_steady_tick(Duration::from_millis(120));
            bar
        }
    };
    Ok(bar)
}

fn cmd_run(model: &str, threads: Option<usize>, context: Option<u32>, predict: u32) -> Result<()> {
    let path = resolve_model(model)?;
    if context == Some(0) {
        bail!("--context must be greater than zero");
    }
    if predict == 0 {
        bail!("--predict must be greater than zero");
    }
    let n = threads.unwrap_or_else(engine::default_threads).max(1);
    llama_backend::run_repl(&path, n, context, predict)
}

fn cmd_status(json: bool) -> Result<()> {
    let sys = System::new_all();
    let logical = engine::default_threads();
    let cpu = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty());
    let total_ram = sys.total_memory();
    let available_ram = sys.available_memory();
    let simd = simd_summary();
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let disk = disk_available(&home);

    if json {
        let out = json!({
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "logical_cpus": logical,
            "cpu_model": cpu,
            "simd": simd,
            "ram": { "total_bytes": total_ram, "available_bytes": available_ram },
            "disk": { "path": home.display().to_string(), "available_bytes": disk },
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("{}", "System status".bold());
    println!("  Arch          : {}", std::env::consts::ARCH);
    println!("  OS            : {}", std::env::consts::OS);
    println!("  Logical CPUs  : {logical}");
    println!(
        "  CPU model     : {}",
        cpu.unwrap_or_else(|| "unknown".into())
    );
    println!("  SIMD          : {simd}");
    println!(
        "  RAM           : {} free of {}",
        format_bytes(available_ram),
        format_bytes(total_ram)
    );
    match disk {
        Some(bytes) => println!("  Disk free     : {} (home volume)", format_bytes(bytes)),
        None => println!("  Disk free     : unavailable"),
    }
    Ok(())
}

fn cmd_doctor(json: bool) -> Result<()> {
    let sys = System::new_all();
    let logical = engine::default_threads();
    let cpu = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty());
    let total_ram = sys.total_memory();
    let available_ram = sys.available_memory();
    let simd = simd_summary();
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let disk = disk_available(&home);

    let tools = [
        (
            "llama-quantize",
            "LLM_COMPRESS_QUANTIZE_BIN",
            "Model compression (requantization)",
        ),
        (
            "llama-imatrix",
            "LLM_COMPRESS_IMATRIX_BIN",
            "Importance matrix calibration",
        ),
        (
            "llama-bench",
            "LLM_COMPRESS_BENCH_BIN",
            "Hardware & model throughput benchmarking",
        ),
        (
            "llama-perplexity",
            "LLM_COMPRESS_PPL_BIN",
            "Perplexity & quantization quality testing",
        ),
    ];

    let mut tool_results = Vec::new();
    let mut all_tools_ready = true;

    for (bin, env_var, desc) in &tools {
        match quantizer::find_tool(bin, env_var) {
            Ok(path) => {
                tool_results.push((bin, env_var, desc, Some(path), None));
            }
            Err(e) => {
                all_tools_ready = false;
                tool_results.push((bin, env_var, desc, None, Some(e.to_string())));
            }
        }
    }

    if json {
        let tools_json: Vec<_> = tool_results
            .iter()
            .map(|(bin, env_var, desc, path, err)| {
                json!({
                    "name": bin,
                    "env_var": env_var,
                    "purpose": desc,
                    "available": path.is_some(),
                    "path": path.as_ref().map(|p| p.display().to_string()),
                    "error": err,
                })
            })
            .collect();

        let out = json!({
            "system": {
                "arch": std::env::consts::ARCH,
                "os": std::env::consts::OS,
                "logical_cpus": logical,
                "cpu_model": cpu,
                "simd": simd,
                "ram": { "total_bytes": total_ram, "available_bytes": available_ram },
                "disk": { "path": home.display().to_string(), "available_bytes": disk },
            },
            "llama_cpp_tools": tools_json,
            "ready": all_tools_ready,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("{}", "=== System Diagnostics (Doctor) ===".bold());
    println!("  Arch          : {}", std::env::consts::ARCH);
    println!("  OS            : {}", std::env::consts::OS);
    println!("  Logical CPUs  : {logical}");
    println!(
        "  CPU model     : {}",
        cpu.unwrap_or_else(|| "unknown".into())
    );
    println!("  SIMD          : {simd}");
    println!(
        "  RAM           : {} free / {} total",
        format_bytes(available_ram),
        format_bytes(total_ram)
    );
    match disk {
        Some(bytes) => println!("  Disk free     : {} (home)", format_bytes(bytes)),
        None => println!("  Disk free     : unavailable"),
    }

    println!("\n{}", "=== External Dependencies (llama.cpp) ===".bold());
    let mut ready_count = 0;
    for (bin, env_var, desc, path, _) in &tool_results {
        match path {
            Some(p) => {
                ready_count += 1;
                println!(
                    "  {} {:<18} : {} ({})",
                    "✓".green().bold(),
                    bin,
                    p.display().to_string().cyan(),
                    desc
                );
            }
            None => {
                println!(
                    "  {} {:<18} : {} (or set {}=/path/to/{bin}) - {}",
                    "✗".red().bold(),
                    bin,
                    "missing from PATH".yellow(),
                    env_var,
                    desc
                );
            }
        }
    }

    println!();
    if ready_count == tool_results.len() {
        println!(
            "{}",
            "All dependencies are satisfied! Your system is fully ready for compression and inference."
                .green()
                .bold()
        );
    } else {
        println!(
            "{} {}/{} tools found.",
            "Notice:".yellow().bold(),
            ready_count,
            tool_results.len()
        );
        println!(
            "To install missing llama.cpp tools:\n  - Ubuntu/Debian: sudo apt install llama.cpp (or build from source at https://github.com/ggerganov/llama.cpp)\n  - macOS: brew install llama.cpp\n  - Or build llama.cpp and add its build directory to your PATH, or set the LLM_COMPRESS_*_BIN environment variables."
        );
    }

    Ok(())
}

/// Accepts a path or a name from `scan`. Exact name matches win over substrings.
fn resolve_model(spec: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(spec);
    if direct.is_file() {
        return Ok(direct);
    }
    let models = detector::scan(&detector::extra_paths_from_env())?;
    if let Some(exact) = models.iter().find(|m| m.name == spec) {
        return Ok(exact.path.clone());
    }
    let matches: Vec<_> = models.iter().filter(|m| m.name.contains(spec)).collect();
    match matches.as_slice() {
        [] => bail!("no model matching '{spec}'. Run `llm-compress scan` to list available models"),
        [one] => Ok(one.path.clone()),
        many => bail!(
            "'{spec}' is ambiguous. Matches: {}",
            many.iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn model_json(m: &detector::ModelInfo) -> serde_json::Value {
    let meta = m.meta.as_ref().map(|meta| {
        json!({
            "version": meta.version,
            "architecture": meta.architecture,
            "file_type": meta.file_type,
            "quant": meta.quant_name(),
            "tensor_count": meta.tensor_count,
            "param_count": meta.param_count,
            "split_count": meta.split_count,
            "split_no": meta.split_no,
        })
    });
    json!({
        "name": m.name,
        "path": m.path.display().to_string(),
        "source": m.source.label(),
        "format": format_label(m.format),
        "size_bytes": m.size_bytes,
        "meta": meta,
    })
}

fn format_label(format: detector::Format) -> &'static str {
    match format {
        detector::Format::Gguf => "gguf",
        detector::Format::Safetensors => "safetensors",
    }
}

fn default_output(input: &Path, preset: Preset) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    input.with_file_name(format!("{stem}-{}.gguf", preset.slug()))
}

fn output_dir(output: &Path) -> &Path {
    output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn temporary_imatrix_path(output: &Path) -> Result<PathBuf> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .context("output path has no file name")?
        .to_string_lossy();
    let pid = std::process::id();
    for attempt in 0..100u32 {
        let candidate = parent.join(format!(".{name}.{pid}.{attempt}.imatrix.dat.tmp"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not allocate a temporary imatrix path beside {}",
        output.display()
    )
}

fn same_file(a: &Path, b: &Path) -> bool {
    fn normalized(path: &Path) -> Option<PathBuf> {
        if let Ok(canonical) = path.canonicalize() {
            return Some(canonical);
        }
        let parent = output_dir(path);
        let name = path.file_name()?;
        parent.canonicalize().ok().map(|dir| dir.join(name))
    }
    match (normalized(a), normalized(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn format_params(count: u64) -> String {
    match count {
        0 => "-".into(),
        n if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.0}M", n as f64 / 1e6),
        n => format!("{n}"),
    }
}

#[cfg(target_arch = "x86_64")]
fn simd_summary() -> String {
    let checks = [
        ("AVX2", is_x86_feature_detected!("avx2")),
        ("FMA", is_x86_feature_detected!("fma")),
        ("AVX-512F", is_x86_feature_detected!("avx512f")),
    ];
    checks
        .iter()
        .map(|(name, on)| format!("{name} {}", if *on { "yes" } else { "no" }))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(target_arch = "aarch64")]
fn simd_summary() -> String {
    "NEON yes".to_string()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn simd_summary() -> String {
    "none detected".to_string()
}

/// Free bytes on the volume containing `path`, via sysinfo.
fn disk_available(path: &Path) -> Option<u64> {
    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if path.starts_with(mount) {
            let len = mount.as_os_str().len();
            if best.is_none_or(|(current, _)| len > current) {
                best = Some((len, disk.available_space()));
            }
        }
    }
    best.map(|(_, bytes)| bytes)
}

/// Currently available RAM in bytes, via sysinfo.
fn available_ram() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    Some(sys.available_memory()).filter(|bytes| *bytes > 0)
}

fn cmd_tokenize(
    model: &str,
    text: Option<&str>,
    ids_str: Option<&str>,
    vocab_info: bool,
    json: bool,
) -> Result<()> {
    let path = resolve_model(model)?;
    let tok = tokenizer::GgufTokenizer::from_file(&path)?;

    if vocab_info || (text.is_none() && ids_str.is_none()) {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "model": tok.model,
                    "vocab_size": tok.vocab_size,
                    "bos_token_id": tok.bos_token_id,
                    "eos_token_id": tok.eos_token_id,
                    "unk_token_id": tok.unk_token_id,
                }))?
            );
            return Ok(());
        }
        println!("{}", "=== Model Tokenizer Vocabulary ===".bold());
        println!("  Tokenizer Model : {}", tok.model);
        println!("  Vocab Size      : {}", tok.vocab_size);
        println!("  BOS Token ID    : {:?}", tok.bos_token_id);
        println!("  EOS Token ID    : {:?}", tok.eos_token_id);
        println!("  UNK Token ID    : {:?}", tok.unk_token_id);
        if text.is_none() && ids_str.is_none() {
            println!("\nPass --text \"...\" to encode text or --ids \"...\" to decode token IDs.");
            return Ok(());
        }
    }

    if let Some(t) = text {
        let tokens = tok.encode(t);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "input_text": t,
                    "token_count": tokens.len(),
                    "token_ids": tokens,
                }))?
            );
            return Ok(());
        }
        println!("\n{}", "=== Encoded Tokens ===".bold());
        println!("  Input text  : \"{t}\"");
        println!("  Token count : {}", tokens.len());
        println!("  Token IDs   : {:?}", tokens);
    }

    if let Some(s) = ids_str {
        let parsed_ids: Result<Vec<u32>> = s
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<u32>()
                    .with_context(|| format!("invalid token ID '{part}'"))
            })
            .collect();
        let parsed_ids = parsed_ids?;
        let decoded = tok.decode(&parsed_ids);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "token_ids": parsed_ids,
                    "decoded_text": decoded,
                }))?
            );
            return Ok(());
        }
        println!("\n{}", "=== Decoded Text ===".bold());
        println!("  Token IDs    : {:?}", parsed_ids);
        println!("  Decoded text : \"{decoded}\"");
    }

    Ok(())
}

fn cmd_clean(paths: &[PathBuf], yes: bool, json: bool) -> Result<()> {
    let candidates = detector::find_cleanable_files(paths)?;
    let total_reclaimable: u64 = candidates.iter().map(|c| c.size_bytes).sum();

    if json {
        let items: Vec<_> = candidates
            .iter()
            .map(|c| {
                json!({
                    "path": c.path.display().to_string(),
                    "size_bytes": c.size_bytes,
                    "reason": c.reason,
                })
            })
            .collect();

        if yes {
            let mut deleted_count = 0usize;
            let mut deleted_bytes = 0u64;
            for c in &candidates {
                if fs::remove_file(&c.path).is_ok() {
                    deleted_count += 1;
                    deleted_bytes += c.size_bytes;
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "mode": "deleted",
                    "deleted_count": deleted_count,
                    "reclaimed_bytes": deleted_bytes,
                    "files": items,
                }))?
            );
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "mode": "dry-run",
                    "cleanable_count": candidates.len(),
                    "reclaimable_bytes": total_reclaimable,
                    "files": items,
                }))?
            );
        }
        return Ok(());
    }

    if candidates.is_empty() {
        println!(
            "{}",
            "No temporary or incomplete files found. Disk is clean!"
                .green()
                .bold()
        );
        return Ok(());
    }

    println!(
        "{}",
        "=== Cleanable Temporary / Incomplete Files ===".bold()
    );
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_header(vec!["File", "Size", "Reason"]);

    for c in &candidates {
        table.add_row(vec![
            c.path.display().to_string(),
            format_bytes(c.size_bytes),
            c.reason.to_string(),
        ]);
    }
    println!("{table}");
    println!(
        "\nTotal disk space that can be reclaimed: {}",
        format_bytes(total_reclaimable).bold().cyan()
    );

    if yes {
        let mut deleted_count = 0usize;
        let mut deleted_bytes = 0u64;
        for c in &candidates {
            match fs::remove_file(&c.path) {
                Ok(_) => {
                    deleted_count += 1;
                    deleted_bytes += c.size_bytes;
                }
                Err(e) => {
                    eprintln!("failed to remove {}: {e}", c.path.display());
                }
            }
        }
        println!(
            "\n{}",
            format!(
                "Successfully deleted {} file(s) and reclaimed {}!",
                deleted_count,
                format_bytes(deleted_bytes)
            )
            .green()
            .bold()
        );
    } else {
        println!(
            "\n{} This was a dry run. To permanently delete these files, run:\n  {}",
            "Notice:".yellow().bold(),
            "llm-compress clean --yes".cyan().bold()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_perplexity_lines() {
        assert_eq!(
            parse_ppl("Final estimate: PPL = 5.1234 +/- 0.0321"),
            Some(5.1234)
        );
        assert_eq!(parse_ppl("PPL = 7.5,"), Some(7.5));
        assert_eq!(parse_ppl("no number here"), None);
    }
}
