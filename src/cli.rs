//! Command-line surface, defined with clap's derive API.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "llm-compress",
    version,
    about = "Compress and run LLMs on low-memory hardware",
    propagate_version = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Discover local models in Ollama, the Hugging Face cache, and custom paths.
    Scan {
        /// Extra directories to search. Repeatable. Also read from LLM_COMPRESS_PATHS (colon-separated).
        #[arg(
            long = "path",
            value_name = "DIR",
            env = "LLM_COMPRESS_PATHS",
            value_delimiter = ':'
        )]
        paths: Vec<PathBuf>,

        /// Print machine-readable JSON instead of a table.
        #[arg(long)]
        json: bool,
    },

    /// Requantize a GGUF model to a smaller preset.
    Compress {
        /// Model file path, or a name as shown by `scan` (e.g. "llama3:8b").
        #[arg(long, value_name = "PATH_OR_NAME")]
        model: String,

        /// Target quantization preset, or `auto` to fit available RAM.
        #[arg(long, value_enum, default_value = "Q4_K_M")]
        preset: PresetArg,

        /// Output file. Defaults to <input>-<preset>.gguf next to the input.
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,

        /// Precomputed importance matrix (.dat). Mutually exclusive with --calibration.
        #[arg(long, value_name = "FILE", conflicts_with = "calibration")]
        imatrix: Option<PathBuf>,

        /// Plain-text calibration corpus. Generates an imatrix with llama-imatrix first.
        #[arg(long, value_name = "TEXT_FILE")]
        calibration: Option<PathBuf>,

        /// Overwrite the output file if it already exists.
        #[arg(long)]
        force: bool,

        /// Worker threads. Defaults to the logical CPU count.
        #[arg(long, env = "LLM_COMPRESS_THREADS")]
        threads: Option<usize>,
    },

    /// Show detailed GGUF metadata and the tensor table.
    Info {
        /// Model file path, or a name as shown by `scan`.
        #[arg(long, value_name = "PATH_OR_NAME")]
        model: String,

        /// Print machine-readable JSON instead of tables/text.
        #[arg(long)]
        json: bool,
    },

    /// Estimate output sizes for every preset, and whether they fit in RAM.
    Estimate {
        /// Model file path, or a name as shown by `scan`.
        #[arg(long, value_name = "PATH_OR_NAME")]
        model: String,

        /// Print machine-readable JSON instead of a table.
        #[arg(long)]
        json: bool,
    },

    /// Benchmark one or more models with llama-bench.
    Bench {
        /// Model file path or name. Repeat to compare several files.
        #[arg(long = "model", value_name = "PATH_OR_NAME", required = true)]
        models: Vec<String>,

        /// Worker threads. Defaults to llama-bench's own default.
        #[arg(long, env = "LLM_COMPRESS_THREADS")]
        threads: Option<usize>,

        /// Repetitions per test.
        #[arg(long, default_value_t = 3)]
        reps: u32,

        /// Pass-through output format for llama-bench (`md`, `csv`, `json`, `sql`).
        #[arg(long, value_name = "FORMAT")]
        output: Option<String>,
    },

    /// Measure perplexity with llama-perplexity and compare against a baseline.
    Perplexity {
        /// Model file path, or a name as shown by `scan`.
        #[arg(long, value_name = "PATH_OR_NAME")]
        model: String,

        /// Text file used for the perplexity evaluation.
        #[arg(long, value_name = "TEXT_FILE")]
        file: PathBuf,

        /// Reference model (e.g. the F16 original) to compute the PPL delta.
        #[arg(long, value_name = "PATH_OR_NAME")]
        baseline: Option<String>,

        /// Worker threads. Defaults to llama-perplexity's own default.
        #[arg(long, env = "LLM_COMPRESS_THREADS")]
        threads: Option<usize>,

        /// Context size (`-c`).
        #[arg(long, value_name = "N")]
        context: Option<u32>,

        /// Number of chunks to evaluate (`--chunks`).
        #[arg(long, value_name = "N")]
        chunks: Option<u32>,

        /// Print machine-readable JSON instead of text.
        #[arg(long)]
        json: bool,
    },

    /// Download a GGUF file from Hugging Face.
    Pull {
        /// Repository id, e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF".
        #[arg(long, value_name = "ORG/REPO")]
        repo: String,

        /// File path inside the repository, e.g. "Llama-3.2-1B-Instruct-Q4_K_M.gguf".
        #[arg(long, value_name = "PATH")]
        file: String,

        /// Destination directory. Defaults to the current directory.
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,

        /// Output file name. Defaults to the source file name.
        #[arg(long, value_name = "FILE")]
        name: Option<String>,

        /// Hugging Face revision (branch, tag, or commit). Defaults to main.
        #[arg(long, value_name = "REV")]
        revision: Option<String>,

        /// Hugging Face access token for gated repositories.
        #[arg(long, env = "HF_TOKEN", hide_env_values = true)]
        token: Option<String>,

        /// Overwrite the output file if it already exists.
        #[arg(long)]
        force: bool,

        /// Print machine-readable JSON instead of text.
        #[arg(long)]
        json: bool,
    },

    /// Memory-map a model and start an interactive chat session.
    Run {
        /// Model file path, or a name as shown by `scan`.
        #[arg(long, value_name = "PATH_OR_NAME")]
        model: String,

        /// Worker threads. Defaults to the logical CPU count.
        #[arg(long, env = "LLM_COMPRESS_THREADS")]
        threads: Option<usize>,
    },

    /// Show RAM, CPU architecture, SIMD support, and free disk space.
    Status {
        /// Print machine-readable JSON instead of tables/text.
        #[arg(long)]
        json: bool,
    },

    /// Diagnose system environment, hardware capabilities, and llama.cpp dependencies.
    Doctor {
        /// Print machine-readable JSON instead of tables/text.
        #[arg(long)]
        json: bool,
    },

    /// Encode text to token IDs or decode token IDs using the GGUF embedded tokenizer.
    Tokenize {
        /// Model file path, or a name as shown by `scan`.
        #[arg(long, value_name = "PATH_OR_NAME")]
        model: String,

        /// Text to encode into token IDs.
        #[arg(long, value_name = "TEXT")]
        text: Option<String>,

        /// Comma-separated token IDs to decode back into text.
        #[arg(long, value_name = "ID,...")]
        ids: Option<String>,

        /// Show vocabulary metadata (size, model type, special tokens).
        #[arg(long)]
        vocab_info: bool,

        /// Print machine-readable JSON instead of tables/text.
        #[arg(long)]
        json: bool,
    },
}

/// Preset as written on the command line, including `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PresetArg {
    #[value(name = "auto")]
    Auto,
    #[value(name = "Q4_K_M")]
    Q4KM,
    #[value(name = "Q3_K_M")]
    Q3KM,
    #[value(name = "IQ3_XXS")]
    Iq3Xxs,
    #[value(name = "Q2_K")]
    Q2K,
}

impl PresetArg {
    /// Resolve `auto` to the highest-quality preset that fits in RAM.
    pub fn resolve(self, params: u64, available_ram: Option<u64>) -> Preset {
        match self {
            PresetArg::Auto => auto_select(params, available_ram),
            PresetArg::Q4KM => Preset::Q4KM,
            PresetArg::Q3KM => Preset::Q3KM,
            PresetArg::Iq3Xxs => Preset::Iq3Xxs,
            PresetArg::Q2K => Preset::Q2K,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Q4KM,
    Q3KM,
    Iq3Xxs,
    Q2K,
}

impl Preset {
    pub fn all() -> [Preset; 4] {
        [Preset::Q4KM, Preset::Q3KM, Preset::Iq3Xxs, Preset::Q2K]
    }

    /// The exact type name llama-quantize expects.
    pub fn llama_name(self) -> &'static str {
        match self {
            Preset::Q4KM => "Q4_K_M",
            Preset::Q3KM => "Q3_K_M",
            Preset::Iq3Xxs => "IQ3_XXS",
            Preset::Q2K => "Q2_K",
        }
    }

    /// Lowercase form used in default output filenames.
    pub fn slug(self) -> String {
        self.llama_name().to_lowercase()
    }

    /// Typical bits per weight for this preset in llama.cpp.
    pub fn bits_per_weight(self) -> f64 {
        match self {
            Preset::Q4KM => 4.85,
            Preset::Q3KM => 3.91,
            Preset::Iq3Xxs => 3.06,
            Preset::Q2K => 2.63,
        }
    }

    /// Rough on-disk size estimate for a model with `params` weights.
    pub fn estimate_bytes(self, params: u64) -> u64 {
        let bytes = params as f64 * self.bits_per_weight() / 8.0;
        (bytes * 1.05) as u64
    }
}

/// `auto` skips IQ3_XXS because it would need an importance matrix.
fn auto_select(params: u64, available_ram: Option<u64>) -> Preset {
    let budget = available_ram.map(|ram| (ram as f64 * 0.85) as u64);
    for preset in [Preset::Q4KM, Preset::Q3KM, Preset::Q2K] {
        if budget.is_none_or(|limit| preset.estimate_bytes(params) <= limit) {
            return preset;
        }
    }
    Preset::Q2K
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: u64 = 8_000_000_000;

    #[test]
    fn auto_without_ram_picks_best_quality() {
        assert_eq!(PresetArg::Auto.resolve(PARAMS, None), Preset::Q4KM);
    }

    #[test]
    fn auto_fits_for_available_ram() {
        assert_eq!(
            PresetArg::Auto.resolve(PARAMS, Some(7_000_000_000)),
            Preset::Q4KM
        );
        assert_eq!(
            PresetArg::Auto.resolve(PARAMS, Some(5_000_000_000)),
            Preset::Q3KM
        );
        assert_eq!(
            PresetArg::Auto.resolve(PARAMS, Some(4_500_000_000)),
            Preset::Q2K
        );
    }

    #[test]
    fn explicit_presets_are_not_rewritten() {
        assert_eq!(PresetArg::Iq3Xxs.resolve(PARAMS, Some(1)), Preset::Iq3Xxs);
    }
}
