//! Model auto-discovery and GGUF header parsing.
//!
//! Only the header is read: key/value metadata and tensor descriptors. Tensor
//! data is never loaded, so scanning a 70 GB model costs a few kilobytes of I/O.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING_LEN: u64 = 1 << 24;
/// Upper bounds that keep a corrupt header from spinning for a long time.
const MAX_KV_COUNT: u64 = 1 << 20;
const MAX_TENSOR_COUNT: u64 = 1 << 24;
const MAX_ARRAY_LEN: u64 = 1 << 24;
const MODEL_LAYER_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Ollama,
    HuggingFace,
    Local,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Ollama => "ollama",
            Source::HuggingFace => "hugging-face",
            Source::Local => "local",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Gguf,
    Safetensors,
}

/// One tensor descriptor from the GGUF header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub elements: u64,
}

/// Metadata extracted from a GGUF header.
#[derive(Debug, Clone)]
pub struct GgufMeta {
    pub version: u32,
    pub architecture: Option<String>,
    /// Raw `general.file_type` value (llama_ftype enum).
    pub file_type: Option<u32>,
    pub tensor_count: u64,
    /// Sum of all tensor element counts, i.e. the parameter count.
    pub param_count: u64,
    /// Number of files this model is split across (`split.count`).
    pub split_count: Option<u32>,
    /// Index of this shard (`split.no`).
    pub split_no: Option<u32>,
    /// `general.alignment`; the data section starts on this boundary.
    pub alignment: u32,
    /// Absolute byte offset of the tensor data section.
    pub data_offset: u64,
    /// All scalar integer metadata, keyed by GGUF key.
    pub ints: BTreeMap<String, i64>,
    /// All scalar floating-point metadata, keyed by GGUF key.
    pub floats: BTreeMap<String, f64>,
    pub tensors: Vec<TensorInfo>,
}

impl Default for GgufMeta {
    fn default() -> Self {
        Self {
            version: 0,
            architecture: None,
            file_type: None,
            tensor_count: 0,
            param_count: 0,
            split_count: None,
            split_no: None,
            alignment: 32,
            data_offset: 0,
            ints: BTreeMap::new(),
            floats: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }
}

impl GgufMeta {
    pub fn quant_name(&self) -> String {
        match self.file_type {
            Some(ft) => file_type_name(ft).to_string(),
            None => "unknown".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub source: Source,
    pub format: Format,
    pub meta: Option<GgufMeta>,
}

#[derive(Deserialize)]
struct Manifest {
    layers: Vec<ManifestLayer>,
}

#[derive(Deserialize)]
struct ManifestLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    size: u64,
}

/// Extra search directories from `LLM_COMPRESS_PATHS` (colon-separated).
pub fn extra_paths_from_env() -> Vec<PathBuf> {
    std::env::var("LLM_COMPRESS_PATHS")
        .map(|v| {
            v.split(':')
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

/// Scan the default locations plus any extra directories.
pub fn scan(extra_paths: &[PathBuf]) -> Result<Vec<ModelInfo>> {
    let home = dirs::home_dir().context("could not determine the home directory")?;
    let mut models = Vec::new();

    let ollama = home.join(".ollama").join("models");
    if ollama.is_dir() {
        models.extend(
            scan_ollama(&ollama)
                .with_context(|| format!("scanning Ollama store at {}", ollama.display()))?,
        );
    }

    let hf = home.join(".cache").join("huggingface").join("hub");
    if hf.is_dir() {
        models.extend(
            scan_huggingface(&hf)
                .with_context(|| format!("scanning Hugging Face cache at {}", hf.display()))?,
        );
    }

    for dir in extra_paths {
        if dir.is_dir() {
            models.extend(scan_local(dir)?);
        } else {
            eprintln!("warning: skipping {} (not a directory)", dir.display());
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(group_splits(models))
}

/// Strip a `-00001-of-00005` suffix from a shard file stem.
fn split_base_stem(stem: &str) -> Option<&str> {
    let (rest, total) = stem.rsplit_once('-')?;
    if total.is_empty() || !total.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (rest, of) = rest.rsplit_once('-')?;
    if of != "of" {
        return None;
    }
    let (base, index) = rest.rsplit_once('-')?;
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(base)
}

fn split_group_key(m: &ModelInfo) -> Option<String> {
    let meta = m.meta.as_ref()?;
    if meta.split_count.unwrap_or(1) <= 1 {
        return None;
    }
    let stem = m.path.file_stem()?.to_string_lossy().into_owned();
    let base = split_base_stem(&stem).unwrap_or(&stem).to_string();
    let parent = m.path.parent().unwrap_or_else(|| Path::new("."));
    Some(format!("{}::{}", parent.display(), base))
}

/// Merge shards of a split GGUF into a single logical model. The first shard is
/// kept as the path, because llama.cpp auto-discovers the remaining shards.
fn group_splits(models: Vec<ModelInfo>) -> Vec<ModelInfo> {
    use std::collections::BTreeMap;

    let mut singles = Vec::new();
    let mut groups: BTreeMap<String, Vec<ModelInfo>> = BTreeMap::new();
    for model in models {
        match split_group_key(&model) {
            Some(key) => groups.entry(key).or_default().push(model),
            None => singles.push(model),
        }
    }

    for (_, mut shards) in groups {
        shards.sort_by_key(|m| m.meta.as_ref().and_then(|x| x.split_no).unwrap_or(0));
        let mut merged = shards.remove(0);
        let stem = merged
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned());
        if let Some(stem) = stem {
            if let Some(base) = split_base_stem(&stem) {
                merged.name = base.to_string();
            }
        }
        for shard in shards {
            merged.size_bytes = merged.size_bytes.saturating_add(shard.size_bytes);
            if let (Some(meta), Some(other)) = (merged.meta.as_mut(), shard.meta) {
                meta.tensor_count = meta.tensor_count.saturating_add(other.tensor_count);
                meta.param_count = meta.param_count.saturating_add(other.param_count);
                meta.tensors.extend(other.tensors);
            }
        }
        if let Some(meta) = merged.meta.as_mut() {
            meta.split_no = Some(0);
        }
        singles.push(merged);
    }

    singles
}

/// Ollama stores manifests at `manifests/<host>/<namespace>/<model>/<tag>`
/// and content blobs at `blobs/sha256-<hex>`. The model weights are the layer
/// with the `application/vnd.ollama.image.model` media type.
fn scan_ollama(root: &Path) -> Result<Vec<ModelInfo>> {
    let manifests = root.join("manifests");
    let blobs = root.join("blobs");
    let mut out = Vec::new();

    for manifest_path in collect_files(&manifests, 6)? {
        let raw = match fs::read(&manifest_path) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let manifest: Manifest = match serde_json::from_slice(&raw) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let Some(layer) = manifest
            .layers
            .iter()
            .find(|l| l.media_type == MODEL_LAYER_MEDIA_TYPE)
        else {
            continue;
        };

        let blob = blobs.join(layer.digest.replace(':', "-"));
        if !blob.is_file() {
            continue;
        }

        let rel = manifest_path.strip_prefix(&manifests)?;
        let parts: Vec<String> = rel
            .iter()
            .map(|c| c.to_string_lossy().into_owned())
            .collect();
        let name = match parts.as_slice() {
            [.., repo, tag] => format!("{repo}:{tag}"),
            _ => continue,
        };

        if let Some(info) = build_info(name, blob, layer.size, Source::Ollama)? {
            out.push(info);
        }
    }
    Ok(out)
}

/// Hugging Face cache layout: `models--<org>--<repo>/snapshots/<rev>/<file>`.
fn scan_huggingface(root: &Path) -> Result<Vec<ModelInfo>> {
    let mut out = Vec::new();

    for entry in fs::read_dir(root)?.flatten() {
        let dir_name = entry.file_name().to_string_lossy().into_owned();
        let Some(encoded) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo = encoded.replacen("--", "/", 1);

        for file in collect_files(&entry.path().join("snapshots"), 4)? {
            let is_model = file.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                e.eq_ignore_ascii_case("gguf") || e.eq_ignore_ascii_case("safetensors")
            });
            if !is_model {
                continue;
            }
            // metadata() follows symlinks; skip dangling ones.
            let Ok(md) = fs::metadata(&file) else {
                continue;
            };
            let file_name = file
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let Some(info) = build_info(
                format!("{repo}/{file_name}"),
                file,
                md.len(),
                Source::HuggingFace,
            )? {
                out.push(info);
            }
        }
    }
    Ok(out)
}

fn scan_local(dir: &Path) -> Result<Vec<ModelInfo>> {
    let mut out = Vec::new();
    for file in collect_files(dir, 4)? {
        let is_model = file.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            e.eq_ignore_ascii_case("gguf") || e.eq_ignore_ascii_case("safetensors")
        });
        if !is_model {
            continue;
        }
        let Ok(md) = fs::metadata(&file) else {
            continue;
        };
        let name = file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Some(info) = build_info(name, file, md.len(), Source::Local)? {
            out.push(info);
        }
    }
    Ok(out)
}

/// Depth-limited, iterative file walk. Unreadable directories are skipped
/// rather than aborting the whole scan.
fn collect_files(dir: &Path, max_depth: usize) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.is_dir() {
        return Ok(files);
    }
    let mut stack = vec![(dir.to_path_buf(), 0usize)];
    while let Some((current, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if depth < max_depth {
                    stack.push((path, depth + 1));
                }
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    Ok(files)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanCandidate {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub reason: &'static str,
}

/// Scan for incomplete downloads and orphaned temporary files.
pub fn find_cleanable_files(extra_paths: &[PathBuf]) -> Result<Vec<CleanCandidate>> {
    let mut candidates = Vec::new();
    let mut search_dirs = Vec::new();

    search_dirs.push(PathBuf::from("."));
    search_dirs.extend_from_slice(extra_paths);

    if let Some(home) = dirs::home_dir() {
        let hf_hub = home.join(".cache").join("huggingface").join("hub");
        if hf_hub.is_dir() {
            search_dirs.push(hf_hub);
        }
    }

    for dir in search_dirs {
        if !dir.is_dir() {
            continue;
        }
        for file in collect_files(&dir, 4)? {
            let file_name = file
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default();

            let reason = if file_name.ends_with(".part") {
                Some("Incomplete download (.part)")
            } else if file_name.ends_with(".tmp") || file_name.ends_with(".temp") {
                Some("Temporary file (.tmp)")
            } else if file_name.ends_with(".imatrix.dat.tmp") {
                Some("Incomplete importance matrix (.tmp)")
            } else {
                None
            };

            if let Some(reason) = reason {
                if let Ok(md) = fs::metadata(&file) {
                    candidates.push(CleanCandidate {
                        path: file,
                        size_bytes: md.len(),
                        reason,
                    });
                }
            }
        }
    }

    candidates.sort_by_key(|a| std::cmp::Reverse(a.size_bytes));
    candidates.dedup_by(|a, b| a.path == b.path);
    Ok(candidates)
}

/// Returns true if the file begins with the GGUF magic bytes.
pub fn is_gguf(path: &Path) -> Result<bool> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut magic = [0u8; 4];
    Ok(f.read_exact(&mut magic).is_ok() && &magic == GGUF_MAGIC)
}

fn sniff_format(path: &Path) -> Option<Format> {
    if is_gguf(path).unwrap_or(false) {
        return Some(Format::Gguf);
    }
    let is_st = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    is_st.then_some(Format::Safetensors)
}

fn build_info(
    name: String,
    path: PathBuf,
    size_bytes: u64,
    source: Source,
) -> Result<Option<ModelInfo>> {
    let Some(format) = sniff_format(&path) else {
        return Ok(None);
    };
    let meta = match format {
        Format::Gguf => match read_gguf_meta(&path) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("warning: could not parse {}: {e:#}", path.display());
                None
            }
        },
        Format::Safetensors => None,
    };
    Ok(Some(ModelInfo {
        name,
        path,
        size_bytes,
        source,
        format,
        meta,
    }))
}

/// Parse a GGUF v2/v3 header. Stops after tensor descriptors; tensor data is untouched.
pub fn read_gguf_meta(path: &Path) -> Result<GgufMeta> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::with_capacity(64 * 1024, file);

    let magic: [u8; 4] = read_array(&mut r)?;
    if &magic != GGUF_MAGIC {
        bail!("bad magic bytes, not a GGUF file");
    }
    let version = read_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version} (expected 2 or 3)");
    }
    let tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if kv_count > MAX_KV_COUNT {
        bail!("GGUF header declares {kv_count} metadata entries (limit {MAX_KV_COUNT})");
    }
    if tensor_count > MAX_TENSOR_COUNT {
        bail!("GGUF header declares {tensor_count} tensors (limit {MAX_TENSOR_COUNT})");
    }

    let mut meta = GgufMeta {
        version,
        tensor_count,
        ..Default::default()
    };

    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let ty = read_u32(&mut r)?;
        let keep_string = key == "general.architecture";
        let value = read_value(&mut r, ty, keep_string)?;
        match &value {
            Value::Int(v) => {
                meta.ints.insert(key.clone(), *v);
            }
            Value::Float(v) => {
                meta.floats.insert(key.clone(), *v);
            }
            Value::Str(_) | Value::Other => {}
        }
        match (key.as_str(), value) {
            ("general.architecture", Value::Str(s)) => meta.architecture = Some(s),
            ("general.file_type", Value::Int(v)) => meta.file_type = Some(v as u32),
            ("split.count", Value::Int(v)) => meta.split_count = Some(v as u32),
            ("split.no", Value::Int(v)) => meta.split_no = Some(v as u32),
            _ => {}
        }
    }

    meta.tensors.reserve(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        if n_dims > 4 {
            bail!("tensor reports {n_dims} dimensions (GGML maximum is 4)");
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = read_u64(&mut r)?;
            dims.push(dim);
            elements = elements.saturating_mul(dim);
        }
        let ggml_type = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;
        meta.param_count = meta.param_count.saturating_add(elements);
        meta.tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            elements,
        });
    }

    let header_end = r
        .stream_position()
        .context("reading the GGUF header position")?;
    let alignment = meta
        .ints
        .get("general.alignment")
        .copied()
        .filter(|value| *value > 0 && *value <= (1 << 20))
        .unwrap_or(32) as u64;
    meta.alignment = alignment as u32;
    meta.data_offset = header_end + (alignment - header_end % alignment) % alignment;

    Ok(meta)
}

enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Other,
}

/// Reads one metadata value. Strings are kept only when `keep` is set.
fn read_value<R: Read>(r: &mut R, ty: u32, keep: bool) -> Result<Value> {
    Ok(match ty {
        0 => Value::Int(read_array::<1, _>(r)?[0] as i64), // UINT8
        1 => Value::Int(i8::from_le_bytes(read_array(r)?) as i64), // INT8
        2 => Value::Int(u16::from_le_bytes(read_array(r)?) as i64), // UINT16
        3 => Value::Int(i16::from_le_bytes(read_array(r)?) as i64), // INT16
        4 => Value::Int(read_u32(r)? as i64),              // UINT32
        5 => Value::Int(i32::from_le_bytes(read_array(r)?) as i64), // INT32
        6 => Value::Float(f32::from_le_bytes(read_array(r)?) as f64), // FLOAT32
        7 => Value::Int(read_array::<1, _>(r)?[0] as i64), // BOOL
        10 => Value::Int(read_u64(r)? as i64),             // UINT64
        11 => Value::Int(i64::from_le_bytes(read_array(r)?)), // INT64
        12 => Value::Float(f64::from_le_bytes(read_array(r)?)), // FLOAT64
        8 => {
            if keep {
                Value::Str(read_string(r)?)
            } else {
                skip_string(r)?;
                Value::Other
            }
        }
        9 => {
            // ARRAY: element type, count, then elements. Skipped without allocation.
            let elem_ty = read_u32(r)?;
            let len = read_u64(r)?;
            if len > MAX_ARRAY_LEN {
                bail!("GGUF array length {len} exceeds the {MAX_ARRAY_LEN}-element limit");
            }
            for _ in 0..len {
                read_value(r, elem_ty, false)?;
            }
            Value::Other
        }
        other => bail!("unknown GGUF value type {other}"),
    })
}

fn read_array<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)
        .context("unexpected end of GGUF header (file truncated?)")?;
    Ok(buf)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_array(r)?))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING_LEN {
        bail!("GGUF string length {len} exceeds the {MAX_STRING_LEN}-byte limit");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).context("GGUF file truncated")?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_string<R: Read>(r: &mut R) -> Result<()> {
    let len = read_u64(r)?;
    if len > MAX_STRING_LEN {
        bail!("GGUF string length {len} exceeds the limit");
    }
    skip(r, len)
}

fn skip<R: Read>(r: &mut R, n: u64) -> Result<()> {
    let copied = io::copy(&mut r.by_ref().take(n), &mut io::sink()).context("reading GGUF")?;
    if copied != n {
        bail!("GGUF file truncated");
    }
    Ok(())
}

/// Maps `general.file_type` to its llama.cpp name.
pub fn file_type_name(ft: u32) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        _ => "unknown",
    }
}

/// Maps a GGUF tensor `ggml_type` to its llama.cpp name.
pub fn ggml_type_name(ty: u32) -> &'static str {
    match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "unknown",
    }
}

/// Approximate on-disk size of a tensor with `elements` weights of type `ty`.
pub fn tensor_size_bytes(ty: u32, elements: u64) -> Option<u64> {
    let (block, bytes) = match ty {
        0 => (1, 4),      // F32
        1 | 30 => (1, 2), // F16, BF16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        _ => return None,
    };
    let blocks = elements.div_ceil(block);
    Some(blocks.saturating_mul(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_suffix_is_stripped() {
        assert_eq!(split_base_stem("model-00001-of-00005"), Some("model"));
        assert_eq!(split_base_stem("llama-3-1-of-4"), Some("llama-3"));
        assert_eq!(split_base_stem("model"), None);
        assert_eq!(split_base_stem("model-of-00005"), None);
    }

    #[test]
    fn tensor_sizes_match_block_layouts() {
        assert_eq!(tensor_size_bytes(0, 8), Some(32));
        assert_eq!(tensor_size_bytes(2, 64), Some(36));
        assert_eq!(tensor_size_bytes(12, 512), Some(288));
        assert_eq!(tensor_size_bytes(999, 512), None);
    }

    #[test]
    fn split_shards_merge_into_one_model() {
        use std::path::PathBuf;

        let shard = |name: &str, no: u32| ModelInfo {
            name: name.to_string(),
            path: PathBuf::from(format!("/tmp/{name}")),
            size_bytes: 10,
            source: Source::Local,
            format: Format::Gguf,
            meta: Some(GgufMeta {
                split_count: Some(2),
                split_no: Some(no),
                tensor_count: 1,
                param_count: 5,
                ..Default::default()
            }),
        };

        let merged = group_splits(vec![
            shard("model-00002-of-00002.gguf", 1),
            shard("model-00001-of-00002.gguf", 0),
        ]);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "model");
        assert_eq!(merged[0].size_bytes, 20);
        assert_eq!(
            merged[0].path,
            PathBuf::from("/tmp/model-00001-of-00002.gguf")
        );
        let meta = merged[0].meta.as_ref().unwrap();
        assert_eq!(meta.param_count, 10);
        assert_eq!(meta.tensor_count, 2);
        assert_eq!(meta.split_no, Some(0));
    }

    #[test]
    fn find_cleanable_files_detects_part_files() {
        let temp_dir = std::env::temp_dir().join(format!("clean-test-{}", std::process::id()));
        fs::create_dir_all(&temp_dir).unwrap();

        let part_file = temp_dir.join("model.gguf.part");
        fs::write(&part_file, vec![0u8; 1024]).unwrap();

        let normal_file = temp_dir.join("model.gguf");
        fs::write(&normal_file, vec![0u8; 2048]).unwrap();

        let candidates = find_cleanable_files(std::slice::from_ref(&temp_dir)).unwrap();
        let found = candidates.iter().any(|c| c.path == part_file);
        assert!(found, "should detect .part file as cleanable candidate");

        let found_normal = candidates.iter().any(|c| c.path == normal_file);
        assert!(
            !found_normal,
            "should NOT mark completed .gguf file as cleanable"
        );

        fs::remove_dir_all(&temp_dir).ok();
    }
}
