//! GGUF embedded tokenizer extraction and text encoding/decoding.
//!
//! GGUF files contain the full model vocabulary and tokenizer configuration
//! embedded directly in the header metadata (e.g. `tokenizer.ggml.tokens`,
//! `tokenizer.ggml.model`, `tokenizer.ggml.bos_token_id`). This module extracts
//! and uses this embedded tokenizer without needing external Python or llama.cpp tools.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::detector::GGUF_MAGIC;

const MAX_STRING_LEN: u64 = 1 << 24;
const MAX_KV_COUNT: u64 = 1 << 20;
const MAX_ARRAY_LEN: u64 = 1 << 24;

/// Tokenizer configuration and vocabulary extracted from a GGUF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTokenizer {
    pub model: String,
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub tokens: Vec<String>,
    #[serde(skip)]
    pub token_to_id: HashMap<String, u32>,
}

impl GgufTokenizer {
    /// Extract the tokenizer from a GGUF file header.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .context("reading GGUF magic")?;
        if &magic != GGUF_MAGIC {
            bail!("{} is not a GGUF file (bad magic)", path.display());
        }

        let version = read_u32(&mut reader)?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version} in {}", path.display());
        }

        let _tensor_count = read_u64(&mut reader)?;
        let kv_count = read_u64(&mut reader)?;
        if kv_count > MAX_KV_COUNT {
            bail!("GGUF metadata count {kv_count} exceeds limit");
        }

        let mut model = "unknown".to_string();
        let mut bos_token_id = None;
        let mut eos_token_id = None;
        let mut unk_token_id = None;
        let mut tokens = Vec::new();

        for _ in 0..kv_count {
            let key = read_string(&mut reader)?;
            let val_type = read_u32(&mut reader)?;

            match key.as_str() {
                "tokenizer.ggml.model" if val_type == 8 => {
                    model = read_string(&mut reader)?;
                }
                "tokenizer.ggml.bos_token_id" => {
                    bos_token_id = read_int_val(&mut reader, val_type).map(|v| v as u32);
                }
                "tokenizer.ggml.eos_token_id" => {
                    eos_token_id = read_int_val(&mut reader, val_type).map(|v| v as u32);
                }
                "tokenizer.ggml.unknown_token_id" => {
                    unk_token_id = read_int_val(&mut reader, val_type).map(|v| v as u32);
                }
                "tokenizer.ggml.tokens" if val_type == 9 => {
                    let elem_ty = read_u32(&mut reader)?;
                    let len = read_u64(&mut reader)?;
                    if len > MAX_ARRAY_LEN {
                        bail!("tokens array length {len} exceeds limit");
                    }
                    if elem_ty == 8 {
                        tokens.reserve(len as usize);
                        for _ in 0..len {
                            tokens.push(read_string(&mut reader)?);
                        }
                    } else {
                        skip_array(&mut reader, elem_ty, len)?;
                    }
                }
                _ => {
                    skip_value(&mut reader, val_type)?;
                }
            }
        }

        if tokens.is_empty() {
            bail!(
                "model '{}' does not contain embedded tokenizer tokens in GGUF metadata",
                path.display()
            );
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }

        let vocab_size = tokens.len();
        Ok(Self {
            model,
            vocab_size,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            tokens,
            token_to_id,
        })
    }

    /// Encode input text into a list of token IDs using greedy longest-match.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();

        // If a BOS token is defined, start with it
        if let Some(bos) = self.bos_token_id {
            result.push(bos);
        }

        if text.is_empty() {
            return result;
        }

        // Standard LLaMA / SentencePiece space prefix representation
        let mut normalized = String::with_capacity(text.len() + 1);
        if !text.starts_with(' ') && !text.starts_with('\u{2581}') {
            normalized.push('\u{2581}');
        }
        normalized.push_str(&text.replace(' ', "\u{2581}"));
        let chars: Vec<char> = normalized.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut matched = false;
            let max_lookahead = (chars.len() - i).min(64);

            for len in (1..=max_lookahead).rev() {
                let candidate: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    result.push(id);
                    i += len;
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Fallback: character-by-character or UNK
                let single_char: String = chars[i..i + 1].iter().collect();
                if let Some(&id) = self.token_to_id.get(&single_char) {
                    result.push(id);
                } else if let Some(unk) = self.unk_token_id {
                    result.push(unk);
                }
                i += 1;
            }
        }

        result
    }

    /// Decode a list of token IDs back into text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if Some(id) == self.bos_token_id || Some(id) == self.eos_token_id {
                continue;
            }
            if let Some(tok) = self.tokens.get(id as usize) {
                // Handle SentencePiece / GPT space prefixes
                let cleaned = tok.replace(['\u{2581}', 'Ġ'], " ");
                out.push_str(&cleaned);
            }
        }
        out
    }
}

fn read_array<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)
        .context("unexpected end of GGUF header")?;
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
        bail!("string length {len} exceeds limit");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).context("reading string")?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_int_val<R: Read>(r: &mut R, ty: u32) -> Option<i64> {
    match ty {
        0 => read_array::<1, _>(r).ok().map(|b| b[0] as i64),
        1 => read_array::<1, _>(r).ok().map(|b| b[0] as i8 as i64),
        2 => read_array::<2, _>(r)
            .ok()
            .map(|b| u16::from_le_bytes(b) as i64),
        3 => read_array::<2, _>(r)
            .ok()
            .map(|b| i16::from_le_bytes(b) as i64),
        4 => read_array::<4, _>(r)
            .ok()
            .map(|b| u32::from_le_bytes(b) as i64),
        5 => read_array::<4, _>(r)
            .ok()
            .map(|b| i32::from_le_bytes(b) as i64),
        7 => read_array::<1, _>(r).ok().map(|b| b[0] as i64),
        10 => read_array::<8, _>(r)
            .ok()
            .map(|b| u64::from_le_bytes(b) as i64),
        11 => read_array::<8, _>(r).ok().map(i64::from_le_bytes),
        _ => {
            skip_value(r, ty).ok();
            None
        }
    }
}

fn skip_value<R: Read>(r: &mut R, ty: u32) -> Result<()> {
    match ty {
        0 | 1 | 7 => {
            read_array::<1, _>(r)?;
        }
        2 | 3 => {
            read_array::<2, _>(r)?;
        }
        4..=6 => {
            read_array::<4, _>(r)?;
        }
        10..=12 => {
            read_array::<8, _>(r)?;
        }
        8 => {
            let len = read_u64(r)?;
            skip_bytes(r, len)?;
        }
        9 => {
            let elem_ty = read_u32(r)?;
            let len = read_u64(r)?;
            skip_array(r, elem_ty, len)?;
        }
        _ => bail!("unknown type {ty}"),
    }
    Ok(())
}

fn skip_array<R: Read>(r: &mut R, elem_ty: u32, len: u64) -> Result<()> {
    if len > MAX_ARRAY_LEN {
        bail!("array length {len} exceeds limit");
    }
    match elem_ty {
        0 | 1 | 7 => skip_bytes(r, len)?,
        2 | 3 => skip_bytes(r, len.saturating_mul(2))?,
        4..=6 => skip_bytes(r, len.saturating_mul(4))?,
        10..=12 => skip_bytes(r, len.saturating_mul(8))?,
        _ => {
            for _ in 0..len {
                skip_value(r, elem_ty)?;
            }
        }
    }
    Ok(())
}

fn skip_bytes<R: Read>(r: &mut R, n: u64) -> Result<()> {
    std::io::copy(&mut r.by_ref().take(n), &mut std::io::sink())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_encode_and_decode() {
        let tokens = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "\u{2581}Hello".to_string(),
            "\u{2581}world".to_string(),
            "!".to_string(),
        ];
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let tokenizer = GgufTokenizer {
            model: "llama".to_string(),
            vocab_size: tokens.len(),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unk_token_id: Some(0),
            tokens,
            token_to_id,
        };

        let encoded = tokenizer.encode("Hello world!");
        assert_eq!(encoded, vec![1, 3, 4, 5]);

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded.trim(), "Hello world!");
    }
}
