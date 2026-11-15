use std::fs;
use std::path::PathBuf;

use llm_compress::detector::{ggml_type_name, read_gguf_meta, tensor_size_bytes, GGUF_MAGIC};
use llm_compress::tokenizer::GgufTokenizer;

fn push_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn push_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("llm-compress-{tag}-{}.gguf", std::process::id()))
}

fn minimal_gguf() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    push_u32(&mut buf, 3);
    push_u64(&mut buf, 1); // tensor count
    push_u64(&mut buf, 4); // metadata count

    push_string(&mut buf, "general.architecture");
    push_u32(&mut buf, 8); // STRING
    push_string(&mut buf, "llama");

    push_string(&mut buf, "general.file_type");
    push_u32(&mut buf, 4); // UINT32
    push_u32(&mut buf, 15); // Q4_K_M

    push_string(&mut buf, "split.count");
    push_u32(&mut buf, 2); // UINT16
    push_u16(&mut buf, 2);

    push_string(&mut buf, "split.no");
    push_u32(&mut buf, 2); // UINT16
    push_u16(&mut buf, 0);

    push_string(&mut buf, "blk.0.attn_q.weight");
    push_u32(&mut buf, 2); // dimensions
    push_u64(&mut buf, 32);
    push_u64(&mut buf, 16);
    push_u32(&mut buf, 2); // Q4_0
    push_u64(&mut buf, 0);

    buf.extend_from_slice(&[0u8; 64]);
    buf
}

#[test]
fn parses_minimal_gguf_header() {
    let path = temp_path("header");
    fs::write(&path, minimal_gguf()).unwrap();
    let meta = read_gguf_meta(&path).unwrap();
    fs::remove_file(&path).ok();

    assert_eq!(meta.version, 3);
    assert_eq!(meta.architecture.as_deref(), Some("llama"));
    assert_eq!(meta.file_type, Some(15));
    assert_eq!(meta.split_count, Some(2));
    assert_eq!(meta.split_no, Some(0));
    assert_eq!(meta.tensor_count, 1);
    assert_eq!(meta.param_count, 512);

    assert_eq!(meta.tensors.len(), 1);
    let tensor = &meta.tensors[0];
    assert_eq!(tensor.name, "blk.0.attn_q.weight");
    assert_eq!(tensor.dims, vec![32, 16]);
    assert_eq!(tensor.elements, 512);
    assert_eq!(tensor.offset, 0);
    assert_eq!(ggml_type_name(tensor.ggml_type), "Q4_0");
    assert_eq!(
        tensor_size_bytes(tensor.ggml_type, tensor.elements),
        Some(288)
    );
}

#[test]
fn rejects_non_gguf_files() {
    let path = temp_path("bad-magic");
    fs::write(&path, b"NOPE this is not a model").unwrap();
    let result = read_gguf_meta(&path);
    fs::remove_file(&path).ok();

    assert!(result.is_err());
}

#[test]
fn parses_tokenizer_from_gguf_file() {
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    push_u32(&mut buf, 3);
    push_u64(&mut buf, 0); // tensor count
    push_u64(&mut buf, 5); // metadata count

    push_string(&mut buf, "tokenizer.ggml.model");
    push_u32(&mut buf, 8); // STRING
    push_string(&mut buf, "llama");

    push_string(&mut buf, "tokenizer.ggml.bos_token_id");
    push_u32(&mut buf, 4); // UINT32
    push_u32(&mut buf, 1);

    push_string(&mut buf, "tokenizer.ggml.eos_token_id");
    push_u32(&mut buf, 4); // UINT32
    push_u32(&mut buf, 2);

    push_string(&mut buf, "tokenizer.ggml.unknown_token_id");
    push_u32(&mut buf, 4); // UINT32
    push_u32(&mut buf, 0);

    push_string(&mut buf, "tokenizer.ggml.tokens");
    push_u32(&mut buf, 9); // ARRAY
    push_u32(&mut buf, 8); // element type STRING
    push_u64(&mut buf, 5); // count
    push_string(&mut buf, "<unk>");
    push_string(&mut buf, "<s>");
    push_string(&mut buf, "</s>");
    push_string(&mut buf, "\u{2581}hello");
    push_string(&mut buf, "\u{2581}world");

    let path = temp_path("tokenizer");
    fs::write(&path, buf).unwrap();
    let tok = GgufTokenizer::from_file(&path).unwrap();
    fs::remove_file(&path).ok();

    assert_eq!(tok.model, "llama");
    assert_eq!(tok.vocab_size, 5);
    assert_eq!(tok.bos_token_id, Some(1));
    assert_eq!(tok.eos_token_id, Some(2));
    assert_eq!(tok.unk_token_id, Some(0));

    let encoded = tok.encode("hello world");
    assert_eq!(encoded, vec![1, 3, 4]);

    let decoded = tok.decode(&encoded);
    assert_eq!(decoded.trim(), "hello world");
}
