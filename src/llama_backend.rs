//! In-process text generation through llama.cpp's C API.
//!
//! The bindings in this module match llama.cpp b8681's public `llama.h` ABI.
//! The library is linked at build time; model formats, tokenization, chat
//! templates, and transformer execution are handled by llama.cpp itself.

use std::ffi::{c_char, c_void, CStr, CString};
use std::io::{self, Write};
use std::path::Path;
use std::ptr;

use anyhow::{anyhow, bail, Context, Result};

#[repr(C)]
struct ModelParams {
    devices: *mut *mut c_void,
    tensor_buft_overrides: *const c_void,
    n_gpu_layers: i32,
    split_mode: i32,
    main_gpu: i32,
    tensor_split: *const f32,
    progress_callback: Option<unsafe extern "C" fn(f32, *mut c_void) -> bool>,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    vocab_only: bool,
    use_mmap: bool,
    use_direct_io: bool,
    use_mlock: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    rope_scaling_type: i32,
    pooling_type: i32,
    attention_type: i32,
    flash_attn_type: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: *const c_void,
    cb_eval_user_data: *mut c_void,
    type_k: i32,
    type_v: i32,
    abort_callback: *const c_void,
    abort_callback_data: *mut c_void,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut c_void,
    n_samplers: usize,
}

#[repr(C)]
struct SamplerChainParams {
    no_perf: bool,
}

#[repr(C)]
struct ChatMessage {
    role: *const c_char,
    content: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Batch {
    n_tokens: i32,
    token: *mut i32,
    embd: *mut f32,
    pos: *mut i32,
    n_seq_id: *mut i32,
    seq_id: *mut *mut i32,
    logits: *mut i8,
}

#[link(name = "llama")]
unsafe extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();
    fn llama_model_default_params() -> ModelParams;
    fn llama_context_default_params() -> ContextParams;
    fn llama_sampler_chain_default_params() -> SamplerChainParams;
    fn llama_model_load_from_file(path: *const c_char, params: ModelParams) -> *mut c_void;
    fn llama_model_free(model: *mut c_void);
    fn llama_model_get_vocab(model: *const c_void) -> *const c_void;
    fn llama_model_chat_template(model: *const c_void, name: *const c_char) -> *const c_char;
    fn llama_model_n_ctx_train(model: *const c_void) -> i32;
    fn llama_init_from_model(model: *mut c_void, params: ContextParams) -> *mut c_void;
    fn llama_free(ctx: *mut c_void);
    fn llama_n_ctx(ctx: *const c_void) -> u32;
    fn llama_get_memory(ctx: *const c_void) -> *mut c_void;
    fn llama_memory_clear(memory: *mut c_void, data: bool);
    fn llama_chat_apply_template(
        template: *const c_char,
        messages: *const ChatMessage,
        n_messages: usize,
        add_assistant: bool,
        output: *mut c_char,
        output_len: i32,
    ) -> i32;
    fn llama_tokenize(
        vocab: *const c_void,
        text: *const c_char,
        text_len: i32,
        tokens: *mut i32,
        max_tokens: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    fn llama_vocab_is_eog(vocab: *const c_void, token: i32) -> bool;
    fn llama_token_to_piece(
        vocab: *const c_void,
        token: i32,
        output: *mut c_char,
        output_len: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> Batch;
    fn llama_batch_free(batch: Batch);
    fn llama_decode(ctx: *mut c_void, batch: Batch) -> i32;
    fn llama_sampler_chain_init(params: SamplerChainParams) -> *mut c_void;
    fn llama_sampler_chain_add(chain: *mut c_void, sampler: *mut c_void);
    fn llama_sampler_init_top_k(k: i32) -> *mut c_void;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut c_void;
    fn llama_sampler_init_temp(temp: f32) -> *mut c_void;
    fn llama_sampler_init_dist(seed: u32) -> *mut c_void;
    fn llama_sampler_sample(sampler: *mut c_void, ctx: *mut c_void, idx: i32) -> i32;
    fn llama_sampler_accept(sampler: *mut c_void, token: i32);
    fn llama_sampler_reset(sampler: *mut c_void);
    fn llama_sampler_free(sampler: *mut c_void);
}

#[link(name = "ggml")]
unsafe extern "C" {
    fn ggml_backend_load_all();
}

struct BackendGuard;

impl BackendGuard {
    fn init() -> Self {
        // SAFETY: llama.cpp documents this as process-wide one-time init.
        unsafe {
            llama_backend_init();
            // Recent llama.cpp packages ship CPU backends as plugins instead
            // of registering them directly from llama_backend_init.
            ggml_backend_load_all();
        }
        Self
    }
}

impl Drop for BackendGuard {
    fn drop(&mut self) {
        // SAFETY: all model/context values are declared after this guard and
        // therefore dropped before the backend.
        unsafe { llama_backend_free() };
    }
}

struct ModelGuard(*mut c_void);

impl Drop for ModelGuard {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.0) };
    }
}

struct ContextGuard(*mut c_void);

impl Drop for ContextGuard {
    fn drop(&mut self) {
        unsafe { llama_free(self.0) };
    }
}

struct SamplerGuard(*mut c_void);

impl Drop for SamplerGuard {
    fn drop(&mut self) {
        unsafe { llama_sampler_free(self.0) };
    }
}

struct BatchGuard(Batch);

impl Drop for BatchGuard {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.0) };
    }
}

/// Start a line-oriented chat loop using llama.cpp inside this process.
pub fn run_repl(
    model_path: &Path,
    threads: usize,
    context: Option<u32>,
    predict: u32,
) -> Result<()> {
    let _backend = BackendGuard::init();
    let path = CString::new(model_path.as_os_str().as_encoded_bytes())
        .context("model path contains an embedded NUL byte")?;

    // SAFETY: params are returned by the same llama.cpp library and passed
    // back unchanged except for fields declared in its public C API.
    let mut model_params = unsafe { llama_model_default_params() };
    model_params.n_gpu_layers = 0;
    let model = unsafe { llama_model_load_from_file(path.as_ptr(), model_params) };
    if model.is_null() {
        bail!("llama.cpp could not load model {}", model_path.display());
    }
    let model = ModelGuard(model);
    let vocab = unsafe { llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        bail!(
            "llama.cpp did not expose a vocabulary for {}",
            model_path.display()
        );
    }

    let mut context_params = unsafe { llama_context_default_params() };
    context_params.n_ctx = context.unwrap_or(0);
    context_params.n_threads = i32::try_from(threads).unwrap_or(i32::MAX);
    context_params.n_threads_batch = context_params.n_threads;
    let ctx = unsafe { llama_init_from_model(model.0, context_params) };
    if ctx.is_null() {
        bail!("llama.cpp could not create an inference context");
    }
    let ctx = ContextGuard(ctx);

    let sampler_params = unsafe { llama_sampler_chain_default_params() };
    let sampler = unsafe { llama_sampler_chain_init(sampler_params) };
    if sampler.is_null() {
        bail!("llama.cpp could not create a sampler");
    }
    let sampler = SamplerGuard(sampler);
    unsafe {
        for component in [
            llama_sampler_init_top_k(40),
            llama_sampler_init_top_p(0.95, 1),
            llama_sampler_init_temp(0.8),
            llama_sampler_init_dist(u32::MAX),
        ] {
            if component.is_null() {
                bail!("llama.cpp could not create a sampler component");
            }
            llama_sampler_chain_add(sampler.0, component);
        }
    }

    let template = unsafe { llama_model_chat_template(model.0, ptr::null()) };
    let template = if template.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(template) }
                .to_string_lossy()
                .into_owned(),
        )
    };
    let train_context = unsafe { llama_model_n_ctx_train(model.0) }.max(0) as u32;
    println!(
        "loaded {} in-process (llama.cpp, context: {}, threads: {})",
        model_path.display(),
        context.unwrap_or(train_context),
        threads
    );
    if template.is_none() {
        eprintln!("model has no chat template; using plain text prompt formatting");
    }
    println!("Type /exit to close the session.");

    let mut history: Vec<(String, String)> = Vec::new();
    let mut line = String::new();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let user_text = line.trim_end_matches(['\r', '\n']);
        if user_text == "/exit" || user_text == "/quit" {
            break;
        }
        if user_text.is_empty() {
            continue;
        }
        history.push(("user".into(), user_text.to_owned()));

        let prompt = render_prompt(template.as_deref(), &history)?;
        let tokens = tokenize(vocab, &prompt)?;
        let ctx_len = unsafe { llama_n_ctx(ctx.0) };
        if tokens.len() >= ctx_len as usize {
            history.pop();
            bail!(
                "prompt is {} tokens, which leaves no room in the {}-token context; shorten the conversation or pass --context",
                tokens.len(),
                ctx_len
            );
        }

        unsafe {
            llama_memory_clear(llama_get_memory(ctx.0), true);
            llama_sampler_reset(sampler.0);
        }
        let assistant = generate(
            ctx.0,
            vocab,
            sampler.0,
            &tokens,
            predict.min(ctx_len.saturating_sub(tokens.len() as u32)),
            context_params.n_batch.max(1),
        )?;
        history.push(("assistant".into(), assistant));
    }

    Ok(())
}

fn render_prompt(template: Option<&str>, history: &[(String, String)]) -> Result<String> {
    let roles = history
        .iter()
        .map(|(role, _)| CString::new(role.as_str()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let contents = history
        .iter()
        .map(|(_, content)| CString::new(content.as_str()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let messages = roles
        .iter()
        .zip(contents.iter())
        .map(|(role, content)| ChatMessage {
            role: role.as_ptr(),
            content: content.as_ptr(),
        })
        .collect::<Vec<_>>();

    if let Some(template) = template {
        let template = CString::new(template)?;
        let total = history
            .iter()
            .map(|(r, c)| r.len() + c.len())
            .sum::<usize>();
        let initial_capacity = total
            .checked_mul(4)
            .and_then(|n| n.checked_add(1024))
            .context("formatted prompt is too large")?;
        let initial_capacity = i32::try_from(initial_capacity)
            .context("formatted prompt exceeds llama.cpp's buffer limit")?;
        let mut output = vec![0u8; initial_capacity as usize];
        let mut written = unsafe {
            llama_chat_apply_template(
                template.as_ptr(),
                messages.as_ptr(),
                messages.len(),
                true,
                output.as_mut_ptr().cast(),
                initial_capacity,
            )
        };
        if written < 0 {
            bail!("llama.cpp could not apply the model chat template");
        }
        if written as usize >= output.len() {
            output.resize(written as usize + 1, 0);
            written = unsafe {
                llama_chat_apply_template(
                    template.as_ptr(),
                    messages.as_ptr(),
                    messages.len(),
                    true,
                    output.as_mut_ptr().cast(),
                    output.len() as i32,
                )
            };
            if written < 0 || written as usize >= output.len() {
                bail!("llama.cpp chat template output exceeded its buffer");
            }
        }
        return Ok(String::from_utf8_lossy(&output[..written as usize]).into_owned());
    }

    Ok(history
        .iter()
        .map(|(role, content)| format!("{role}: {content}\n"))
        .collect())
}

fn tokenize(vocab: *const c_void, prompt: &str) -> Result<Vec<i32>> {
    let prompt = CString::new(prompt).context("prompt contains an embedded NUL byte")?;
    let text_len = i32::try_from(prompt.as_bytes().len()).context("prompt is too large")?;
    let needed = unsafe {
        llama_tokenize(
            vocab,
            prompt.as_ptr(),
            text_len,
            ptr::null_mut(),
            0,
            true,
            true,
        )
    };
    if needed >= 0 {
        return Ok(Vec::new());
    }
    let count = needed
        .checked_neg()
        .context("token count overflow from llama.cpp")?;
    let mut tokens = vec![0i32; count as usize];
    let written = unsafe {
        llama_tokenize(
            vocab,
            prompt.as_ptr(),
            text_len,
            tokens.as_mut_ptr(),
            count,
            true,
            true,
        )
    };
    if written < 0 {
        bail!("llama.cpp failed to tokenize the prompt");
    }
    tokens.truncate(written as usize);
    Ok(tokens)
}

fn generate(
    ctx: *mut c_void,
    vocab: *const c_void,
    sampler: *mut c_void,
    tokens: &[i32],
    max_tokens: u32,
    batch_capacity: u32,
) -> Result<String> {
    if tokens.is_empty() {
        bail!("the prompt produced no tokens");
    }
    let batch_capacity = batch_capacity.max(1).min(i32::MAX as u32) as i32;
    let mut batch = BatchGuard(unsafe { llama_batch_init(batch_capacity, 0, 1) });
    if batch.0.token.is_null()
        || batch.0.pos.is_null()
        || batch.0.n_seq_id.is_null()
        || batch.0.seq_id.is_null()
        || batch.0.logits.is_null()
    {
        bail!("llama.cpp could not allocate an inference batch");
    }

    let mut position = 0i32;
    for chunk in tokens.chunks(batch_capacity as usize) {
        let is_last = position as usize + chunk.len() == tokens.len();
        fill_batch(&mut batch.0, chunk, position, is_last);
        let status = unsafe { llama_decode(ctx, batch.0) };
        if status != 0 {
            bail!("llama.cpp failed to evaluate the prompt (decode status {status})");
        }
        position += chunk.len() as i32;
    }

    let mut output = Vec::new();
    for generated in 0..max_tokens {
        let token = unsafe { llama_sampler_sample(sampler, ctx, -1) };
        if unsafe { llama_vocab_is_eog(vocab, token) } {
            break;
        }
        unsafe { llama_sampler_accept(sampler, token) };
        append_piece(vocab, token, &mut output)?;

        if generated + 1 == max_tokens {
            break;
        }
        let pos = position;
        fill_batch(&mut batch.0, &[token], pos, true);
        let status = unsafe { llama_decode(ctx, batch.0) };
        if status != 0 {
            bail!("llama.cpp failed during token generation (decode status {status})");
        }
        position += 1;
    }
    io::stdout().write_all(&output)?;
    io::stdout().flush()?;
    Ok(String::from_utf8_lossy(&output).into_owned())
}

fn fill_batch(batch: &mut Batch, tokens: &[i32], start_pos: i32, last_output: bool) {
    unsafe {
        batch.n_tokens = tokens.len() as i32;
        for (i, &token) in tokens.iter().enumerate() {
            *batch.token.add(i) = token;
            *batch.pos.add(i) = start_pos + i as i32;
            *batch.n_seq_id.add(i) = 1;
            *(*batch.seq_id.add(i)) = 0;
            *batch.logits.add(i) = if last_output && i + 1 == tokens.len() {
                1
            } else {
                0
            };
        }
    }
}

fn append_piece(vocab: *const c_void, token: i32, output: &mut Vec<u8>) -> Result<()> {
    let mut buf = [0u8; 256];
    let n = unsafe {
        llama_token_to_piece(
            vocab,
            token,
            buf.as_mut_ptr().cast(),
            buf.len() as i32,
            0,
            true,
        )
    };
    if n >= 0 {
        output.extend_from_slice(&buf[..n as usize]);
        return Ok(());
    }
    let size = n
        .checked_neg()
        .ok_or_else(|| anyhow!("token piece size overflow"))? as usize;
    let mut buf = vec![0u8; size];
    let written = unsafe {
        llama_token_to_piece(vocab, token, buf.as_mut_ptr().cast(), size as i32, 0, true)
    };
    if written < 0 {
        bail!("llama.cpp could not decode token {token}");
    }
    output.extend_from_slice(&buf[..written as usize]);
    Ok(())
}
