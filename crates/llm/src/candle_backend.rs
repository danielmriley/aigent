//! Candle local inference backend.
//!
//! When the `candle` feature is enabled, this module provides a
//! [`CandleBackend`] that runs text generation locally using HuggingFace
//! Candle.  It supports GGUF-quantized models for low memory usage and loads
//! them from a local path or downloads them from HuggingFace Hub.
//!
//! CPU-intensive work (model forward passes, token sampling) is always
//! executed inside `tokio::task::spawn_blocking` so the async executor
//! stays responsive.

#[cfg(feature = "candle")]
use anyhow::{Context, Result};
#[cfg(feature = "candle")]
use serde::{Deserialize, Serialize};

// -- Configuration -----------------------------------------------------------

/// Runtime configuration for the Candle inference backend.
///
/// Maps directly from the `[inference]` section of `AppConfig`.
#[cfg(feature = "candle")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleConfig {
    /// HuggingFace model ID (e.g. `"Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"`).
    pub model_id: String,
    /// Specific GGUF filename inside the repo.
    pub gguf_file: String,
    /// Local override path to a `.gguf` file (skips download if set).
    pub model_path: Option<String>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Temperature for sampling (0.0 = greedy).
    pub temperature: f64,
    /// Top-p nucleus sampling threshold.
    pub top_p: f64,
    /// Repeat penalty.
    pub repeat_penalty: f32,
    /// Window size for repeat penalty.
    pub repeat_penalty_last_n: usize,
    /// Device selection: "cpu" or "cuda".
    pub device: String,
}

#[cfg(feature = "candle")]
impl Default for CandleConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF".to_string(),
            gguf_file: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string(),
            model_path: None,
            max_seq_len: 4096,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_penalty_last_n: 64,
            device: "cpu".to_string(),
        }
    }
}

// -- Backend -----------------------------------------------------------------

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};
#[cfg(feature = "candle")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "candle")]
use candle_transformers::models::quantized_qwen2::ModelWeights;
#[cfg(feature = "candle")]
use tokenizers::Tokenizer;

/// Resolved device from config string.
#[cfg(feature = "candle")]
fn resolve_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "candle-cuda")]
        "cuda" | "gpu" => Ok(Device::new_cuda(0).context("failed to open CUDA device 0")?),
        #[cfg(feature = "candle-metal")]
        "metal" | "gpu" => Ok(Device::new_metal(0).context("failed to open Metal device 0")?),
        other => {
            tracing::warn!(device = other, "unknown device, falling back to CPU");
            Ok(Device::Cpu)
        }
    }
}

/// Softmax implementation for logit sampling.
#[cfg(feature = "candle")]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Sample a token index from logits using temperature + top-p.
#[cfg(feature = "candle")]
fn sample_token(logits: &[f32], temperature: f64, top_p: f64) -> usize {
    use std::cmp::Ordering;

    if temperature <= 1e-8 {
        // Greedy: pick the argmax.
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Scale logits by temperature.
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature as f32).collect();
    let probs = softmax(&scaled);

    // Top-p (nucleus) sampling.
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut cumulative = 0.0_f32;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p as f32 {
            cutoff = i + 1;
            break;
        }
    }
    let candidates = &indexed[..cutoff];

    // Re-normalize and sample.
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r: f32 = rand_f32() * total;
    let mut acc = 0.0_f32;
    for &(idx, p) in candidates {
        acc += p;
        if acc >= r {
            return idx;
        }
    }
    candidates.last().map(|&(i, _)| i).unwrap_or(0)
}

/// Simple pseudo-random f32 in [0, 1) using thread-local state.
#[cfg(feature = "candle")]
fn rand_f32() -> f32 {
    use std::cell::Cell;
    use std::time::SystemTime;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        );
    }
    STATE.with(|s| {
        // xorshift64
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f32) / (u64::MAX as f32)
    })
}

/// Apply repeat penalty to logits for the last N generated tokens.
#[cfg(feature = "candle")]
fn apply_repeat_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
    for &tok in recent_tokens {
        if (tok as usize) < logits.len() {
            let l = &mut logits[tok as usize];
            if *l > 0.0 {
                *l /= penalty;
            } else {
                *l *= penalty;
            }
        }
    }
}

// -- Synchronous token generation (runs on blocking thread) ------------------

/// Generate tokens synchronously.  This function contains the entire CPU/GPU
/// intensive computation and must be called from `tokio::task::spawn_blocking`.
#[cfg(feature = "candle")]
#[allow(clippy::too_many_arguments)]
fn generate_tokens_sync(
    model: &mut ModelWeights,
    device: &Device,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f32,
    repeat_penalty_last_n: usize,
    eos_token_id: Option<u32>,
    token_tx: Option<&std::sync::mpsc::Sender<u32>>,
) -> Result<Vec<u32>> {
    if prompt_tokens.is_empty() {
        return Ok(vec![]);
    }

    // Process prompt (prefill).
    let input = Tensor::new(prompt_tokens, device)
        .context("failed to create prompt tensor")?
        .unsqueeze(0)
        .map_err(|e| anyhow::anyhow!("unsqueeze failed: {e}"))?;

    let logits = model
        .forward(&input, 0)
        .map_err(|e| anyhow::anyhow!("forward pass (prefill) failed: {e}"))?;

    let logits = logits
        .squeeze(0)
        .map_err(|e| anyhow::anyhow!("squeeze failed: {e}"))?;
    let mut logits_vec: Vec<f32> = logits
        .to_vec1()
        .map_err(|e| anyhow::anyhow!("logits to vec failed: {e}"))?;

    // Track recent tokens for repeat penalty.
    let mut recent: Vec<u32> = prompt_tokens
        .iter()
        .rev()
        .take(repeat_penalty_last_n)
        .copied()
        .collect();

    apply_repeat_penalty(&mut logits_vec, &recent, repeat_penalty);
    let mut next_token = sample_token(&logits_vec, temperature, top_p) as u32;

    let mut generated_tokens: Vec<u32> = vec![next_token];
    if let Some(tx) = token_tx {
        let _ = tx.send(next_token);
    }

    // Autoregressive generation loop.
    for step in 1..max_tokens {
        if Some(next_token) == eos_token_id {
            break;
        }

        let input = Tensor::new(&[next_token], device)
            .map_err(|e| anyhow::anyhow!("token tensor failed at step {step}: {e}"))?
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("unsqueeze failed: {e}"))?;

        let logits = model
            .forward(&input, prompt_tokens.len() + step)
            .map_err(|e| anyhow::anyhow!("forward pass (step {step}) failed: {e}"))?;

        let logits = logits
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("squeeze: {e}"))?;
        let mut logits_vec: Vec<f32> = logits
            .to_vec1()
            .map_err(|e| anyhow::anyhow!("logits to vec: {e}"))?;

        // Update recent tokens ring buffer.
        recent.push(next_token);
        if recent.len() > repeat_penalty_last_n {
            recent.remove(0);
        }
        apply_repeat_penalty(&mut logits_vec, &recent, repeat_penalty);

        next_token = sample_token(&logits_vec, temperature, top_p) as u32;
        generated_tokens.push(next_token);
        if let Some(tx) = token_tx {
            let _ = tx.send(next_token);
        }
    }

    // Strip EOS if it ended up in the output.
    if let Some(&last) = generated_tokens.last() {
        if Some(last) == eos_token_id {
            generated_tokens.pop();
        }
    }

    Ok(generated_tokens)
}

// -- Chat template detection -------------------------------------------------

/// Known chat template families.
#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateKind {
    /// `<|im_start|>role\ncontent<|im_end|>` (Qwen, Yi, etc.)
    ChatML,
    /// `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` (Llama 3)
    Llama3,
    /// `[INST] content [/INST]` (Mistral, Llama 2)
    Mistral,
    /// `<start_of_turn>role\ncontent<end_of_turn>` (Gemma)
    Gemma,
    /// `<|role|>\ncontent<|end|>` (Phi-3)
    Phi3,
}

/// Format chat messages according to the detected template.
#[cfg(feature = "candle")]
pub fn apply_chat_template(
    kind: ChatTemplateKind,
    messages: &[(String, String)],
) -> String {
    let mut prompt = String::new();
    match kind {
        ChatTemplateKind::ChatML => {
            for (role, content) in messages {
                prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            prompt.push_str("<|im_start|>assistant\n");
        }
        ChatTemplateKind::Llama3 => {
            prompt.push_str("<|begin_of_text|>");
            for (role, content) in messages {
                prompt.push_str(&format!(
                    "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                ));
            }
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ChatTemplateKind::Mistral => {
            let mut first = true;
            for (role, content) in messages {
                if role == "system" {
                    // Mistral prepends system to first user message.
                    prompt.push_str(content);
                    prompt.push_str("\n\n");
                } else if role == "user" {
                    if first {
                        prompt.push_str(&format!("<s>[INST] {content} [/INST]"));
                        first = false;
                    } else {
                        prompt.push_str(&format!("[INST] {content} [/INST]"));
                    }
                } else if role == "assistant" {
                    prompt.push_str(&format!(" {content}</s>"));
                }
            }
        }
        ChatTemplateKind::Gemma => {
            for (role, content) in messages {
                let gemma_role = if role == "assistant" { "model" } else { role.as_str() };
                prompt.push_str(&format!(
                    "<start_of_turn>{gemma_role}\n{content}<end_of_turn>\n"
                ));
            }
            prompt.push_str("<start_of_turn>model\n");
        }
        ChatTemplateKind::Phi3 => {
            for (role, content) in messages {
                prompt.push_str(&format!("<|{role}|>\n{content}<|end|>\n"));
            }
            prompt.push_str("<|assistant|>\n");
        }
    }
    prompt
}

/// Detect the chat template from the tokenizer_config.json in the HF repo.
///
/// Falls back to ChatML if detection fails (the default model uses ChatML).
#[cfg(feature = "candle")]
fn detect_chat_template(config: &CandleConfig) -> ChatTemplateKind {
    let api = match hf_hub::api::sync::Api::new() {
        Ok(api) => api,
        Err(_) => return ChatTemplateKind::ChatML,
    };

    // Try the GGUF repo first, then the base repo.
    let repos = [
        config.model_id.clone(),
        config
            .model_id
            .strip_suffix("-GGUF")
            .unwrap_or(&config.model_id)
            .to_string(),
    ];

    for repo_id in &repos {
        let repo = api.repo(hf_hub::Repo::new(repo_id.clone(), hf_hub::RepoType::Model));
        if let Ok(path) = repo.get("tokenizer_config.json") {
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) {
                    if let Some(tmpl) = json.get("chat_template").and_then(|v| v.as_str()) {
                        return classify_template(tmpl);
                    }
                }
            }
        }
    }

    ChatTemplateKind::ChatML
}

/// Classify a Jinja2 chat template string into a known family by marker tokens.
#[cfg(feature = "candle")]
fn classify_template(tmpl: &str) -> ChatTemplateKind {
    if tmpl.contains("<|im_start|>") {
        ChatTemplateKind::ChatML
    } else if tmpl.contains("<|begin_of_text|>") || tmpl.contains("<|eot_id|>") {
        ChatTemplateKind::Llama3
    } else if tmpl.contains("[INST]") {
        ChatTemplateKind::Mistral
    } else if tmpl.contains("<start_of_turn>") {
        ChatTemplateKind::Gemma
    } else if tmpl.contains("<|end|>") {
        ChatTemplateKind::Phi3
    } else {
        // Unknown template — ChatML is the safest default.
        ChatTemplateKind::ChatML
    }
}

// -- Backend struct ----------------------------------------------------------

/// Local inference engine powered by Candle (quantized GGUF models).
///
/// Supports Qwen2, Llama, Phi and similar architectures via quantized GGUF
/// format for efficient CPU/GPU inference with minimal memory footprint.
///
/// All CPU/GPU-intensive work is offloaded to `tokio::task::spawn_blocking`
/// to avoid starving the async executor.
#[cfg(feature = "candle")]
pub struct CandleBackend {
    pub config: CandleConfig,
    model: Option<ModelWeights>,
    tokenizer: Option<Tokenizer>,
    device: Device,
    eos_token_id: Option<u32>,
    chat_template: ChatTemplateKind,
}

#[cfg(feature = "candle")]
impl CandleBackend {
    /// Create a new (unloaded) backend with the given config.
    pub fn new(config: CandleConfig) -> Result<Self> {
        let device = resolve_device(&config.device)?;
        Ok(Self {
            config,
            model: None,
            tokenizer: None,
            device,
            eos_token_id: None,
            chat_template: ChatTemplateKind::ChatML,
        })
    }

    /// Load the model weights and tokenizer into memory.
    ///
    /// This is intentionally separate from `new()` so callers can defer the
    /// expensive I/O until the first inference request.
    pub async fn load(&mut self) -> Result<()> {
        let config = self.config.clone();
        let device = self.device.clone();

        // Run the blocking model load on a dedicated thread.
        let (model, tokenizer, eos_id, template) = tokio::task::spawn_blocking(move || {
            let (model, tokenizer, eos_id) = Self::load_model_sync(&config, &device)?;
            let template = detect_chat_template(&config);
            Ok::<_, anyhow::Error>((model, tokenizer, eos_id, template))
        })
        .await
        .context("model loading task panicked")??;

        self.model = Some(model);
        self.tokenizer = Some(tokenizer);
        self.eos_token_id = eos_id;
        self.chat_template = template;

        tracing::info!(
            model = %self.config.model_id,
            gguf = %self.config.gguf_file,
            device = %self.config.device,
            template = ?self.chat_template,
            "candle backend loaded"
        );
        Ok(())
    }

    /// Synchronous model loading (runs on a blocking thread).
    fn load_model_sync(
        config: &CandleConfig,
        device: &Device,
    ) -> Result<(ModelWeights, Tokenizer, Option<u32>)> {
        let gguf_path = if let Some(ref local) = config.model_path {
            std::path::PathBuf::from(local)
        } else {
            Self::download_gguf(config)?
        };

        tracing::info!(path = %gguf_path.display(), "loading GGUF model");

        let mut file = std::fs::File::open(&gguf_path)
            .with_context(|| format!("cannot open GGUF file: {}", gguf_path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("failed to parse GGUF: {e}"))?;

        let model = ModelWeights::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("failed to load model weights: {e}"))?;

        // Load the tokenizer from the same repo (non-GGUF repo for tokenizer).
        let tokenizer = Self::load_tokenizer(config)?;

        // Resolve EOS token ID.
        let eos_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .or_else(|| tokenizer.token_to_id("<|eot_id|>"));

        Ok((model, tokenizer, eos_id))
    }

    /// Download a GGUF file from HuggingFace Hub.
    fn download_gguf(config: &CandleConfig) -> Result<std::path::PathBuf> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let api = Api::new().context("failed to initialise HuggingFace Hub API")?;
        let repo = api.repo(Repo::new(config.model_id.clone(), RepoType::Model));
        let path = repo
            .get(&config.gguf_file)
            .with_context(|| format!(
                "failed to download {}/{} from HuggingFace Hub",
                config.model_id, config.gguf_file
            ))?;
        Ok(path)
    }

    /// Load the tokenizer from HuggingFace Hub.
    ///
    /// Attempts to load from the GGUF repo first, then falls back to the
    /// base (non-GGUF) model repo.
    fn load_tokenizer(config: &CandleConfig) -> Result<Tokenizer> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let api = Api::new().context("failed to initialise HuggingFace Hub API")?;

        // Try the GGUF repo first.
        let repo = api.repo(Repo::new(config.model_id.clone(), RepoType::Model));
        if let Ok(path) = repo.get("tokenizer.json") {
            if let Ok(tok) = Tokenizer::from_file(&path) {
                return Ok(tok);
            }
        }

        // Fall back to base model repo (strip -GGUF suffix).
        let base_id = config
            .model_id
            .strip_suffix("-GGUF")
            .unwrap_or(&config.model_id);
        let base_repo = api.repo(Repo::new(base_id.to_string(), RepoType::Model));
        let path = base_repo
            .get("tokenizer.json")
            .with_context(|| format!("failed to download tokenizer for {base_id}"))?;
        let tok = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        Ok(tok)
    }

    /// Generate a completion for the given prompt.
    ///
    /// CPU/GPU work is offloaded to a blocking thread via
    /// `tokio::task::spawn_blocking` so the async executor stays responsive.
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("candle backend: tokenizer not loaded"))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let max_tokens = max_tokens.min(self.config.max_seq_len.saturating_sub(prompt_tokens.len()));
        if max_tokens == 0 {
            return Ok(String::new());
        }

        // Take model out to move into the blocking thread.
        let mut model = self
            .model
            .take()
            .ok_or_else(|| anyhow::anyhow!("candle backend: model not loaded -- call load() first"))?;

        let device = self.device.clone();
        let temperature = self.config.temperature;
        let top_p = self.config.top_p;
        let repeat_penalty = self.config.repeat_penalty;
        let repeat_last_n = self.config.repeat_penalty_last_n;
        let eos_id = self.eos_token_id;

        let (tokens_result, model_back) = tokio::task::spawn_blocking(move || {
            let r = generate_tokens_sync(
                &mut model,
                &device,
                &prompt_tokens,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                repeat_last_n,
                eos_id,
                None,
            );
            (r, model)
        })
        .await
        .context("generation task panicked")?;

        // Always return the model, even on generation error.
        self.model = Some(model_back);
        let generated_tokens = tokens_result?;

        let decoded = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("decoding failed: {e}"))?;

        Ok(decoded)
    }

    /// Generate with true token-by-token streaming.
    ///
    /// Each generated token is decoded and sent through `tx` as soon as it
    /// is sampled.  The full response text is returned at the end.
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("candle backend: tokenizer not loaded"))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let max_tokens = max_tokens.min(self.config.max_seq_len.saturating_sub(prompt_tokens.len()));
        if max_tokens == 0 {
            return Ok(String::new());
        }

        // Take model out to move into the blocking thread.
        let mut model = self
            .model
            .take()
            .ok_or_else(|| anyhow::anyhow!("candle backend: model not loaded"))?;

        let device = self.device.clone();
        let temperature = self.config.temperature;
        let top_p = self.config.top_p;
        let repeat_penalty = self.config.repeat_penalty;
        let repeat_last_n = self.config.repeat_penalty_last_n;
        let eos_id = self.eos_token_id;
        let tok_clone = tokenizer.clone();

        // Use std::sync::mpsc for the blocking thread → async bridge.
        let (token_tx, token_rx) = std::sync::mpsc::channel::<u32>();

        let gen_handle = tokio::task::spawn_blocking(move || {
            let r = generate_tokens_sync(
                &mut model,
                &device,
                &prompt_tokens,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                repeat_last_n,
                eos_id,
                Some(&token_tx),
            );
            drop(token_tx); // Signal end of stream.
            (r, model)
        });

        // Forward decoded tokens to the async channel.
        let mut full_text = String::new();
        // Process tokens from the sync channel, decoding and forwarding each.
        while let Ok(token_id) = token_rx.recv() {
            if let Some(eos) = eos_id {
                if token_id == eos {
                    break;
                }
            }
            if let Ok(piece) = tok_clone.decode(&[token_id], true) {
                if !piece.is_empty() {
                    full_text.push_str(&piece);
                    let _ = tx.send(piece).await;
                }
            }
        }

        let (tokens_result, model_back) = gen_handle
            .await
            .context("generation task panicked")?;

        self.model = Some(model_back);
        // We already have full_text from streaming; ignore the token vec on success.
        tokens_result?;

        Ok(full_text)
    }

    /// The detected chat template for this model.
    pub fn chat_template(&self) -> ChatTemplateKind {
        self.chat_template
    }

    /// Whether the model has been loaded into memory.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// The model identifier.
    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    /// The GGUF filename.
    pub fn gguf_file(&self) -> &str {
        &self.config.gguf_file
    }
}

// -- Complexity router -------------------------------------------------------

/// Estimate task complexity from the conversation so far.
///
/// Returns a score in `[0.0, 1.0]` where 0.0 = trivial (route to Candle)
/// and 1.0 = complex (route to remote provider).
pub fn estimate_complexity(messages: &[crate::ChatMessage]) -> f32 {
    let total_tokens: usize = messages.iter().map(|m| m.content.as_deref().unwrap_or("").split_whitespace().count()).sum();
    let num_turns = messages.len();
    let has_code = messages.iter().any(|m| {
        m.content.as_deref().unwrap_or("").contains("```") || m.content.as_deref().unwrap_or("").contains("fn ") || m.content.as_deref().unwrap_or("").contains("def ")
    });

    let mut score: f32 = 0.0;

    // Long conversations are more complex.
    if total_tokens > 2000 {
        score += 0.3;
    } else if total_tokens > 500 {
        score += 0.15;
    }

    // Multi-turn implies context-dependent reasoning.
    if num_turns > 6 {
        score += 0.2;
    } else if num_turns > 3 {
        score += 0.1;
    }

    // Code-related tasks benefit from larger models.
    if has_code {
        score += 0.25;
    }

    score.min(1.0)
}

/// Decide whether a tool call should be routed to the local Candle backend.
///
/// Returns `true` if the tool is in the fast-tools whitelist AND complexity
/// is below the threshold.
pub fn should_use_candle(
    tool_name: &str,
    complexity: f32,
    threshold: f32,
    fast_tools: &[String],
) -> bool {
    complexity < threshold && fast_tools.iter().any(|t| t == tool_name)
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complexity_empty_is_low() {
        assert_eq!(estimate_complexity(&[]), 0.0);
    }

    #[test]
    fn complexity_long_conversation() {
        let msgs: Vec<crate::ChatMessage> = (0..10)
            .map(|i| crate::ChatMessage::user(format!("Message {i} with enough words to count")))
            .collect();
        let score = estimate_complexity(&msgs);
        assert!(score > 0.1, "expected > 0.1, got {score}");
    }

    #[test]
    fn complexity_code_present() {
        let msgs = vec![
            crate::ChatMessage::user("Here is some code:\n```rust\nfn main() {}\n```".to_string()),
        ];
        let score = estimate_complexity(&msgs);
        assert!(score >= 0.25, "expected >= 0.25, got {score}");
    }

    #[test]
    fn should_use_candle_fast_tool() {
        let fast = vec!["list_dir".to_string(), "read_file".to_string()];
        assert!(should_use_candle("list_dir", 0.1, 0.3, &fast));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_config_defaults() {
        let cfg = CandleConfig::default();
        assert!(cfg.model_id.contains("Qwen"));
        assert_eq!(cfg.max_seq_len, 4096);
        assert!(cfg.gguf_file.contains("q4_k_m"));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn sample_greedy_picks_max() {
        let logits = vec![0.1_f32, 0.2, 0.5, 0.9, 0.3];
        let idx = sample_token(&logits, 0.0, 1.0);
        assert_eq!(idx, 3, "greedy should pick index 3 (0.9)");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn repeat_penalty_reduces_logits() {
        let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        apply_repeat_penalty(&mut logits, &[2], 1.5);
        assert!(logits[2] < 3.0, "positive logit should decrease with penalty");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn resolve_cpu_device() {
        let d = resolve_device("cpu").unwrap();
        assert!(matches!(d, Device::Cpu));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn classify_chatml_template() {
        let tmpl = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";
        assert_eq!(classify_template(tmpl), ChatTemplateKind::ChatML);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn classify_llama3_template() {
        let tmpl = "{% for message in messages %}<|begin_of_text|><|start_header_id|>{{ message.role }}<|end_header_id|>{{ message.content }}<|eot_id|>{% endfor %}";
        assert_eq!(classify_template(tmpl), ChatTemplateKind::Llama3);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn classify_mistral_template() {
        let tmpl = "{% for message in messages %}[INST] {{ message.content }} [/INST]{% endfor %}";
        assert_eq!(classify_template(tmpl), ChatTemplateKind::Mistral);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn classify_gemma_template() {
        let tmpl = "{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n{% endfor %}";
        assert_eq!(classify_template(tmpl), ChatTemplateKind::Gemma);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn classify_phi3_template() {
        let tmpl = "{% for message in messages %}<|{{ message.role }}|>\n{{ message.content }}<|end|>\n{% endfor %}";
        assert_eq!(classify_template(tmpl), ChatTemplateKind::Phi3);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn chatml_format_roundtrip() {
        let msgs = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let prompt = apply_chat_template(ChatTemplateKind::ChatML, &msgs);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn llama3_format_roundtrip() {
        let msgs = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let prompt = apply_chat_template(ChatTemplateKind::Llama3, &msgs);
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }
}
