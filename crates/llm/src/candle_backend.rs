//! Candle local inference backend.
//!
//! When the `candle` feature is enabled, this module provides a
//! [`CandleBackend`] that runs text generation locally using HuggingFace
//! Candle.  It supports GGUF-quantized models for low memory usage and loads
//! them from a local path or downloads them from HuggingFace Hub.
//!
//! The complexity router in [`InferenceConfig`] decides when to use Candle
//! vs. a remote provider based on estimated task complexity and a whitelist
//! of "fast tools" that benefit from <50ms local latency.

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

/// Local inference engine powered by Candle (quantized GGUF models).
///
/// Supports Qwen2, Llama, Phi and similar architectures via quantized GGUF
/// format for efficient CPU/GPU inference with minimal memory footprint.
#[cfg(feature = "candle")]
pub struct CandleBackend {
    pub config: CandleConfig,
    model: Option<ModelWeights>,
    tokenizer: Option<Tokenizer>,
    device: Device,
    eos_token_id: Option<u32>,
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
        let (model, tokenizer, eos_id) = tokio::task::spawn_blocking(move || {
            Self::load_model_sync(&config, &device)
        })
        .await
        .context("model loading task panicked")??;

        self.model = Some(model);
        self.tokenizer = Some(tokenizer);
        self.eos_token_id = eos_id;

        tracing::info!(
            model = %self.config.model_id,
            gguf = %self.config.gguf_file,
            device = %self.config.device,
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
            .or_else(|| tokenizer.token_to_id("<eos>"));

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
    /// Returns the generated text (excluding the prompt).
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("candle backend: model not loaded -- call load() first"))?;
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

        let temperature = self.config.temperature;
        let top_p = self.config.top_p;
        let repeat_penalty = self.config.repeat_penalty;
        let repeat_last_n = self.config.repeat_penalty_last_n;
        let eos_id = self.eos_token_id;
        let device = &self.device;

        // Process prompt (prefill)
        let input = Tensor::new(prompt_tokens.as_slice(), device)
            .context("failed to create prompt tensor")?
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("unsqueeze failed: {e}"))?;

        let logits = model
            .forward(&input, 0)
            .map_err(|e| anyhow::anyhow!("forward pass (prefill) failed: {e}"))?;

        let logits = logits.squeeze(0).map_err(|e| anyhow::anyhow!("squeeze failed: {e}"))?;
        let mut logits_vec: Vec<f32> = logits
            .to_vec1()
            .map_err(|e| anyhow::anyhow!("logits to vec failed: {e}"))?;

        // Track recent tokens for repeat penalty.
        let mut recent: Vec<u32> = prompt_tokens
            .iter()
            .rev()
            .take(repeat_last_n)
            .copied()
            .collect();

        apply_repeat_penalty(&mut logits_vec, &recent, repeat_penalty);
        let mut next_token = sample_token(&logits_vec, temperature, top_p) as u32;

        let mut generated_tokens: Vec<u32> = vec![next_token];

        // Autoregressive generation loop.
        for step in 1..max_tokens {
            if Some(next_token) == eos_id {
                break;
            }

            let input = Tensor::new(&[next_token], device)
                .map_err(|e| anyhow::anyhow!("token tensor failed at step {step}: {e}"))?
                .unsqueeze(0)
                .map_err(|e| anyhow::anyhow!("unsqueeze failed: {e}"))?;

            let logits = model
                .forward(&input, prompt_tokens.len() + step)
                .map_err(|e| anyhow::anyhow!("forward pass (step {step}) failed: {e}"))?;

            let logits = logits.squeeze(0).map_err(|e| anyhow::anyhow!("squeeze: {e}"))?;
            let mut logits_vec: Vec<f32> = logits
                .to_vec1()
                .map_err(|e| anyhow::anyhow!("logits to vec: {e}"))?;

            // Update recent tokens ring buffer.
            recent.push(next_token);
            if recent.len() > repeat_last_n {
                recent.remove(0);
            }
            apply_repeat_penalty(&mut logits_vec, &recent, repeat_penalty);

            next_token = sample_token(&logits_vec, temperature, top_p) as u32;
            generated_tokens.push(next_token);
        }

        // Strip EOS if it ended up in the output.
        if let Some(&last) = generated_tokens.last() {
            if Some(last) == eos_id {
                generated_tokens.pop();
            }
        }

        let decoded = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("decoding failed: {e}"))?;

        Ok(decoded)
    }

    /// Whether the model has been loaded into memory.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// The model identifier.
    pub fn model_id(&self) -> &str {
        &self.config.model_id
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
}
