//! Candle local inference backend.
//!
//! When the `candle` feature is enabled, this module provides a
//! [`CandleBackend`] that runs text generation locally using HuggingFace
//! Candle.  It is designed for small, fast models (≤3B params) that can
//! handle simple tool calls without requiring an external API.
//!
//! The complexity router in [`InferenceConfig`] decides when to use Candle
//! vs. a remote provider based on estimated task complexity and a whitelist
//! of "fast tools" that benefit from <50ms local latency.

#[cfg(feature = "candle")]
use anyhow::{Context, Result};
#[cfg(feature = "candle")]
use serde::{Deserialize, Serialize};

// ── Configuration ──────────────────────────────────────────────────────────────

/// Runtime configuration for the Candle inference backend.
///
/// Maps directly from the `[inference]` section of `AppConfig`.
#[cfg(feature = "candle")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleConfig {
    /// HuggingFace model ID.
    pub model_id: String,
    /// Local override path (safetensors / GGUF).
    pub model_path: Option<String>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Temperature for sampling.
    pub temperature: f64,
    /// Top-p nucleus sampling threshold.
    pub top_p: f64,
    /// Repeat penalty.
    pub repeat_penalty: f32,
}

#[cfg(feature = "candle")]
impl Default for CandleConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-Coder-1.5B".to_string(),
            model_path: None,
            max_seq_len: 4096,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
        }
    }
}

// ── Backend ────────────────────────────────────────────────────────────────────

/// Local inference engine powered by Candle.
///
/// Currently a scaffolding struct — the actual model loading and generation
/// loop will be implemented once the Candle dependency graph stabilises and
/// we confirm hardware support (AVX2 / NEON / Metal).
#[cfg(feature = "candle")]
pub struct CandleBackend {
    config: CandleConfig,
    _loaded: bool,
}

#[cfg(feature = "candle")]
impl CandleBackend {
    /// Create a new (unloaded) backend with the given config.
    pub fn new(config: CandleConfig) -> Self {
        Self {
            config,
            _loaded: false,
        }
    }

    /// Load the model weights into memory.
    ///
    /// This is intentionally separate from `new()` so callers can defer the
    /// expensive I/O until the first inference request.
    pub async fn load(&mut self) -> Result<()> {
        // TODO: Implement actual model loading via candle_transformers.
        //
        // Rough plan:
        // 1. Resolve model_path or download from HF Hub.
        // 2. Load tokenizer via `tokenizers::Tokenizer::from_file()`.
        // 3. Load safetensors weights via `candle_core::safetensors`.
        // 4. Build the model graph (Qwen2 / Llama / Mistral architecture).
        // 5. Store model + tokenizer in self for generation.
        tracing::info!(
            model = %self.config.model_id,
            "candle backend: model loading not yet implemented"
        );
        self._loaded = true;
        Ok(())
    }

    /// Generate a completion for the given prompt.
    ///
    /// Returns the generated text (excluding the prompt).
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        if !self._loaded {
            anyhow::bail!("candle backend: model not loaded — call load() first");
        }

        // TODO: Implement actual token-by-token generation.
        //
        // Rough plan:
        // 1. Tokenize prompt.
        // 2. Run forward pass in a loop up to max_tokens.
        // 3. Sample from logits with temperature/top_p.
        // 4. Detect EOS and break.
        // 5. Decode token IDs back to text.
        let _ = (prompt, max_tokens);
        anyhow::bail!(
            "candle backend: generation not yet implemented for {}",
            self.config.model_id
        )
    }

    /// Whether the model has been loaded into memory.
    pub fn is_loaded(&self) -> bool {
        self._loaded
    }

    /// The model identifier.
    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }
}

// ── Complexity router ──────────────────────────────────────────────────────────

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

// ── Tests ──────────────────────────────────────────────────────────────────────

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
        let msgs = vec![crate::ChatMessage::user("```rust\nfn main() {}\n```")];
        let score = estimate_complexity(&msgs);
        assert!(score >= 0.25, "expected >= 0.25, got {score}");
    }

    #[test]
    fn should_use_candle_fast_tool() {
        let fast = vec!["list_dir".to_string(), "read_file".to_string()];
        assert!(should_use_candle("list_dir", 0.1, 0.3, &fast));
        assert!(!should_use_candle("list_dir", 0.5, 0.3, &fast));
        assert!(!should_use_candle("write_file", 0.1, 0.3, &fast));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_config_defaults() {
        let cfg = CandleConfig::default();
        assert!(cfg.model_id.contains("Qwen"));
        assert_eq!(cfg.max_seq_len, 4096);
    }
}
