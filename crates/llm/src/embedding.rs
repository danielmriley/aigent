//! Embedding backend trait and providers.
//!
//! Provides a unified [`EmbeddingClient`] trait with concrete implementations
//! for Ollama's `/api/embeddings` endpoint and (feature-gated) Candle local
//! inference.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ── Trait ──────────────────────────────────────────────────────────────────────

/// A backend that converts text into dense vector embeddings.
#[allow(async_fn_in_trait)]
pub trait EmbeddingClient: Send + Sync {
    /// Embed a single text string.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of texts.  Default implementation calls [`embed`] in a
    /// loop; providers may override with a native batch endpoint.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }

    /// The dimensionality of vectors produced by this model.
    fn dimensions(&self) -> usize;

    /// Short human-readable model identifier.
    fn model_name(&self) -> &str;
}

// ── Ollama provider ────────────────────────────────────────────────────────────

/// Configuration for the Ollama embedding provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingConfig {
    /// Ollama model name (e.g. `nomic-embed-text`, `mxbai-embed-large`).
    pub model: String,
    /// Base URL for the Ollama API (default: `http://localhost:11434`).
    pub base_url: String,
    /// Expected embedding dimensions (used for pre-allocation).
    pub dimensions: usize,
}

impl Default for OllamaEmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "nomic-embed-text".to_string(),
            base_url: "http://localhost:11434".to_string(),
            dimensions: 768,
        }
    }
}

/// Embedding provider backed by a local Ollama instance.
pub struct OllamaEmbeddingClient {
    config: OllamaEmbeddingConfig,
    http: reqwest::Client,
}

impl OllamaEmbeddingClient {
    pub fn new(config: OllamaEmbeddingConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
        }
    }
}

/// JSON body sent to Ollama's `/api/embeddings`.
#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

/// JSON response from Ollama's `/api/embeddings`.
#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

impl EmbeddingClient for OllamaEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.config.base_url);
        let body = OllamaEmbedRequest {
            model: &self.config.model,
            prompt: text,
        };
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("ollama embedding request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("ollama embedding HTTP {status}: {body_text}");
        }

        let parsed: OllamaEmbedResponse = resp
            .json()
            .await
            .context("failed to parse ollama embedding response")?;

        Ok(parsed.embedding)
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

// ── Candle local embedding (feature-gated stub) ───────────────────────────────

/// Configuration for a Candle-based local embedding model.
#[cfg(feature = "candle")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleEmbeddingConfig {
    /// HuggingFace model ID (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
    pub model_id: String,
    /// Override local path instead of downloading from HF Hub.
    pub model_path: Option<String>,
    /// Expected embedding dimensions.
    pub dimensions: usize,
}

#[cfg(feature = "candle")]
impl Default for CandleEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            model_path: None,
            dimensions: 384,
        }
    }
}

/// Placeholder Candle-based embedding client.
///
/// A full implementation would load the ONNX / safetensors model via
/// `candle_transformers` and run inference on the CPU/GPU.  For now this
/// serves as a compilable stub that forwards to Ollama.
#[cfg(feature = "candle")]
pub struct CandleEmbeddingClient {
    config: CandleEmbeddingConfig,
    /// Falls back to Ollama until native inference is implemented.
    fallback: OllamaEmbeddingClient,
}

#[cfg(feature = "candle")]
impl CandleEmbeddingClient {
    pub fn new(config: CandleEmbeddingConfig) -> Self {
        let fallback_cfg = OllamaEmbeddingConfig {
            model: "nomic-embed-text".to_string(),
            base_url: "http://localhost:11434".to_string(),
            dimensions: config.dimensions,
        };
        Self {
            config,
            fallback: OllamaEmbeddingClient::new(fallback_cfg),
        }
    }
}

#[cfg(feature = "candle")]
impl EmbeddingClient for CandleEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // TODO: Replace with actual Candle model inference.
        self.fallback.embed(text).await
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn model_name(&self) -> &str {
        &self.config.model_id
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_config_default() {
        let cfg = OllamaEmbeddingConfig::default();
        assert_eq!(cfg.model, "nomic-embed-text");
        assert_eq!(cfg.dimensions, 768);
        assert!(cfg.base_url.contains("11434"));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_config_default() {
        let cfg = CandleEmbeddingConfig::default();
        assert_eq!(cfg.dimensions, 384);
        assert!(cfg.model_id.contains("MiniLM"));
    }

    #[test]
    fn embed_request_serializes() {
        let req = OllamaEmbedRequest {
            model: "nomic-embed-text",
            prompt: "hello world",
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("nomic-embed-text"));
        assert!(json.contains("hello world"));
    }
}
