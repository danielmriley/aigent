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

/// Candle-based local embedding client using a BERT model.
///
/// Loads a sentence-transformers model (e.g. all-MiniLM-L6-v2) via Candle
/// and runs mean-pooled inference on CPU/GPU for fully offline embeddings.
#[cfg(feature = "candle")]
pub struct CandleEmbeddingClient {
    config: CandleEmbeddingConfig,
    model: std::sync::Mutex<Option<CandleEmbeddingModel>>,
}

/// Loaded BERT model + tokenizer for embedding inference.
#[cfg(feature = "candle")]
struct CandleEmbeddingModel {
    model: candle_transformers::models::bert::BertModel,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "candle")]
impl CandleEmbeddingClient {
    pub fn new(config: CandleEmbeddingConfig) -> Self {
        Self {
            config,
            model: std::sync::Mutex::new(None),
        }
    }

    /// Lazily load the model on first embed() call.
    fn ensure_loaded(&self) -> Result<()> {
        let mut guard = self.model.lock().map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        if guard.is_some() {
            return Ok(());
        }

        use hf_hub::{api::sync::Api, Repo, RepoType};
        use candle_core::Device;
        use candle_nn::VarBuilder;
        use candle_transformers::models::bert::{BertModel, Config as BertConfig};

        let model_id = if let Some(ref p) = self.config.model_path {
            if !p.is_empty() {
                p.clone()
            } else {
                self.config.model_id.clone()
            }
        } else {
            self.config.model_id.clone()
        };

        tracing::info!(model = %model_id, "loading candle embedding model");

        let api = Api::new().context("hf-hub API init")?;
        let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

        let config_path = repo.get("config.json").context("download config.json")?;
        let tokenizer_path = repo.get("tokenizer.json").context("download tokenizer.json")?;
        let weights_path = repo.get("model.safetensors").context("download model.safetensors")?;

        let config_data = std::fs::read_to_string(&config_path)?;
        let bert_config: BertConfig = serde_json::from_str(&config_data)
            .context("parse BERT config.json")?;

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
                .context("load safetensors")?
        };

        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| anyhow::anyhow!("load BERT model: {e}"))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        *guard = Some(CandleEmbeddingModel { model, tokenizer });
        tracing::info!("candle embedding model loaded");
        Ok(())
    }

    /// Run a single forward pass through the BERT model and return mean-pooled output.
    fn embed_sync(&self, text: &str) -> Result<Vec<f32>> {
        self.ensure_loaded()?;

        let guard = self.model.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        let loaded = guard.as_ref().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;

        let encoding = loaded.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();

        let device = &loaded.model.device;
        let input_ids = candle_core::Tensor::new(ids, device)
            .map_err(|e| anyhow::anyhow!("tensor ids: {e}"))?
            .unsqueeze(0).map_err(|e| anyhow::anyhow!("unsqueeze: {e}"))?;
        let token_type_ids = candle_core::Tensor::new(type_ids, device)
            .map_err(|e| anyhow::anyhow!("tensor type_ids: {e}"))?
            .unsqueeze(0).map_err(|e| anyhow::anyhow!("unsqueeze: {e}"))?;

        let output = loaded.model.forward(&input_ids, &token_type_ids, None)
            .map_err(|e| anyhow::anyhow!("forward: {e}"))?;

        // Mean pooling over the sequence dimension (dim=1).
        let seq_len = output.dim(1).map_err(|e| anyhow::anyhow!("dim: {e}"))?;
        let sum = output.sum(1).map_err(|e| anyhow::anyhow!("sum: {e}"))?;
        let mean = (sum / (seq_len as f64))
            .map_err(|e| anyhow::anyhow!("div: {e}"))?;

        // L2-normalize the embedding.
        let norm = mean.sqr()
            .map_err(|e| anyhow::anyhow!("sqr: {e}"))?
            .sum_keepdim(1)
            .map_err(|e| anyhow::anyhow!("sum_keepdim: {e}"))?
            .sqrt()
            .map_err(|e| anyhow::anyhow!("sqrt: {e}"))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| anyhow::anyhow!("clamp: {e}"))?;
        let normalized = mean.broadcast_div(&norm)
            .map_err(|e| anyhow::anyhow!("div norm: {e}"))?;

        let vec: Vec<f32> = normalized.squeeze(0)
            .map_err(|e| anyhow::anyhow!("squeeze: {e}"))?
            .to_vec1()
            .map_err(|e| anyhow::anyhow!("to_vec1: {e}"))?;

        Ok(vec)
    }
}

#[cfg(feature = "candle")]
impl EmbeddingClient for CandleEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_sync(text)
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
