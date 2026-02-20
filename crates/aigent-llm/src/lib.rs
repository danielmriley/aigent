use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::process::Command;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub user: String,
    pub assistant: String,
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat(&self, prompt: &str) -> Result<String>;
    async fn chat_stream(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<String>;
}

#[derive(Debug)]
pub struct OllamaClient;
#[derive(Debug)]
pub struct OpenRouterClient;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Provider {
    Ollama,
    OpenRouter,
}

#[derive(Debug)]
pub struct LlmRouter {
    ollama: OllamaClient,
    openrouter: OpenRouterClient,
}

const OPENROUTER_FALLBACK_MODELS: &[&str] = &[
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat",
];

pub async fn list_ollama_models() -> Result<Vec<String>> {
    let output = Command::new("ollama").arg("list").output();
    let output = match output {
        Ok(output) => output,
        Err(_) => {
            return Ok(vec![
                "ollama not found in PATH".to_string(),
                "install ollama and run: ollama pull <model>".to_string(),
            ]);
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            return Ok(vec!["failed to read ollama models".to_string()]);
        }
        return Ok(vec![format!("failed to read ollama models: {stderr}")]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut models = stdout
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    models.sort();
    models.dedup();

    if models.is_empty() {
        models.push("no models installed (run: ollama pull <model>)".to_string());
    }

    Ok(models)
}

pub async fn list_openrouter_models() -> Result<Vec<String>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(6))
        .build()?;
    let response = client
        .get("https://openrouter.ai/api/v1/models")
        .send()
        .await;

    let models = match response {
        Ok(response) if response.status().is_success() => {
            let body: serde_json::Value = response.json().await?;
            let mut items = body
                .get("data")
                .and_then(|data| data.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.get("id").and_then(|id| id.as_str()))
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            items.sort();
            items.dedup();
            if items.is_empty() {
                OPENROUTER_FALLBACK_MODELS
                    .iter()
                    .map(|model| (*model).to_string())
                    .collect::<Vec<_>>()
            } else {
                items
            }
        }
        _ => OPENROUTER_FALLBACK_MODELS
            .iter()
            .map(|model| (*model).to_string())
            .collect::<Vec<_>>(),
    };

    Ok(models)
}

impl Default for LlmRouter {
    fn default() -> Self {
        Self {
            ollama: OllamaClient,
            openrouter: OpenRouterClient,
        }
    }
}

impl LlmRouter {
    pub async fn chat_with_fallback(
        &self,
        primary: Provider,
        model: &str,
        prompt: &str,
    ) -> Result<(Provider, String)> {
        let should_force_fallback = prompt.to_lowercase().contains("/fallback");

        match primary {
            Provider::Ollama if !should_force_fallback => Ok((
                Provider::Ollama,
                self.ollama.chat_model(model, prompt).await?,
            )),
            Provider::Ollama => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model(model, prompt).await?,
            )),
            Provider::OpenRouter => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model(model, prompt).await?,
            )),
        }
    }

    pub async fn chat_stream_with_fallback(
        &self,
        primary: Provider,
        model: &str,
        prompt: &str,
        tx: mpsc::Sender<String>,
    ) -> Result<(Provider, String)> {
        let should_force_fallback = prompt.to_lowercase().contains("/fallback");

        match primary {
            Provider::Ollama if !should_force_fallback => Ok((
                Provider::Ollama,
                self.ollama.chat_model_stream(model, prompt, tx).await?,
            )),
            Provider::Ollama => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model_stream(model, prompt, tx).await?,
            )),
            Provider::OpenRouter => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model_stream(model, prompt, tx).await?,
            )),
        }
    }
}

#[async_trait]
impl LlmClient for OllamaClient {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_model("llama3.1:8b", prompt).await
    }
    async fn chat_stream(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<String> {
        self.chat_model_stream("llama3.1:8b", prompt, tx).await
    }
}

impl OllamaClient {
    async fn chat_model(&self, model: &str, prompt: &str) -> Result<String> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let endpoint = format!("{}/api/generate", base_url.trim_end_matches('/'));

        let payload = json!({
            "model": model,
            "prompt": prompt,
            "stream": false
        });

        let client = reqwest::Client::new();
        let response = client.post(endpoint).json(&payload).send().await;

        match response {
            Ok(response) => {
                let status = response.status();
                let body: serde_json::Value = response.json().await?;
                if !status.is_success() {
                    return Ok(format!("Ollama error ({status}): {body}"));
                }

                if let Some(content) = body.get("response").and_then(|value| value.as_str()) {
                    return Ok(content.to_string());
                }

                Ok(format!("Ollama response missing text: {body}"))
            }
            Err(error) => Ok(format!(
                "Ollama unavailable at {base_url}. Start Ollama and ensure model '{model}' is installed. Error: {error}"
            )),
        }
    }

    async fn chat_model_stream(&self, model: &str, prompt: &str, tx: mpsc::Sender<String>) -> Result<String> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let endpoint = format!("{}/api/generate", base_url.trim_end_matches('/'));

        let payload = json!({
            "model": model,
            "prompt": prompt,
            "stream": true
        });

        let client = reqwest::Client::new();
        let mut response = client.post(endpoint).json(&payload).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body: serde_json::Value = response.json().await?;
            return Ok(format!("Ollama error ({status}): {body}"));
        }

        let mut full_response = String::new();
        while let Some(chunk) = response.chunk().await? {
            let chunk_str = String::from_utf8_lossy(&chunk);
            for line in chunk_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(content) = json.get("response").and_then(|v| v.as_str()) {
                        full_response.push_str(content);
                        let _ = tx.send(content.to_string()).await;
                    }
                }
            }
        }

        Ok(full_response)
    }
}

#[async_trait]
impl LlmClient for OpenRouterClient {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_model("openai/gpt-4o-mini", prompt).await
    }
    async fn chat_stream(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<String> {
        self.chat_model_stream("openai/gpt-4o-mini", prompt, tx).await
    }
}

impl OpenRouterClient {
    async fn chat_model(&self, model: &str, prompt: &str) -> Result<String> {
        let api_key = std::env::var("OPENROUTER_API_KEY").ok();
        if let Some(api_key) = api_key {
            if !api_key.trim().is_empty() {
                let client = reqwest::Client::new();
                let payload = json!({
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                });

                let response = client
                    .post("https://openrouter.ai/api/v1/chat/completions")
                    .bearer_auth(api_key)
                    .header("HTTP-Referer", "https://aigent.local")
                    .header("X-Title", "Aigent")
                    .json(&payload)
                    .send()
                    .await?;

                let status = response.status();
                let body: serde_json::Value = response.json().await?;
                if !status.is_success() {
                    return Ok(format!("OpenRouter error ({status}): {body}"));
                }

                if let Some(content) = body
                    .get("choices")
                    .and_then(|choices| choices.get(0))
                    .and_then(|choice| choice.get("message"))
                    .and_then(|message| message.get("content"))
                    .and_then(|content| content.as_str())
                {
                    return Ok(content.to_string());
                }
            }
        }

        Ok(
            "OpenRouter key missing or response empty. Set OPENROUTER_API_KEY or switch to /model provider ollama."
                .to_string(),
        )
    }

    async fn chat_model_stream(&self, model: &str, prompt: &str, tx: mpsc::Sender<String>) -> Result<String> {
        let api_key = std::env::var("OPENROUTER_API_KEY").ok();
        if let Some(api_key) = api_key {
            if !api_key.trim().is_empty() {
                let client = reqwest::Client::new();
                let payload = json!({
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": true
                });

                let mut response = client
                    .post("https://openrouter.ai/api/v1/chat/completions")
                    .bearer_auth(api_key)
                    .header("HTTP-Referer", "https://aigent.local")
                    .header("X-Title", "Aigent")
                    .json(&payload)
                    .send()
                    .await?;

                let status = response.status();
                if !status.is_success() {
                    let body: serde_json::Value = response.json().await?;
                    return Ok(format!("OpenRouter error ({status}): {body}"));
                }

                let mut full_response = String::new();
                while let Some(chunk) = response.chunk().await? {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    for line in chunk_str.lines() {
                        let line = line.trim();
                        if line.is_empty() || line == "data: [DONE]" {
                            continue;
                        }
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                                if let Some(content) = json
                                    .get("choices")
                                    .and_then(|choices| choices.get(0))
                                    .and_then(|choice| choice.get("delta"))
                                    .and_then(|delta| delta.get("content"))
                                    .and_then(|content| content.as_str())
                                {
                                    full_response.push_str(content);
                                    let _ = tx.send(content.to_string()).await;
                                }
                            }
                        }
                    }
                }

                return Ok(full_response);
            }
        }

        Ok(
            "OpenRouter key missing or response empty. Set OPENROUTER_API_KEY or switch to /model provider ollama."
                .to_string(),
        )
    }
}
