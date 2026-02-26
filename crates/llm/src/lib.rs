use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::process::Command;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    client: reqwest::Client,
}

#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for OpenRouterClient {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Provider {
    Ollama,
    OpenRouter,
}

#[derive(Debug, Clone)]
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
            ollama: OllamaClient::new(),
            openrouter: OpenRouterClient::new(),
        }
    }
}

impl LlmRouter {
    pub async fn chat_with_fallback(
        &self,
        primary: Provider,
        ollama_model: &str,
        openrouter_model: &str,
        prompt: &str,
    ) -> Result<(Provider, String)> {
        let should_force_fallback = prompt.to_lowercase().contains("/fallback");

        match primary {
            Provider::Ollama if !should_force_fallback => Ok((
                Provider::Ollama,
                self.ollama.chat_model(ollama_model, prompt).await?,
            )),
            Provider::Ollama => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model(openrouter_model, prompt).await?,
            )),
            Provider::OpenRouter => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model(openrouter_model, prompt).await?,
            )),
        }
    }

    pub async fn chat_stream_with_fallback(
        &self,
        primary: Provider,
        ollama_model: &str,
        openrouter_model: &str,
        prompt: &str,
        tx: mpsc::Sender<String>,
    ) -> Result<(Provider, String)> {
        let should_force_fallback = prompt.to_lowercase().contains("/fallback");

        match primary {
            Provider::Ollama if !should_force_fallback => Ok((
                Provider::Ollama,
                self.ollama.chat_model_stream(ollama_model, prompt, tx).await?,
            )),
            Provider::Ollama => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model_stream(openrouter_model, prompt, tx).await?,
            )),
            Provider::OpenRouter => Ok((
                Provider::OpenRouter,
                self.openrouter.chat_model_stream(openrouter_model, prompt, tx).await?,
            )),
        }
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

        let client = self.client.clone();
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

    async fn chat_model_stream(
        &self,
        model: &str,
        prompt: &str,
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let endpoint = format!("{}/api/generate", base_url.trim_end_matches('/'));

        let payload = json!({
            "model": model,
            "prompt": prompt,
            "stream": true
        });

        let client = self.client.clone();
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
                        if content.is_empty() {
                            continue;
                        }
                        full_response.push_str(content);
                        let _ = tx.send(content.to_string()).await;
                    }
                }
            }
        }

        Ok(full_response)
    }
}

impl OpenRouterClient {
    async fn chat_model(&self, model: &str, prompt: &str) -> Result<String> {
        let api_key = std::env::var("OPENROUTER_API_KEY").ok();
        if let Some(api_key) = api_key {
            if !api_key.trim().is_empty() {
                let client = self.client.clone();
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

    async fn chat_model_stream(
        &self,
        model: &str,
        prompt: &str,
        tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let api_key = std::env::var("OPENROUTER_API_KEY").ok();
        if let Some(api_key) = api_key {
            if !api_key.trim().is_empty() {
                let client = self.client.clone();
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
                                    if content.is_empty() {
                                        continue;
                                    }
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

// ── Structured output extraction ──────────────────────────────────────────────

/// Structured fields that an LLM may embed in a fenced `json` code block
/// inside its reply.
///
/// The agent can instruct the model to wrap structured actions in:
/// ` ```json\n{ "action": "...", "params": {...}, "reply": "..." }\n` ``` `
///
/// The runtime extracts this via [`extract_json_output`] and acts on
/// `action`/`params` while displaying only `reply` to the user.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructuredOutput {
    /// Logical action name (e.g. `"record_memory"`, `"tool_call"`).
    #[serde(default)]
    pub action: Option<String>,
    /// Free-form parameters associated with the action.
    #[serde(default)]
    pub params: serde_json::Value,
    /// Human-readable rationale for the action.
    #[serde(default)]
    pub rationale: Option<String>,
    /// The user-facing portion of the reply.  When present, callers should
    /// display this instead of the raw LLM text.
    #[serde(default)]
    pub reply: Option<String>,
}

/// Extract the first valid JSON fenced code block from an LLM response.
///
/// Looks for ` ```json\n...\n` ``` ` delimiters.  Returns `None` when the
/// response contains no such block or the block is not valid JSON.
///
/// # Usage
///
/// ```rust
/// use aigent_llm::{extract_json_output, StructuredOutput};
///
/// let raw = "Sure!\n```json\n{\"action\":\"record_belief\",\"reply\":\"Got it\"}\n```";
/// if let Some(out) = extract_json_output::<StructuredOutput>(raw) {
///     println!("action: {:?}", out.action);
/// }
/// ```
pub fn extract_json_output<T: serde::de::DeserializeOwned>(response: &str) -> Option<T> {
    // Strategy 1: fenced ```json ... ``` blocks.
    if let Some(fence_start) = response.find("```json") {
        let after_fence = &response[fence_start + "```json".len()..];
        if let Some(json_start) = after_fence.find(|c: char| !c.is_whitespace()) {
            let json_body = &after_fence[json_start..];
            if let Some(fence_end) = json_body.find("```") {
                let json_str = json_body[..fence_end].trim();
                if let Ok(val) = serde_json::from_str(json_str) {
                    return Some(val);
                }
            }
        }
    }

    // Strategy 2: bare JSON object — find the first '{' and its matching '}'.
    let trimmed = response.trim();
    if let Some(start) = trimmed.find('{') {
        // Walk from the end to find the last matching '}'.
        if let Some(end) = trimmed.rfind('}') {
            if end > start {
                let candidate = &trimmed[start..=end];
                if let Ok(val) = serde_json::from_str(candidate) {
                    return Some(val);
                }
            }
        }
    }

    None
}
