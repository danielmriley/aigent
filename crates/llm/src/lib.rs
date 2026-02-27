use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::process::Command;
use std::time::Duration;
use tokio::sync::mpsc;

// ── Chat message types for structured tool calling ───────────────────────────

/// Role in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in a chat conversation.
///
/// Used with the structured chat APIs (`/api/chat` for Ollama,
/// `/chat/completions` for OpenRouter) that support native tool calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Option<String>,
    /// Tool calls requested by the assistant (only present on assistant messages).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// When role == Tool, identifies which tool call this result is for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: Some(content.into()), tool_calls: vec![], tool_call_id: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: Some(content.into()), tool_calls: vec![], tool_call_id: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: Some(content.into()), tool_calls: vec![], tool_call_id: None }
    }
    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self { role: ChatRole::Assistant, content: None, tool_calls, tool_call_id: None }
    }
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: ChatRole::Tool, content: Some(content.into()), tool_calls: vec![], tool_call_id: Some(tool_call_id.into()) }
    }
}

/// A tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this call (used to correlate tool results).
    /// Ollama may not always provide one, in which case we generate one.
    #[serde(default)]
    pub id: String,
    /// Always "function" for OpenAI-compatible APIs.
    #[serde(default = "default_tool_call_type")]
    pub r#type: String,
    pub function: ToolCallFunction,
}

fn default_tool_call_type() -> String {
    "function".to_string()
}

/// The function name and arguments within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// Arguments as a JSON string (OpenRouter) or parsed object (Ollama).
    /// We normalize to a parsed `HashMap` for downstream consumers.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Response from a structured chat call.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The provider that actually handled the request.
    pub provider: Provider,
    /// Text content of the assistant's response (may be empty if tool_calls present).
    pub content: String,
    /// Tool calls the assistant wants to make (empty if a normal text response).
    pub tool_calls: Vec<ToolCall>,
    /// Finish reason: "stop", "tool_calls", "length", etc.
    pub finish_reason: String,
}

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

#[derive(Debug, Clone, Default)]
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

    /// Send structured chat messages with optional tool definitions.
    ///
    /// Uses the native chat API (Ollama `/api/chat`, OpenRouter `/chat/completions`)
    /// with the `tools` parameter so models that support function calling can
    /// return structured `tool_calls` in their response.
    pub async fn chat_messages(
        &self,
        primary: Provider,
        ollama_model: &str,
        openrouter_model: &str,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<ChatResponse> {
        match primary {
            Provider::Ollama => {
                let (content, tool_calls, finish_reason) = self.ollama
                    .chat_messages(ollama_model, messages, tools).await?;
                Ok(ChatResponse {
                    provider: Provider::Ollama,
                    content,
                    tool_calls,
                    finish_reason,
                })
            }
            Provider::OpenRouter => {
                let (content, tool_calls, finish_reason) = self.openrouter
                    .chat_messages(openrouter_model, messages, tools).await?;
                Ok(ChatResponse {
                    provider: Provider::OpenRouter,
                    content,
                    tool_calls,
                    finish_reason,
                })
            }
        }
    }

    /// Send structured chat messages with streaming and optional tool definitions.
    ///
    /// Text tokens are streamed via `tx` as they arrive. If the model returns
    /// tool calls, they are accumulated and returned in the final `ChatResponse`.
    pub async fn chat_messages_stream(
        &self,
        primary: Provider,
        ollama_model: &str,
        openrouter_model: &str,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        tx: mpsc::Sender<String>,
    ) -> Result<ChatResponse> {
        match primary {
            Provider::Ollama => {
                let (content, tool_calls, finish_reason) = self.ollama
                    .chat_messages_stream(ollama_model, messages, tools, tx).await?;
                Ok(ChatResponse {
                    provider: Provider::Ollama,
                    content,
                    tool_calls,
                    finish_reason,
                })
            }
            Provider::OpenRouter => {
                let (content, tool_calls, finish_reason) = self.openrouter
                    .chat_messages_stream(openrouter_model, messages, tools, tx).await?;
                Ok(ChatResponse {
                    provider: Provider::OpenRouter,
                    content,
                    tool_calls,
                    finish_reason,
                })
            }
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

    /// Structured chat using Ollama's `/api/chat` endpoint with optional tools.
    async fn chat_messages(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<(String, Vec<ToolCall>, String)> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let endpoint = format!("{}/api/chat", base_url.trim_end_matches('/'));

        let ollama_messages = messages_to_ollama(messages);
        let mut payload = json!({
            "model": model,
            "messages": ollama_messages,
            "stream": false
        });
        if let Some(tools_val) = tools {
            payload["tools"] = tools_val.clone();
        }

        let response = self.client.clone().post(&endpoint).json(&payload).send().await;
        match response {
            Ok(response) => {
                let status = response.status();
                let body: serde_json::Value = response.json().await?;
                if !status.is_success() {
                    return Ok((format!("Ollama error ({status}): {body}"), vec![], "error".to_string()));
                }
                parse_ollama_chat_response(&body)
            }
            Err(error) => Ok((
                format!("Ollama unavailable at {base_url}. Error: {error}"),
                vec![],
                "error".to_string(),
            )),
        }
    }

    /// Streaming structured chat using Ollama's `/api/chat` endpoint with optional tools.
    async fn chat_messages_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        tx: mpsc::Sender<String>,
    ) -> Result<(String, Vec<ToolCall>, String)> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let endpoint = format!("{}/api/chat", base_url.trim_end_matches('/'));

        let ollama_messages = messages_to_ollama(messages);
        let mut payload = json!({
            "model": model,
            "messages": ollama_messages,
            "stream": true
        });
        if let Some(tools_val) = tools {
            payload["tools"] = tools_val.clone();
        }

        let mut response = self.client.clone().post(&endpoint).json(&payload).send().await?;
        let status = response.status();
        if !status.is_success() {
            let body: serde_json::Value = response.json().await?;
            return Ok((format!("Ollama error ({status}): {body}"), vec![], "error".to_string()));
        }

        let mut full_response = String::new();
        let mut tool_calls: Vec<ToolCall> = vec![];
        let mut finish_reason = "stop".to_string();

        while let Some(chunk) = response.chunk().await? {
            let chunk_str = String::from_utf8_lossy(&chunk);
            for line in chunk_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                    // Extract streamed text content
                    if let Some(content) = json.get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|v| v.as_str())
                    {
                        if !content.is_empty() {
                            full_response.push_str(content);
                            let _ = tx.send(content.to_string()).await;
                        }
                    }
                    // Extract tool calls from the final chunk
                    if json.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                        if let Some(calls) = json.get("message")
                            .and_then(|m| m.get("tool_calls"))
                            .and_then(|v| v.as_array())
                        {
                            tool_calls = parse_ollama_tool_calls(calls);
                            if !tool_calls.is_empty() {
                                finish_reason = "tool_calls".to_string();
                            }
                        }
                    }
                }
            }
        }

        Ok((full_response, tool_calls, finish_reason))
    }
}

/// Convert our `ChatMessage` array to Ollama's message format.
fn messages_to_ollama(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    messages.iter().map(|m| {
        let role = match m.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        let mut msg = json!({ "role": role });
        if let Some(ref content) = m.content {
            msg["content"] = json!(content);
        }
        if !m.tool_calls.is_empty() {
            let calls: Vec<serde_json::Value> = m.tool_calls.iter().map(|tc| {
                json!({
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
            }).collect();
            msg["tool_calls"] = json!(calls);
        }
        // Ollama expects tool-role messages to carry the correlated tool_call_id.
        if let Some(ref id) = m.tool_call_id {
            msg["tool_call_id"] = json!(id);
        }
        msg
    }).collect()
}

/// Parse Ollama's `/api/chat` non-streaming response.
fn parse_ollama_chat_response(body: &serde_json::Value) -> Result<(String, Vec<ToolCall>, String)> {
    let content = body.get("message")
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let tool_calls = body.get("message")
        .and_then(|m| m.get("tool_calls"))
        .and_then(|v| v.as_array())
        .map(|calls| parse_ollama_tool_calls(calls))
        .unwrap_or_default();

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls".to_string()
    } else {
        "stop".to_string()
    };

    Ok((content, tool_calls, finish_reason))
}

/// Parse Ollama tool_calls array into our `ToolCall` type.
fn parse_ollama_tool_calls(calls: &[serde_json::Value]) -> Vec<ToolCall> {
    calls.iter().enumerate().filter_map(|(i, call)| {
        let func = call.get("function")?;
        let name = func.get("name")?.as_str()?.to_string();
        let arguments = func.get("arguments").cloned().unwrap_or(json!({}));
        Some(ToolCall {
            id: format!("call_{i}"),
            r#type: "function".to_string(),
            function: ToolCallFunction { name, arguments },
        })
    }).collect()
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

    /// Structured chat using OpenRouter's `/chat/completions` endpoint with optional tools.
    async fn chat_messages(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<(String, Vec<ToolCall>, String)> {
        let api_key = std::env::var("OPENROUTER_API_KEY").ok();
        let Some(api_key) = api_key.filter(|k| !k.trim().is_empty()) else {
            return Ok(("OpenRouter key missing. Set OPENROUTER_API_KEY.".to_string(), vec![], "error".to_string()));
        };

        let openai_messages = messages_to_openai(messages);
        let mut payload = json!({
            "model": model,
            "messages": openai_messages
        });
        if let Some(tools_val) = tools {
            payload["tools"] = tools_val.clone();
        }

        let response = self.client.clone()
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&api_key)
            .header("HTTP-Referer", "https://aigent.local")
            .header("X-Title", "Aigent")
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        let body: serde_json::Value = response.json().await?;
        if !status.is_success() {
            return Ok((format!("OpenRouter error ({status}): {body}"), vec![], "error".to_string()));
        }
        parse_openai_chat_response(&body)
    }

    /// Streaming structured chat using OpenRouter's `/chat/completions` endpoint with tools.
    async fn chat_messages_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        tx: mpsc::Sender<String>,
    ) -> Result<(String, Vec<ToolCall>, String)> {
        let api_key = std::env::var("OPENROUTER_API_KEY").ok();
        let Some(api_key) = api_key.filter(|k| !k.trim().is_empty()) else {
            return Ok(("OpenRouter key missing. Set OPENROUTER_API_KEY.".to_string(), vec![], "error".to_string()));
        };

        let openai_messages = messages_to_openai(messages);
        let mut payload = json!({
            "model": model,
            "messages": openai_messages,
            "stream": true
        });
        if let Some(tools_val) = tools {
            payload["tools"] = tools_val.clone();
        }

        let mut response = self.client.clone()
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&api_key)
            .header("HTTP-Referer", "https://aigent.local")
            .header("X-Title", "Aigent")
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body: serde_json::Value = response.json().await?;
            return Ok((format!("OpenRouter error ({status}): {body}"), vec![], "error".to_string()));
        }

        let mut full_response = String::new();
        // Accumulate tool call deltas by index
        let mut tool_call_map: HashMap<usize, (String, String, String)> = HashMap::new(); // (id, name, arguments)
        let mut finish_reason = "stop".to_string();

        while let Some(chunk) = response.chunk().await? {
            let chunk_str = String::from_utf8_lossy(&chunk);
            for line in chunk_str.lines() {
                let line = line.trim();
                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }
                let Some(data) = line.strip_prefix("data: ") else { continue };
                let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else { continue };

                let choice = json.get("choices").and_then(|c| c.get(0));
                let Some(choice) = choice else { continue };

                // Check finish_reason
                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    finish_reason = fr.to_string();
                }

                let delta = choice.get("delta");
                let Some(delta) = delta else { continue };

                // Accumulate text content
                if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() {
                        full_response.push_str(content);
                        let _ = tx.send(content.to_string()).await;
                    }
                }

                // Accumulate tool call deltas
                if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let entry = tool_call_map.entry(idx).or_insert_with(|| (String::new(), String::new(), String::new()));
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            entry.0 = id.to_string();
                        }
                        if let Some(func) = tc.get("function") {
                            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                // Name is sent once in the first delta, not
                                // incrementally — assign rather than append.
                                entry.1 = name.to_string();
                            }
                            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                entry.2.push_str(args);
                            }
                        }
                    }
                }
            }
        }

        // Convert accumulated tool call deltas to ToolCall structs
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        if !tool_call_map.is_empty() {
            let mut indices: Vec<usize> = tool_call_map.keys().copied().collect();
            indices.sort();
            for idx in indices {
                let (id, name, args_str) = &tool_call_map[&idx];
                let arguments = serde_json::from_str(args_str).unwrap_or(json!({}));
                tool_calls.push(ToolCall {
                    id: if id.is_empty() { format!("call_{idx}") } else { id.clone() },
                    r#type: "function".to_string(),
                    function: ToolCallFunction { name: name.clone(), arguments },
                });
            }
            if finish_reason == "stop" {
                finish_reason = "tool_calls".to_string();
            }
        }

        Ok((full_response, tool_calls, finish_reason))
    }
}

/// Convert our `ChatMessage` array to OpenAI-compatible message format.
fn messages_to_openai(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    messages.iter().map(|m| {
        let role = match m.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        let mut msg = json!({ "role": role });
        if let Some(ref content) = m.content {
            msg["content"] = json!(content);
        } else {
            msg["content"] = json!(null);
        }
        if !m.tool_calls.is_empty() {
            let calls: Vec<serde_json::Value> = m.tool_calls.iter().map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": if tc.function.arguments.is_string() {
                            tc.function.arguments.clone()
                        } else {
                            json!(tc.function.arguments.to_string())
                        }
                    }
                })
            }).collect();
            msg["tool_calls"] = json!(calls);
        }
        if let Some(ref id) = m.tool_call_id {
            msg["tool_call_id"] = json!(id);
        }
        msg
    }).collect()
}

/// Parse an OpenAI-compatible `/chat/completions` non-streaming response.
fn parse_openai_chat_response(body: &serde_json::Value) -> Result<(String, Vec<ToolCall>, String)> {
    let choice = body.get("choices").and_then(|c| c.get(0));
    let message = choice.and_then(|c| c.get("message"));

    let content = message
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .unwrap_or("stop")
        .to_string();

    let tool_calls = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(|v| v.as_array())
        .map(|calls| {
            calls.iter().enumerate().filter_map(|(i, tc)| {
                let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let func = tc.get("function")?;
                let name = func.get("name")?.as_str()?.to_string();
                let arguments = func.get("arguments")
                    .map(|v| {
                        if let Some(s) = v.as_str() {
                            serde_json::from_str(s).unwrap_or(json!({}))
                        } else {
                            v.clone()
                        }
                    })
                    .unwrap_or(json!({}));
                Some(ToolCall {
                    id: if id.is_empty() { format!("call_{i}") } else { id },
                    r#type: "function".to_string(),
                    function: ToolCallFunction { name, arguments },
                })
            }).collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok((content, tool_calls, finish_reason))
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StructuredOutput defaults ──────────────────────────────────────────

    #[test]
    fn structured_output_default_all_none() {
        let out = StructuredOutput::default();
        assert!(out.action.is_none());
        assert!(out.rationale.is_none());
        assert!(out.reply.is_none());
        assert_eq!(out.params, serde_json::Value::Null);
    }

    // ── extract_json_output: fenced code block ─────────────────────────────

    #[test]
    fn extract_fenced_json() {
        let raw = "Sure!\n```json\n{\"action\":\"record_belief\",\"reply\":\"Got it\"}\n```";
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("record_belief"));
        assert_eq!(out.reply.as_deref(), Some("Got it"));
    }

    #[test]
    fn extract_fenced_json_with_extra_text() {
        let raw = "Here is the result:\n\n```json\n{\"action\":\"tool_call\",\"params\":{\"name\":\"read\"},\"reply\":\"done\"}\n```\n\nHope that helps!";
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("tool_call"));
        assert_eq!(out.reply.as_deref(), Some("done"));
        assert_eq!(out.params["name"], "read");
    }

    #[test]
    fn extract_fenced_json_with_leading_newlines() {
        let raw = "```json\n\n  {\"action\":\"test\"}\n```";
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("test"));
    }

    // ── extract_json_output: bare JSON ─────────────────────────────────────

    #[test]
    fn extract_bare_json() {
        let raw = r#"{"action":"hello","reply":"world"}"#;
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("hello"));
        assert_eq!(out.reply.as_deref(), Some("world"));
    }

    #[test]
    fn extract_bare_json_with_surrounding_text() {
        let raw = "some preamble {\"action\":\"x\"} some epilogue";
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("x"));
    }

    #[test]
    fn extract_bare_json_nested_braces() {
        let raw = r#"{"action":"call","params":{"cmd":"echo {}"},"reply":"ok"}"#;
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("call"));
        assert_eq!(out.reply.as_deref(), Some("ok"));
    }

    // ── extract_json_output: failure cases ─────────────────────────────────

    #[test]
    fn extract_returns_none_for_plain_text() {
        let raw = "Hello, this is a plain text response with no JSON.";
        assert!(extract_json_output::<StructuredOutput>(raw).is_none());
    }

    #[test]
    fn extract_returns_none_for_empty_string() {
        assert!(extract_json_output::<StructuredOutput>("").is_none());
    }

    #[test]
    fn extract_returns_none_for_malformed_json_in_fence() {
        let raw = "```json\n{not valid json}\n```";
        assert!(extract_json_output::<StructuredOutput>(raw).is_none());
    }

    #[test]
    fn extract_returns_none_for_lone_braces() {
        let raw = "Something { that } is not really JSON";
        assert!(extract_json_output::<StructuredOutput>(raw).is_none());
    }

    // ── extract_json_output: all StructuredOutput fields ───────────────────

    #[test]
    fn extract_all_structured_fields() {
        let raw = r#"```json
{
  "action": "record_memory",
  "params": {"tier": "semantic", "content": "User likes Rust"},
  "rationale": "Long-term preference worth remembering",
  "reply": "Noted, you like Rust!"
}
```"#;
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("record_memory"));
        assert_eq!(out.rationale.as_deref(), Some("Long-term preference worth remembering"));
        assert_eq!(out.reply.as_deref(), Some("Noted, you like Rust!"));
        assert_eq!(out.params["tier"], "semantic");
        assert_eq!(out.params["content"], "User likes Rust");
    }

    // ── Provider enum serde ────────────────────────────────────────────────

    #[test]
    fn provider_serde_roundtrip() {
        for provider in [Provider::Ollama, Provider::OpenRouter] {
            let json = serde_json::to_string(&provider).unwrap();
            let back: Provider = serde_json::from_str(&json).unwrap();
            assert_eq!(back, provider);
        }
    }

    // ── extract_json_output edge cases ─────────────────────────────────────

    /// When two bare JSON objects appear in one response the bare strategy
    /// grabs first '{' to last '}', which may span across both objects.
    /// Fenced blocks don't have this problem, so verify fenced takes
    /// precedence even when bare objects are also present.
    #[test]
    fn extract_fenced_takes_precedence_over_bare() {
        let raw = r#"Bare: {"action":"wrong","reply":"no"}
```json
{"action":"right","reply":"yes"}
```
"#;
        let out = extract_json_output::<StructuredOutput>(raw).unwrap();
        assert_eq!(out.action.as_deref(), Some("right"));
        assert_eq!(out.reply.as_deref(), Some("yes"));
    }

    /// Two bare JSON objects with no fence — the strategy spans first '{' to
    /// last '}' which combines them into invalid JSON. Ensure we return None
    /// rather than silently merging.
    #[test]
    fn extract_two_bare_objects_returns_none() {
        let raw = r#"Here: {"action":"a"} and also {"action":"b"}"#;
        // first '{' to last '}' = `{"action":"a"} and also {"action":"b"}`
        // which is invalid JSON, so we should get None.
        assert!(extract_json_output::<StructuredOutput>(raw).is_none());
    }
}
