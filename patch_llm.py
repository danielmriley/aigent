import re

with open('crates/aigent-llm/src/lib.rs', 'r') as f:
    content = f.read()

# Add tokio::sync::mpsc to imports
content = content.replace('use std::time::Duration;', 'use std::time::Duration;\nuse tokio::sync::mpsc;')

# Add chat_stream to LlmClient trait
trait_old = """#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat(&self, prompt: &str) -> Result<String>;
}"""

trait_new = """#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat(&self, prompt: &str) -> Result<String>;
    async fn chat_stream(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<String>;
}"""

content = content.replace(trait_old, trait_new)

# Add chat_stream_with_fallback to LlmRouter
router_old = """    pub async fn chat_with_fallback(
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
    }"""

router_new = """    pub async fn chat_with_fallback(
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
    }"""

content = content.replace(router_old, router_new)

# Add chat_stream to OllamaClient
ollama_trait_old = """#[async_trait]
impl LlmClient for OllamaClient {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_model("llama3.1:8b", prompt).await
    }
}"""

ollama_trait_new = """#[async_trait]
impl LlmClient for OllamaClient {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_model("llama3.1:8b", prompt).await
    }
    async fn chat_stream(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<String> {
        self.chat_model_stream("llama3.1:8b", prompt, tx).await
    }
}"""

content = content.replace(ollama_trait_old, ollama_trait_new)

# Add chat_model_stream to OllamaClient
ollama_impl_old = """    async fn chat_model(&self, model: &str, prompt: &str) -> Result<String> {
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
    }"""

ollama_impl_new = """    async fn chat_model(&self, model: &str, prompt: &str) -> Result<String> {
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
    }"""

content = content.replace(ollama_impl_old, ollama_impl_new)

# Add chat_stream to OpenRouterClient
openrouter_trait_old = """#[async_trait]
impl LlmClient for OpenRouterClient {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_model("openai/gpt-4o-mini", prompt).await
    }
}"""

openrouter_trait_new = """#[async_trait]
impl LlmClient for OpenRouterClient {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_model("openai/gpt-4o-mini", prompt).await
    }
    async fn chat_stream(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<String> {
        self.chat_model_stream("openai/gpt-4o-mini", prompt, tx).await
    }
}"""

content = content.replace(openrouter_trait_old, openrouter_trait_new)

# Add chat_model_stream to OpenRouterClient
openrouter_impl_old = """    async fn chat_model(&self, model: &str, prompt: &str) -> Result<String> {
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
    }"""

openrouter_impl_new = """    async fn chat_model(&self, model: &str, prompt: &str) -> Result<String> {
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
    }"""

content = content.replace(openrouter_impl_old, openrouter_impl_new)

with open('crates/aigent-llm/src/lib.rs', 'w') as f:
    f.write(content)
