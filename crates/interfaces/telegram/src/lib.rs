use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use anyhow::{Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;

use aigent_runtime::{BackendEvent, DaemonClient};
use aigent_llm::{list_ollama_models, list_openrouter_models};
use tokio::sync::mpsc;

pub async fn start_bot(client_ipc: DaemonClient) -> Result<()> {
    let token = std::env::var("TELEGRAM_BOT_TOKEN")
        .map_err(|_| anyhow::anyhow!("TELEGRAM_BOT_TOKEN is not set"))?;
    if token.trim().is_empty() {
        bail!("TELEGRAM_BOT_TOKEN is empty");
    }

    let client = Client::new();
    let base_url = format!("https://api.telegram.org/bot{token}");
    let mut offset: i64 = 0;
    let mut recent_turns: HashMap<i64, VecDeque<String>> = HashMap::new();

    println!("telegram mode initialized");
    println!("listening for updates...");

    loop {
        let updates = match fetch_updates(&client, &base_url, offset).await {
            Ok(u) => u,
            Err(err) => {
                let err_str = err.to_string();
                if err_str.contains("409") {
                    // Another instance is polling — back off and let it win.
                    eprintln!("[telegram] 409 Conflict: another bot instance is running; waiting 15s before retrying");
                    tokio::time::sleep(Duration::from_secs(15)).await;
                } else {
                    eprintln!("[telegram] getUpdates error: {err} — retrying in 5s");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
                continue;
            }
        };

        for update in updates {
            offset = update.update_id + 1;

            let Some(message) = update.message else {
                continue;
            };
            let Some(text) = message.text else {
                continue;
            };

            let chat_id = message.chat.id;

            // ── Typing indicator ────────────────────────────────────────────────────
            // Fire a background task that periodically sends "typing" until
            // the response is ready.  Telegram shows the indicator for ~5 s
            // after each sendChatAction call, so we refresh every 4 s.
            let (cancel_typing_tx, cancel_typing_rx) = tokio::sync::oneshot::channel::<()>();
            {
                let typing_client = client.clone();
                let typing_url = base_url.clone();
                tokio::spawn(async move {
                    // Immediate first ping — user sees typing right away.
                    let _ = send_chat_action(&typing_client, &typing_url, chat_id, "typing").await;
                    let mut interval = tokio::time::interval(Duration::from_secs(4));
                    // Avoid burst-retrying if a tick was missed (e.g. during a slow tool call).
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    interval.tick().await; // consume the immediately-firing first tick
                    let mut cancel = cancel_typing_rx;
                    loop {
                        tokio::select! {
                            biased;
                            _ = &mut cancel => break,
                            _ = interval.tick() => {
                                // Re-send every 4 s so the indicator survives long tool calls,
                                // reflection passes, and multi-step reasoning.
                                let _ = send_chat_action(&typing_client, &typing_url, chat_id, "typing").await;
                            }
                        }
                    }
                });
            }

            let response = match handle_telegram_input(
                &client_ipc,
                chat_id,
                text.trim(),
                &mut recent_turns,
            )
            .await
            {
                Ok(r) => r,
                Err(err) => {
                    eprintln!("[telegram] handler error for chat {chat_id}: {err}");
                    format!("⚠️ error: {err}")
                }
            };

            // Cancel the typing indicator before sending the reply.
            let _ = cancel_typing_tx.send(());

            for chunk in chunk_message(&response, 3500) {
                if let Err(err) = send_message(&client, &base_url, chat_id, &chunk).await {
                    eprintln!("[telegram] sendMessage error for chat {chat_id}: {err}");
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

async fn handle_telegram_input(
    daemon: &DaemonClient,
    chat_id: i64,
    line: &str,
    recent_turns_by_chat: &mut HashMap<i64, VecDeque<String>>,
) -> Result<String> {
    let line = normalize_telegram_command(line);
    if line == "/start" || line == "/help" {
        return Ok([
            "/help",
            "/status",
            "/context",
            "/model show",
            "/model list [ollama|openrouter]",
            "/model provider <ollama|openrouter>",
            "/model set <model>",
            "/model test",
            "/think <low|balanced|deep>",
            "",
            "Send any other message to chat with Aigent.",
        ]
        .join("\n"));
    }

    if line == "/status" {
        let status = daemon.get_status().await?;
        return Ok([
            format!("bot: {}", status.bot_name),
            format!("provider: {}", status.provider),
            format!("model: {}", status.model),
            format!("thinking: {}", status.thinking_level),
            format!("stored memories: {}", status.memory_total),
            format!("daemon uptime: {}s", status.uptime_secs),
        ]
        .join("\n"));
    }

    if line == "/context" {
        return Ok(
            "/context is provided by daemon tools in unified mode; try asking directly in chat."
                .to_string(),
        );
    }

    if line == "/model show" {
        let status = daemon.get_status().await?;
        return Ok(format!(
            "provider: {}\nmodel: {}",
            status.provider, status.model
        ));
    }

    if line == "/model list" || line.starts_with("/model list ") {
        let provider = parse_model_provider_from_command(&line);
        let models = collect_model_lines(provider).await?;
        return Ok(models.join("\n"));
    }

    if let Some(provider) = line.strip_prefix("/model provider ") {
        let _ = provider;
        return Ok(
            "/model provider is not supported from Telegram in unified daemon mode".to_string(),
        );
    }

    if let Some(model) = line.strip_prefix("/model set ") {
        let _ = model;
        return Ok("/model set is not supported from Telegram in unified daemon mode".to_string());
    }

    if line == "/model test" {
        return Ok("/model test is not supported from Telegram in unified daemon mode".to_string());
    }

    if line.starts_with("/model key ") {
        return Ok("/model key is disabled in Telegram mode; update .env locally.".to_string());
    }

    if let Some(level) = line.strip_prefix("/think ") {
        let _ = level;
        return Ok("/think is not supported from Telegram in unified daemon mode".to_string());
    }

    if line.starts_with('/') {
        return Ok("unknown command. use /help".to_string());
    }

    let recent_turns = recent_turns_by_chat.entry(chat_id).or_default();
    recent_turns.push_back(line.to_string());
    while recent_turns.len() > 8 {
        let _ = recent_turns.pop_front();
    }

    let (tx, mut rx) = mpsc::unbounded_channel();
    // stream_submit reads the socket until Done/Error, then returns.
    // We drive both concurrently: collecting tokens while the streaming task runs.
    let submit_task = tokio::spawn({
        let daemon = daemon.clone();
        let line = line.to_string();
        async move { daemon.stream_submit(line, "telegram", tx).await }
    });

    let mut out = String::new();
    let mut tools_used: Vec<String> = Vec::new();
    while let Some(event) = rx.recv().await {
        match event {
            BackendEvent::Token(chunk) => out.push_str(&chunk),
            BackendEvent::Error(err) => {
                let _ = submit_task.await;
                return Ok(format!("error: {err}"));
            }
            BackendEvent::Done => break,
            // Lifecycle / meta events — safe to ignore in a per-request
            // Telegram response context (not a persistent subscriber).
            BackendEvent::Thinking
            | BackendEvent::SleepCycleRunning
            | BackendEvent::SleepProgress(_)
            | BackendEvent::MemoryUpdated
            | BackendEvent::ToolCallStart(_)
            | BackendEvent::ExternalTurn { .. }
            | BackendEvent::ClearStream => {}
            BackendEvent::ToolCallEnd(result) => {
                tools_used.push(result.name.clone());
            }
            // Phase-2 events: log at debug level.  These arrive when inline
            // reflection adds beliefs or insights after the turn.  The Telegram
            // handler does not render them inline (they reach Telegram users via
            // the broadcast Subscribe channel).
            BackendEvent::ReflectionInsight(ref insight) => {
                debug!(insight, "telegram: reflection insight received");
            }
            BackendEvent::BeliefAdded { ref claim, confidence } => {
                debug!(claim, confidence, "telegram: belief added");
            }
            BackendEvent::ProactiveMessage { ref content } => {
                // Proactive messages on a per-request stream are unexpected;
                // they normally arrive via the background broadcast channel.
                debug!(len = content.len(), "telegram: unexpected proactive message on request stream");
            }
        }
    }
    // Propagate IPC-level errors from stream_submit.
    match submit_task.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => return Ok(format!("error: {err}")),
        Err(err) => return Ok(format!("error: task panicked: {err}")),
    }

    if !tools_used.is_empty() {
        let footnote = format!("\n\n_\u{1f527} Tools used: {}_", tools_used.join(", "));
        out.push_str(&footnote);
    }

    if out.trim().is_empty() {
        Ok("(no response)".to_string())
    } else {
        Ok(out)
    }
}

async fn fetch_updates(
    client: &Client,
    base_url: &str,
    offset: i64,
) -> Result<Vec<TelegramUpdate>> {
    let url = format!("{base_url}/getUpdates");
    let response = client
        .get(url)
        .query(&[("timeout", "25"), ("offset", &offset.to_string())])
        .send()
        .await?
        .error_for_status()?;

    let payload: TelegramResponse<Vec<TelegramUpdate>> = response.json().await?;
    if !payload.ok {
        let description = payload
            .description
            .unwrap_or_else(|| "telegram getUpdates failed".to_string());
        bail!(description);
    }

    Ok(payload.result.unwrap_or_default())
}

async fn send_message(client: &Client, base_url: &str, chat_id: i64, text: &str) -> Result<()> {
    let url = format!("{base_url}/sendMessage");
    let body = SendMessageRequest {
        chat_id,
        text,
        disable_web_page_preview: true,
    };

    let response = client
        .post(url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;

    let payload: TelegramResponse<serde_json::Value> = response.json().await?;
    if !payload.ok {
        let description = payload
            .description
            .unwrap_or_else(|| "telegram sendMessage failed".to_string());
        bail!(description);
    }

    Ok(())
}

/// Send a `sendChatAction` request to Telegram (best-effort; errors are silently ignored).
async fn send_chat_action(
    client: &Client,
    base_url: &str,
    chat_id: i64,
    action: &str,
) -> Result<()> {
    let url = format!("{base_url}/sendChatAction");
    let body = serde_json::json!({ "chat_id": chat_id, "action": action });
    // Fire and forget — a failed typing indicator must never surface to the user.
    let _ = client.post(url).json(&body).send().await;
    Ok(())
}

fn normalize_telegram_command(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with('/') {
        return trimmed.to_string();
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let rest = parts.next().unwrap_or("").trim();

    let command = command
        .split_once('@')
        .map(|(base, _)| base)
        .unwrap_or(command);

    if rest.is_empty() {
        command.to_string()
    } else {
        format!("{command} {rest}")
    }
}

#[derive(Debug, Clone, Copy)]
enum ModelProviderFilter {
    All,
    Ollama,
    Openrouter,
}

fn parse_model_provider_from_command(line: &str) -> ModelProviderFilter {
    if let Some(raw) = line.strip_prefix("/model list") {
        let provider = raw.trim().to_lowercase();
        if provider == "ollama" {
            return ModelProviderFilter::Ollama;
        }
        if provider == "openrouter" {
            return ModelProviderFilter::Openrouter;
        }
    }
    ModelProviderFilter::All
}

async fn collect_model_lines(provider: ModelProviderFilter) -> Result<Vec<String>> {
    let mut lines = Vec::new();

    if matches!(
        provider,
        ModelProviderFilter::All | ModelProviderFilter::Ollama
    ) {
        let ollama = list_ollama_models().await?;
        lines.push(format!("ollama models ({})", ollama.len()));
        lines.extend(ollama.into_iter().map(|model| format!("- {model}")));
    }

    if matches!(
        provider,
        ModelProviderFilter::All | ModelProviderFilter::Openrouter
    ) {
        let openrouter = list_openrouter_models().await?;
        lines.push(format!("openrouter models ({})", openrouter.len()));
        lines.extend(openrouter.into_iter().map(|model| format!("- {model}")));
    }

    Ok(lines)
}

fn chunk_message(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_len = 0;

    for line in text.lines() {
        let line_len = line.chars().count() + 1;
        if current_len > 0 && current_len + line_len > max_chars {
            chunks.push(current.trim_end().to_string());
            current.clear();
            current_len = 0;
        }
        current.push_str(line);
        current.push('\n');
        current_len += line_len;
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim_end().to_string());
    }

    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

#[derive(Debug, Deserialize)]
struct TelegramResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
}

#[derive(Debug, Deserialize)]
struct TelegramMessage {
    chat: TelegramChat,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramChat {
    id: i64,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest<'a> {
    chat_id: i64,
    text: &'a str,
    disable_web_page_preview: bool,
}

#[cfg(test)]
mod tests {
    use super::normalize_telegram_command;

    #[test]
    fn normalizes_bot_mentions_in_commands() {
        assert_eq!(normalize_telegram_command("/status@aigent_bot"), "/status");
        assert_eq!(
            normalize_telegram_command("/think@aigent_bot deep"),
            "/think deep"
        );
        assert_eq!(normalize_telegram_command(" hello "), "hello");
    }
}
