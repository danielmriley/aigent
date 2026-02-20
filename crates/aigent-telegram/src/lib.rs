use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use anyhow::{Result, bail};
use chrono::{Local, Timelike};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use aigent_daemon::{AgentRuntime, ConversationTurn};
use aigent_llm::{list_ollama_models, list_openrouter_models};
use aigent_memory::{MemoryManager, MemoryTier};

pub async fn start_bot(runtime: &mut AgentRuntime, memory: &mut MemoryManager) -> Result<()> {
    let token = std::env::var("TELEGRAM_BOT_TOKEN")
        .map_err(|_| anyhow::anyhow!("TELEGRAM_BOT_TOKEN is not set"))?;
    if token.trim().is_empty() {
        bail!("TELEGRAM_BOT_TOKEN is empty");
    }

    let client = Client::new();
    let base_url = format!("https://api.telegram.org/bot{token}");
    let mut offset: i64 = 0;
    let mut recent_turns: HashMap<i64, VecDeque<ConversationTurn>> = HashMap::new();
    let mut turn_counts: HashMap<i64, usize> = HashMap::new();

    println!("telegram mode initialized");
    println!("listening for updates...");

    loop {
        let updates = fetch_updates(&client, &base_url, offset).await?;
        for update in updates {
            offset = update.update_id + 1;

            let Some(message) = update.message else {
                continue;
            };
            let Some(text) = message.text else {
                continue;
            };

            let chat_id = message.chat.id;
            let response = handle_telegram_input(
                runtime,
                memory,
                chat_id,
                text.trim(),
                &mut recent_turns,
                &mut turn_counts,
            )
            .await?;

            for chunk in chunk_message(&response, 3500) {
                send_message(&client, &base_url, chat_id, &chunk).await?;
            }
        }

        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

async fn handle_telegram_input(
    runtime: &mut AgentRuntime,
    memory: &mut MemoryManager,
    chat_id: i64,
    line: &str,
    recent_turns_by_chat: &mut HashMap<i64, VecDeque<ConversationTurn>>,
    turn_counts: &mut HashMap<i64, usize>,
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
        let recent = recent_turns_by_chat
            .get(&chat_id)
            .map(|turns| turns.len())
            .unwrap_or(0);
        return Ok([
            format!("bot: {}", runtime.config.agent.name),
            format!("provider: {}", runtime.config.llm.provider),
            format!("model: {}", runtime.config.active_model()),
            format!("thinking: {}", runtime.config.agent.thinking_level),
            format!("stored memories: {}", memory.all().len()),
            format!("recent conversation turns: {recent}"),
            format!("sleep mode: {}", runtime.config.memory.auto_sleep_mode),
            format!(
                "night sleep window: {:02}:00-{:02}:00 (local)",
                runtime.config.memory.night_sleep_start_hour,
                runtime.config.memory.night_sleep_end_hour
            ),
        ]
        .join("\n"));
    }

    if line == "/context" {
        let recent = recent_turns_by_chat
            .get(&chat_id)
            .map(|turns| turns.len())
            .unwrap_or(0);
        let snapshot = runtime.environment_snapshot(memory, recent);
        return Ok(format!("environment context:\n{snapshot}"));
    }

    if line == "/model show" {
        return Ok(format!(
            "provider: {}\nmodel: {}",
            runtime.config.llm.provider,
            runtime.config.active_model()
        ));
    }

    if line == "/model list" || line.starts_with("/model list ") {
        let provider = parse_model_provider_from_command(&line);
        let models = collect_model_lines(provider).await?;
        return Ok(models.join("\n"));
    }

    if let Some(provider) = line.strip_prefix("/model provider ") {
        let provider = provider.trim().to_lowercase();
        if provider == "ollama" || provider == "openrouter" {
            runtime.config.llm.provider = provider;
            runtime.config.save_to("config/default.toml")?;
            return Ok("provider updated".to_string());
        }
        return Ok("invalid provider, expected ollama or openrouter".to_string());
    }

    if let Some(model) = line.strip_prefix("/model set ") {
        let model = model.trim();
        if model.is_empty() {
            return Ok("model cannot be empty".to_string());
        }
        if runtime
            .config
            .llm
            .provider
            .eq_ignore_ascii_case("openrouter")
        {
            runtime.config.llm.openrouter_model = model.to_string();
        } else {
            runtime.config.llm.ollama_model = model.to_string();
        }
        runtime.config.save_to("config/default.toml")?;
        return Ok("model updated".to_string());
    }

    if line == "/model test" {
        let message = match runtime.test_model_connection().await {
            Ok(result) => format!("model test ok: {result}"),
            Err(error) => format!("model test failed: {error}"),
        };
        return Ok(message);
    }

    if line.starts_with("/model key ") {
        return Ok("/model key is disabled in Telegram mode; update .env locally.".to_string());
    }

    if let Some(level) = line.strip_prefix("/think ") {
        return match parse_thinking_level(level) {
            Some(level) => {
                runtime.config.agent.thinking_level = level.to_string();
                runtime.config.save_to("config/default.toml")?;
                Ok("thinking level updated".to_string())
            }
            None => Ok("invalid thinking level, expected low, balanced, or deep".to_string()),
        };
    }

    if line.starts_with('/') {
        return Ok("unknown command. use /help".to_string());
    }

    let recent_turns = recent_turns_by_chat.entry(chat_id).or_default();
    let recent_context = recent_turns.iter().cloned().collect::<Vec<_>>();

    memory.record(
        MemoryTier::Episodic,
        line.to_string(),
        format!("telegram:user:chat={chat_id}"),
    )?;

    let reply = runtime
        .respond_and_remember(memory, &line, &recent_context)
        .await?;

    memory.record(
        MemoryTier::Episodic,
        reply.clone(),
        format!("telegram:assistant:chat={chat_id}"),
    )?;

    recent_turns.push_back(ConversationTurn {
        user: line.to_string(),
        assistant: reply.clone(),
    });
    while recent_turns.len() > 8 {
        let _ = recent_turns.pop_front();
    }

    let turn_count = turn_counts.entry(chat_id).or_default();
    *turn_count += 1;

    if should_run_sleep_cycle(runtime, memory, *turn_count) {
        let summary = memory.run_sleep_cycle()?;
        return Ok(format!(
            "{reply}\n\ninternal sleep cycle: {}",
            summary.distilled
        ));
    }

    Ok(reply)
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

fn parse_thinking_level(raw: &str) -> Option<&'static str> {
    match raw.trim().to_lowercase().as_str() {
        "low" => Some("low"),
        "balanced" => Some("balanced"),
        "deep" => Some("deep"),
        _ => None,
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

fn should_run_sleep_cycle(
    runtime: &AgentRuntime,
    memory: &MemoryManager,
    turn_count: usize,
) -> bool {
    let mode = runtime.config.memory.auto_sleep_mode.trim().to_lowercase();

    if mode == "nightly" {
        let now = Local::now();
        let start = runtime.config.memory.night_sleep_start_hour.min(23);
        let end = runtime.config.memory.night_sleep_end_hour.min(23);
        let in_window = if start == end {
            true
        } else if start < end {
            (start..end).contains(&(now.hour() as u8))
        } else {
            (now.hour() as u8) >= start || (now.hour() as u8) < end
        };

        if !in_window {
            return false;
        }

        let today = now.date_naive();
        let already_slept_today = memory.all().iter().any(|entry| {
            entry.source.starts_with("sleep:cycle")
                && entry.created_at.with_timezone(&Local).date_naive() == today
        });

        return !already_slept_today;
    }

    let interval = runtime.config.memory.auto_sleep_turn_interval.max(1);
    turn_count % interval == 0
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
