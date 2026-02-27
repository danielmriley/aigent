use std::io;
use std::io::IsTerminal;

use anyhow::{Result, bail};

use aigent_config::AppConfig;
use aigent_runtime::{BackendEvent, DaemonClient};
use aigent_llm::{list_ollama_models, list_openrouter_models};

use crate::CliModelProvider;

pub(crate) async fn run_interactive_session(config: &AppConfig, daemon: DaemonClient) -> Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return run_interactive_line_session(daemon).await;
    }

    let (backend_tx, backend_rx) = aigent_ui::tui::create_backend_channel();
    let mut app = aigent_ui::App::new(backend_rx, config);
    app.push_assistant_message(format!(
        "{} is online. Type /help for commands.",
        config.agent.name
    ));

    // Subscribe to broadcast events from external sources (e.g. Telegram)
    {
        let backend_tx_sub = backend_tx.clone();
        let daemon_sub = daemon.clone();
        tokio::spawn(async move {
            if let Err(err) = daemon_sub.subscribe(backend_tx_sub).await {
                tracing::debug!("subscribe task ended: {err}");
            }
        });
    }

    aigent_ui::tui::run_app_with(&mut app, |command| {
        let backend_tx = backend_tx.clone();
        let daemon = daemon.clone();

        async move {
            match command {
                aigent_ui::UiCommand::Quit => {}
                aigent_ui::UiCommand::Submit(line) => {
                    if line == "/help" {
                        let _ = backend_tx.send(BackendEvent::Token(
                            "Commands: /help, /status, /memory, /sleep, /dedup, /tools, /tools run <name> {args}, /model show, /model list [ollama|openrouter], /model provider <ollama|openrouter>, /model set <model>, /think <low|balanced|deep>, /exit".to_string(),
                        ));
                        let _ = backend_tx.send(BackendEvent::Done);
                        return Ok(());
                    }
                    if line == "/status" {
                        match daemon.get_status().await {
                            Ok(status) => {
                                let text = format!(
                                    "bot: {}\nprovider: {}\nmodel: {}\nthinking: {}\nmemory: {} total (core={} profile={} reflective={} semantic={} episodic={})\nuptime: {}s",
                                    status.bot_name,
                                    status.provider,
                                    status.model,
                                    status.thinking_level,
                                    status.memory_total,
                                    status.memory_core,
                                    status.memory_user_profile,
                                    status.memory_reflective,
                                    status.memory_semantic,
                                    status.memory_episodic,
                                    status.uptime_secs
                                );
                                let _ = backend_tx.send(BackendEvent::Token(text));
                                let _ = backend_tx.send(BackendEvent::Done);
                            }
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(err.to_string()));
                            }
                        }
                        return Ok(());
                    }
                    if line == "/memory" {
                        match daemon.get_memory_peek(5).await {
                            Ok(peek) => {
                                let text = if peek.is_empty() {
                                    "(no memory entries)".to_string()
                                } else {
                                    peek.join("\n")
                                };
                                let _ = backend_tx.send(BackendEvent::Token(text));
                                let _ = backend_tx.send(BackendEvent::Done);
                            }
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(err.to_string()));
                            }
                        }
                        return Ok(());
                    }

                    if line == "/sleep" {
                        // Show spinner immediately so the user knows we're working.
                        let _ = backend_tx.send(BackendEvent::SleepCycleRunning);
                        // Spawn into a background task so the TUI event loop
                        // keeps running (ticks, redraws, key events).  Without
                        // this the `tokio::select!` loop in tui.rs is blocked
                        // for the entire duration of the sleep cycle.
                        let tx_spawn = backend_tx.clone();
                        tokio::spawn(async move {
                            let tx_progress = tx_spawn.clone();
                            match daemon
                                .run_sleep_cycle_with_progress(|msg| {
                                    let _ = tx_progress
                                        .send(BackendEvent::SleepProgress(msg.to_string()));
                                })
                                .await
                            {
                                Ok(msg) => {
                                    let _ = tx_spawn.send(BackendEvent::Token(msg));
                                    let _ = tx_spawn.send(BackendEvent::Done);
                                }
                                Err(err) => {
                                    let _ = tx_spawn.send(BackendEvent::Error(err.to_string()));
                                }
                            }
                        });
                        return Ok(());
                    }

                    if line == "/dedup" {
                        let _ = backend_tx.send(BackendEvent::SleepProgress(
                            "Running content deduplication…".to_string(),
                        ));
                        let tx_spawn = backend_tx.clone();
                        tokio::spawn(async move {
                            match daemon.deduplicate_memory().await {
                                Ok(msg) => {
                                    let _ = tx_spawn.send(BackendEvent::Token(msg));
                                    let _ = tx_spawn.send(BackendEvent::Done);
                                }
                                Err(err) => {
                                    let _ = tx_spawn.send(BackendEvent::Error(err.to_string()));
                                }
                            }
                        });
                        return Ok(());
                    }

                    if line == "/model show" {
                        match daemon.get_status().await {
                            Ok(status) => {
                                let text = format!(
                                    "provider: {}\nmodel: {}\nthinking: {}",
                                    status.provider, status.model, status.thinking_level
                                );
                                let _ = backend_tx.send(BackendEvent::Token(text));
                                let _ = backend_tx.send(BackendEvent::Done);
                            }
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(err.to_string()));
                            }
                        }
                        return Ok(());
                    }

                    if line == "/model list" || line.starts_with("/model list ") {
                        let provider = parse_model_provider_from_command(&line);
                        match collect_model_lines(provider).await {
                            Ok(lines) => {
                                let _ = backend_tx.send(BackendEvent::Token(lines.join("\n")));
                                let _ = backend_tx.send(BackendEvent::Done);
                            }
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(err.to_string()));
                            }
                        }
                        return Ok(());
                    }

                    if let Some(provider) = line.strip_prefix("/model provider ") {
                        let msg = update_model_provider(provider.trim(), &daemon).await?;
                        let _ = backend_tx.send(BackendEvent::Token(msg));
                        let _ = backend_tx.send(BackendEvent::Done);
                        return Ok(());
                    }

                    if let Some(model) = line.strip_prefix("/model set ") {
                        let msg = update_model_selection(model.trim(), &daemon).await?;
                        let _ = backend_tx.send(BackendEvent::Token(msg));
                        let _ = backend_tx.send(BackendEvent::Done);
                        return Ok(());
                    }

                    if let Some(level) = line.strip_prefix("/think ") {
                        let msg = update_thinking_level(level.trim(), &daemon).await?;
                        let _ = backend_tx.send(BackendEvent::Token(msg));
                        let _ = backend_tx.send(BackendEvent::Done);
                        return Ok(());
                    }

                    if line == "/tools" {
                        match daemon.list_tools().await {
                            Ok(specs) => {
                                if specs.is_empty() {
                                    let _ = backend_tx.send(BackendEvent::Token("(no tools registered)".to_string()));
                                } else {
                                    let mut text = String::from("**Available tools:**\n");
                                    for spec in &specs {
                                        text.push_str(&format!("- **{}** — {}\n", spec.name, spec.description));
                                        for p in &spec.params {
                                            let req = if p.required { " (required)" } else { "" };
                                            text.push_str(&format!("  - `{}`{}: {}\n", p.name, req, p.description));
                                        }
                                    }
                                    let _ = backend_tx.send(BackendEvent::Token(text));
                                }
                                let _ = backend_tx.send(BackendEvent::Done);
                            }
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(err.to_string()));
                            }
                        }
                        return Ok(());
                    }

                    if line.starts_with("/tools run ") {
                        let rest = line.strip_prefix("/tools run ").unwrap().trim();
                        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                        let tool_name = parts[0].to_string();
                        let raw_args = parts.get(1).unwrap_or(&"{}");
                        let args: std::collections::HashMap<String, String> = match serde_json::from_str(raw_args) {
                            Ok(a) => a,
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(format!("invalid args JSON: {err}")));
                                return Ok(());
                            }
                        };
                        match daemon.execute_tool(&tool_name, args).await {
                            Ok((success, output)) => {
                                let status = if success { "success" } else { "failed" };
                                let text = format!("**Tool {status}:**\n```\n{output}\n```");
                                let _ = backend_tx.send(BackendEvent::Token(text));
                                let _ = backend_tx.send(BackendEvent::Done);
                            }
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(err.to_string()));
                            }
                        }
                        return Ok(());
                    }

                    tokio::spawn(async move {
                        let _ = daemon.stream_submit(line, "tui", backend_tx.clone()).await;
                    });
                }
            }
            Ok(())
        }
    })
    .await
}

pub(crate) async fn run_interactive_line_session(daemon: DaemonClient) -> Result<()> {
    println!("interactive mode");
    println!("commands: /model show|list [ollama|openrouter]|provider <ollama|openrouter>|set <model>");
    println!("          /think <low|balanced|deep>, /status, /memory, /help, /exit");
    println!("or type any message to chat with Aigent");

    let stdin = io::stdin();
    loop {
        let mut line = String::new();
        let bytes = stdin.read_line(&mut line)?;
        if bytes == 0 {
            println!("session closed");
            break;
        }
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if line == "/exit" {
            println!("session closed");
            break;
        }

        if line == "/help" {
            println!("/help");
            println!("/status");
            println!("/memory");
            println!("/sleep  -- trigger an agentic sleep cycle now");
            println!("/dedup  -- remove content-duplicate memory entries");
            println!("/model show");
            println!("/model list [ollama|openrouter]");
            println!("/model provider <ollama|openrouter>");
            println!("/model set <model>");
            println!("/think <low|balanced|deep>");
            println!("/exit");
            continue;
        }

        if line == "/status" {
            let status = daemon.get_status().await?;
            println!("bot: {}", status.bot_name);
            println!("provider: {}", status.provider);
            println!("model: {}", status.model);
            println!("thinking: {}", status.thinking_level);
            println!(
                "memory: {} total (core={} profile={} reflective={} semantic={} episodic={})",
                status.memory_total,
                status.memory_core,
                status.memory_user_profile,
                status.memory_reflective,
                status.memory_semantic,
                status.memory_episodic
            );
            println!("uptime: {}s", status.uptime_secs);
            continue;
        }

        if line == "/memory" {
            let peek = daemon.get_memory_peek(5).await?;
            if peek.is_empty() {
                println!("(no memory entries)");
            } else {
                println!("{}", peek.join("\n"));
            }
            continue;
        }

        if line == "/sleep" {
            println!("Starting sleep cycle…");
            match daemon
                .run_sleep_cycle_with_progress(|msg| println!("  {msg}"))
                .await
            {
                Ok(msg) => println!("{msg}"),
                Err(err) => eprintln!("error: {err}"),
            }
            continue;
        }

        if line == "/dedup" {
            println!("Running content deduplication…");
            match daemon.deduplicate_memory().await {
                Ok(msg) => println!("{msg}"),
                Err(err) => eprintln!("error: {err}"),
            }
            continue;
        }

        if line == "/model show" {
            let status = daemon.get_status().await?;
            println!("provider: {}", status.provider);
            println!("model: {}", status.model);
            println!("thinking: {}", status.thinking_level);
            continue;
        }

        if line == "/model list" || line.starts_with("/model list ") {
            let provider = parse_model_provider_from_command(line);
            let lines = collect_model_lines(provider).await?;
            println!("{}", lines.join("\n"));
            continue;
        }

        if let Some(provider) = line.strip_prefix("/model provider ") {
            println!("{}", update_model_provider(provider.trim(), &daemon).await?);
            continue;
        }

        if let Some(model) = line.strip_prefix("/model set ") {
            println!("{}", update_model_selection(model.trim(), &daemon).await?);
            continue;
        }

        if let Some(level) = line.strip_prefix("/think ") {
            println!("{}", update_thinking_level(level.trim(), &daemon).await?);
            continue;
        }

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        daemon.stream_submit(line.to_string(), "tui", tx).await?;
        while let Ok(event) = rx.try_recv() {
            match event {
                BackendEvent::Token(chunk) => print!("{chunk}"),
                BackendEvent::Done => {
                    println!();
                    break;
                }
                BackendEvent::Error(err) => {
                    println!("error: {err}");
                    break;
                }
                _ => {}
            }
        }
    }

    Ok(())
}

pub(crate) fn parse_model_provider_from_command(line: &str) -> CliModelProvider {
    if let Some(raw) = line.strip_prefix("/model list") {
        let provider = raw.trim().to_lowercase();
        if provider == "ollama" {
            return CliModelProvider::Ollama;
        }
        if provider == "openrouter" {
            return CliModelProvider::Openrouter;
        }
    }
    CliModelProvider::All
}

pub(crate) async fn update_model_provider(raw: &str, daemon: &DaemonClient) -> Result<String> {
    let provider = raw.trim().to_lowercase();
    if provider != "ollama" && provider != "openrouter" {
        bail!("invalid provider, expected ollama or openrouter");
    }

    let mut config = AppConfig::load_from("config/default.toml")?;
    config.llm.provider = provider.clone();
    config.save_to("config/default.toml")?;
    daemon.reload_config().await?;
    let status = daemon.get_status().await?;

    Ok(format!(
        "provider updated to {} • active model: {} • thinking: {}",
        status.provider, status.model, status.thinking_level
    ))
}

pub(crate) async fn update_model_selection(raw: &str, daemon: &DaemonClient) -> Result<String> {
    let model = raw.trim();
    if model.is_empty() {
        bail!("model cannot be empty");
    }

    let mut config = AppConfig::load_from("config/default.toml")?;
    if config.llm.provider.eq_ignore_ascii_case("openrouter") {
        config.llm.openrouter_model = model.to_string();
    } else {
        config.llm.ollama_model = model.to_string();
    }
    config.save_to("config/default.toml")?;
    daemon.reload_config().await?;
    let status = daemon.get_status().await?;

    Ok(format!(
        "model updated to {} • active provider: {} • active model: {}",
        model, status.provider, status.model
    ))
}

pub(crate) async fn update_thinking_level(raw: &str, daemon: &DaemonClient) -> Result<String> {
    let level = match raw.trim().to_lowercase().as_str() {
        "low" => "low",
        "balanced" => "balanced",
        "deep" => "deep",
        _ => bail!("invalid thinking level, expected low, balanced, or deep"),
    };

    let mut config = AppConfig::load_from("config/default.toml")?;
    config.agent.thinking_level = level.to_string();
    config.save_to("config/default.toml")?;
    daemon.reload_config().await?;
    let status = daemon.get_status().await?;

    Ok(format!(
        "thinking level updated to {} • active provider: {} • active model: {}",
        status.thinking_level, status.provider, status.model
    ))
}

pub(crate) async fn collect_model_lines(provider: CliModelProvider) -> Result<Vec<String>> {
    let mut lines = Vec::new();

    if matches!(provider, CliModelProvider::All | CliModelProvider::Ollama) {
        let ollama = list_ollama_models().await?;
        lines.push(format!("ollama models ({})", ollama.len()));
        lines.extend(ollama.into_iter().map(|model| format!("- {model}")));
    }

    if matches!(
        provider,
        CliModelProvider::All | CliModelProvider::Openrouter
    ) {
        let openrouter = list_openrouter_models().await?;
        lines.push(format!("openrouter models ({})", openrouter.len()));
        lines.extend(openrouter.into_iter().map(|model| format!("- {model}")));
    }

    Ok(lines)
}

