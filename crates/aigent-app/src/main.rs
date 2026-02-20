mod tui;

use arboard::Clipboard;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::fs::OpenOptions;
use std::io;
use std::io::IsTerminal;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use chrono::{Local, Timelike};
use clap::{Parser, Subcommand, ValueEnum};
use crossterm::event::{
    self, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tracing_subscriber::EnvFilter;
use ignore::WalkBuilder;

use aigent_config::AppConfig;
use aigent_daemon::{AgentRuntime, ConversationTurn};
use aigent_llm::{list_ollama_models, list_openrouter_models};
use aigent_memory::event_log::{MemoryEventLog, MemoryRecordEvent};
use aigent_memory::{MemoryManager, MemoryTier};
use aigent_telegram::start_bot;
use aigent_ui::onboard::{run_configuration, run_onboarding};

#[derive(Debug, Parser)]
#[command(name = "aigent", version, about = "A persistent memory-centric AI agent")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Onboard,
    #[command(
        name = "configuration",
        visible_alias = "config",
        about = "Update bot settings via configuration wizard"
    )]
    Configuration,
    Start,
    #[command(hide = true)]
    Run,
    Telegram,
    Doctor {
        #[arg(long)]
        review_gate: bool,
        #[arg(long)]
        model_catalog: bool,
        #[arg(long, value_enum, default_value = "all")]
        provider: CliModelProvider,
        #[arg(long)]
        report: Option<String>,
    },
    Memory {
        #[command(subcommand)]
        command: MemoryCommands,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliThinkingLevel {
    Low,
    Balanced,
    Deep,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMemoryLayer {
    All,
    Episodic,
    Semantic,
    Procedural,
    Core,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliModelProvider {
    All,
    Ollama,
    Openrouter,
}

#[derive(Debug, Clone, Copy)]
enum CliDaemonChannel {
    Telegram,
}

#[derive(Debug, Subcommand)]
enum MemoryCommands {
    Wipe {
        #[arg(long, value_enum, default_value = "all")]
        layer: CliMemoryLayer,
        #[arg(long)]
        yes: bool,
    },
    Stats,
    InspectCore {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    Promotions {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    ExportVault {
        #[arg(long)]
        path: Option<String>,
    },
}

async fn fetch_available_models() -> aigent_ui::onboard::AvailableModels {
    println!("Fetching available models...");
    let ollama = aigent_llm::list_ollama_models().await.unwrap_or_default();
    let openrouter = aigent_llm::list_openrouter_models().await.unwrap_or_default();
    aigent_ui::onboard::AvailableModels { ollama, openrouter }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let mut config = AppConfig::load_from("config/default.toml")?;
    let config_exists = Path::new("config/default.toml").exists();
    let memory_log_path = Path::new(".aigent").join("memory").join("events.jsonl");

    if let Some(channel) = std::env::var("AIGENT_DAEMON_WORKER")
        .ok()
        .and_then(|raw| parse_daemon_channel(&raw).ok())
    {
        run_daemon_worker(channel, config, &memory_log_path).await?;
        return Ok(());
    }

    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if raw_args.first().map(|arg| arg == "daemon").unwrap_or(false) {
        run_daemon_command_from_args(&raw_args[1..])?;
        return Ok(());
    }

    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Start) {
        Commands::Onboard => {
            let models = fetch_available_models().await;
            run_onboarding(&mut config, models)?;
            config.save_to("config/default.toml")?;
            println!("{}", aigent_ui::tui::banner());
        }
        Commands::Configuration => {
            let models = fetch_available_models().await;
            run_configuration(&mut config, models)?;
            config.save_to("config/default.toml")?;
            println!("configuration updated");
        }
        Commands::Start | Commands::Run => {
            if !config_exists || config.needs_onboarding() {
                let models = fetch_available_models().await;
                run_onboarding(&mut config, models)?;
                config.save_to("config/default.toml")?;
            }

            run_start_mode(config, &memory_log_path).await?;
        }
        Commands::Telegram => {
            if !config_exists || config.needs_onboarding() {
                let models = fetch_available_models().await;
                run_onboarding(&mut config, models)?;
                config.save_to("config/default.toml")?;
            }

            run_telegram_runtime(config, &memory_log_path).await?;
        }
        Commands::Doctor {
            review_gate,
            model_catalog,
            provider,
            report,
        } => {
            let mut memory = MemoryManager::with_event_log(&memory_log_path)?;
            if model_catalog {
                let lines = collect_model_lines(provider).await?;
                for line in lines {
                    println!("{line}");
                }
            } else if review_gate {
                run_phase_review_gate(
                    &mut config,
                    &mut memory,
                    &memory_log_path,
                    report.as_deref(),
                )?;
            } else {
                println!("aigent doctor");
                println!("- memory log path: .aigent/memory/events.jsonl");
                println!("- memory entries loaded: {}", memory.all().len());
                println!("- provider: {}", config.llm.provider);
                println!("- model: {}", config.active_model());
                println!("- thinking level: {}", config.agent.thinking_level);
            }
        }
        Commands::Memory { command } => match command {
            MemoryCommands::Wipe { layer, yes } => {
                let mut memory = MemoryManager::with_event_log(memory_log_path)?;
                run_memory_wipe(&mut memory, layer, yes)?;
            }
            MemoryCommands::Stats => {
                let memory = MemoryManager::with_event_log(memory_log_path)?;
                run_memory_stats(&memory);
            }
            MemoryCommands::InspectCore { limit } => {
                let memory = MemoryManager::with_event_log(memory_log_path)?;
                run_memory_inspect_core(&memory, limit.max(1));
            }
            MemoryCommands::Promotions { limit } => {
                let memory = MemoryManager::with_event_log(memory_log_path)?;
                run_memory_promotions(&memory, limit.max(1));
            }
            MemoryCommands::ExportVault { path } => {
                let memory = MemoryManager::with_event_log(memory_log_path)?;
                let target = path.unwrap_or_else(|| ".aigent/vault".to_string());
                run_memory_export_vault(&memory, &target)?;
            }
        },
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct DaemonPaths {
    runtime_dir: PathBuf,
    pid_file: PathBuf,
    log_file: PathBuf,
    mode_file: PathBuf,
}

fn daemon_paths() -> DaemonPaths {
    let runtime_dir = Path::new(".aigent").join("runtime");
    DaemonPaths {
        pid_file: runtime_dir.join("daemon.pid"),
        log_file: runtime_dir.join("daemon.log"),
        mode_file: runtime_dir.join("daemon.mode"),
        runtime_dir,
    }
}

fn run_daemon_command_from_args(args: &[String]) -> Result<()> {
    if args.is_empty() {
        print_daemon_help();
        return Ok(());
    }

    let command = args[0].as_str();
    let force = args.iter().any(|arg| arg == "--force");

    match command {
        "start" => daemon_start(force),
        "stop" => daemon_stop(),
        "restart" => {
            daemon_stop()?;
            daemon_start(true)
        }
        "status" => daemon_status(),
        "help" | "--help" | "-h" => {
            print_daemon_help();
            Ok(())
        }
        _ => {
            print_daemon_help();
            bail!("unknown daemon command: {command}")
        }
    }
}

fn print_daemon_help() {
    println!("Manage background aigent service");
    println!("Usage: aigent daemon <start|stop|restart|status> [--force]");
}

fn parse_daemon_channel(raw: &str) -> Result<CliDaemonChannel> {
    if raw.eq_ignore_ascii_case("telegram") {
        Ok(CliDaemonChannel::Telegram)
    } else {
        bail!("unsupported daemon channel: {raw}")
    }
}

fn daemon_start(force: bool) -> Result<()> {
    let config = AppConfig::load_from("config/default.toml")?;
    if config.needs_onboarding() {
        bail!("onboarding not complete; run `aigent onboard` first");
    }

    if !config.integrations.telegram_enabled {
        bail!("no messaging integrations enabled; enable Telegram in `aigent configuration`");
    }

    if !telegram_token_configured() {
        bail!(
            "TELEGRAM_BOT_TOKEN is missing; add it via `aigent configuration` or set it in .env"
        );
    }

    let paths = daemon_paths();
    fs::create_dir_all(&paths.runtime_dir)?;

    if let Some(pid) = read_pid(&paths.pid_file)? {
        if is_pid_running(pid) {
            if !force {
                bail!(
                    "daemon already running with pid {pid}; use `aigent daemon restart` or `aigent daemon start --force`"
                );
            }
            terminate_pid(pid)?;
        }
        let _ = fs::remove_file(&paths.pid_file);
    }

    let exe = std::env::current_exe()?;
    let out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log_file)?;
    let err = out.try_clone()?;

    let child = Command::new(exe)
        .env("AIGENT_DAEMON_WORKER", daemon_channel_label(CliDaemonChannel::Telegram))
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()?;

    fs::write(&paths.pid_file, child.id().to_string())?;
    fs::write(&paths.mode_file, "telegram")?;

    println!("daemon started");
    println!("- pid: {}", child.id());
    println!("- integrations: telegram");
    println!("- log: {}", paths.log_file.display());
    Ok(())
}

fn daemon_stop() -> Result<()> {
    let paths = daemon_paths();
    let Some(pid) = read_pid(&paths.pid_file)? else {
        println!("daemon is not running");
        return Ok(());
    };

    if !is_pid_running(pid) {
        let _ = fs::remove_file(&paths.pid_file);
        println!("daemon was not running (stale pid file cleaned)");
        return Ok(());
    }

    terminate_pid(pid)?;
    let _ = fs::remove_file(&paths.pid_file);
    println!("daemon stopped (pid {pid})");
    Ok(())
}

fn daemon_status() -> Result<()> {
    let paths = daemon_paths();
    let mode = fs::read_to_string(&paths.mode_file)
        .unwrap_or_else(|_| "unknown".to_string())
        .trim()
        .to_string();

    if let Some(pid) = read_pid(&paths.pid_file)? {
        if is_pid_running(pid) {
            println!("daemon status: running");
            println!("- pid: {pid}");
            println!("- channel: {mode}");
            println!("- log: {}", paths.log_file.display());
            return Ok(());
        }
    }

    println!("daemon status: stopped");
    println!("- channel: {mode}");
    println!("- log: {}", paths.log_file.display());
    Ok(())
}

fn read_pid(path: &Path) -> Result<Option<u32>> {
    if !path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(path)?;
    let pid = raw.trim().parse::<u32>().ok();
    Ok(pid)
}

fn is_pid_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

fn terminate_pid(pid: u32) -> Result<()> {
    #[cfg(unix)]
    {
        let status = Command::new("kill").arg(pid.to_string()).status()?;
        if !status.success() {
            bail!("failed to terminate daemon pid {pid}");
        }
        Ok(())
    }

    #[cfg(not(unix))]
    {
        let _ = pid;
        bail!("daemon stop is only implemented on unix in this build")
    }
}

async fn run_daemon_worker(
    channel: CliDaemonChannel,
    config: AppConfig,
    memory_log_path: &Path,
) -> Result<()> {
    match channel {
        CliDaemonChannel::Telegram => run_telegram_runtime(config, memory_log_path).await,
    }
}

async fn run_start_mode(config: AppConfig, memory_log_path: &Path) -> Result<()> {
    let interactive_terminal = io::stdin().is_terminal() && io::stdout().is_terminal();

    if interactive_terminal {
        let mut service_worker = spawn_connected_service_worker(&config)?;

        let mut memory = MemoryManager::with_event_log(memory_log_path)?;
        seed_identity_memory(&mut memory, &config.agent.name)?;

        let mut runtime = AgentRuntime::new(config);
        runtime.run().await?;
        println!("{}", aigent_ui::tui::banner());
        let session_result = run_interactive_session(&mut runtime, &mut memory).await;

        if let Some(child) = service_worker.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }

        return session_result;
    }

    if config.integrations.telegram_enabled {
        return run_telegram_runtime(config, memory_log_path).await;
    }

    bail!(
        "no interactive terminal detected and no messaging integrations enabled; run in a terminal or enable Telegram in `aigent configuration`"
    )
}

fn spawn_connected_service_worker(config: &AppConfig) -> Result<Option<Child>> {
    if !config.integrations.telegram_enabled {
        return Ok(None);
    }

    if !telegram_token_configured() {
        println!(
            "warning: Telegram integration enabled but TELEGRAM_BOT_TOKEN is missing; running local TUI only"
        );
        return Ok(None);
    }

    let paths = daemon_paths();
    fs::create_dir_all(&paths.runtime_dir)?;
    let out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log_file)?;
    let err = out.try_clone()?;

    let child = Command::new(std::env::current_exe()?)
        .env("AIGENT_DAEMON_WORKER", "telegram")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()?;

    println!("connected services: telegram worker started (pid {})", child.id());
    Ok(Some(child))
}

async fn run_telegram_runtime(config: AppConfig, memory_log_path: &Path) -> Result<()> {
    if config.needs_onboarding() {
        bail!("onboarding not complete; run `aigent onboard` first");
    }

    if !config.integrations.telegram_enabled {
        bail!(
            "telegram integration is disabled; run `aigent configuration` and enable Telegram first"
        );
    }

    let mut memory = MemoryManager::with_event_log(memory_log_path)?;
    seed_identity_memory(&mut memory, &config.agent.name)?;

    let mut runtime = AgentRuntime::new(config);
    runtime.run().await?;
    start_bot(&mut runtime, &mut memory).await
}

fn daemon_channel_label(channel: CliDaemonChannel) -> &'static str {
    match channel {
        CliDaemonChannel::Telegram => "telegram",
    }
}

fn telegram_token_configured() -> bool {
    std::env::var("TELEGRAM_BOT_TOKEN")
        .ok()
        .map(|token| !token.trim().is_empty())
        .unwrap_or(false)
}

struct ExternalFeedState {
    known_event_ids: HashSet<String>,
    last_scan: Instant,
}

impl ExternalFeedState {
    fn from_log(event_log: &MemoryEventLog) -> Result<Self> {
        let known_event_ids = event_log
            .load()?
            .into_iter()
            .map(|event| event.event_id.to_string())
            .collect::<HashSet<_>>();

        Ok(Self {
            known_event_ids,
            last_scan: Instant::now(),
        })
    }
}

fn pull_telegram_events_into_transcript(
    event_log: &MemoryEventLog,
    state: &mut ExternalFeedState,
    transcript: &mut Vec<String>,
    auto_follow: bool,
    viewport_start_line: &mut usize,
) -> Result<()> {
    if state.last_scan.elapsed() < Duration::from_millis(300) {
        return Ok(());
    }
    state.last_scan = Instant::now();

    let mut new_lines = event_log
        .load()?
        .into_iter()
        .filter(|event| state.known_event_ids.insert(event.event_id.to_string()))
        .filter_map(telegram_transcript_line)
        .collect::<Vec<_>>();

    if new_lines.is_empty() {
        return Ok(());
    }

    transcript.append(&mut new_lines);
    if auto_follow {
        *viewport_start_line = usize::MAX;
    }
    Ok(())
}

fn telegram_transcript_line(event: MemoryRecordEvent) -> Option<String> {
    let source = event.entry.source;
    if source.starts_with("telegram:user:") {
        return Some(format!("you> [telegram] {}", event.entry.content));
    }
    if source.starts_with("telegram:assistant:") {
        return Some(format!("aigent> [telegram] {}", event.entry.content));
    }
    None
}

fn seed_identity_memory(memory: &mut MemoryManager, bot_name: &str) -> Result<()> {
    let marker = format!("my name is {bot_name}").to_lowercase();
    let already_seeded = memory
        .entries_by_tier(MemoryTier::Core)
        .iter()
        .any(|entry| entry.content.to_lowercase().contains(&marker));

    if !already_seeded {
        let identity = format!("My name is {bot_name}. I am your evolving AI companion.");
        memory.record(MemoryTier::Core, identity, "onboarding-identity")?;
    }

    Ok(())
}

async fn run_interactive_session(runtime: &mut AgentRuntime, memory: &mut MemoryManager) -> Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return run_interactive_line_session(runtime, memory).await;
    }

    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut transcript: Vec<String> = aigent_ui::tui::banner()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(ToString::to_string)
        .collect();
    transcript.push("Type /help for commands. Press Tab for autocomplete.".to_string());
    
    let workspace_files = collect_workspace_files(Path::new("."))?;
    let mut app = tui::App::new(transcript, workspace_files);
    let mut turn_count: usize = 0;
    let mut recent_turns: VecDeque<ConversationTurn> = VecDeque::new();
    let event_log = MemoryEventLog::new(Path::new(".aigent").join("memory").join("events.jsonl"));
    let mut external_feed = ExternalFeedState::from_log(&event_log)?;

    let (tx, mut rx) = tokio::sync::mpsc::channel(100);
    let tick_rate = Duration::from_millis(120);

    // Event listener task
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        loop {
            if event::poll(tick_rate).unwrap_or(false) {
                if let Ok(event) = event::read() {
                    match event {
                        Event::Key(key) => {
                            let _ = tx_clone.send(tui::Event::Key(key)).await;
                        }
                        Event::Paste(text) => {
                            let _ = tx_clone.send(tui::Event::Paste(text)).await;
                        }
                        _ => {}
                    }
                }
            } else {
                let _ = tx_clone.send(tui::Event::Tick).await;
            }
        }
    });

    let result = async {
        loop {
            pull_telegram_events_into_transcript(
                &event_log,
                &mut external_feed,
                &mut app.chat_panel.transcript,
                app.chat_panel.auto_follow,
                &mut app.chat_panel.viewport_start_line,
            )?;

            app.refresh_file_popup();
            let suggestions = command_suggestions(&app.input_box.text());
            let viewport = app.draw(&mut terminal, &suggestions, &runtime.config)?;

            if let Some(event) = rx.recv().await {
                match event {
                    tui::Event::Key(key) => {
                        if key.kind != KeyEventKind::Press {
                            continue;
                        }

                        if app.handle_file_popup_key(key) {
                            continue;
                        }

                        if app.chat_panel.history_mode {
                            match key.code {
                                KeyCode::Esc => {
                                    app.chat_panel.exit_history_mode();
                                }
                                KeyCode::Up => {
                                    app.chat_panel.select_prev_message();
                                }
                                KeyCode::Down => {
                                    app.chat_panel.select_next_message();
                                }
                                KeyCode::Char('c') => {
                                    if let Some(selected) = app.chat_panel.selected_transcript_entry() {
                                        if let Some(assistant) = selected.strip_prefix("aigent> ") {
                                            if let Some(code) = tui::extract_first_code_block(assistant) {
                                                match Clipboard::new().and_then(|mut cb| cb.set_text(code.clone())) {
                                                    Ok(_) => {
                                                        app.chat_panel
                                                            .transcript
                                                            .push("aigent> copied code block to clipboard".to_string());
                                                    }
                                                    Err(err) => {
                                                        app.chat_panel
                                                            .transcript
                                                            .push(format!("aigent> clipboard error: {err}"));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                KeyCode::Char('a') => {
                                    if let Some(selected) = app.chat_panel.selected_transcript_entry() {
                                        if let Some(assistant) = selected.strip_prefix("aigent> ") {
                                            if let Some(code) = tui::extract_first_code_block(assistant) {
                                                let dir = Path::new(".aigent").join("snippets");
                                                fs::create_dir_all(&dir)?;
                                                let file_name = format!(
                                                    "applied_{}.txt",
                                                    Local::now().format("%Y%m%d_%H%M%S")
                                                );
                                                let path = dir.join(file_name);
                                                fs::write(&path, code)?;
                                                app.chat_panel.transcript.push(format!(
                                                    "aigent> applied code block to {}",
                                                    path.display()
                                                ));
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                            continue;
                        }

                        match key.code {
                            KeyCode::Esc => {
                                app.chat_panel.enter_history_mode();
                            }
                            KeyCode::PageUp => {
                                let page_step = usize::from(terminal.size()?.height.saturating_sub(6)).max(1);
                                let from_line = if app.chat_panel.auto_follow { viewport.max_scroll } else { viewport.start };
                                app.chat_panel.viewport_start_line = from_line.saturating_sub(page_step);
                                app.chat_panel.auto_follow = false;
                            }
                            KeyCode::PageDown => {
                                let page_step = usize::from(terminal.size()?.height.saturating_sub(6)).max(1);
                                let from_line = if app.chat_panel.auto_follow { viewport.max_scroll } else { viewport.start };
                                app.chat_panel.viewport_start_line = from_line.saturating_add(page_step).min(viewport.max_scroll);
                                if app.chat_panel.viewport_start_line >= viewport.max_scroll {
                                    app.chat_panel.auto_follow = true;
                                }
                            }
                            KeyCode::End => {
                                app.chat_panel.viewport_start_line = viewport.max_scroll;
                                app.chat_panel.auto_follow = true;
                            }
                            KeyCode::Tab => {
                                let input = app.input_box.text();
                                if let Some(suggestion) = command_suggestions(&input).first() {
                                    app.input_box.textarea = tui_textarea::TextArea::default();
                                    app.input_box.textarea.insert_str(*suggestion);
                                }
                            }
                            KeyCode::Enter => {
                                if key.modifiers.contains(KeyModifiers::ALT) {
                                    app.input_box.textarea.insert_newline();
                                    continue;
                                }

                                let line = app.input_box.textarea.lines().join("\n").trim().to_string();
                                app.input_box.textarea = tui_textarea::TextArea::default();
                                if line.is_empty() {
                                    continue;
                                }

                                app.chat_panel.transcript.push(format!("you> {line}"));
                                let is_chat_turn = !line.starts_with('/');
                                if is_chat_turn {
                                    app.chat_panel.is_thinking = true;
                                    app.chat_panel.thinking_spinner_tick = 0;
                                }
                                if app.chat_panel.auto_follow {
                                    app.chat_panel.viewport_start_line = viewport.max_scroll;
                                }

                                // Spawn LLM task
                                let tx_llm = tx.clone();
                                // We need to clone runtime config and memory for the background task,
                                // but we can't easily do that since they are mutable references.
                                // For now, we'll block the main loop on the LLM call, but we'll use tokio::select!
                                // to keep processing Tick events for the spinner.

                                let config_clone = runtime.config.clone();
                                let (tx_chunk, mut rx_chunk) = tokio::sync::mpsc::channel(100);
                                let tx_llm_chunk = tx.clone();
                                tokio::spawn(async move {
                                    while let Some(chunk) = rx_chunk.recv().await {
                                        let _ = tx_llm_chunk.send(tui::Event::LlmChunk(chunk)).await;
                                    }
                                });

                                let mut outcome_future = Box::pin(handle_interactive_input(
                                    runtime,
                                    memory,
                                    &line,
                                    &mut turn_count,
                                    &mut recent_turns,
                                    Some(tx_chunk),
                                ));

                                let outcome = loop {
                                    tokio::select! {
                                        res = &mut outcome_future => {
                                            break res;
                                        }
                                        Some(inner_event) = rx.recv() => {
                                            app.handle_event(inner_event);
                                            app.refresh_file_popup();
                                            let suggestions = command_suggestions(&app.input_box.text());
                                            app.draw(&mut terminal, &suggestions, &config_clone)?;
                                        }
                                    }
                                };

                                match outcome {
                                    Ok(outcome) => {
                                        let _ = tx_llm.try_send(tui::Event::LlmDone(outcome.messages));
                                        if outcome.exit_requested {
                                            app.should_quit = true;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx_llm.try_send(tui::Event::Error(e.to_string()));
                                    }
                                }
                            }
                            _ => {
                                app.handle_event(tui::Event::Key(key));
                            }
                        }
                    }
                    _ => {
                        app.handle_event(event);
                    }
                }
            }

            if app.should_quit {
                break;
            }
        }
        Ok(()) as Result<()>
    }
    .await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

async fn run_interactive_line_session(runtime: &mut AgentRuntime, memory: &mut MemoryManager) -> Result<()> {
    println!("interactive mode");
    println!("commands: /model show|provider <ollama|openrouter>|set <model>|key <api-key>|test");
    println!("          /think <low|balanced|deep>, /status, /context, /help, /exit");
    println!("or type any message to chat with Aigent");

    let stdin = io::stdin();
    let mut turn_count: usize = 0;
    let mut recent_turns: VecDeque<ConversationTurn> = VecDeque::new();

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

        let outcome =
            handle_interactive_input(runtime, memory, line, &mut turn_count, &mut recent_turns, None)
                .await?;
        for msg in outcome.messages {
            println!("{msg}");
        }
        if outcome.exit_requested {
            break;
        }
    }

    Ok(())
}

struct InputOutcome {
    messages: Vec<String>,
    exit_requested: bool,
}

async fn handle_interactive_input(
    runtime: &mut AgentRuntime,
    memory: &mut MemoryManager,
    line: &str,
    turn_count: &mut usize,
    recent_turns: &mut VecDeque<ConversationTurn>,
    tx: Option<tokio::sync::mpsc::Sender<String>>,
) -> Result<InputOutcome> {
    if line == "/exit" {
        return Ok(InputOutcome {
            messages: vec!["session closed".to_string()],
            exit_requested: true,
        });
    }

    if line == "/help" {
        return Ok(InputOutcome {
            messages: vec![
                "/model show".to_string(),
                "/model list [ollama|openrouter]".to_string(),
                "/model provider <ollama|openrouter>".to_string(),
                "/model set <model>".to_string(),
                "/model key <openrouter-api-key>".to_string(),
                "/model test".to_string(),
                "/think <low|balanced|deep>".to_string(),
                "/status".to_string(),
                "/context".to_string(),
                "/exit".to_string(),
            ],
            exit_requested: false,
        });
    }

    if line == "/status" {
        return Ok(InputOutcome {
            messages: vec![
                format!("bot: {}", runtime.config.agent.name),
                format!("provider: {}", runtime.config.llm.provider),
                format!("model: {}", runtime.config.active_model()),
                format!("thinking: {}", runtime.config.agent.thinking_level),
                format!("stored memories: {}", memory.all().len()),
                format!("recent conversation turns: {}", recent_turns.len()),
                format!("sleep mode: {}", runtime.config.memory.auto_sleep_mode),
                format!(
                    "night sleep window: {:02}:00-{:02}:00 (local)",
                    runtime.config.memory.night_sleep_start_hour,
                    runtime.config.memory.night_sleep_end_hour
                ),
            ],
            exit_requested: false,
        });
    }

    if line == "/context" {
        let snapshot = runtime.environment_snapshot(memory, recent_turns.len());
        let mut messages = vec!["environment context:".to_string()];
        messages.extend(snapshot.lines().map(ToString::to_string));
        return Ok(InputOutcome {
            messages,
            exit_requested: false,
        });
    }

    if line == "/model show" {
        return Ok(InputOutcome {
            messages: vec![
                format!("provider: {}", runtime.config.llm.provider),
                format!("model: {}", runtime.config.active_model()),
            ],
            exit_requested: false,
        });
    }

    if line == "/model list" || line.starts_with("/model list ") {
        let provider = parse_model_provider_from_command(line);
        let messages = collect_model_lines(provider).await?;
        return Ok(InputOutcome {
            messages,
            exit_requested: false,
        });
    }

    if let Some(provider) = line.strip_prefix("/model provider ") {
        let provider = provider.trim().to_lowercase();
        if provider == "ollama" || provider == "openrouter" {
            runtime.config.llm.provider = provider;
            runtime.config.save_to("config/default.toml")?;
            return Ok(InputOutcome {
                messages: vec!["provider updated".to_string()],
                exit_requested: false,
            });
        }
        return Ok(InputOutcome {
            messages: vec!["invalid provider, expected ollama or openrouter".to_string()],
            exit_requested: false,
        });
    }

    if let Some(model) = line.strip_prefix("/model set ") {
        let model = model.trim();
        if model.is_empty() {
            return Ok(InputOutcome {
                messages: vec!["model cannot be empty".to_string()],
                exit_requested: false,
            });
        }
        if runtime.config.llm.provider.eq_ignore_ascii_case("openrouter") {
            runtime.config.llm.openrouter_model = model.to_string();
        } else {
            runtime.config.llm.ollama_model = model.to_string();
        }
        runtime.config.save_to("config/default.toml")?;
        return Ok(InputOutcome {
            messages: vec!["model updated".to_string()],
            exit_requested: false,
        });
    }

    if let Some(api_key) = line.strip_prefix("/model key ") {
        let api_key = api_key.trim();
        if api_key.is_empty() {
            return Ok(InputOutcome {
                messages: vec!["api key cannot be empty".to_string()],
                exit_requested: false,
            });
        }
        upsert_env_value(Path::new(".env"), "OPENROUTER_API_KEY", api_key)?;
        dotenvy::from_path_override(".env").ok();
        return Ok(InputOutcome {
            messages: vec!["openrouter key saved to .env".to_string()],
            exit_requested: false,
        });
    }

    if line == "/model test" {
        let message = match runtime.test_model_connection().await {
            Ok(result) => format!("aigent> model test ok: {result}"),
            Err(error) => format!("aigent> model test failed: {error}"),
        };
        return Ok(InputOutcome {
            messages: vec![message],
            exit_requested: false,
        });
    }

    if let Some(level) = line.strip_prefix("/think ") {
        let parsed = parse_thinking_level(level);
        return Ok(InputOutcome {
            messages: match parsed {
                Some(level) => {
                    runtime.config.agent.thinking_level = thinking_level_label(level).to_string();
                    runtime.config.save_to("config/default.toml")?;
                    vec!["thinking level updated".to_string()]
                }
                None => vec!["invalid thinking level, expected low, balanced, or deep".to_string()],
            },
            exit_requested: false,
        });
    }

    let mut messages = Vec::new();
    let recent_context = recent_turns.iter().cloned().collect::<Vec<_>>();
    let reply = if let Some(tx) = tx {
        runtime
            .respond_and_remember_stream(memory, line, &recent_context, tx)
            .await?
    } else {
        runtime
            .respond_and_remember(memory, line, &recent_context)
            .await?
    };
    messages.push(format!("aigent> {reply}"));

    recent_turns.push_back(ConversationTurn {
        user: line.to_string(),
        assistant: reply.clone(),
    });
    while recent_turns.len() > 8 {
        let _ = recent_turns.pop_front();
    }

    *turn_count += 1;
    if should_run_sleep_cycle(runtime, memory, *turn_count) {
        let summary = memory.run_sleep_cycle()?;
        messages.push(format!("aigent> internal sleep cycle: {}", summary.distilled));
    }

    Ok(InputOutcome {
        messages,
        exit_requested: false,
    })
}

const COMMAND_CATALOG: [&str; 10] = [
    "/model show",
    "/model list",
    "/model provider ",
    "/model set ",
    "/model key ",
    "/model test",
    "/think ",
    "/status",
    "/context",
    "/exit",
];

fn command_suggestions(input: &str) -> Vec<&'static str> {
    if !input.starts_with('/') {
        return Vec::new();
    }
    COMMAND_CATALOG
        .iter()
        .copied()
        .filter(|candidate| candidate.starts_with(input))
        .take(4)
        .collect()
}

fn collect_workspace_files(root: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in WalkBuilder::new(root).hidden(false).git_ignore(true).build() {
        let Ok(entry) = entry else {
            continue;
        };
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(root) else {
            continue;
        };
        files.push(rel.display().to_string());
    }
    files.sort();
    Ok(files)
}

fn parse_model_provider_from_command(line: &str) -> CliModelProvider {
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

async fn collect_model_lines(provider: CliModelProvider) -> Result<Vec<String>> {
    let mut lines = Vec::new();

    if matches!(provider, CliModelProvider::All | CliModelProvider::Ollama) {
        let ollama = list_ollama_models().await?;
        lines.push(format!("ollama models ({})", ollama.len()));
        lines.extend(ollama.into_iter().map(|model| format!("- {model}")));
    }

    if matches!(provider, CliModelProvider::All | CliModelProvider::Openrouter) {
        let openrouter = list_openrouter_models().await?;
        lines.push(format!("openrouter models ({})", openrouter.len()));
        lines.extend(openrouter.into_iter().map(|model| format!("- {model}")));
    }

    Ok(lines)
}

fn run_memory_wipe(memory: &mut MemoryManager, layer: CliMemoryLayer, yes: bool) -> Result<()> {
    let targets = layer_to_tiers(layer);
    let total = memory.all().len();
    let target_count = if matches!(layer, CliMemoryLayer::All) {
        total
    } else {
        memory
            .all()
            .iter()
            .filter(|entry| targets.contains(&entry.tier))
            .count()
    };

    println!("⚠️  destructive operation: memory wipe");
    println!("- selected layer: {}", memory_layer_label(layer));
    println!("- targeted entries: {target_count}");
    println!("- total entries: {total}");
    println!(
        "- by tier: episodic={} semantic={} procedural={} core={}",
        memory.entries_by_tier(MemoryTier::Episodic).len(),
        memory.entries_by_tier(MemoryTier::Semantic).len(),
        memory.entries_by_tier(MemoryTier::Procedural).len(),
        memory.entries_by_tier(MemoryTier::Core).len(),
    );

    if target_count == 0 {
        println!("no matching memory entries to wipe");
        return Ok(());
    }

    if !yes {
        if !io::stdin().is_terminal() {
            bail!(
                "refusing destructive wipe in non-interactive mode without --yes (or pass --yes)"
            );
        }

        let expected = format!("WIPE {}", memory_layer_label(layer).to_uppercase());
        print!(
            "This permanently deletes memory from .aigent/memory/events.jsonl. Type '{expected}' to continue: "
        );
        io::stdout().flush()?;

        let mut confirmation = String::new();
        io::stdin().read_line(&mut confirmation)?;
        if confirmation.trim() != expected {
            println!("memory wipe cancelled");
            return Ok(());
        }
    }

    let removed = if matches!(layer, CliMemoryLayer::All) {
        memory.wipe_all()?
    } else {
        memory.wipe_tiers(&targets)?
    };

    println!("memory wipe complete: removed {removed} entries");
    println!("remaining entries: {}", memory.all().len());
    Ok(())
}

fn run_memory_stats(memory: &MemoryManager) {
    let stats = memory.stats();
    println!("memory stats");
    println!("- total: {}", stats.total);
    println!("- core: {}", stats.core);
    println!("- semantic: {}", stats.semantic);
    println!("- episodic: {}", stats.episodic);
    println!("- procedural: {}", stats.procedural);
}

fn run_memory_inspect_core(memory: &MemoryManager, limit: usize) {
    let mut entries = memory.entries_by_tier(MemoryTier::Core);
    entries.sort_by(|left, right| right.created_at.cmp(&left.created_at));

    println!("core memories (latest {limit})");
    for (index, entry) in entries.into_iter().take(limit).enumerate() {
        println!("{}. [{}] {}", index + 1, entry.created_at, entry.content);
    }
}

fn run_memory_promotions(memory: &MemoryManager, limit: usize) {
    let entries = memory.recent_promotions(limit);
    println!("memory promotions (latest {})", entries.len());
    for (index, entry) in entries.into_iter().enumerate() {
        println!(
            "{}. [{}] {:?} {} (source={})",
            index + 1,
            entry.created_at,
            entry.tier,
            entry.content,
            entry.source
        );
    }
}

fn run_memory_export_vault(memory: &MemoryManager, path: &str) -> Result<()> {
    let summary = memory.export_vault(path)?;
    println!("memory vault export complete");
    println!("- root: {}", summary.root);
    println!("- notes: {}", summary.note_count);
    println!("- topics: {}", summary.topic_count);
    println!("- daily notes: {}", summary.daily_note_count);
    Ok(())
}

#[derive(Debug, Clone)]
struct GateCheck {
    name: &'static str,
    passed: bool,
    details: String,
}

fn run_phase_review_gate(
    config: &mut AppConfig,
    memory: &mut MemoryManager,
    memory_log_path: &Path,
    report_path: Option<&str>,
) -> Result<()> {
    let mut config_changed = false;
    if !config.memory.backend.eq_ignore_ascii_case("eventlog") {
        config.memory.backend = "eventlog".to_string();
        config_changed = true;
    }
    if !config.memory.auto_sleep_mode.eq_ignore_ascii_case("nightly") {
        config.memory.auto_sleep_mode = "nightly".to_string();
        config_changed = true;
    }
    if config_changed {
        config.save_to("config/default.toml")?;
    }

    if !Path::new(".aigent/vault/index.md").exists() {
        let _ = memory.export_vault(".aigent/vault")?;
    }

    let has_sleep_marker = memory
        .all()
        .iter()
        .any(|entry| entry.source.starts_with("sleep:"));
    if !has_sleep_marker && !memory.all().is_empty() {
        let _ = memory.run_sleep_cycle()?;
    }

    let event_log = MemoryEventLog::new(memory_log_path);
    let events = event_log.load()?;
    let promotions = memory
        .all()
        .iter()
        .filter(|entry| entry.source.starts_with("sleep:"))
        .count();
    let vault_index = Path::new(".aigent/vault/index.md");
    let telegram_token_present = std::env::var("TELEGRAM_BOT_TOKEN")
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);

    let checks = vec![
        GateCheck {
            name: "config memory backend",
            passed: config.memory.backend.eq_ignore_ascii_case("eventlog"),
            details: format!("backend={}", config.memory.backend),
        },
        GateCheck {
            name: "sleep mode nightly",
            passed: config.memory.auto_sleep_mode.eq_ignore_ascii_case("nightly"),
            details: format!(
                "mode={} window={:02}:00-{:02}:00",
                config.memory.auto_sleep_mode,
                config.memory.night_sleep_start_hour,
                config.memory.night_sleep_end_hour
            ),
        },
        GateCheck {
            name: "memory event log readable",
            passed: memory_log_path.exists(),
            details: format!("path={} events={}", memory_log_path.display(), events.len()),
        },
        GateCheck {
            name: "core identity seeded",
            passed: !memory.entries_by_tier(MemoryTier::Core).is_empty(),
            details: format!(
                "core_entries={}",
                memory.entries_by_tier(MemoryTier::Core).len()
            ),
        },
        GateCheck {
            name: "vault projection exists",
            passed: vault_index.exists(),
            details: format!("index_path={}", vault_index.display()),
        },
        GateCheck {
            name: "sleep promotion evidence",
            passed: promotions > 0,
            details: format!("promotions={promotions}"),
        },
        GateCheck {
            name: "telegram token configured",
            passed: telegram_token_present,
            details: "env=TELEGRAM_BOT_TOKEN".to_string(),
        },
    ];

    let passed = checks.iter().filter(|check| check.passed).count();
    let total = checks.len();

    println!("phase review gate (phase 0-2)");
    println!("- auto-remediation: enabled for fixable checks");
    for check in &checks {
        let marker = if check.passed {
            "PASS"
        } else {
            "FAIL"
        };
        println!("- [{marker}] {} ({})", check.name, check.details);
    }
    println!("- summary: {passed}/{total} checks passed");

    if let Some(path) = report_path {
        let now = Local::now().to_rfc3339();
        let mut rendered = String::new();
        rendered.push_str("# Aigent Phase Review Gate\n\n");
        rendered.push_str(&format!("Generated: {now}\n\n"));
        rendered.push_str("## Checks\n");
        for check in &checks {
            let marker = if check.passed { "PASS" } else { "FAIL" };
            rendered.push_str(&format!(
                "- [{marker}] {} ({})\n",
                check.name, check.details
            ));
        }
        rendered.push_str(&format!("\n## Summary\n- {passed}/{total} checks passed\n"));
        if passed == total {
            rendered.push_str("- Gate result: PASS\n");
        } else {
            rendered.push_str("- Gate result: FAIL\n");
        }

        let output = Path::new(path);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output, rendered)?;
        println!("- report written: {}", output.display());
    }

    if passed != total {
        bail!("phase review gate failed: resolve failing checks before advancing");
    }

    println!("phase review gate passed: ready for phase 3");
    Ok(())
}

fn layer_to_tiers(layer: CliMemoryLayer) -> Vec<MemoryTier> {
    match layer {
        CliMemoryLayer::All => vec![
            MemoryTier::Episodic,
            MemoryTier::Semantic,
            MemoryTier::Procedural,
            MemoryTier::Core,
        ],
        CliMemoryLayer::Episodic => vec![MemoryTier::Episodic],
        CliMemoryLayer::Semantic => vec![MemoryTier::Semantic],
        CliMemoryLayer::Procedural => vec![MemoryTier::Procedural],
        CliMemoryLayer::Core => vec![MemoryTier::Core],
    }
}

fn memory_layer_label(layer: CliMemoryLayer) -> &'static str {
    match layer {
        CliMemoryLayer::All => "all",
        CliMemoryLayer::Episodic => "episodic",
        CliMemoryLayer::Semantic => "semantic",
        CliMemoryLayer::Procedural => "procedural",
        CliMemoryLayer::Core => "core",
    }
}

fn should_run_sleep_cycle(runtime: &AgentRuntime, memory: &MemoryManager, turn_count: usize) -> bool {
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

fn parse_thinking_level(raw: &str) -> Option<CliThinkingLevel> {
    match raw.trim().to_lowercase().as_str() {
        "low" => Some(CliThinkingLevel::Low),
        "balanced" => Some(CliThinkingLevel::Balanced),
        "deep" => Some(CliThinkingLevel::Deep),
        _ => None,
    }
}

fn thinking_level_label(level: CliThinkingLevel) -> &'static str {
    match level {
        CliThinkingLevel::Low => "low",
        CliThinkingLevel::Balanced => "balanced",
        CliThinkingLevel::Deep => "deep",
    }
}

fn upsert_env_value(path: &Path, key: &str, value: &str) -> Result<()> {
    let existing = fs::read_to_string(path).unwrap_or_default();
    let mut updated = Vec::new();
    let mut replaced = false;

    for line in existing.lines() {
        if line.trim_start().starts_with(&format!("{key}=")) {
            updated.push(format!("{key}={value}"));
            replaced = true;
        } else {
            updated.push(line.to_string());
        }
    }

    if !replaced {
        updated.push(format!("{key}={value}"));
    }

    let content = format!("{}\n", updated.join("\n"));
    fs::write(path, content)?;
    Ok(())
}
