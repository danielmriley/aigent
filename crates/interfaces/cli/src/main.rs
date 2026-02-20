use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::IsTerminal;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Result, bail};
use chrono::Local;
use clap::{Parser, Subcommand, ValueEnum};
use fs2::FileExt;
use tracing_subscriber::EnvFilter;

use aigent_config::AppConfig;
use aigent_daemon::{
    BackendEvent, DaemonClient, run_unified_daemon,
};
use aigent_llm::{list_ollama_models, list_openrouter_models};
use aigent_memory::event_log::MemoryEventLog;
use aigent_memory::{MemoryManager, MemoryTier};
use aigent_telegram::start_bot;
use aigent_ui::onboard::{run_configuration, run_onboarding};

#[derive(Debug, Parser)]
#[command(
    name = "aigent",
    version,
    about = "A persistent memory-centric AI agent"
)]
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
    let openrouter = aigent_llm::list_openrouter_models()
        .await
        .unwrap_or_default();
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

    if std::env::var("AIGENT_DAEMON_PROCESS").ok().as_deref() == Some("1") {
        run_daemon_process(config, &memory_log_path).await?;
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
    lock_file: PathBuf,
}

fn daemon_paths() -> DaemonPaths {
    let runtime_dir = Path::new(".aigent").join("runtime");
    DaemonPaths {
        pid_file: runtime_dir.join("daemon.pid"),
        log_file: runtime_dir.join("daemon.log"),
        mode_file: runtime_dir.join("daemon.mode"),
        lock_file: runtime_dir.join("daemon.lock"),
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

fn daemon_start(force: bool) -> Result<()> {
    let config = AppConfig::load_from("config/default.toml")?;
    if config.needs_onboarding() {
        bail!("onboarding not complete; run `aigent onboard` first");
    }

    let paths = daemon_paths();
    fs::create_dir_all(&paths.runtime_dir)?;
    let socket_path = PathBuf::from(&config.daemon.socket_path);

    if is_socket_live(&socket_path) {
        if !force {
            bail!(
                "daemon already running on socket {}; use `aigent daemon restart`",
                socket_path.display()
            );
        }
    }

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

    if socket_path.exists() {
        let _ = fs::remove_file(&socket_path);
    }

    let exe = std::env::current_exe()?;
    let out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log_file)?;
    let err = out.try_clone()?;

    let child = Command::new(exe)
        .env("AIGENT_DAEMON_PROCESS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()?;

    fs::write(&paths.pid_file, child.id().to_string())?;
    fs::write(&paths.mode_file, "unified")?;

    println!("daemon started");
    println!("- pid: {}", child.id());
    println!("- socket: {}", socket_path.display());
    println!("- log: {}", paths.log_file.display());
    Ok(())
}

fn daemon_stop() -> Result<()> {
    let config = AppConfig::load_from("config/default.toml")?;
    let paths = daemon_paths();
    let client = DaemonClient::new(&config.daemon.socket_path);

    if tokio::runtime::Runtime::new()?
        .block_on(client.graceful_shutdown())
        .is_ok()
    {
        println!("daemon stop requested gracefully");
    }

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
    let config = AppConfig::load_from("config/default.toml")?;
    let paths = daemon_paths();
    let socket_path = PathBuf::from(&config.daemon.socket_path);
    let mode = fs::read_to_string(&paths.mode_file)
        .unwrap_or_else(|_| "unknown".to_string())
        .trim()
        .to_string();

    let socket_live = is_socket_live(&socket_path);

    if let Some(pid) = read_pid(&paths.pid_file)? {
        if is_pid_running(pid) || socket_live {
            println!("daemon status: running");
            println!("- pid: {pid}");
            println!("- channel: {mode}");
            println!("- socket: {}", socket_path.display());
            println!("- log: {}", paths.log_file.display());
            return Ok(());
        }
    }

    println!("daemon status: stopped");
    println!("- channel: {mode}");
    println!("- socket: {}", socket_path.display());
    println!("- log: {}", paths.log_file.display());
    Ok(())
}

fn is_socket_live(path: &Path) -> bool {
    std::os::unix::net::UnixStream::connect(path).is_ok()
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

async fn run_daemon_process(config: AppConfig, memory_log_path: &Path) -> Result<()> {
    let paths = daemon_paths();
    fs::create_dir_all(&paths.runtime_dir)?;
    let lock_file = File::create(&paths.lock_file)?;
    lock_file
        .try_lock_exclusive()
        .map_err(|_| anyhow::anyhow!("another daemon instance already holds the lock"))?;

    fs::write(&paths.pid_file, std::process::id().to_string())?;
    fs::write(&paths.mode_file, "unified")?;

    let socket_path = config.daemon.socket_path.clone();
    let daemon = run_unified_daemon(config, memory_log_path, &socket_path);

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate())?;
        let mut sigint = signal(SignalKind::interrupt())?;
        tokio::select! {
            _ = sigterm.recv() => {},
            _ = sigint.recv() => {},
        }
        Ok::<(), anyhow::Error>(())
    };

    #[cfg(not(unix))]
    let terminate = async {
        tokio::signal::ctrl_c().await?;
        Ok::<(), anyhow::Error>(())
    };

    tokio::select! {
        result = daemon => {
            result?;
        }
        result = terminate => {
            result?;
            let client = DaemonClient::new(&socket_path);
            let _ = client.graceful_shutdown().await;
        }
    }

    let _ = fs::remove_file(&paths.pid_file);
    let _ = fs::remove_file(&paths.lock_file);
    Ok(())
}

async fn run_start_mode(config: AppConfig, memory_log_path: &Path) -> Result<()> {
    let interactive_terminal = io::stdin().is_terminal() && io::stdout().is_terminal();

    ensure_daemon_running(&config)?;

    if interactive_terminal {
        println!("{}", aigent_ui::tui::banner());
        let client = DaemonClient::new(&config.daemon.socket_path);
        return run_interactive_session(&config, client).await;
    }

    if config.integrations.telegram_enabled {
        return run_telegram_runtime(config, memory_log_path).await;
    }

    bail!(
        "no interactive terminal detected and no messaging integrations enabled; run in a terminal or enable Telegram in `aigent configuration`"
    )
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

    let _ = memory_log_path;
    ensure_daemon_running(&config)?;
    let client = DaemonClient::new(&config.daemon.socket_path);
    start_bot(client).await
}

fn ensure_daemon_running(config: &AppConfig) -> Result<()> {
    let socket_path = PathBuf::from(&config.daemon.socket_path);
    if is_socket_live(&socket_path) {
        return Ok(());
    }

    daemon_start(false)?;
    for _ in 0..40 {
        if is_socket_live(&socket_path) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    bail!(
        "daemon did not become ready on socket {}",
        socket_path.display()
    )
}

async fn run_interactive_session(config: &AppConfig, daemon: DaemonClient) -> Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return run_interactive_line_session(daemon).await;
    }

    let (backend_tx, backend_rx) = aigent_ui::tui::create_backend_channel();
    let mut app = aigent_ui::App::new(backend_rx, config);
    app.push_assistant_message(format!(
        "{} is online. Type /help for commands.",
        config.agent.name
    ));

    aigent_ui::tui::run_app_with(&mut app, |command| {
        let backend_tx = backend_tx.clone();
        let daemon = daemon.clone();

        async move {
            match command {
                aigent_ui::UiCommand::Quit => {}
                aigent_ui::UiCommand::Submit(line) => {
                    if line == "/help" {
                        let _ = backend_tx.send(BackendEvent::Token(
                            "Commands: /help, /status, /memory, /exit, or ask anything".to_string(),
                        ));
                        let _ = backend_tx.send(BackendEvent::Done);
                        return Ok(());
                    }
                    if line == "/status" {
                        match daemon.get_status().await {
                            Ok(status) => {
                                let text = format!(
                                    "bot: {}\nprovider: {}\nmodel: {}\nthinking: {}\nmemories: {}\nuptime: {}s",
                                    status.bot_name,
                                    status.provider,
                                    status.model,
                                    status.thinking_level,
                                    status.memory_total,
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

                    tokio::spawn(async move {
                        let _ = daemon.stream_submit(line, backend_tx.clone()).await;
                    });
                }
            }
            Ok(())
        }
    })
    .await
}

async fn run_interactive_line_session(daemon: DaemonClient) -> Result<()> {
    println!("interactive mode");
    println!("commands: /model show|provider <ollama|openrouter>|set <model>|key <api-key>|test");
    println!("          /think <low|balanced|deep>, /status, /context, /help, /exit");
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

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        daemon.stream_submit(line.to_string(), tx).await?;
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

async fn collect_model_lines(provider: CliModelProvider) -> Result<Vec<String>> {
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
    if !config
        .memory
        .auto_sleep_mode
        .eq_ignore_ascii_case("nightly")
    {
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
            passed: config
                .memory
                .auto_sleep_mode
                .eq_ignore_ascii_case("nightly"),
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
        let marker = if check.passed { "PASS" } else { "FAIL" };
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

