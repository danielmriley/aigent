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
use aigent_exec::sandbox;
use aigent_runtime::{
    BackendEvent, DaemonClient, run_unified_daemon,
};
use aigent_runtime::history as chat_history;
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
    /// Manage and invoke tools registered in the daemon.
    Tool {
        #[command(subcommand)]
        command: ToolCommands,
    },
    Reset {
        #[arg(long)]
        hard: bool,
        #[arg(long)]
        yes: bool,
    },
    /// Manage TUI chat history.
    #[command(about = "Manage TUI chat history (persisted daily JSONL)")]
    History {
        #[command(subcommand)]
        command: HistoryCommands,
    },
    /// Show daemon and sleep cycle status at a glance.
    #[command(about = "Show daemon status: uptime, memory, sleep schedule")]
    Status,
}

#[derive(Debug, Subcommand)]
enum HistoryCommands {
    /// Delete today's history file.
    Clear,
    /// Export today's history to a file.
    Export {
        /// Destination path for the exported JSONL.
        #[arg(value_name = "PATH")]
        path: String,
    },
    /// Show the path to today's history file.
    Path,
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
    /// Proactive mode commands (trigger a check or view statistics).
    Proactive {
        #[command(subcommand)]
        command: ProactiveCommands,
    },
}

#[derive(Debug, Subcommand)]
enum ProactiveCommands {
    /// Immediately run a proactive check (bypasses DND and the configured interval).
    Check,
    /// Display proactive mode statistics.
    Stats,
}

#[derive(Debug, Subcommand)]
enum ToolCommands {
    /// List all tools registered in the running daemon.
    List,
    /// Execute a tool directly (key=value arguments).
    /// Example: aigent tool call web_search query="Rust programming"
    Call {
        /// Tool name to invoke
        name: String,
        /// Arguments as key=value pairs (e.g. path=README.md max_bytes=1024)
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Build WASM guest tools from `extensions/tools-src/`.
    /// Requires the `wasm32-wasip1` toolchain target.
    Build,
    /// Show per-tool runtime status: WASM binary present or native fallback.
    Status,
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
        run_daemon_command_from_args(&raw_args[1..]).await?;
        return Ok(());
    }

    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Start) {
        Commands::Onboard => {
            let models = fetch_available_models().await;
            run_onboarding(&mut config, models)?;
            config.save_to("config/default.toml")?;
            seed_identity_from_config(&config, &memory_log_path).await?;
            println!("{}", aigent_ui::tui::banner());
        }
        Commands::Configuration => {
            let models = fetch_available_models().await;
            run_configuration(&mut config, models)?;
            config.save_to("config/default.toml")?;
            seed_identity_from_config(&config, &memory_log_path).await?;
            println!("configuration updated");
        }
        Commands::Start | Commands::Run => {
            if !config_exists || config.needs_onboarding() {
                let models = fetch_available_models().await;
                run_onboarding(&mut config, models)?;
                config.save_to("config/default.toml")?;
                seed_identity_from_config(&config, &memory_log_path).await?;
            }

            run_start_mode(config, &memory_log_path).await?;
        }
        Commands::Telegram => {
            if !config_exists || config.needs_onboarding() {
                let models = fetch_available_models().await;
                run_onboarding(&mut config, models)?;
                config.save_to("config/default.toml")?;
                seed_identity_from_config(&config, &memory_log_path).await?;
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
                ).await?;
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
                run_memory_wipe(&mut memory, layer, yes).await?;
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
            MemoryCommands::Proactive { command } => {
                let client = DaemonClient::new(&config.daemon.socket_path);
                match command {
                    ProactiveCommands::Check => {
                        match client.trigger_proactive().await {
                            Ok(msg) => println!("{msg}"),
                            Err(err) => eprintln!("proactive check failed: {err}"),
                        }
                    }
                    ProactiveCommands::Stats => {
                        match client.get_proactive_stats().await {
                            Ok(stats) => {
                                println!("── proactive stats ──────────────────────────────────");
                                println!("  interval_minutes : {}", stats.interval_minutes);
                                println!("  dnd window       : {:02}:00 – {:02}:00", stats.dnd_start_hour, stats.dnd_end_hour);
                                println!("  total_sent       : {}", stats.total_sent);
                                println!("  last_sent_at     : {}", stats.last_proactive_at.as_deref().unwrap_or("(never)"));
                            }
                            Err(err) => eprintln!("failed to fetch stats: {err}"),
                        }
                    }
                }
            }
        },
        Commands::Tool { command } => {
            let client = DaemonClient::new(&config.daemon.socket_path);
            match command {
                ToolCommands::List => {
                    match client.list_tools().await {
                        Ok(specs) => {
                            println!("── registered tools ─────────────────────────────────");
                            for spec in &specs {
                                println!("  {} — {}", spec.name, spec.description);
                                for p in &spec.params {
                                    println!(
                                        "      {} [{}] — {}",
                                        p.name,
                                        if p.required { "required" } else { "optional" },
                                        p.description
                                    );
                                }
                            }
                            println!("  ({} tools total)", specs.len());
                        }
                        Err(err) => eprintln!("error listing tools: {err}"),
                    }
                }
                ToolCommands::Call { name, args } => {
                    // Parse "key=value" arguments.
                    let mut parsed: std::collections::HashMap<String, String> =
                        std::collections::HashMap::new();
                    for item in &args {
                        if let Some((k, v)) = item.split_once('=') {
                            parsed.insert(k.to_string(), v.to_string());
                        } else {
                            eprintln!("warning: skipping malformed arg '{}' (expected key=value)", item);
                        }
                    }
                    match client.execute_tool(&name, parsed).await {
                        Ok((success, output)) => {
                            let status = if success { "succeeded" } else { "failed" };
                            println!("tool '{}' {status}:", name);
                            println!("{output}");
                        }
                        Err(err) => eprintln!("error calling tool '{}': {err}", name),
                    }
                }
                ToolCommands::Build => {
                    // Step 1: ensure wasm32-wasip1 target is installed.
                    let status1 = Command::new("rustup")
                        .args(["target", "add", "wasm32-wasip1"])
                        .status()
                        .map_err(|e| anyhow::anyhow!("failed to run rustup: {e}"))?;
                    if !status1.success() {
                        bail!("rustup target add wasm32-wasip1 failed");
                    }
                    // Step 2: cargo build in extensions/tools-src/
                    let tools_src = Path::new("extensions").join("tools-src");
                    if !tools_src.exists() {
                        bail!(
                            "extensions/tools-src/ not found — create WASM guest crates there first"
                        );
                    }
                    let status2 = Command::new("cargo")
                        .args(["build", "--release"])
                        .env("CARGO_BUILD_TARGET", "wasm32-wasip1")
                        .current_dir(&tools_src)
                        .status()
                        .map_err(|e| anyhow::anyhow!("failed to run cargo: {e}"))?;
                    if !status2.success() {
                        bail!("cargo build --release failed in extensions/tools-src/");
                    }
                    println!("WASM guest tools built successfully.");
                    println!("Restart the daemon to activate (`aigent start`).");
                }
                ToolCommands::Status => {
                    const KNOWN_TOOLS: &[&str] = &[
                        "read_file", "write_file", "run_shell",
                        "calendar_add_event", "web_search", "draft_email",
                        "remind_me", "git_rollback",
                    ];
                    let extensions_dir = Path::new("extensions");

                    println!("── tool runtime status ───────────────────────────────");
                    for &name in KNOWN_TOOLS {
                        let wasm = find_wasm_binary(extensions_dir, name);
                        if let Some(ref p) = wasm {
                            println!("  {name:<22}  WASM  ({})", p.display());
                        } else {
                            println!("  {name:<22}  native (run `aigent tools build` to activate WASM)");
                        }
                    }

                    println!();
                    println!("── sandbox status ────────────────────────────────────");
                    let compiled_in = sandbox::is_active();
                    println!(
                        "  compiled-in     : {}",
                        if compiled_in { "yes" } else { "no (rebuild with --features sandbox)" }
                    );
                    println!(
                        "  config enabled  : {}",
                        if config.tools.sandbox_enabled { "yes" } else { "no (sandbox_enabled = false in config)" }
                    );
                    println!(
                        "  effective       : {}",
                        if compiled_in && config.tools.sandbox_enabled { "ACTIVE" } else { "disabled" }
                    );
                }
            }
        },
        Commands::Reset { hard, yes } => {
            run_reset_command(hard, yes).await?;
        }
        Commands::History { command } => match command {
            HistoryCommands::Clear => {
                chat_history::clear_history()?;
                println!("today's chat history cleared");
            }
            HistoryCommands::Export { path } => {
                let dest = std::path::Path::new(&path);
                chat_history::export_history(dest)?;
                println!("history exported to {path}");
            }
            HistoryCommands::Path => {
                let p = chat_history::history_file_path();
                println!("{}", p.display());
            }
        },
        Commands::Status => {
            let socket_path = config.daemon.socket_path.clone();
            let client = aigent_runtime::DaemonClient::new(&socket_path);
            match client.get_status().await {
                Ok(status) => {
                    println!("── daemon status ─────────────────────────────────────");
                    println!("  bot:       {}", status.bot_name);
                    println!("  provider:  {}", status.provider);
                    println!("  model:     {}", status.model);
                    println!("  thinking:  {}", status.thinking_level);
                    println!("  uptime:    {}s", status.uptime_secs);
                    println!("  memory:    {} total ({} core, {} user, {} reflective, {} semantic, {} episodic)",
                        status.memory_total, status.memory_core, status.memory_user_profile,
                        status.memory_reflective, status.memory_semantic, status.memory_episodic);
                    println!("  tools:     {}", status.available_tools.join(", "));
                }
                Err(err) => {
                    eprintln!("daemon not reachable: {err}");
                    eprintln!("is the daemon running? try: aigent daemon start");
                    std::process::exit(1);
                }
            }
            match client.get_sleep_status().await {
                Ok(sleep) => {
                    println!();
                    println!("── sleep schedule ────────────────────────────────────");
                    println!("  mode:      {}", sleep.auto_sleep_mode);
                    println!("  passive:   every {}h", sleep.passive_interval_hours);
                    println!("  last passive:  {}", sleep.last_passive_sleep_at.as_deref().unwrap_or("never"));
                    println!("  last nightly:  {}", sleep.last_nightly_sleep_at.as_deref().unwrap_or("never"));
                    println!("  quiet window:  {:02}:00–{:02}:00 {}",
                        sleep.quiet_window_start, sleep.quiet_window_end, sleep.timezone);
                    println!("  in window now: {}", if sleep.in_quiet_window { "yes" } else { "no" });
                }
                Err(err) => {
                    eprintln!("  (sleep status unavailable: {err})");
                }
            }
        },
    }

    Ok(())
}

/// Resolve the `.wasm` binary path for a given tool name, checking both the
/// direct layout (`extensions/<name>.wasm`) and the sub-workspace layout
/// (`extensions/tools-src/<crate>/target/wasm32-wasip1/release/<name>.wasm`).
fn find_wasm_binary(extensions_dir: &Path, tool_name: &str) -> Option<PathBuf> {
    // Layout 1: direct binary next to extensions/
    let direct = extensions_dir.join(format!("{tool_name}.wasm"));
    if direct.exists() {
        return Some(direct);
    }
    // Layout 2: sub-workspace built with `aigent tools build`
    let crate_name = tool_name.replace('_', "-");
    let sub = extensions_dir
        .join("tools-src")
        .join(&crate_name)
        .join("target")
        .join("wasm32-wasip1")
        .join("release")
        .join(format!("{tool_name}.wasm"));
    if sub.exists() {
        return Some(sub);
    }
    None
}

async fn seed_identity_from_config(config: &AppConfig, memory_log_path: &Path) -> Result<()> {
    if config.agent.user_name.trim().is_empty() || config.agent.name.trim().is_empty() {
        return Ok(());
    }

    let mut memory = MemoryManager::with_event_log(memory_log_path)?;
    memory.seed_core_identity(&config.agent.user_name, &config.agent.name).await?;
    Ok(())
}

async fn run_reset_command(hard: bool, yes: bool) -> Result<()> {
    if !hard {
        bail!("reset requires --hard (for now, only full reset is supported)");
    }

    if !yes {
        if !io::stdin().is_terminal() {
            bail!("refusing hard reset in non-interactive mode without --yes");
        }

        print!("This will stop daemon, wipe .aigent state, and require onboarding again. Type 'RESET HARD' to continue: ");
        io::stdout().flush()?;
        let mut confirmation = String::new();
        io::stdin().read_line(&mut confirmation)?;
        if confirmation.trim() != "RESET HARD" {
            println!("reset cancelled");
            return Ok(());
        }
    }

    let config = AppConfig::load_from("config/default.toml").ok();
    if let Some(config) = &config {
        let client = DaemonClient::new(&config.daemon.socket_path);
        let _ = client.graceful_shutdown().await;
    }

    let paths = daemon_paths();
    if let Some(pid) = read_pid(&paths.pid_file)? {
        if is_pid_running(pid) {
            let _ = terminate_pid(pid);
            wait_for_pid_exit(pid, Duration::from_secs(4));
        }
    }

    clear_dir_contents(Path::new(".aigent").join("memory").as_path())?;
    clear_dir_contents(Path::new(".aigent").join("vault").as_path())?;
    clear_dir_contents(Path::new(".aigent").join("runtime").as_path())?;

    if let Some(mut config) = config {
        config.onboarding.completed = false;
        config.save_to("config/default.toml")?;
    }

    println!("hard reset complete");
    println!("- daemon stopped");
    println!("- state wiped: .aigent/memory, .aigent/vault, .aigent/runtime");
    println!("- onboarding required on next start");
    Ok(())
}

fn clear_dir_contents(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            fs::remove_dir_all(entry.path())?;
        } else {
            fs::remove_file(entry.path())?;
        }
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
    telegram_pid_file: PathBuf,
    telegram_lock_file: PathBuf,
}

fn daemon_paths() -> DaemonPaths {
    let runtime_dir = Path::new(".aigent").join("runtime");
    DaemonPaths {
        pid_file: runtime_dir.join("daemon.pid"),
        log_file: runtime_dir.join("daemon.log"),
        mode_file: runtime_dir.join("daemon.mode"),
        lock_file: runtime_dir.join("daemon.lock"),
        telegram_pid_file: runtime_dir.join("telegram.pid"),
        telegram_lock_file: runtime_dir.join("telegram.lock"),
        runtime_dir,
    }
}

async fn run_daemon_command_from_args(args: &[String]) -> Result<()> {
    if args.is_empty() {
        print_daemon_help();
        return Ok(());
    }

    let command = args[0].as_str();
    let force = args.iter().any(|arg| arg == "--force");

    match command {
        "start" => daemon_start(force),
        "stop" => daemon_stop().await,
        "restart" => {
            daemon_stop().await?;
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

    if is_socket_live(&socket_path) && !force {
        bail!(
            "daemon already running on socket {}; use `aigent daemon restart`",
            socket_path.display()
        );
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

    // Remove stale lock file so a previously crashed or unclean daemon doesn't
    // block startup. Safe because we've already confirmed no live process holds it.
    if force && paths.lock_file.exists() {
        let _ = fs::remove_file(&paths.lock_file);
    }

    let exe = std::env::current_exe()?;
    let out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log_file)?;
    let err = out.try_clone()?;

    let mut child = Command::new(exe)
        .env("AIGENT_DAEMON_PROCESS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()?;

    fs::write(&paths.pid_file, child.id().to_string())?;
    fs::write(&paths.mode_file, "unified")?;

    for _ in 0..40 {
        if is_socket_live(&socket_path) {
            println!("daemon started");
            println!("- pid: {}", child.id());
            println!("- socket: {}", socket_path.display());
            println!("- log: {}", paths.log_file.display());
            return Ok(());
        }

        if let Some(status) = child.try_wait()? {
            let _ = fs::remove_file(&paths.pid_file);
            bail!(
                "daemon exited during startup with status {status}; check {}",
                paths.log_file.display()
            );
        }

        std::thread::sleep(Duration::from_millis(100));
    }

    let _ = fs::remove_file(&paths.pid_file);
    bail!(
        "daemon did not become ready on socket {}; check {}",
        socket_path.display(),
        paths.log_file.display()
    )
}

async fn daemon_stop() -> Result<()> {
    let config = AppConfig::load_from("config/default.toml")?;
    let paths = daemon_paths();
    let client = DaemonClient::new(&config.daemon.socket_path);

    if client.graceful_shutdown().await.is_ok() {
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
    wait_for_pid_exit(pid, Duration::from_secs(4));
    let _ = fs::remove_file(&paths.pid_file);
    let _ = fs::remove_file(&paths.lock_file);
    println!("daemon stopped (pid {pid})");
    Ok(())
}

fn wait_for_pid_exit(pid: u32, timeout: Duration) {
    let step = Duration::from_millis(50);
    let mut waited = Duration::from_millis(0);
    while waited < timeout {
        if !is_pid_running(pid) {
            return;
        }
        std::thread::sleep(step);
        waited += step;
    }
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
        } else {
            println!("daemon status: stopped");
            println!("- channel: {mode}");
            println!("- socket: {}", socket_path.display());
            println!("- log: {}", paths.log_file.display());
        }
    } else {
        println!("daemon status: stopped");
        println!("- channel: {mode}");
        println!("- socket: {}", socket_path.display());
        println!("- log: {}", paths.log_file.display());
    }

    // Telegram bot status
    if let Some(tg_pid) = read_pid(&paths.telegram_pid_file)? {
        if is_pid_running(tg_pid) {
            println!("telegram status: running");
            println!("- pid: {tg_pid}");
        } else {
            println!("telegram status: stopped (stale pid {tg_pid})");
            let _ = fs::remove_file(&paths.telegram_pid_file);
        }
    } else {
        println!("telegram status: stopped");
    }

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
        // If Telegram is enabled, run the bot in the background alongside the TUI.
        // Skip silently if another process already holds the lock (e.g. `aigent telegram`).
        if config.integrations.telegram_enabled {
            let tg_config = config.clone();
            let tg_memory_log = memory_log_path.to_path_buf();
            tokio::spawn(async move {
                if let Err(err) = run_telegram_runtime_guarded(tg_config, &tg_memory_log, true).await {
                    eprintln!("[telegram] bot exited: {err}");
                }
            });
        }

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
    run_telegram_runtime_guarded(config, memory_log_path, false).await
}

/// `silent_if_locked`: when true (background mode), skip without error if another instance
/// already holds the lock. When false (explicit `aigent telegram`), error immediately.
async fn run_telegram_runtime_guarded(
    config: AppConfig,
    memory_log_path: &Path,
    silent_if_locked: bool,
) -> Result<()> {
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

    // Acquire an exclusive lock to prevent duplicate polling instances.
    let paths = daemon_paths();
    fs::create_dir_all(&paths.runtime_dir)?;
    let lock_file = File::create(&paths.telegram_lock_file)?;
    match lock_file.try_lock_exclusive() {
        Ok(()) => {}
        Err(_) => {
            if silent_if_locked {
                return Ok(());
            }
            bail!(
                "another Telegram bot instance is already running (lock held at {})",
                paths.telegram_lock_file.display()
            );
        }
    }

    fs::write(&paths.telegram_pid_file, std::process::id().to_string())?;

    let client = DaemonClient::new(&config.daemon.socket_path);
    let result = start_bot(client).await;

    // Cleanup regardless of success/failure.
    let _ = fs::remove_file(&paths.telegram_pid_file);
    // lock_file drop releases the OS lock automatically.

    result
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
                            "Commands: /help, /status, /memory, /sleep, /tools, /tools run <name> {args}, /model show, /model list [ollama|openrouter], /model provider <ollama|openrouter>, /model set <model>, /think <low|balanced|deep>, /exit".to_string(),
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
                        let tx_clone = backend_tx.clone();
                        match daemon
                            .run_sleep_cycle_with_progress(|msg| {
                                let _ = tx_clone
                                    .send(BackendEvent::Token(format!("[sleep] {msg}")));
                            })
                            .await
                        {
                            Ok(msg) => {
                                let _ = backend_tx.send(BackendEvent::Token(msg));
                                let _ = backend_tx.send(BackendEvent::Done);
                            }
                            Err(err) => {
                                let _ = backend_tx.send(BackendEvent::Error(err.to_string()));
                            }
                        }
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

async fn run_interactive_line_session(daemon: DaemonClient) -> Result<()> {
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

async fn update_model_provider(raw: &str, daemon: &DaemonClient) -> Result<String> {
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

async fn update_model_selection(raw: &str, daemon: &DaemonClient) -> Result<String> {
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

async fn update_thinking_level(raw: &str, daemon: &DaemonClient) -> Result<String> {
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

async fn run_memory_wipe(memory: &mut MemoryManager, layer: CliMemoryLayer, yes: bool) -> Result<()> {
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
        memory.wipe_all().await?
    } else {
        memory.wipe_tiers(&targets).await?
    };

    println!("memory wipe complete: removed {removed} entries");
    println!("remaining entries: {}", memory.all().len());
    Ok(())
}

fn run_memory_stats(memory: &MemoryManager) {
    let stats = memory.stats();
    println!("── memory stats ─────────────────────────────────────");
    println!("  total:        {}", stats.total);
    println!("  core:         {}", stats.core);
    println!("  user_profile: {}", stats.user_profile);
    println!("  reflective:   {}", stats.reflective);
    println!("  semantic:     {}", stats.semantic);
    println!("  procedural:   {}", stats.procedural);
    println!("  episodic:     {}", stats.episodic);

    // ── tool execution stats ──────────────────────────────────────
    {
        let tool_entries = memory.entries_by_tier(MemoryTier::Procedural);
        let tool_execs: Vec<_> = tool_entries
            .iter()
            .filter(|e| e.source.starts_with("tool-use:"))
            .collect();
        let tool_total = tool_execs.len();
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        let tool_today = tool_execs.iter().filter(|e| e.created_at > cutoff).count();
        println!();
        println!("── tool executions ──────────────────────────────────");
        println!("  today (24h): {tool_today}");
        println!("  all time:    {tool_total}");
        if tool_total > 0 {
            // Count by tool name for a per-tool breakdown.
            let mut by_tool: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for e in &tool_execs {
                let tool_name = e.source.trim_start_matches("tool-use:");
                *by_tool.entry(tool_name).or_insert(0) += 1;
            }
            for (t, n) in &by_tool {
                println!("    {t}: {n}");
            }
        }
    }

    println!();
    println!("── redb index ───────────────────────────────────────");
    match (stats.index_size, stats.index_cache) {
        (Some(size), Some(cache)) => {
            println!("  entries:    {size}");
            println!("  cache cap:  {}", cache.capacity);
            println!("  cache len:  {}", cache.len);
            println!("  hits:       {}", cache.hits);
            println!("  misses:     {}", cache.misses);
            println!("  hit rate:   {:.1}%", cache.hit_rate_pct);
        }
        _ => println!("  (index not enabled — run with daemon to activate)"),
    }

    println!();
    println!("── vault checksums ──────────────────────────────────");
    if stats.vault_files.is_empty() {
        println!("  (vault not configured)");
    } else {
        for f in &stats.vault_files {
            let status = match (f.exists, f.checksum_valid) {
                (false, _) => "MISSING",
                (true, true) => "OK",
                (true, false) => "MODIFIED (human edit detected)",
            };
            println!("  {:<28}  {status}", f.filename);
        }
    }
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

async fn run_phase_review_gate(
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
        let _ = memory.run_sleep_cycle().await?;
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

