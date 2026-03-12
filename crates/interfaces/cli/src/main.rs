mod daemon;
mod interactive;
mod memory_cmds;
mod sleep_cmds;

use std::fs;
use std::io;
use std::io::IsTerminal;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use aigent_config::{AppConfig, DebugConfig};
use aigent_exec::sandbox;
use aigent_runtime::DaemonClient;
use aigent_runtime::history as chat_history;
use aigent_memory::MemoryManager;
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
    /// Sleep cycle control and accelerated-learning test tools.
    #[command(about = "Sleep cycle control and accelerated-learning test tools")]
    Sleep {
        #[command(subcommand)]
        command: SleepCommands,
    },
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

#[derive(Debug, Subcommand)]
enum SleepCommands {
    /// Force-trigger the nightly multi-agent sleep cycle immediately.
    /// Bypasses all time/quiet-window guards.
    Run,
    /// Show sleep schedule, last-run times, and quiet-window status.
    Status,
    /// Seed synthetic episodic memories about a topic to accelerate testing.
    ///
    /// Injects N observations containing clear preference language so the
    /// Identity specialist can detect and form an opinion after one `run`.
    Seed {
        /// The topic or theme to build memories around.
        /// Example: "Rust programming", "test-driven development"
        #[arg(value_name = "THEME")]
        theme: String,
        /// Number of memories to inject (1–10, default 7).
        #[arg(long, default_value_t = 7)]
        count: usize,
        /// Valence direction of the seeded memories.
        #[arg(long, value_enum, default_value = "positive")]
        valence: sleep_cmds::SeedValence,
        /// Immediately run the full multi-agent sleep cycle after seeding.
        #[arg(long)]
        run: bool,
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
    Candle,
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
    /// Inspect active beliefs, their confidence levels, and graph connections.
    Beliefs {
        /// Filter by belief kind (empirical, procedural, self-model, opinion).
        #[arg(long)]
        kind: Option<String>,
        /// Filter by memory tier (episodic, semantic, reflective, core).
        #[arg(long)]
        tier: Option<String>,
        /// Show only beliefs with a current confidence at or below this threshold.
        #[arg(long)]
        max_confidence: Option<f32>,
        #[arg(long, default_value_t = 30)]
        limit: usize,
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
    /// Hot-reload dynamic modules from the modules directory.
    Reload,
}

async fn fetch_available_models() -> aigent_ui::onboard::AvailableModels {
    println!("Fetching available models...");
    let ollama = aigent_llm::list_ollama_models().await.unwrap_or_default();
    let openrouter = aigent_llm::list_openrouter_models()
        .await
        .unwrap_or_default();
    aigent_ui::onboard::AvailableModels { ollama, openrouter }
}

/// Initialise the tracing subscriber using the `[debug]` config section.
///
/// - `RUST_LOG` env var, if set, **always** takes precedence.
/// - Otherwise, `debug.log_level` from config is used.
/// - When `debug.log_to_file` is `true` **and** this is the daemon subprocess
///   (`AIGENT_DAEMON_PROCESS=1`), an additional file layer appends to
///   `.aigent/runtime/debug.log`.  The parent CLI process never writes to the
///   file — it exits too quickly to produce useful output and two concurrent
///   `O_TRUNC` opens on the same inode would corrupt the file with null bytes.
fn init_tracing(debug: &DebugConfig) {
    // Honour RUST_LOG if explicitly set; otherwise use config level.
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::new(&debug.log_level)
    };

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_thread_ids(false);

    // Only attach the file layer inside the actual daemon subprocess.
    // The parent CLI process exits after spawning the daemon, so writing a
    // file log there adds no value and races with the daemon's own open.
    let is_daemon = std::env::var("AIGENT_DAEMON_PROCESS").ok().as_deref() == Some("1");

    if debug.log_to_file && is_daemon {
        let log_dir = Path::new(".aigent").join("runtime");
        let _ = fs::create_dir_all(&log_dir);
        // Append so previous runs are preserved across daemon restarts.
        let log_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_dir.join("debug.log"))
            .expect("failed to open debug.log");
        let file_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(log_file));

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(file_layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let mut config = AppConfig::load_from(AppConfig::config_path())?;
    init_tracing(&config.debug);

    let config_exists = Path::new(&AppConfig::config_path()).exists();
    // Resolve to an absolute path so that the daemon and the CLI always
    // refer to the same file regardless of the working directory each was
    // launched from.
    let memory_log_path = {
        let rel = Path::new(".aigent").join("memory").join("events.jsonl");
        std::env::current_dir().map(|cwd| cwd.join(&rel)).unwrap_or(rel)
    };

    if std::env::var("AIGENT_DAEMON_PROCESS").ok().as_deref() == Some("1") {
        daemon::run_daemon_process(config, &memory_log_path).await?;
        return Ok(());
    }

    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if raw_args.first().map(|arg| arg == "daemon").unwrap_or(false) {
        daemon::run_daemon_command_from_args(&raw_args[1..]).await?;
        return Ok(());
    }

    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Start) {
        Commands::Onboard => {
            let models = fetch_available_models().await;
            run_onboarding(&mut config, models)?;
            config.save_to(AppConfig::config_path())?;
            seed_identity_from_config(&config, &memory_log_path).await?;
            println!("{}", aigent_ui::tui::banner());
        }
        Commands::Configuration => {
            let models = fetch_available_models().await;
            run_configuration(&mut config, models)?;
            config.save_to(AppConfig::config_path())?;
            seed_identity_from_config(&config, &memory_log_path).await?;
            println!("configuration updated");
        }
        Commands::Start | Commands::Run => {
            if !config_exists || config.needs_onboarding() {
                let models = fetch_available_models().await;
                run_onboarding(&mut config, models)?;
                config.save_to(AppConfig::config_path())?;
                seed_identity_from_config(&config, &memory_log_path).await?;
            }

            daemon::run_start_mode(config, &memory_log_path).await?;
        }
        Commands::Telegram => {
            if !config_exists || config.needs_onboarding() {
                let models = fetch_available_models().await;
                run_onboarding(&mut config, models)?;
                config.save_to(AppConfig::config_path())?;
                seed_identity_from_config(&config, &memory_log_path).await?;
            }

            daemon::run_telegram_runtime(config, &memory_log_path).await?;
        }
        Commands::Doctor {
            review_gate,
            model_catalog,
            provider,
            report,
        } => {
            let mut memory = MemoryManager::with_event_log(&memory_log_path).await?;
            if model_catalog {
                let lines = interactive::collect_model_lines(provider).await?;
                for line in lines {
                    println!("{line}");
                }
            } else if review_gate {
                memory_cmds::run_phase_review_gate(
                    &mut config,
                    &mut memory,
                    &memory_log_path,
                    report.as_deref(),
                ).await?;
            } else {
                println!("aigent doctor");
                println!("- memory log path: {}", memory_log_path.display());
                println!("- memory entries loaded: {}", memory.all().len());
                println!("- provider: {}", config.llm.provider);
                println!("- model: {}", config.active_model());
                println!("- thinking level: {}", config.agent.thinking_level);
            }
        }
        Commands::Memory { command } => match command {
            MemoryCommands::Wipe { layer, yes } => {
                let mut memory = MemoryManager::with_event_log(&memory_log_path).await?;
                memory_cmds::run_memory_wipe(&mut memory, layer, yes).await?;
                // Best-effort: notify the running daemon to resync its in-memory store.
                let client = DaemonClient::new(&config.daemon.socket_path);
                let _ = client.reload_memory().await;
            }
            MemoryCommands::Stats => {
                let memory = MemoryManager::with_event_log(memory_log_path).await?;
                memory_cmds::run_memory_stats(&memory);
            }
            MemoryCommands::InspectCore { limit } => {
                let memory = MemoryManager::with_event_log(memory_log_path).await?;
                memory_cmds::run_memory_inspect_core(&memory, limit.max(1));
            }
            MemoryCommands::Promotions { limit } => {
                let memory = MemoryManager::with_event_log(memory_log_path).await?;
                memory_cmds::run_memory_promotions(&memory, limit.max(1));
            }
            MemoryCommands::ExportVault { path } => {
                let memory = MemoryManager::with_event_log(memory_log_path).await?;
                let target = path.unwrap_or_else(|| ".aigent/vault".to_string());
                memory_cmds::run_memory_export_vault(&memory, &target)?;
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
            MemoryCommands::Beliefs { kind, tier, max_confidence, limit } => {
                let memory = MemoryManager::with_event_log(memory_log_path).await?;
                memory_cmds::run_memory_beliefs(&memory, kind, tier, max_confidence, limit.max(1));
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
                ToolCommands::Reload => {
                    match client.reload_tools().await {
                        Ok(msg) => println!("{msg}"),
                        Err(err) => eprintln!("error reloading tools: {err}"),
                    }
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
        Commands::Sleep { command } => {
            let socket = &config.daemon.socket_path;
            match command {
                SleepCommands::Run => {
                    sleep_cmds::run_sleep_run(socket).await?;
                }
                SleepCommands::Status => {
                    sleep_cmds::run_sleep_status(socket).await?;
                }
                SleepCommands::Seed { theme, count, valence, run } => {
                    sleep_cmds::run_sleep_seed(socket, &theme, count, valence, run).await?;
                }
            }
        }
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

    let mut memory = MemoryManager::with_event_log(memory_log_path).await?;
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

    let config = AppConfig::load_from(AppConfig::config_path()).ok();
    if let Some(config) = &config {
        let client = DaemonClient::new(&config.daemon.socket_path);
        let _ = client.graceful_shutdown().await;
    }

    let paths = daemon::daemon_paths();
    if let Some(pid) = daemon::read_pid(&paths.pid_file)? {
        if daemon::is_pid_running(pid) {
            let _ = daemon::terminate_pid(pid);
            daemon::wait_for_pid_exit(pid, Duration::from_secs(4));
        }
    }

    clear_dir_contents(Path::new(".aigent").join("memory").as_path())?;
    clear_dir_contents(Path::new(".aigent").join("vault").as_path())?;
    clear_dir_contents(Path::new(".aigent").join("runtime").as_path())?;

    if let Some(mut config) = config {
        config.onboarding.completed = false;
        config.save_to(AppConfig::config_path())?;
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

