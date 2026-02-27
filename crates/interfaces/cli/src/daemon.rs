use std::fs;
use std::io;
use std::io::IsTerminal;
use std::fs::File;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Result, bail};
use fs2::FileExt;

use aigent_config::AppConfig;
use aigent_runtime::{DaemonClient, run_unified_daemon};
use aigent_telegram::start_bot;

use crate::interactive::run_interactive_session;

#[derive(Debug, Clone)]
pub(crate) struct DaemonPaths {
    runtime_dir: PathBuf,
    pub(crate) pid_file: PathBuf,
    log_file: PathBuf,
    mode_file: PathBuf,
    pub(crate) lock_file: PathBuf,
    telegram_pid_file: PathBuf,
    telegram_lock_file: PathBuf,
}

pub(crate) fn daemon_paths() -> DaemonPaths {
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

pub(crate) async fn run_daemon_command_from_args(args: &[String]) -> Result<()> {
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

pub(crate) fn print_daemon_help() {
    println!("Manage background aigent service");
    println!("Usage: aigent daemon <start|stop|restart|status> [--force]");
}

pub(crate) fn daemon_start(force: bool) -> Result<()> {
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

pub(crate) async fn daemon_stop() -> Result<()> {
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

pub(crate) fn wait_for_pid_exit(pid: u32, timeout: Duration) {
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

pub(crate) fn daemon_status() -> Result<()> {
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

pub(crate) fn is_socket_live(path: &Path) -> bool {
    std::os::unix::net::UnixStream::connect(path).is_ok()
}

pub(crate) fn read_pid(path: &Path) -> Result<Option<u32>> {
    if !path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(path)?;
    let pid = raw.trim().parse::<u32>().ok();
    Ok(pid)
}

pub(crate) fn is_pid_running(pid: u32) -> bool {
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

pub(crate) fn terminate_pid(pid: u32) -> Result<()> {
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

pub(crate) async fn run_daemon_process(config: AppConfig, memory_log_path: &Path) -> Result<()> {
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

pub(crate) async fn run_start_mode(config: AppConfig, memory_log_path: &Path) -> Result<()> {
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

pub(crate) async fn run_telegram_runtime(config: AppConfig, memory_log_path: &Path) -> Result<()> {
    run_telegram_runtime_guarded(config, memory_log_path, false).await
}

/// `silent_if_locked`: when true (background mode), skip without error if another instance
/// already holds the lock. When false (explicit `aigent telegram`), error immediately.
pub(crate) async fn run_telegram_runtime_guarded(
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

pub(crate) fn ensure_daemon_running(config: &AppConfig) -> Result<()> {
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

