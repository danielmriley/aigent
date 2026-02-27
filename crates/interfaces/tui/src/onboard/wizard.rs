//! Wizard logic: entry points, prompts, and configuration helpers.

use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use std::fs;
use std::io::IsTerminal;

use aigent_config::{AppConfig, ApprovalMode};
use aigent_runtime::AgentRuntime;

use super::state::*;
use super::ui;

pub fn run_onboarding(config: &mut AppConfig, models: AvailableModels) -> Result<()> {
    run_setup_wizard(config, SetupMode::Onboarding, models)
}

pub fn run_configuration(config: &mut AppConfig, models: AvailableModels) -> Result<()> {
    run_setup_wizard(config, SetupMode::Configuration, models)
}

pub(super) fn run_setup_wizard(
    config: &mut AppConfig,
    mode: SetupMode,
    models: AvailableModels,
) -> Result<()> {
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return ui::run_onboarding_tui(config, mode, models);
    }

    run_onboarding_prompt(config, mode, models)
}

pub(super) fn run_onboarding_prompt(
    config: &mut AppConfig,
    mode: SetupMode,
    _models: AvailableModels,
) -> Result<()> {
    if mode == SetupMode::Configuration {
        println!("Configuration Wizard");
        loop {
            let section = prompt_config_section()?;
            match section {
                ConfigSection::Identity => prompt_identity_settings(config, false)?,
                ConfigSection::Llm => prompt_llm_settings(config)?,
                ConfigSection::Telegram => prompt_telegram_settings(config)?,
                ConfigSection::DataMemory => prompt_data_memory_settings(config)?,
                ConfigSection::Safety => prompt_safety_settings(config)?,
                ConfigSection::FullWizard => {
                    prompt_identity_settings(config, false)?;
                    prompt_llm_settings(config)?;
                    prompt_data_memory_settings(config)?;
                    prompt_safety_settings(config)?;
                    prompt_telegram_settings(config)?;
                }
            }

            if !prompt_bool("Configure another section? (yes/no)", false)? {
                break;
            }
        }
    } else {
        println!("Onboarding Wizard");
        prompt_identity_settings(config, true)?;
        prompt_llm_settings(config)?;
        prompt_data_memory_settings(config)?;
        prompt_safety_settings(config)?;
        prompt_telegram_settings(config)?;
    }

    config.onboarding.completed = true;

    Ok(())
}

pub(super) fn command_exists(command: &str) -> bool {
    std::process::Command::new(command)
        .arg("--version")
        .output()
        .is_ok()
}

pub(super) fn mask_secret(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        "•".repeat(value.chars().count())
    }
}

pub(super) fn test_llm_connection_for_draft(draft: &OnboardingDraft) -> Result<String> {
    let mut config = AppConfig::default();
    draft.apply_partial(&mut config)?;
    let runtime = AgentRuntime::new(config);
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(runtime.test_model_connection())
}

pub(super) fn test_telegram_connection(token: &str) -> Result<()> {
    let token = token.trim();
    if token.is_empty() {
        bail!("telegram token is empty")
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let url = format!("https://api.telegram.org/bot{token}/getMe");
        let response = reqwest::Client::new().get(url).send().await?;
        let payload: serde_json::Value = response.json().await?;
        if payload
            .get("ok")
            .and_then(|ok| ok.as_bool())
            .unwrap_or(false)
        {
            Ok(())
        } else {
            bail!(
                "telegram token rejected: {}",
                payload
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("unknown error")
            )
        }
    })
}

pub(super) fn parse_hour(raw: &str) -> Result<u8> {
    let value: u8 = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("hour must be an integer between 0 and 23"))?;
    if value > 23 {
        bail!("hour must be between 0 and 23");
    }
    Ok(value)
}

pub(super) fn apply_safety_profile(config: &mut AppConfig, profile: &str) {
    if profile.eq_ignore_ascii_case("power-user") {
        config.safety.approval_required = true;
        config.safety.allow_shell = true;
        config.safety.allow_wasm = true;
    } else {
        config.safety.approval_required = true;
        config.safety.allow_shell = false;
        config.safety.allow_wasm = false;
    }
}

pub(super) fn normalize_provider(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("openrouter") {
        Ok("openrouter".to_string())
    } else if raw.eq_ignore_ascii_case("ollama") {
        Ok("ollama".to_string())
    } else {
        bail!("provider must be one of: ollama, openrouter");
    }
}

pub(super) fn normalize_thinking_level(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("low") {
        Ok("low".to_string())
    } else if raw.eq_ignore_ascii_case("balanced") {
        Ok("balanced".to_string())
    } else if raw.eq_ignore_ascii_case("deep") {
        Ok("deep".to_string())
    } else {
        bail!("thinking level must be one of: low, balanced, deep");
    }
}

pub(super) fn normalize_safety_profile(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("strict") {
        Ok("strict".to_string())
    } else if raw.eq_ignore_ascii_case("power-user") || raw.eq_ignore_ascii_case("poweruser") {
        Ok("power-user".to_string())
    } else {
        bail!("safety profile must be one of: strict, power-user");
    }
}

pub(super) fn normalize_approval_mode(raw: &str) -> Result<String> {
    match raw.trim().to_lowercase().as_str() {
        "safer" => Ok("safer".to_string()),
        "balanced" => Ok("balanced".to_string()),
        "autonomous" => Ok("autonomous".to_string()),
        _ => bail!("approval mode must be one of: safer, balanced, autonomous"),
    }
}

pub(super) fn apply_approval_mode(config: &mut AppConfig, approval_mode: &str) {
    config.tools.approval_mode = match approval_mode.to_lowercase().as_str() {
        "safer" => ApprovalMode::Safer,
        "autonomous" => ApprovalMode::Autonomous,
        _ => ApprovalMode::Balanced,
    };
}

pub(super) fn detect_project_root() -> Option<PathBuf> {
    let mut cursor = std::env::current_dir().ok()?;
    loop {
        if cursor.join("Cargo.toml").exists() {
            return Some(cursor);
        }
        if !cursor.pop() {
            return None;
        }
    }
}

pub(super) fn is_forbidden_data_dir(path: &Path) -> bool {
    let Some(root) = detect_project_root() else {
        return false;
    };

    let path = path.to_path_buf();
    if path == root {
        return true;
    }

    let forbidden = [
        "crates",
        "src",
        "target",
        ".git",
        "config",
        "docs",
        "skills-src",
        "wit",
        ".aigent",
    ];

    forbidden
        .iter()
        .map(|segment| root.join(segment))
        .any(|blocked| path.starts_with(blocked))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResolveDataDirMode {
    AskToCreate,
    CreateIfMissing,
}

pub(super) fn resolve_data_directory(raw: &str, mode: ResolveDataDirMode) -> Result<PathBuf> {
    let trimmed = raw.trim();
    let chosen = if trimmed.is_empty() {
        "aigent-data"
    } else {
        trimmed
    };

    let mut path = PathBuf::from(chosen);
    if !path.is_absolute() {
        path = std::env::current_dir()?.join(path);
    }

    if is_forbidden_data_dir(&path) {
        bail!("forbidden data directory; choose a path outside project source folders")
    }

    if !path.exists() {
        match mode {
            ResolveDataDirMode::AskToCreate => {
                let create = prompt_bool("Directory does not exist. Create it? (Y/n)", true)?;
                if !create {
                    bail!("data directory does not exist")
                }
            }
            ResolveDataDirMode::CreateIfMissing => {}
        }
        fs::create_dir_all(&path)?;
    }

    if !path.is_dir() {
        bail!("data directory must be a directory: {}", path.display());
    }

    Ok(path)
}

pub(super) fn apply_memory_contract(config: &mut AppConfig, workspace: &Path) -> Result<()> {
    config.memory.backend = "eventlog".to_string();
    config.memory.auto_sleep_mode = "nightly".to_string();
    config.memory.core_rewrite_requires_approval = true;

    let memory_dir = workspace.join(".aigent").join("memory");
    fs::create_dir_all(&memory_dir)?;
    Ok(())
}

pub(super) fn prompt_config_section() -> Result<ConfigSection> {
    println!("\nChoose a section to configure:");
    for (index, section) in ConfigSection::all().iter().enumerate() {
        println!("  {}. {}", index + 1, section.label());
    }

    loop {
        let raw = prompt("Section number", "6")?;
        if let Ok(number) = raw.trim().parse::<usize>() {
            if (1..=ConfigSection::all().len()).contains(&number) {
                return Ok(ConfigSection::all()[number - 1]);
            }
        }
        println!("Enter a valid number from 1 to 6.");
    }
}

fn prompt_identity_settings(config: &mut AppConfig, require_user_name: bool) -> Result<()> {
    let user_default = if config.agent.user_name.trim().is_empty() {
        ""
    } else {
        &config.agent.user_name
    };
    let user_name = prompt("What is your name?", user_default)?;
    if require_user_name && user_name.trim().is_empty() {
        bail!("user name is required");
    }
    if !user_name.trim().is_empty() {
        config.agent.user_name = user_name;
    }

    let bot_name = prompt(
        "Choose a name for your bot",
        if config.agent.name.trim().is_empty() {
            "Aigent"
        } else {
            &config.agent.name
        },
    )?;
    config.agent.name = bot_name;
    Ok(())
}

fn prompt_llm_settings(config: &mut AppConfig) -> Result<()> {
    let provider = prompt(
        "LLM provider (ollama/openrouter)",
        if config.llm.provider.trim().is_empty() {
            "ollama"
        } else {
            &config.llm.provider
        },
    )?;
    let provider = normalize_provider(&provider)?;
    if provider == "openrouter" {
        config.llm.provider = "openrouter".to_string();
        let model = prompt("OpenRouter model", &config.llm.openrouter_model)?;
        config.llm.openrouter_model = model;

        let key = prompt_optional("OpenRouter API key (leave blank to keep existing)")?;
        if let Some(key) = key {
            upsert_env_value(Path::new(".env"), "OPENROUTER_API_KEY", &key)?;
        }
    } else {
        config.llm.provider = "ollama".to_string();
        let model = prompt("Ollama model", &config.llm.ollama_model)?;
        config.llm.ollama_model = model;
    }

    let thinking_level = prompt(
        "Thinking level (low/balanced/deep)",
        &config.agent.thinking_level,
    )?;
    config.agent.thinking_level = normalize_thinking_level(&thinking_level)?;
    Ok(())
}

fn prompt_data_memory_settings(config: &mut AppConfig) -> Result<()> {
    let data_dir_default = if config.agent.workspace_path.trim().is_empty()
        || config.agent.workspace_path.trim() == "."
    {
        "aigent-data"
    } else {
        &config.agent.workspace_path
    };
    let workspace = prompt("Data directory", data_dir_default)?;
    let workspace = resolve_data_directory(&workspace, ResolveDataDirMode::AskToCreate)?;
    config.agent.workspace_path = workspace.display().to_string();

    let sleep_start = prompt(
        "Night sleep start hour (0-23)",
        &config.memory.night_sleep_start_hour.to_string(),
    )?;
    config.memory.night_sleep_start_hour = parse_hour(&sleep_start)?;
    let sleep_end = prompt(
        "Night sleep end hour (0-23)",
        &config.memory.night_sleep_end_hour.to_string(),
    )?;
    config.memory.night_sleep_end_hour = parse_hour(&sleep_end)?;

    apply_memory_contract(config, &workspace)?;
    Ok(())
}

fn prompt_safety_settings(config: &mut AppConfig) -> Result<()> {
    let safety_profile = prompt("Safety profile (strict/power-user)", "strict")?;
    let safety_profile = normalize_safety_profile(&safety_profile)?;
    apply_safety_profile(config, &safety_profile);

    let approval_mode = prompt(
        "Tool approval mode (safer/balanced/autonomous)",
        "balanced",
    )?;
    let approval_mode = normalize_approval_mode(&approval_mode)?;
    apply_approval_mode(config, &approval_mode);

    let brave_key = prompt_optional("Brave Search API key (leave blank to use DuckDuckGo)")?;
    if let Some(key) = brave_key {
        config.tools.brave_api_key = key;
    }

    Ok(())
}

fn prompt_telegram_settings(config: &mut AppConfig) -> Result<()> {
    let telegram_enabled = prompt_bool(
        "Enable Telegram integration (yes/no)",
        config.integrations.telegram_enabled,
    )?;
    config.integrations.telegram_enabled = telegram_enabled;
    if telegram_enabled {
        let token = prompt_optional("Telegram bot token (leave blank to keep existing)")?;
        if let Some(token) = token {
            upsert_env_value(Path::new(".env"), "TELEGRAM_BOT_TOKEN", &token)?;
        }
    }
    Ok(())
}

pub(super) fn prompt(label: &str, default_value: &str) -> Result<String> {
    print!("{} [{}]: ", label, default_value);
    io::stdout().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(default_value.to_string());
    }

    Ok(trimmed.to_string())
}

pub(super) fn prompt_optional(label: &str) -> Result<Option<String>> {
    print!("{}: ", label);
    io::stdout().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    Ok(Some(trimmed.to_string()))
}

pub(super) fn prompt_bool(label: &str, default_value: bool) -> Result<bool> {
    let default_label = if default_value { "yes" } else { "no" };
    let value = prompt(label, default_label)?;
    parse_bool_like(&value)
}

pub(super) fn parse_bool_like(raw: &str) -> Result<bool> {
    let normalized = raw.trim().to_lowercase();
    match normalized.as_str() {
        "y" | "yes" | "true" | "1" | "enabled" | "on" => Ok(true),
        "n" | "no" | "false" | "0" | "disabled" | "off" => Ok(false),
        _ => bail!("expected yes/no"),
    }
}

pub(super) fn upsert_env_value(path: &Path, key: &str, value: &str) -> Result<()> {
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

