use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, bail};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, Paragraph, Wrap};

use aigent_config::{AppConfig, ApprovalMode};
use aigent_exec;
use aigent_runtime::AgentRuntime;

use crate::theme::Theme;

pub fn center(area: Rect, horizontal: Constraint, vertical: Constraint) -> Rect {
    let [area] = Layout::horizontal([horizontal])
        .flex(Flex::Center)
        .areas(area);
    let [area] = Layout::vertical([vertical]).flex(Flex::Center).areas(area);
    area
}

#[derive(Debug, Clone, Default)]
pub struct AvailableModels {
    pub ollama: Vec<String>,
    pub openrouter: Vec<String>,
}

pub fn run_onboarding(config: &mut AppConfig, models: AvailableModels) -> Result<()> {
    run_setup_wizard(config, SetupMode::Onboarding, models)
}

pub fn run_configuration(config: &mut AppConfig, models: AvailableModels) -> Result<()> {
    run_setup_wizard(config, SetupMode::Configuration, models)
}

fn run_setup_wizard(
    config: &mut AppConfig,
    mode: SetupMode,
    models: AvailableModels,
) -> Result<()> {
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return run_onboarding_tui(config, mode, models);
    }

    run_onboarding_prompt(config, mode, models)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SetupMode {
    Onboarding,
    Configuration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigSection {
    Identity,
    Llm,
    Telegram,
    DataMemory,
    Safety,
    FullWizard,
}

impl ConfigSection {
    fn label(self) -> &'static str {
        match self {
            ConfigSection::Identity => "Identity",
            ConfigSection::Llm => "LLM settings",
            ConfigSection::Telegram => "Telegram settings",
            ConfigSection::DataMemory => "Data directory & memory",
            ConfigSection::Safety => "Safety profile",
            ConfigSection::FullWizard => "Run full setup",
        }
    }

    fn all() -> [ConfigSection; 6] {
        [
            ConfigSection::Identity,
            ConfigSection::Llm,
            ConfigSection::Telegram,
            ConfigSection::DataMemory,
            ConfigSection::Safety,
            ConfigSection::FullWizard,
        ]
    }
}

fn run_onboarding_prompt(
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

fn run_onboarding_tui(
    config: &mut AppConfig,
    mode: SetupMode,
    models: AvailableModels,
) -> Result<()> {
    let mut draft = OnboardingDraft::from_config(config, models);

    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut step = if mode == SetupMode::Configuration {
        WizardStep::ConfigMenu
    } else {
        WizardStep::Welcome
    };
    let mut input = draft.value_for_step(step);
    let mut error_message: Option<String> = None;

    let result = (|| -> Result<()> {
        loop {
            draw_onboarding(
                &mut terminal,
                step,
                &draft,
                &input,
                error_message.as_deref(),
                mode,
            )?;

            if !event::poll(Duration::from_millis(120))? {
                continue;
            }

            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Esc => {
                    draft.apply_partial(config)?;
                    if draft.provider == "openrouter" && !draft.openrouter_key.trim().is_empty() {
                        upsert_env_value(
                            Path::new(".env"),
                            "OPENROUTER_API_KEY",
                            draft.openrouter_key.trim(),
                        )?;
                    }
                    if draft.telegram_enabled && !draft.telegram_token.trim().is_empty() {
                        upsert_env_value(
                            Path::new(".env"),
                            "TELEGRAM_BOT_TOKEN",
                            draft.telegram_token.trim(),
                        )?;
                    }
                    if !draft.brave_api_key.trim().is_empty() {
                        upsert_env_value(
                            Path::new(".env"),
                            "BRAVE_API_KEY",
                            draft.brave_api_key.trim(),
                        )?;
                    }
                    return Ok(());
                }
                KeyCode::Backspace => {
                    if step.is_text_step() {
                        input.pop();
                    }
                }
                KeyCode::Left => {
                    step = prev_step(step, &draft, mode);
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                KeyCode::Up => {
                    draft.choice_prev(step);
                    input = draft.value_for_step(step);
                }
                KeyCode::Down => {
                    draft.choice_next(step);
                    input = draft.value_for_step(step);
                }
                KeyCode::Char('b') if !step.is_text_step() => {
                    step = prev_step(step, &draft, mode);
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                KeyCode::Char('t') if !step.is_text_step() => {
                    let tested = match step {
                        WizardStep::Model | WizardStep::OpenRouterKey => {
                            Some(test_llm_connection_for_draft(&draft).map(|ok| {
                                format!("Connected ✓ {ok}")
                            }))
                        }
                        WizardStep::TelegramToken => Some(test_telegram_connection(&draft.telegram_token).map(|_| {
                            "Connected ✓ Telegram token looks valid".to_string()
                        })),
                        _ => None,
                    };

                    if let Some(result) = tested {
                        error_message = Some(match result {
                            Ok(msg) => msg,
                            Err(err) => format!("Connection failed: {err}"),
                        });
                    }
                }
                KeyCode::Char(ch) => {
                    if step.is_text_step() {
                        input.push(ch);
                    }
                }
                KeyCode::Enter => {
                    if step == WizardStep::Summary {
                        if let Err(err) = draft.apply(config) {
                            error_message = Some(err.to_string());
                            continue;
                        }
                        if draft.provider == "openrouter" && !draft.openrouter_key.trim().is_empty()
                        {
                            upsert_env_value(
                                Path::new(".env"),
                                "OPENROUTER_API_KEY",
                                draft.openrouter_key.trim(),
                            )?;
                        }
                        if draft.telegram_enabled && !draft.telegram_token.trim().is_empty() {
                            upsert_env_value(
                                Path::new(".env"),
                                "TELEGRAM_BOT_TOKEN",
                                draft.telegram_token.trim(),
                            )?;
                        }
                        if !draft.brave_api_key.trim().is_empty() {
                            upsert_env_value(
                                Path::new(".env"),
                                "BRAVE_API_KEY",
                                draft.brave_api_key.trim(),
                            )?;
                        }
                        break;
                    }

                    if step == WizardStep::Welcome || step == WizardStep::ConfigMenu {
                        step = next_step(step, &draft, mode);
                        input = draft.value_for_step(step);
                        error_message = None;
                        continue;
                    }

                    if let Err(err) = draft.commit_step(step, &input) {
                        error_message = Some(err.to_string());
                        continue;
                    }
                    step = next_step(step, &draft, mode);
                    if step == WizardStep::ConfigMenu && mode == SetupMode::Configuration {
                        let _ = draft.apply_partial(config);
                    }
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                KeyCode::Tab => {
                    draft.choice_next(step);
                    input = draft.value_for_step(step);
                    error_message = None;
                }
                _ => {}
            }
        }

        Ok(())
    })();

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WizardStep {
    ConfigMenu,
    Welcome,
    UserName,
    BotName,
    DataDirectory,
    Provider,
    Model,
    OpenRouterKey,
    Thinking,
    NightSleepStart,
    NightSleepEnd,
    Safety,
    /// Choose approval mode: safer / balanced / autonomous
    ApprovalMode,
    /// Enter optional API keys (Brave Search, etc.)
    ApiKeys,
    Messaging,
    TelegramToken,
    Summary,
}

impl WizardStep {
    fn is_text_step(self) -> bool {
        matches!(
            self,
            WizardStep::BotName
                | WizardStep::UserName
                | WizardStep::DataDirectory
                | WizardStep::OpenRouterKey
                | WizardStep::ApiKeys
                | WizardStep::NightSleepStart
                | WizardStep::NightSleepEnd
                | WizardStep::TelegramToken
        )
    }
}

#[derive(Debug, Clone)]
struct OnboardingDraft {
    config_section: ConfigSection,
    user_name: String,
    bot_name: String,
    workspace_path: String,
    provider: String,
    ollama_model: String,
    openrouter_model: String,
    openrouter_key: String,
    thinking_level: String,
    night_sleep_start_hour: u8,
    night_sleep_end_hour: u8,
    safety_profile: String,
    /// Tool approval mode: "safer" | "balanced" | "autonomous"
    approval_mode: String,
    /// Optional Brave Search API key (masked in UI; stored to config / .env)
    brave_api_key: String,
    telegram_enabled: bool,
    telegram_token: String,
    available_models: AvailableModels,
}

impl OnboardingDraft {
    fn from_config(config: &AppConfig, models: AvailableModels) -> Self {
        Self {
            config_section: ConfigSection::Identity,
            user_name: config.agent.user_name.clone(),
            bot_name: config.agent.name.clone(),
            workspace_path: config.agent.workspace_path.clone(),
            provider: config.llm.provider.clone(),
            ollama_model: config.llm.ollama_model.clone(),
            openrouter_model: config.llm.openrouter_model.clone(),
            openrouter_key: String::new(),
            thinking_level: config.agent.thinking_level.clone(),
            night_sleep_start_hour: config.memory.night_sleep_start_hour,
            night_sleep_end_hour: config.memory.night_sleep_end_hour,
            safety_profile: if config.safety.allow_shell || config.safety.allow_wasm {
                "power-user".to_string()
            } else {
                "strict".to_string()
            },
            approval_mode: match &config.tools.approval_mode {
                ApprovalMode::Safer => "safer".to_string(),
                ApprovalMode::Balanced => "balanced".to_string(),
                ApprovalMode::Autonomous => "autonomous".to_string(),
            },
            brave_api_key: config.tools.brave_api_key.clone(),
            telegram_enabled: config.integrations.telegram_enabled,
            telegram_token: String::new(),
            available_models: models,
        }
    }

    fn value_for_step(&self, step: WizardStep) -> String {
        match step {
            WizardStep::ConfigMenu => self.config_section.label().to_string(),
            WizardStep::Welcome => String::new(),
            WizardStep::UserName => self.user_name.clone(),
            WizardStep::BotName => self.bot_name.clone(),
            WizardStep::DataDirectory => self.workspace_path.clone(),
            WizardStep::Provider => self.provider.clone(),
            WizardStep::Model => {
                if self.provider.eq_ignore_ascii_case("openrouter") {
                    self.openrouter_model.clone()
                } else {
                    self.ollama_model.clone()
                }
            }
            WizardStep::OpenRouterKey => self.openrouter_key.clone(),
            WizardStep::Thinking => self.thinking_level.clone(),
            WizardStep::NightSleepStart => self.night_sleep_start_hour.to_string(),
            WizardStep::NightSleepEnd => self.night_sleep_end_hour.to_string(),
            WizardStep::Safety => self.safety_profile.clone(),
            WizardStep::ApprovalMode => self.approval_mode.clone(),
            WizardStep::ApiKeys => self.brave_api_key.clone(),
            WizardStep::Messaging => {
                if self.telegram_enabled {
                    "enabled".to_string()
                } else {
                    "disabled".to_string()
                }
            }
            WizardStep::TelegramToken => self.telegram_token.clone(),
            WizardStep::Summary => String::new(),
        }
    }

    fn commit_step(&mut self, step: WizardStep, input: &str) -> Result<()> {
        let value = input.trim();
        match step {
            WizardStep::ConfigMenu => {
                if let Some((index, _)) = ConfigSection::all()
                    .iter()
                    .enumerate()
                    .find(|(_, section)| section.label().eq_ignore_ascii_case(value))
                {
                    self.config_section = ConfigSection::all()[index];
                }
            }
            WizardStep::UserName => {
                if value.is_empty() {
                    bail!("user name cannot be empty");
                }
                self.user_name = value.to_string();
            }
            WizardStep::BotName => {
                if value.is_empty() {
                    bail!("bot name cannot be empty");
                }
                self.bot_name = value.to_string();
            }
            WizardStep::DataDirectory => {
                if value.is_empty() {
                    bail!("data directory cannot be empty");
                }
                let path = resolve_data_directory(value, ResolveDataDirMode::CreateIfMissing)?;
                self.workspace_path = path.display().to_string();
            }
            WizardStep::Model => {
                if value.is_empty() {
                    bail!("model cannot be empty");
                }
                if self.provider.eq_ignore_ascii_case("openrouter") {
                    self.openrouter_model = value.to_string();
                } else {
                    self.ollama_model = value.to_string();
                }
            }
            WizardStep::OpenRouterKey => {
                self.openrouter_key = value.to_string();
            }
            WizardStep::NightSleepStart => {
                self.night_sleep_start_hour = parse_hour(value)?;
            }
            WizardStep::NightSleepEnd => {
                self.night_sleep_end_hour = parse_hour(value)?;
            }
            WizardStep::TelegramToken => {
                self.telegram_token = value.to_string();
            }
            WizardStep::ApprovalMode => {
                self.approval_mode = normalize_approval_mode(value)
                    .unwrap_or_else(|_| "balanced".to_string());
            }
            WizardStep::ApiKeys => {
                // Blank input = keep existing value; non-blank = set new value.
                if !value.is_empty() {
                    self.brave_api_key = value.to_string();
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn choice_next(&mut self, step: WizardStep) {
        match step {
            WizardStep::ConfigMenu => {
                let all = ConfigSection::all();
                let idx = all
                    .iter()
                    .position(|section| *section == self.config_section)
                    .unwrap_or(0);
                self.config_section = all[(idx + 1) % all.len()];
            }
            WizardStep::Provider => {
                self.provider = if self.provider.eq_ignore_ascii_case("openrouter") {
                    "ollama".to_string()
                } else {
                    "openrouter".to_string()
                }
            }
            WizardStep::Model => {
                let is_openrouter = self.provider.eq_ignore_ascii_case("openrouter");
                let models = if is_openrouter {
                    &self.available_models.openrouter
                } else {
                    &self.available_models.ollama
                };

                if models.is_empty() {
                    return;
                }

                let current = if is_openrouter {
                    &self.openrouter_model
                } else {
                    &self.ollama_model
                };

                let idx = models.iter().position(|m| m == current).unwrap_or(0);
                let next_idx = (idx + 1) % models.len();
                let next_model = models[next_idx].clone();

                if is_openrouter {
                    self.openrouter_model = next_model;
                } else {
                    self.ollama_model = next_model;
                }
            }
            WizardStep::Thinking => {
                self.thinking_level = match self.thinking_level.to_lowercase().as_str() {
                    "low" => "balanced".to_string(),
                    "balanced" => "deep".to_string(),
                    _ => "low".to_string(),
                }
            }
            WizardStep::Safety => {
                self.safety_profile = if self.safety_profile.eq_ignore_ascii_case("power-user") {
                    "strict".to_string()
                } else {
                    "power-user".to_string()
                }
            }
            WizardStep::ApprovalMode => {
                self.approval_mode = match self.approval_mode.to_lowercase().as_str() {
                    "safer" => "balanced".to_string(),
                    "balanced" => "autonomous".to_string(),
                    _ => "safer".to_string(),
                }
            }
            WizardStep::Messaging => {
                self.telegram_enabled = !self.telegram_enabled;
            }
            _ => {}
        }
    }

    fn choice_prev(&mut self, step: WizardStep) {
        match step {
            WizardStep::ConfigMenu => {
                let all = ConfigSection::all();
                let idx = all
                    .iter()
                    .position(|section| *section == self.config_section)
                    .unwrap_or(0);
                let prev = if idx == 0 { all.len() - 1 } else { idx - 1 };
                self.config_section = all[prev];
            }
            WizardStep::Model => {
                let is_openrouter = self.provider.eq_ignore_ascii_case("openrouter");
                let models = if is_openrouter {
                    &self.available_models.openrouter
                } else {
                    &self.available_models.ollama
                };

                if models.is_empty() {
                    return;
                }

                let current = if is_openrouter {
                    &self.openrouter_model
                } else {
                    &self.ollama_model
                };

                let idx = models.iter().position(|m| m == current).unwrap_or(0);
                let prev_idx = if idx == 0 { models.len() - 1 } else { idx - 1 };
                let prev_model = models[prev_idx].clone();

                if is_openrouter {
                    self.openrouter_model = prev_model;
                } else {
                    self.ollama_model = prev_model;
                }
            }
            WizardStep::Thinking => {
                self.thinking_level = match self.thinking_level.to_lowercase().as_str() {
                    "low" => "deep".to_string(),
                    "balanced" => "low".to_string(),
                    _ => "balanced".to_string(),
                }
            }
            WizardStep::ApprovalMode => {
                self.approval_mode = match self.approval_mode.to_lowercase().as_str() {
                    "safer" => "autonomous".to_string(),
                    "balanced" => "safer".to_string(),
                    _ => "balanced".to_string(),
                }
            }
            _ => self.choice_next(step),
        }
    }

    fn apply(&self, config: &mut AppConfig) -> Result<()> {
        config.agent.user_name = self.user_name.clone();
        config.agent.name = self.bot_name.clone();
        config.agent.workspace_path = self.workspace_path.clone();
        config.llm.provider = normalize_provider(&self.provider)?;
        config.llm.ollama_model = self.ollama_model.clone();
        config.llm.openrouter_model = self.openrouter_model.clone();
        config.agent.thinking_level = normalize_thinking_level(&self.thinking_level)?;
        config.memory.night_sleep_start_hour = self.night_sleep_start_hour;
        config.memory.night_sleep_end_hour = self.night_sleep_end_hour;
        let safety_profile = normalize_safety_profile(&self.safety_profile)?;
        apply_safety_profile(config, &safety_profile);
        apply_approval_mode(config, &self.approval_mode);
        if !self.brave_api_key.trim().is_empty() {
            config.tools.brave_api_key = self.brave_api_key.trim().to_string();
        }
        config.integrations.telegram_enabled = self.telegram_enabled;

        let workspace = resolve_data_directory(
            &config.agent.workspace_path,
            ResolveDataDirMode::CreateIfMissing,
        )?;
        apply_memory_contract(config, &workspace)?;
        config.onboarding.completed = true;

        Ok(())
    }

    fn apply_partial(&self, config: &mut AppConfig) -> Result<()> {
        if !self.user_name.trim().is_empty() {
            config.agent.user_name = self.user_name.clone();
        }
        if !self.bot_name.trim().is_empty() {
            config.agent.name = self.bot_name.clone();
        }
        if !self.workspace_path.trim().is_empty() {
            config.agent.workspace_path = self.workspace_path.clone();
        }
        config.llm.provider = normalize_provider(&self.provider)?;
        config.llm.ollama_model = self.ollama_model.clone();
        config.llm.openrouter_model = self.openrouter_model.clone();
        if let Ok(level) = normalize_thinking_level(&self.thinking_level) {
            config.agent.thinking_level = level;
        }
        config.memory.night_sleep_start_hour = self.night_sleep_start_hour;
        config.memory.night_sleep_end_hour = self.night_sleep_end_hour;
        if let Ok(profile) = normalize_safety_profile(&self.safety_profile) {
            apply_safety_profile(config, &profile);
        }
        apply_approval_mode(config, &self.approval_mode);
        if !self.brave_api_key.trim().is_empty() {
            config.tools.brave_api_key = self.brave_api_key.trim().to_string();
        }
        config.integrations.telegram_enabled = self.telegram_enabled;
        Ok(())
    }
}

fn next_step(current: WizardStep, draft: &OnboardingDraft, mode: SetupMode) -> WizardStep {
    if mode == SetupMode::Configuration && draft.config_section != ConfigSection::FullWizard {
        return match (draft.config_section, current) {
            (ConfigSection::Identity, WizardStep::ConfigMenu) => WizardStep::UserName,
            (ConfigSection::Identity, WizardStep::UserName) => WizardStep::BotName,
            (ConfigSection::Identity, WizardStep::BotName) => WizardStep::ConfigMenu,

            (ConfigSection::Llm, WizardStep::ConfigMenu) => WizardStep::Provider,
            (ConfigSection::Llm, WizardStep::Provider) => WizardStep::Model,
            (ConfigSection::Llm, WizardStep::Model) => {
                if draft.provider.eq_ignore_ascii_case("openrouter") {
                    WizardStep::OpenRouterKey
                } else {
                    WizardStep::ConfigMenu
                }
            }
            (ConfigSection::Llm, WizardStep::OpenRouterKey) => WizardStep::ConfigMenu,

            (ConfigSection::Telegram, WizardStep::ConfigMenu) => WizardStep::Messaging,
            (ConfigSection::Telegram, WizardStep::Messaging) => {
                if draft.telegram_enabled {
                    WizardStep::TelegramToken
                } else {
                    WizardStep::ConfigMenu
                }
            }
            (ConfigSection::Telegram, WizardStep::TelegramToken) => WizardStep::ConfigMenu,

            (ConfigSection::DataMemory, WizardStep::ConfigMenu) => WizardStep::DataDirectory,
            (ConfigSection::DataMemory, WizardStep::DataDirectory) => WizardStep::NightSleepStart,
            (ConfigSection::DataMemory, WizardStep::NightSleepStart) => WizardStep::NightSleepEnd,
            (ConfigSection::DataMemory, WizardStep::NightSleepEnd) => WizardStep::ConfigMenu,

            (ConfigSection::Safety, WizardStep::ConfigMenu) => WizardStep::Safety,
            (ConfigSection::Safety, WizardStep::Safety) => WizardStep::ApprovalMode,
            (ConfigSection::Safety, WizardStep::ApprovalMode) => WizardStep::ApiKeys,
            (ConfigSection::Safety, WizardStep::ApiKeys) => WizardStep::ConfigMenu,

            _ => WizardStep::ConfigMenu,
        };
    }

    match current {
        WizardStep::ConfigMenu => WizardStep::Welcome,
        WizardStep::Welcome => WizardStep::UserName,
        WizardStep::UserName => WizardStep::BotName,
        WizardStep::BotName => WizardStep::DataDirectory,
        WizardStep::DataDirectory => WizardStep::Provider,
        WizardStep::Provider => WizardStep::Model,
        WizardStep::Model => {
            if draft.provider.eq_ignore_ascii_case("openrouter") {
                WizardStep::OpenRouterKey
            } else {
                WizardStep::Thinking
            }
        }
        WizardStep::OpenRouterKey => WizardStep::Thinking,
        WizardStep::Thinking => WizardStep::NightSleepStart,
        WizardStep::NightSleepStart => WizardStep::NightSleepEnd,
        WizardStep::NightSleepEnd => WizardStep::Safety,
        WizardStep::Safety => WizardStep::ApprovalMode,
        WizardStep::ApprovalMode => WizardStep::ApiKeys,
        WizardStep::ApiKeys => WizardStep::Messaging,
        WizardStep::Messaging => {
            if draft.telegram_enabled {
                WizardStep::TelegramToken
            } else {
                WizardStep::Summary
            }
        }
        WizardStep::TelegramToken => WizardStep::Summary,
        WizardStep::Summary => WizardStep::Summary,
    }
}

fn prev_step(current: WizardStep, draft: &OnboardingDraft, mode: SetupMode) -> WizardStep {
    if mode == SetupMode::Configuration && draft.config_section != ConfigSection::FullWizard {
        return match (draft.config_section, current) {
            (_, WizardStep::ConfigMenu) => WizardStep::ConfigMenu,
            (ConfigSection::Identity, WizardStep::UserName) => WizardStep::ConfigMenu,
            (ConfigSection::Identity, WizardStep::BotName) => WizardStep::UserName,

            (ConfigSection::Llm, WizardStep::Provider) => WizardStep::ConfigMenu,
            (ConfigSection::Llm, WizardStep::Model) => WizardStep::Provider,
            (ConfigSection::Llm, WizardStep::OpenRouterKey) => WizardStep::Model,

            (ConfigSection::Telegram, WizardStep::Messaging) => WizardStep::ConfigMenu,
            (ConfigSection::Telegram, WizardStep::TelegramToken) => WizardStep::Messaging,

            (ConfigSection::DataMemory, WizardStep::DataDirectory) => WizardStep::ConfigMenu,
            (ConfigSection::DataMemory, WizardStep::NightSleepStart) => WizardStep::DataDirectory,
            (ConfigSection::DataMemory, WizardStep::NightSleepEnd) => {
                WizardStep::NightSleepStart
            }

            (ConfigSection::Safety, WizardStep::Safety) => WizardStep::ConfigMenu,
            (ConfigSection::Safety, WizardStep::ApprovalMode) => WizardStep::Safety,
            (ConfigSection::Safety, WizardStep::ApiKeys) => WizardStep::ApprovalMode,
            _ => WizardStep::ConfigMenu,
        };
    }

    match current {
        WizardStep::ConfigMenu => WizardStep::ConfigMenu,
        WizardStep::Welcome => WizardStep::Welcome,
        WizardStep::UserName => WizardStep::Welcome,
        WizardStep::BotName => WizardStep::UserName,
        WizardStep::DataDirectory => WizardStep::BotName,
        WizardStep::Provider => WizardStep::DataDirectory,
        WizardStep::Model => WizardStep::Provider,
        WizardStep::OpenRouterKey => WizardStep::Model,
        WizardStep::Thinking => {
            if draft.provider.eq_ignore_ascii_case("openrouter") {
                WizardStep::OpenRouterKey
            } else {
                WizardStep::Model
            }
        }
        WizardStep::NightSleepStart => WizardStep::Thinking,
        WizardStep::NightSleepEnd => WizardStep::NightSleepStart,
        WizardStep::Safety => WizardStep::NightSleepEnd,
        WizardStep::ApprovalMode => WizardStep::Safety,
        WizardStep::ApiKeys => WizardStep::ApprovalMode,
        WizardStep::Messaging => WizardStep::ApiKeys,
        WizardStep::TelegramToken => WizardStep::Messaging,
        WizardStep::Summary => {
            if draft.telegram_enabled {
                WizardStep::TelegramToken
            } else {
                WizardStep::Messaging
            }
        }
    }
}

fn active_steps(mode: SetupMode, draft: &OnboardingDraft) -> Vec<WizardStep> {
    if mode == SetupMode::Configuration {
        return match draft.config_section {
            ConfigSection::Identity => {
                vec![WizardStep::ConfigMenu, WizardStep::UserName, WizardStep::BotName]
            }
            ConfigSection::Llm => {
                let mut steps = vec![WizardStep::ConfigMenu, WizardStep::Provider, WizardStep::Model];
                if draft.provider.eq_ignore_ascii_case("openrouter") {
                    steps.push(WizardStep::OpenRouterKey);
                }
                steps
            }
            ConfigSection::Telegram => {
                let mut steps = vec![WizardStep::ConfigMenu, WizardStep::Messaging];
                if draft.telegram_enabled {
                    steps.push(WizardStep::TelegramToken);
                }
                steps
            }
            ConfigSection::DataMemory => vec![
                WizardStep::ConfigMenu,
                WizardStep::DataDirectory,
                WizardStep::NightSleepStart,
                WizardStep::NightSleepEnd,
            ],
            ConfigSection::Safety => vec![
                WizardStep::ConfigMenu,
                WizardStep::Safety,
                WizardStep::ApprovalMode,
                WizardStep::ApiKeys,
            ],
            ConfigSection::FullWizard => {
                let mut steps = vec![WizardStep::ConfigMenu];
                steps.extend(active_steps(SetupMode::Onboarding, draft));
                steps
            }
        };
    }

    let mut steps = vec![
        WizardStep::Welcome,
        WizardStep::UserName,
        WizardStep::BotName,
        WizardStep::DataDirectory,
        WizardStep::Provider,
        WizardStep::Model,
    ];
    if draft.provider.eq_ignore_ascii_case("openrouter") {
        steps.push(WizardStep::OpenRouterKey);
    }
    steps.extend([
        WizardStep::Thinking,
        WizardStep::NightSleepStart,
        WizardStep::NightSleepEnd,
        WizardStep::Safety,
        WizardStep::ApprovalMode,
        WizardStep::ApiKeys,
        WizardStep::Messaging,
    ]);
    if draft.telegram_enabled {
        steps.push(WizardStep::TelegramToken);
    }
    steps.push(WizardStep::Summary);
    steps
}

fn step_progress(step: WizardStep, mode: SetupMode, draft: &OnboardingDraft) -> (usize, usize) {
    let steps = active_steps(mode, draft);
    let total = steps.len();
    let index = steps
        .iter()
        .position(|candidate| *candidate == step)
        .unwrap_or(0)
        + 1;
    (index, total)
}

fn draw_onboarding(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    step: WizardStep,
    draft: &OnboardingDraft,
    input: &str,
    error_message: Option<&str>,
    mode: SetupMode,
) -> Result<()> {
    let (title, description, input_display) = step_content(step, draft, input);
    let theme = Theme::default();
    let (step_index, step_total) = step_progress(step, mode, draft);
    let progress = if step_total == 0 {
        0.0
    } else {
        step_index as f64 / step_total as f64
    };

    terminal.draw(|frame| {
        let area = frame.area();

        let wizard_area = center(area, Constraint::Length(92), Constraint::Length(26));

        let header_title = if mode == SetupMode::Onboarding {
            " Aigent Onboarding "
        } else {
            " Aigent Configuration "
        };

        // Draw background block
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray))
            .title_top(
                ratatui::text::Line::from(Span::styled(
                    header_title,
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ))
                .centered(),
            )
            .title_bottom(
                ratatui::text::Line::from(Span::styled(
                    format!(" Step {step_index}/{step_total} "),
                    Style::default().fg(Color::DarkGray),
                ))
                .right_aligned(),
            );

        frame.render_widget(block.clone(), wizard_area);

        let inner_area = wizard_area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });

        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(24), Constraint::Min(40)])
            .split(inner_area);

        let sidebar_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.muted))
            .title(" Steps ");
        let sidebar_inner = sidebar_block.inner(columns[0]);
        frame.render_widget(sidebar_block, columns[0]);

        let labels = active_steps(mode, draft)
            .into_iter()
            .map(|candidate| {
                let label = match candidate {
                    WizardStep::ConfigMenu => "Configuration Menu",
                    WizardStep::Welcome => "Welcome",
                    WizardStep::UserName => "User Name",
                    WizardStep::BotName => "Bot Name",
                    WizardStep::DataDirectory => "Data Directory",
                    WizardStep::Provider => "Provider",
                    WizardStep::Model => "Model",
                    WizardStep::OpenRouterKey => "OpenRouter Key",
                    WizardStep::Thinking => "Thinking",
                    WizardStep::NightSleepStart => "Sleep Start",
                    WizardStep::NightSleepEnd => "Sleep End",
                    WizardStep::Safety => "Safety",
                    WizardStep::ApprovalMode => "Approval Mode",
                    WizardStep::ApiKeys => "API Keys",
                    WizardStep::Messaging => "Messaging",
                    WizardStep::TelegramToken => "Telegram Token",
                    WizardStep::Summary => "Summary",
                };

                if candidate == step {
                    Line::from(Span::styled(
                        format!("> {label}"),
                        Style::default()
                            .fg(theme.accent)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(Span::styled(
                        format!("  {label}"),
                        Style::default().fg(theme.foreground),
                    ))
                }
            })
            .collect::<Vec<_>>();
        frame.render_widget(Paragraph::new(labels), sidebar_inner);

        let content_area = columns[1];

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // progress
                Constraint::Length(2), // Title
                Constraint::Min(3),    // Description
                Constraint::Length(5), // Input box
                Constraint::Length(1), // Spacer
                Constraint::Length(2), // Footer/Error
            ])
            .split(content_area);

        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(theme.accent).bg(theme.background))
            .ratio(progress)
            .label(format!("Step {step_index} of {step_total}"));
        frame.render_widget(gauge, chunks[0]);

        // Title
        let title_widget = Paragraph::new(title)
            .style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(title_widget, chunks[1]);

        // Description
        let desc_widget = Paragraph::new(description)
            .style(Style::default().fg(Color::Gray))
            .wrap(Wrap { trim: true })
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(desc_widget, chunks[2]);

        // Input box
        if step != WizardStep::Welcome && step != WizardStep::Summary {
            let input_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme.accent));
            let input_widget = Paragraph::new(input_display)
                .block(input_block)
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: false })
                .alignment(ratatui::layout::Alignment::Center);

            // Center the input box horizontally
            let input_area = center(chunks[3], Constraint::Percentage(80), Constraint::Length(5));
            frame.render_widget(input_widget, input_area);
        } else if step == WizardStep::Summary {
            // For summary, we use the input box area for more description
            let summary_widget = Paragraph::new(input_display)
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true })
                .alignment(ratatui::layout::Alignment::Left);

            let summary_area = chunks[2].union(chunks[3]).union(chunks[4]);
            frame.render_widget(summary_widget, summary_area);
        }

        // Footer
        let footer_text = if let Some(message) = error_message {
            let is_success = message.starts_with("Connected ✓");
            if is_success {
                Line::from(Span::styled(
                    message,
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(vec![
                    Span::styled(
                        "Error: ",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(message, Style::default().fg(Color::LightRed)),
                ])
            }
        } else {
            Line::from(Span::styled(
                "Enter=next • Left/b=back • t=test connection • Esc=save draft & exit",
                Style::default().fg(Color::DarkGray),
            ))
        };

        let footer_widget =
            Paragraph::new(footer_text).alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(footer_widget, chunks[5]);
    })?;

    Ok(())
}

fn format_choice_list<'a>(options: &[&'a str], current: &str) -> ratatui::text::Text<'a> {
    if options.is_empty() {
        return ratatui::text::Text::from(vec![
            Line::from(""),
            Line::from(Span::styled(
                "No options available",
                Style::default().fg(Color::Red),
            )),
            Line::from(""),
        ]);
    }

    let idx = options.iter().position(|&o| o == current).unwrap_or(0);
    let prev_idx = if idx == 0 { options.len() - 1 } else { idx - 1 };
    let next_idx = (idx + 1) % options.len();

    let mut lines = Vec::new();

    if options.len() == 1 {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(
                "  > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                options[0],
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " <  ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
    } else if options.len() == 2 {
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
            Span::styled(options[prev_idx], Style::default().fg(Color::DarkGray)),
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(
                "  > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                options[idx],
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " <  ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
    } else {
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
            Span::styled(options[prev_idx], Style::default().fg(Color::DarkGray)),
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(
                "  > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                options[idx],
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " <  ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
            Span::styled(options[next_idx], Style::default().fg(Color::DarkGray)),
            Span::styled("    ", Style::default().fg(Color::DarkGray)),
        ]));
    }

    ratatui::text::Text::from(lines)
}

fn step_content<'a>(
    step: WizardStep,
    draft: &'a OnboardingDraft,
    input: &'a str,
) -> (&'static str, String, ratatui::text::Text<'a>) {
    match step {
        WizardStep::ConfigMenu => {
            let options = ConfigSection::all()
                .into_iter()
                .map(ConfigSection::label)
                .collect::<Vec<_>>();
            (
                "Configuration Menu",
                "Select a settings area, then press Enter.".to_string(),
                format_choice_list(&options, draft.config_section.label()),
            )
        }
        WizardStep::Welcome => (
            "Welcome to Aigent",
            "This wizard configures identity, memory policy, and safety.\n\n\
             Aigent ships with 8 built-in tools (web_search, read_file, write_file, \n\
             list_directory, run_shell, calendar_add_event, remind_me, draft_email) \n\
             running in WASM isolation by default (seccomp/sandbox active on Linux & macOS).\n\
             You'll choose an approval mode to control how much autonomy the agent has.\n\n\
             Press Enter to begin.".to_string(),
            ratatui::text::Text::from(""),
        ),
        WizardStep::BotName => (
            "Bot Identity",
            "Choose a name for your bot. This is how it will identify itself.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::UserName => (
            "User Identity",
            "What is your name? (the agent stores this in Core memory).".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::DataDirectory => (
            "Data Directory",
            "Where Aigent stores memory files, event logs, vault exports, and config (default: aigent-data).".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::Provider => (
            "LLM Provider",
            "Choose the LLM provider (Up/Down/Tab to cycle):\n\nOllama = local-first\nOpenRouter = hosted API".to_string(),
            format_choice_list(&["ollama", "openrouter"], &draft.provider),
        ),
        WizardStep::Model => {
            let is_openrouter = draft.provider.eq_ignore_ascii_case("openrouter");
            let models = if is_openrouter {
                &draft.available_models.openrouter
            } else {
                &draft.available_models.ollama
            };
            let current = if is_openrouter {
                &draft.openrouter_model
            } else {
                &draft.ollama_model
            };
            let model_refs: Vec<&str> = models.iter().map(|s| s.as_str()).collect();
            (
                "Model Selection",
                format!("Set the model to use for provider {}:", draft.provider),
                format_choice_list(&model_refs, current),
            )
        }
        WizardStep::OpenRouterKey => (
            "OpenRouter API Key",
            "Optional OpenRouter API key (saved to .env if set). Leave blank to keep existing.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", mask_secret(input))),
        ),
        WizardStep::Thinking => (
            "Thinking Level",
            "Choose default thinking depth (Up/Down/Tab to cycle):".to_string(),
            format_choice_list(&["low", "balanced", "deep"], &draft.thinking_level),
        ),
        WizardStep::NightSleepStart => (
            "Night Sleep Start",
            "Local hour (0-23) when the nightly sleep window starts:".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::NightSleepEnd => (
            "Night Sleep End",
            "Local hour (0-23) when the nightly sleep window ends:".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", input)),
        ),
        WizardStep::Safety => (
            "Safety Profile",
            "Choose safety profile (Up/Down/Tab to cycle):\n\nstrict: deny shell/wasm by default\npower-user: allow shell/wasm with approvals".to_string(),
            format_choice_list(&["strict", "power-user"], &draft.safety_profile),
        ),
        WizardStep::ApprovalMode => (
            "Tool Approval Mode",
            "How should the agent ask for approval before running tools?\n\nsafer     — ask for every tool call\nbalanced  — read-only tools free; writes/shell require approval (default)\nautonomous — never ask; all gated tools auto-approved within workspace".to_string(),
            format_choice_list(
                &["safer", "balanced", "autonomous"],
                &draft.approval_mode,
            ),
        ),
        WizardStep::ApiKeys => (
            "API Keys (optional)",
            "Brave Search API key for high-quality web search results.\nLeave blank to use the free DuckDuckGo fallback instead.\nKey is saved to config (never logged).".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", mask_secret(input))),
        ),
        WizardStep::Messaging => (
            "Messaging Integration",
            "Toggle Telegram integration (Up/Down/Tab to cycle):".to_string(),
            format_choice_list(&["enabled", "disabled"], if draft.telegram_enabled { "enabled" } else { "disabled" }),
        ),
        WizardStep::TelegramToken => (
            "Telegram Token",
            "Optional Telegram bot token (saved to .env if set). Leave blank to keep existing.".to_string(),
            ratatui::text::Text::from(format!("\n{}\n", mask_secret(input))),
        ),
        WizardStep::Summary => {
            let docker_ok = command_exists("docker");
            let memory_path = PathBuf::from(&draft.workspace_path)
                .join(".aigent")
                .join("memory");
            let sandbox_status = if aigent_exec::sandbox::is_active() { "active (platform sandbox compiled in)" } else { "inactive (rebuild with sandbox feature)" };
            (
                "Setup Summary",
                String::new(),
                ratatui::text::Text::from(format!(
                    "User: {}\nBot: {}\nProvider: {}\nModel: {}\nThinking: {}\nData directory: {}\nSleep mode: nightly ({}:00-{}:00 local)\nSafety: {}\nApproval mode: {}\nBrave Search key: {}\nTelegram: {}\nSandbox: {}\nDocker available: {}\nMemory path: {}\n\nMemory contract:\n- Event-log memory is canonical\n- Nightly sleep distills and promotes important memories\n- Core memories shape personality over time\n\nPress Enter to finalize setup.",
                    draft.user_name,
                    draft.bot_name,
                    draft.provider,
                    if draft.provider.eq_ignore_ascii_case("openrouter") {
                        &draft.openrouter_model
                    } else {
                        &draft.ollama_model
                    },
                    draft.thinking_level,
                    draft.workspace_path,
                    draft.night_sleep_start_hour,
                    draft.night_sleep_end_hour,
                    draft.safety_profile,
                    draft.approval_mode,
                    if draft.brave_api_key.trim().is_empty() { "not set (DuckDuckGo fallback)" } else { "configured" },
                    if draft.telegram_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    },
                    sandbox_status,
                    if docker_ok { "yes" } else { "no" },
                    memory_path.display()
                ) + "\n\nBuilt-in tools (8): web_search, read_file, write_file, list_directory, \nrun_shell, calendar_add_event, remind_me, draft_email"),
            )
        }
    }
}

fn command_exists(command: &str) -> bool {
    std::process::Command::new(command)
        .arg("--version")
        .output()
        .is_ok()
}

fn mask_secret(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        "•".repeat(value.chars().count())
    }
}

fn test_llm_connection_for_draft(draft: &OnboardingDraft) -> Result<String> {
    let mut config = AppConfig::default();
    draft.apply_partial(&mut config)?;
    let runtime = AgentRuntime::new(config);
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(runtime.test_model_connection())
}

fn test_telegram_connection(token: &str) -> Result<()> {
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

fn parse_hour(raw: &str) -> Result<u8> {
    let value: u8 = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("hour must be an integer between 0 and 23"))?;
    if value > 23 {
        bail!("hour must be between 0 and 23");
    }
    Ok(value)
}

fn apply_safety_profile(config: &mut AppConfig, profile: &str) {
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

fn normalize_provider(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("openrouter") {
        Ok("openrouter".to_string())
    } else if raw.eq_ignore_ascii_case("ollama") {
        Ok("ollama".to_string())
    } else {
        bail!("provider must be one of: ollama, openrouter");
    }
}

fn normalize_thinking_level(raw: &str) -> Result<String> {
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

fn normalize_safety_profile(raw: &str) -> Result<String> {
    if raw.eq_ignore_ascii_case("strict") {
        Ok("strict".to_string())
    } else if raw.eq_ignore_ascii_case("power-user") || raw.eq_ignore_ascii_case("poweruser") {
        Ok("power-user".to_string())
    } else {
        bail!("safety profile must be one of: strict, power-user");
    }
}

fn normalize_approval_mode(raw: &str) -> Result<String> {
    match raw.trim().to_lowercase().as_str() {
        "safer" => Ok("safer".to_string()),
        "balanced" => Ok("balanced".to_string()),
        "autonomous" => Ok("autonomous".to_string()),
        _ => bail!("approval mode must be one of: safer, balanced, autonomous"),
    }
}

fn apply_approval_mode(config: &mut AppConfig, approval_mode: &str) {
    config.tools.approval_mode = match approval_mode.to_lowercase().as_str() {
        "safer" => ApprovalMode::Safer,
        "autonomous" => ApprovalMode::Autonomous,
        _ => ApprovalMode::Balanced,
    };
}

fn detect_project_root() -> Option<PathBuf> {
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

fn is_forbidden_data_dir(path: &Path) -> bool {
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
enum ResolveDataDirMode {
    AskToCreate,
    CreateIfMissing,
}

fn resolve_data_directory(raw: &str, mode: ResolveDataDirMode) -> Result<PathBuf> {
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

fn apply_memory_contract(config: &mut AppConfig, workspace: &Path) -> Result<()> {
    config.memory.backend = "eventlog".to_string();
    config.memory.auto_sleep_mode = "nightly".to_string();
    config.memory.core_rewrite_requires_approval = true;

    let memory_dir = workspace.join(".aigent").join("memory");
    fs::create_dir_all(&memory_dir)?;
    Ok(())
}

fn prompt_config_section() -> Result<ConfigSection> {
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

fn prompt(label: &str, default_value: &str) -> Result<String> {
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

fn prompt_optional(label: &str) -> Result<Option<String>> {
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

fn prompt_bool(label: &str, default_value: bool) -> Result<bool> {
    let default_label = if default_value { "yes" } else { "no" };
    let value = prompt(label, default_label)?;
    parse_bool_like(&value)
}

fn parse_bool_like(raw: &str) -> Result<bool> {
    let normalized = raw.trim().to_lowercase();
    match normalized.as_str() {
        "y" | "yes" | "true" | "1" | "enabled" | "on" => Ok(true),
        "n" | "no" | "false" | "0" | "disabled" | "off" => Ok(false),
        _ => bail!("expected yes/no"),
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

#[cfg(test)]
mod tests {
    use super::{
        normalize_provider, normalize_safety_profile, normalize_thinking_level, parse_bool_like,
        parse_hour,
    };

    #[test]
    fn normalizes_provider_values() {
        assert_eq!(normalize_provider("OLLAMA").unwrap(), "ollama");
        assert_eq!(normalize_provider("openrouter").unwrap(), "openrouter");
        assert!(normalize_provider("something-else").is_err());
    }

    #[test]
    fn normalizes_thinking_values() {
        assert_eq!(normalize_thinking_level("LOW").unwrap(), "low");
        assert_eq!(normalize_thinking_level("balanced").unwrap(), "balanced");
        assert_eq!(normalize_thinking_level("deep").unwrap(), "deep");
        assert!(normalize_thinking_level("medium").is_err());
    }

    #[test]
    fn normalizes_safety_profile_values() {
        assert_eq!(normalize_safety_profile("strict").unwrap(), "strict");
        assert_eq!(normalize_safety_profile("poweruser").unwrap(), "power-user");
        assert!(normalize_safety_profile("unsafe").is_err());
    }

    #[test]
    fn validates_sleep_hour_bounds() {
        assert_eq!(parse_hour("0").unwrap(), 0);
        assert_eq!(parse_hour("23").unwrap(), 23);
        assert!(parse_hour("24").is_err());
        assert!(parse_hour("-1").is_err());
    }

    #[test]
    fn parses_boolean_prompt_values() {
        assert!(parse_bool_like("yes").unwrap());
        assert!(parse_bool_like("enabled").unwrap());
        assert!(!parse_bool_like("no").unwrap());
        assert!(!parse_bool_like("off").unwrap());
        assert!(parse_bool_like("maybe").is_err());
    }
}
