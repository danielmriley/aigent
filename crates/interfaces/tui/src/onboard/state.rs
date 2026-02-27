//! Setup state: types, enums, and navigation logic.


use anyhow::{Result, bail};
use aigent_config::{AppConfig, ApprovalMode};

#[derive(Debug, Clone, Default)]
pub struct AvailableModels {
    pub ollama: Vec<String>,
    pub openrouter: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SetupMode {
    Onboarding,
    Configuration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConfigSection {
    Identity,
    Llm,
    Telegram,
    DataMemory,
    Safety,
    FullWizard,
}

impl ConfigSection {
    pub(super) fn label(self) -> &'static str {
        match self {
            ConfigSection::Identity => "Identity",
            ConfigSection::Llm => "LLM settings",
            ConfigSection::Telegram => "Telegram settings",
            ConfigSection::DataMemory => "Data directory & memory",
            ConfigSection::Safety => "Safety profile",
            ConfigSection::FullWizard => "Run full setup",
        }
    }

    pub(super) fn all() -> [ConfigSection; 6] {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WizardStep {
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
    pub(super) fn is_text_step(self) -> bool {
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
pub(super) struct OnboardingDraft {
    pub(super) config_section: ConfigSection,
    pub(super) user_name: String,
    pub(super) bot_name: String,
    pub(super) workspace_path: String,
    pub(super) provider: String,
    pub(super) ollama_model: String,
    pub(super) openrouter_model: String,
    pub(super) openrouter_key: String,
    pub(super) thinking_level: String,
    pub(super) night_sleep_start_hour: u8,
    pub(super) night_sleep_end_hour: u8,
    pub(super) safety_profile: String,
    /// Tool approval mode: "safer" | "balanced" | "autonomous"
    pub(super) approval_mode: String,
    /// Optional Brave Search API key (masked in UI; stored to config / .env)
    pub(super) brave_api_key: String,
    pub(super) telegram_enabled: bool,
    pub(super) telegram_token: String,
    pub(super) available_models: AvailableModels,
}

impl OnboardingDraft {
    pub(super) fn from_config(config: &AppConfig, models: AvailableModels) -> Self {
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

    pub(super) fn value_for_step(&self, step: WizardStep) -> String {
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

    pub(super) fn commit_step(&mut self, step: WizardStep, input: &str) -> Result<()> {
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
                let path = super::wizard::resolve_data_directory(value, super::wizard::ResolveDataDirMode::CreateIfMissing)?;
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
                self.night_sleep_start_hour = super::wizard::parse_hour(value)?;
            }
            WizardStep::NightSleepEnd => {
                self.night_sleep_end_hour = super::wizard::parse_hour(value)?;
            }
            WizardStep::TelegramToken => {
                self.telegram_token = value.to_string();
            }
            WizardStep::ApprovalMode => {
                self.approval_mode = super::wizard::normalize_approval_mode(value)
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

    pub(super) fn choice_next(&mut self, step: WizardStep) {
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

    pub(super) fn choice_prev(&mut self, step: WizardStep) {
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

    pub(super) fn apply(&self, config: &mut AppConfig) -> Result<()> {
        config.agent.user_name = self.user_name.clone();
        config.agent.name = self.bot_name.clone();
        config.agent.workspace_path = self.workspace_path.clone();
        config.llm.provider = super::wizard::normalize_provider(&self.provider)?;
        config.llm.ollama_model = self.ollama_model.clone();
        config.llm.openrouter_model = self.openrouter_model.clone();
        config.agent.thinking_level = super::wizard::normalize_thinking_level(&self.thinking_level)?;
        config.memory.night_sleep_start_hour = self.night_sleep_start_hour;
        config.memory.night_sleep_end_hour = self.night_sleep_end_hour;
        let safety_profile = super::wizard::normalize_safety_profile(&self.safety_profile)?;
        super::wizard::apply_safety_profile(config, &safety_profile);
        super::wizard::apply_approval_mode(config, &self.approval_mode);
        if !self.brave_api_key.trim().is_empty() {
            config.tools.brave_api_key = self.brave_api_key.trim().to_string();
        }
        config.integrations.telegram_enabled = self.telegram_enabled;

        let workspace = super::wizard::resolve_data_directory(
            &config.agent.workspace_path,
            super::wizard::ResolveDataDirMode::CreateIfMissing,
        )?;
        super::wizard::apply_memory_contract(config, &workspace)?;
        config.onboarding.completed = true;

        Ok(())
    }

    pub(super) fn apply_partial(&self, config: &mut AppConfig) -> Result<()> {
        if !self.user_name.trim().is_empty() {
            config.agent.user_name = self.user_name.clone();
        }
        if !self.bot_name.trim().is_empty() {
            config.agent.name = self.bot_name.clone();
        }
        if !self.workspace_path.trim().is_empty() {
            config.agent.workspace_path = self.workspace_path.clone();
        }
        config.llm.provider = super::wizard::normalize_provider(&self.provider)?;
        config.llm.ollama_model = self.ollama_model.clone();
        config.llm.openrouter_model = self.openrouter_model.clone();
        if let Ok(level) = super::wizard::normalize_thinking_level(&self.thinking_level) {
            config.agent.thinking_level = level;
        }
        config.memory.night_sleep_start_hour = self.night_sleep_start_hour;
        config.memory.night_sleep_end_hour = self.night_sleep_end_hour;
        if let Ok(profile) = super::wizard::normalize_safety_profile(&self.safety_profile) {
            super::wizard::apply_safety_profile(config, &profile);
        }
        super::wizard::apply_approval_mode(config, &self.approval_mode);
        if !self.brave_api_key.trim().is_empty() {
            config.tools.brave_api_key = self.brave_api_key.trim().to_string();
        }
        config.integrations.telegram_enabled = self.telegram_enabled;
        Ok(())
    }
}

pub(super) fn next_step(current: WizardStep, draft: &OnboardingDraft, mode: SetupMode) -> WizardStep {
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

pub(super) fn prev_step(current: WizardStep, draft: &OnboardingDraft, mode: SetupMode) -> WizardStep {
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

pub(super) fn active_steps(mode: SetupMode, draft: &OnboardingDraft) -> Vec<WizardStep> {
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

pub(super) fn step_progress(step: WizardStep, mode: SetupMode, draft: &OnboardingDraft) -> (usize, usize) {
    let steps = active_steps(mode, draft);
    let total = steps.len();
    let index = steps
        .iter()
        .position(|candidate| *candidate == step)
        .unwrap_or(0)
        + 1;
    (index, total)
}

