use std::env;
use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub name: String,
    pub user_name: String,
    pub workspace_path: String,
    pub thinking_level: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Aigent".to_string(),
            user_name: String::new(),
            workspace_path: ".".to_string(),
            thinking_level: "balanced".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub provider: String,
    pub ollama_model: String,
    pub openrouter_model: String,
    pub ollama_local_first: bool,
    /// Base URL for the Ollama API.  Overridden at runtime by the
    /// `OLLAMA_BASE_URL` environment variable when set.
    pub ollama_base_url: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            ollama_model: "llama3.1:8b".to_string(),
            openrouter_model: "openai/gpt-4o-mini".to_string(),
            ollama_local_first: true,
            ollama_base_url: "http://localhost:11434".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub backend: String,
    pub auto_sleep_minutes: u64,
    pub auto_sleep_turn_interval: usize,
    pub auto_sleep_mode: String,
    pub night_sleep_start_hour: u8,
    pub night_sleep_end_hour: u8,
    pub core_rewrite_requires_approval: bool,
    /// Number of non-anchor entries per multi-agent sleep batch.
    /// Core and UserProfile entries are always replicated into every batch.
    pub multi_agent_sleep_batch_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            backend: "eventlog".to_string(),
            auto_sleep_minutes: 120,
            auto_sleep_turn_interval: 6,
            auto_sleep_mode: "nightly".to_string(),
            night_sleep_start_hour: 22,
            night_sleep_end_hour: 6,
            core_rewrite_requires_approval: true,
            multi_agent_sleep_batch_size: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SafetyConfig {
    pub approval_required: bool,
    pub allow_shell: bool,
    pub allow_wasm: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            approval_required: true,
            allow_shell: false,
            allow_wasm: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetryConfig {
    pub log_level: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct OnboardingConfig {
    pub completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct IntegrationsConfig {
    pub telegram_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DaemonConfig {
    pub socket_path: String,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            socket_path: "/tmp/aigent.sock".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub agent: AgentConfig,
    pub llm: LlmConfig,
    pub memory: MemoryConfig,
    pub safety: SafetyConfig,
    pub telemetry: TelemetryConfig,
    pub onboarding: OnboardingConfig,
    pub integrations: IntegrationsConfig,
    pub daemon: DaemonConfig,
}

impl AppConfig {
    pub fn load_from(path: impl AsRef<Path>) -> Result<Self> {
        let mut config = Self::default();
        if let Ok(raw) = fs::read_to_string(path) {
            config = toml::from_str(&raw)?;
        }

        if let Ok(value) = env::var("OLLAMA_BASE_URL") {
            if !value.is_empty() {
                config.llm.provider = "ollama".to_string();
            }
        }

        Ok(config)
    }

    pub fn save_to(&self, path: impl AsRef<Path>) -> Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }

        let rendered = toml::to_string_pretty(self)?;
        fs::write(path, rendered)?;
        Ok(())
    }

    pub fn active_model(&self) -> &str {
        if self.llm.provider.eq_ignore_ascii_case("openrouter") {
            &self.llm.openrouter_model
        } else {
            &self.llm.ollama_model
        }
    }

    pub fn needs_onboarding(&self) -> bool {
        !self.onboarding.completed
    }
}
