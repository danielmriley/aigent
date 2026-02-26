use std::env;
use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ── Tool approval mode ────────────────────────────────────────────────────────

/// Controls how aggressively the agent asks for permission before running tools.
///
/// | Mode       | Behaviour                                                     |
/// |------------|---------------------------------------------------------------|
/// | `safer`    | Every tool invocation triggers an interactive approval prompt.|
/// | `balanced` | Read-only tools run freely; write / shell / HTTP require approval. |
/// | `autonomous` | No approval prompts (still workspace-sandboxed).           |
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    Safer,
    #[default]
    Balanced,
    Autonomous,
}

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
    /// Maximum entries per tier written into the YAML KV vault summaries.
    /// Lower values reduce vault file size; higher values expose more detail.
    pub kv_tier_limit: usize,
    /// IANA timezone name (e.g. `"America/New_York"`, `"Europe/London"`).
    /// Used to determine the local time when evaluating the nightly sleep window.
    /// Falls back to UTC when the name is unrecognised.
    pub timezone: String,
    /// Episodic entries older than this many days whose confidence is below
    /// `forget_min_confidence` are pruned during the sleep cycle.
    /// Set to `0` (the default) to disable.
    pub forget_episodic_after_days: u64,
    /// Confidence ceiling for lightweight forgetting.  Only entries *below* this
    /// threshold are eligible for pruning.  Has no effect when
    /// `forget_episodic_after_days` is `0`.
    pub forget_min_confidence: f32,
    /// How often (in minutes) the daemon runs a proactive check to see if it
    /// should send an unprompted message.  `0` (the default) disables proactive
    /// mode entirely.
    pub proactive_interval_minutes: u64,
    /// Do-not-disturb window start hour in local time.  Proactive messages are
    /// suppressed between `proactive_dnd_start_hour` and `proactive_dnd_end_hour`.
    pub proactive_dnd_start_hour: u8,
    /// Do-not-disturb window end hour in local time.
    pub proactive_dnd_end_hour: u8,
    /// Maximum number of beliefs to inject into each conversation prompt.
    /// Beliefs are sorted by composite score (confidence × recency × valence) before truncation.
    /// `0` means unlimited (not recommended for long-running agents).
    pub max_beliefs_in_prompt: usize,
    /// Minimum gap in minutes between proactive messages actually sent.
    /// Prevents rapid-fire messages when the agent becomes very active.
    /// Only checked after a message has been sent; a first message is never blocked.
    pub proactive_cooldown_minutes: u64,
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
            kv_tier_limit: 15,
            timezone: "UTC".to_string(),
            forget_episodic_after_days: 0,
            forget_min_confidence: 0.3,
            proactive_interval_minutes: 0,
            proactive_dnd_start_hour: 22,
            proactive_dnd_end_hour: 8,
            max_beliefs_in_prompt: 5,
            proactive_cooldown_minutes: 5,
        }
    }
}

// ── Tools config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    /// How aggressively the agent asks for approval before invoking tools.
    /// See [`ApprovalMode`] for semantics.
    pub approval_mode: ApprovalMode,
    /// Brave Search API key.  When non-empty the `web_search` tool uses the
    /// Brave Search REST API instead of DuckDuckGo Instant Answers.
    /// Can also be set via `BRAVE_API_KEY` env var (env takes precedence).
    pub brave_api_key: String,
    /// Automatically run `git add -A && git commit` after every successful
    /// `write_file` or `run_shell` tool call.  Requires git to be installed
    /// and the workspace to be a git repository (or `git init` to have been
    /// run during onboarding).
    pub git_auto_commit: bool,
    /// Apply platform-level process sandboxing to child processes spawned by
    /// `run_shell`.  When `true` (the default) the daemon will install a
    /// seccomp BPF filter on Linux (x86-64) and call `sandbox_init` on macOS
    /// before executing the shell command.  Has no effect on platforms where
    /// the `sandbox` Cargo feature is not supported.
    ///
    /// Set to `false` to disable sandboxing at runtime (the binary can still
    /// be compiled without the `sandbox` feature for a permanent opt-out).
    #[serde(default = "default_sandbox_enabled")]
    pub sandbox_enabled: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            approval_mode: ApprovalMode::Balanced,
            brave_api_key: String::new(),
            git_auto_commit: false,
            sandbox_enabled: true,
        }
    }
}

fn default_sandbox_enabled() -> bool {
    true
}

// ── Safety config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SafetyConfig {
    pub approval_required: bool,
    pub allow_shell: bool,
    pub allow_wasm: bool,
    /// Explicit allow-list of tool names.  Empty (the default) means all tools
    /// are allowed, subject to `allow_shell` / `allow_wasm` capability gates.
    pub tool_allowlist: Vec<String>,
    /// Explicit deny-list of tool names.  Takes precedence over `tool_allowlist`.
    pub tool_denylist: Vec<String>,
    /// Tools that bypass the interactive approval flow even when
    /// `approval_required = true`.  Defaults to the four safe built-in tools.
    pub approval_exempt_tools: Vec<String>,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            approval_required: true,
            allow_shell: false,
            allow_wasm: false,
            tool_allowlist: vec![],
            tool_denylist: vec![],
            approval_exempt_tools: vec![
                "calendar_add_event".to_string(),
                "remind_me".to_string(),
                "draft_email".to_string(),
                "web_search".to_string(),
            ],
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

/// User-interface appearance settings exposed in `[ui]` config section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    /// Named colour theme.  Recognised values: `catppuccin-mocha` (default),
    /// `tokyo-night`, `nord`.
    pub theme: String,
    /// Show the collapsible sidebar on startup.
    pub show_sidebar: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "catppuccin-mocha".to_string(),
            show_sidebar: true,
        }
    }
}

/// Git integration (`gait`) security and trust configuration.
///
/// Controls which repositories the agent can write to and whether read-only
/// git operations are allowed against arbitrary system paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GitConfig {
    /// Repository URLs or local paths considered "trusted" for identity.
    /// Used for commit authorship verification and push safety.
    pub trusted_repos: Vec<String>,
    /// Absolute paths where git write operations (commit, checkout, merge,
    /// reset, clone, etc.) are permitted.  `agent.workspace_path` and the
    /// detected Aigent source directory are **always** included at runtime
    /// even if not listed here.
    pub trusted_write_paths: Vec<String>,
    /// When `true` (the default), read-only git operations (`status`, `log`,
    /// `diff`, `blame`, `ls-remote`, etc.) are allowed on any accessible
    /// repository.  Set to `false` in `safer` mode to restrict reads to
    /// `trusted_write_paths` only.
    pub allow_system_read: bool,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            trusted_repos: vec!["https://github.com/danielmriley/aigent".to_string()],
            trusted_write_paths: vec![],
            allow_system_read: true,
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
    pub tools: ToolsConfig,
    pub telemetry: TelemetryConfig,
    pub onboarding: OnboardingConfig,
    pub integrations: IntegrationsConfig,
    pub daemon: DaemonConfig,
    pub ui: UiConfig,
    pub git: GitConfig,
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

        // Brave API key env override (takes precedence over config file).
        if let Ok(key) = env::var("BRAVE_API_KEY") {
            if !key.is_empty() {
                config.tools.brave_api_key = key;
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // ── Security-critical defaults ────────────────────────────────────────
    // These protect against foot-guns. Changing any of these values should
    // be a deliberate, reviewed decision.

    #[test]
    fn security_defaults_require_approval_and_deny_shell() {
        let cfg = AppConfig::default();
        assert!(
            cfg.safety.approval_required,
            "approval_required must default to true"
        );
        assert!(
            !cfg.safety.allow_shell,
            "allow_shell must default to false"
        );
        assert_eq!(
            cfg.tools.approval_mode,
            ApprovalMode::Balanced,
            "approval_mode must default to Balanced"
        );
        assert!(
            cfg.tools.sandbox_enabled,
            "sandbox_enabled must default to true"
        );
    }

    // ── Cosmetic / functional defaults ─────────────────────────────────────

    #[test]
    fn cosmetic_defaults() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.agent.name, "Aigent");
        assert_eq!(cfg.llm.provider, "ollama");
        assert_eq!(cfg.llm.ollama_model, "llama3.1:8b");
        assert_eq!(cfg.llm.ollama_base_url, "http://localhost:11434");
        assert_eq!(cfg.memory.backend, "eventlog");
        assert_eq!(cfg.memory.auto_sleep_minutes, 120);
        assert_eq!(cfg.telemetry.log_level, "info");
        assert!(!cfg.onboarding.completed);
        assert!(!cfg.integrations.telegram_enabled);
        assert_eq!(cfg.daemon.socket_path, "/tmp/aigent.sock");
        assert_eq!(cfg.ui.theme, "catppuccin-mocha");
        assert!(cfg.ui.show_sidebar);
    }

    #[test]
    fn default_git_config() {
        let git = GitConfig::default();
        assert_eq!(
            git.trusted_repos,
            vec!["https://github.com/danielmriley/aigent".to_string()]
        );
        assert!(git.trusted_write_paths.is_empty());
        assert!(git.allow_system_read);
    }

    #[test]
    fn default_safety_exempt_tools() {
        let safety = SafetyConfig::default();
        assert_eq!(safety.approval_exempt_tools.len(), 4);
        assert!(safety.approval_exempt_tools.contains(&"web_search".to_string()));
        assert!(safety.approval_exempt_tools.contains(&"remind_me".to_string()));
    }

    // ── load_from ──────────────────────────────────────────────────────────

    #[test]
    fn load_from_missing_file_returns_defaults() {
        let dir = TempDir::new().unwrap();
        let cfg = AppConfig::load_from(dir.path().join("nonexistent.toml")).unwrap();
        assert_eq!(cfg.agent.name, "Aigent");
        assert_eq!(cfg.llm.provider, "ollama");
    }

    #[test]
    fn load_from_valid_toml() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.toml");
        fs::write(
            &path,
            r#"
[agent]
name = "TestBot"
user_name = "Alice"
workspace_path = "/tmp/ws"
thinking_level = "deep"

[llm]
provider = "openrouter"
ollama_model = "custom:7b"
openrouter_model = "anthropic/claude-3.5-sonnet"

[tools]
approval_mode = "autonomous"
git_auto_commit = true

[onboarding]
completed = true
"#,
        )
        .unwrap();

        let cfg = AppConfig::load_from(&path).unwrap();
        assert_eq!(cfg.agent.name, "TestBot");
        assert_eq!(cfg.agent.user_name, "Alice");
        assert_eq!(cfg.agent.workspace_path, "/tmp/ws");
        assert_eq!(cfg.agent.thinking_level, "deep");
        assert_eq!(cfg.llm.provider, "openrouter");
        assert_eq!(cfg.llm.ollama_model, "custom:7b");
        assert_eq!(cfg.llm.openrouter_model, "anthropic/claude-3.5-sonnet");
        assert_eq!(cfg.tools.approval_mode, ApprovalMode::Autonomous);
        assert!(cfg.tools.git_auto_commit);
        assert!(cfg.onboarding.completed);
        // Unspecified sections should have defaults
        assert_eq!(cfg.memory.backend, "eventlog");
    }

    #[test]
    fn load_from_partial_toml_fills_defaults() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("partial.toml");
        fs::write(
            &path,
            r#"
[agent]
name = "Partial"
"#,
        )
        .unwrap();

        let cfg = AppConfig::load_from(&path).unwrap();
        assert_eq!(cfg.agent.name, "Partial");
        // Everything else should be default
        assert_eq!(cfg.llm.provider, "ollama");
        assert_eq!(cfg.tools.approval_mode, ApprovalMode::Balanced);
    }

    #[test]
    fn load_from_invalid_toml_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.toml");
        fs::write(&path, "this is not valid toml {{{{").unwrap();
        assert!(AppConfig::load_from(&path).is_err());
    }

    // ── save_to + roundtrip ────────────────────────────────────────────────

    #[test]
    fn save_and_reload_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sub/config.toml");

        let mut cfg = AppConfig::default();
        cfg.agent.name = "RoundTrip".to_string();
        cfg.llm.provider = "openrouter".to_string();
        cfg.tools.approval_mode = ApprovalMode::Safer;
        cfg.memory.timezone = "America/New_York".to_string();
        cfg.git.trusted_write_paths = vec!["/home/test".to_string()];

        cfg.save_to(&path).unwrap();
        assert!(path.exists());

        let loaded = AppConfig::load_from(&path).unwrap();
        assert_eq!(loaded.agent.name, "RoundTrip");
        assert_eq!(loaded.llm.provider, "openrouter");
        assert_eq!(loaded.tools.approval_mode, ApprovalMode::Safer);
        assert_eq!(loaded.memory.timezone, "America/New_York");
        assert_eq!(loaded.git.trusted_write_paths, vec!["/home/test".to_string()]);
    }

    #[test]
    fn save_creates_parent_directories() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a/b/c/config.toml");
        let cfg = AppConfig::default();
        cfg.save_to(&path).unwrap();
        assert!(path.exists());
    }

    // ── active_model ───────────────────────────────────────────────────────

    #[test]
    fn active_model_returns_ollama_by_default() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.active_model(), "llama3.1:8b");
    }

    #[test]
    fn active_model_returns_openrouter_model_when_provider_is_openrouter() {
        let mut cfg = AppConfig::default();
        cfg.llm.provider = "openrouter".to_string();
        assert_eq!(cfg.active_model(), "openai/gpt-4o-mini");
    }

    #[test]
    fn active_model_case_insensitive() {
        let mut cfg = AppConfig::default();
        cfg.llm.provider = "OpenRouter".to_string();
        assert_eq!(cfg.active_model(), "openai/gpt-4o-mini");
    }

    // ── needs_onboarding ───────────────────────────────────────────────────

    #[test]
    fn needs_onboarding_true_by_default() {
        let cfg = AppConfig::default();
        assert!(cfg.needs_onboarding());
    }

    #[test]
    fn needs_onboarding_false_after_completion() {
        let mut cfg = AppConfig::default();
        cfg.onboarding.completed = true;
        assert!(!cfg.needs_onboarding());
    }

    // ── ApprovalMode serde ─────────────────────────────────────────────────

    #[test]
    fn approval_mode_serde_roundtrip() {
        for (mode, label) in [
            (ApprovalMode::Safer, "\"safer\""),
            (ApprovalMode::Balanced, "\"balanced\""),
            (ApprovalMode::Autonomous, "\"autonomous\""),
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(json, label);
            let back: ApprovalMode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, mode);
        }
    }

    // ── Env var overrides ──────────────────────────────────────────────────

    #[test]
    fn env_ollama_base_url_forces_ollama_provider() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("env.toml");
        fs::write(
            &path,
            r#"
[llm]
provider = "openrouter"
"#,
        )
        .unwrap();

        // Set the env var before loading.
        // SAFETY: test is single-threaded for this env var.
        unsafe { env::set_var("OLLAMA_BASE_URL", "http://custom:11434") };
        let cfg = AppConfig::load_from(&path).unwrap();
        // Env var forces provider back to "ollama"
        assert_eq!(cfg.llm.provider, "ollama");
        unsafe { env::remove_var("OLLAMA_BASE_URL") };
    }

    #[test]
    fn env_brave_api_key_overrides_config() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("brave.toml");
        fs::write(
            &path,
            r#"
[tools]
brave_api_key = "from-file"
"#,
        )
        .unwrap();

        // SAFETY: test is single-threaded for this env var.
        unsafe { env::set_var("BRAVE_API_KEY", "from-env") };
        let cfg = AppConfig::load_from(&path).unwrap();
        assert_eq!(cfg.tools.brave_api_key, "from-env");
        unsafe { env::remove_var("BRAVE_API_KEY") };
    }

    // ── Git config through TOML ────────────────────────────────────────────

    #[test]
    fn git_config_from_toml() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("git.toml");
        fs::write(
            &path,
            r#"
[git]
trusted_repos = ["https://github.com/example/repo"]
trusted_write_paths = ["/tmp/safe"]
allow_system_read = false
"#,
        )
        .unwrap();

        let cfg = AppConfig::load_from(&path).unwrap();
        assert_eq!(cfg.git.trusted_repos, vec!["https://github.com/example/repo"]);
        assert_eq!(cfg.git.trusted_write_paths, vec!["/tmp/safe"]);
        assert!(!cfg.git.allow_system_read);
    }

    // ── Memory config edge values ──────────────────────────────────────────

    #[test]
    fn memory_config_forgetfulness_defaults() {
        let mem = MemoryConfig::default();
        assert_eq!(mem.forget_episodic_after_days, 0); // disabled
        assert!((mem.forget_min_confidence - 0.3).abs() < f32::EPSILON);
        assert_eq!(mem.max_beliefs_in_prompt, 5);
        assert_eq!(mem.proactive_cooldown_minutes, 5);
    }
}
