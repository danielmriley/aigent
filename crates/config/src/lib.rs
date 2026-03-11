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
    /// When `true`, the agent uses externalized JSON reasoning instead of
    /// letting the model think internally.  This forces short structured
    /// outputs, preventing long internal monologues and timeouts with large
    /// local models.  Default: `false`.
    #[serde(default)]
    pub external_thinking: bool,
    /// Per-step timeout in seconds for the external thinking loop.
    /// If the LLM takes longer than this, the step is retried.
    /// Only applies when `external_thinking = true`.
    #[serde(default = "default_step_timeout_seconds")]
    pub step_timeout_seconds: u64,
    /// Maximum number of JSON reasoning steps per turn.
    /// Prevents infinite loops in external thinking mode.
    #[serde(default = "default_max_steps_per_turn")]
    pub max_steps_per_turn: usize,
    /// Maximum wall-clock seconds a single tool execution may take before it
    /// is cancelled and an error observation is injected into the loop.
    /// Separate from `step_timeout_seconds` (which covers LLM inference).
    /// Default: 60.
    #[serde(default = "default_tool_timeout_secs")]
    pub tool_timeout_secs: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Aigent".to_string(),
            user_name: String::new(),
            workspace_path: ".".to_string(),
            thinking_level: "balanced".to_string(),
            external_thinking: false,
            step_timeout_seconds: default_step_timeout_seconds(),
            max_steps_per_turn: default_max_steps_per_turn(),
            tool_timeout_secs: default_tool_timeout_secs(),
        }
    }
}

fn default_step_timeout_seconds() -> u64 {
    120
}

fn default_max_steps_per_turn() -> usize {
    10
}

fn default_tool_timeout_secs() -> u64 {
    60
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
    /// How many concurrent inference slots Ollama has available.
    /// Set this to match `OLLAMA_NUM_PARALLEL` in your Ollama service
    /// configuration.  When `None`, the daemon falls back to reading the
    /// `OLLAMA_NUM_PARALLEL` environment variable from its own process.
    /// Sub-agent parallel execution requires ≥ 3 (recommend 4–5).
    pub ollama_num_parallel: Option<u32>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            ollama_model: "llama3.1:8b".to_string(),
            openrouter_model: "openai/gpt-4o-mini".to_string(),
            ollama_local_first: true,
            ollama_base_url: "http://localhost:11434".to_string(),
            ollama_num_parallel: None,
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
    /// Maximum number of entries **per bucket** (USER, MY_BELIEFS, OUR_DYNAMIC)
    /// in the relational matrix injected into the prompt.
    /// Items are deduplicated by content and sorted most-recent-first before
    /// the cap is applied.  `0` means unlimited (dangerous for production).
    /// Default: `20`.
    pub max_relational_in_prompt: usize,
    /// Hard ceiling on the total system prompt size in characters.
    /// When the assembled prompt exceeds this limit, the builder truncates the
    /// longest expandable section (relational → conversation → memory context)
    /// until the prompt fits.  `0` disables the guard.
    /// Default: `48000` (~12K tokens — leaves room for the model's response).
    pub max_prompt_chars: usize,
    /// Minimum gap in minutes between proactive messages actually sent.
    /// Prevents rapid-fire messages when the agent becomes very active.
    /// Only checked after a message has been sent; a first message is never blocked.
    pub proactive_cooldown_minutes: u64,
    /// Run an inline reflection pass (extra LLM call) after every TOOLS turn
    /// to extract beliefs and insights.  Disable to reduce LLM token usage on
    /// resource-constrained hardware.  Default: `true`.
    #[serde(default = "default_reflection_enabled")]
    pub reflection_enabled: bool,
    /// Persist each reasoning step's `thought` field to `MemoryTier::Reflective`
    /// after every agent turn.  Produces a searchable trace of the agent's
    /// chain-of-thought reasoning across conversations.
    /// Default: `false` (off by default to save memory writes on every turn).
    #[serde(default)]
    pub store_reasoning_traces: bool,
    /// Maximum number of Active entries allowed in the working memory store.
    /// When the count exceeds 90% of this limit the confidence sleep cycle
    /// triggers immediately (capacity pressure trigger) instead of waiting for
    /// the nightly quiet window.  Default: 2000.
    #[serde(default = "default_sleep_capacity_limit")]
    pub sleep_capacity_limit: usize,
    /// Learning rate parameters: initial confidence, confirmation/contradiction
    /// deltas, stale decay rates, and tier rendering thresholds.
    #[serde(default)]
    pub learning: LearningConfig,
    /// Sleep-cycle tuning: hot window duration and opinion synthesis gate.
    #[serde(default)]
    pub sleep: MemorySleepConfig,
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
            max_relational_in_prompt: 20,
            max_prompt_chars: 48_000,
            proactive_cooldown_minutes: 5,
            reflection_enabled: default_reflection_enabled(),
            store_reasoning_traces: false,
            sleep_capacity_limit: default_sleep_capacity_limit(),
            learning: LearningConfig::default(),
            sleep: MemorySleepConfig::default(),
        }
    }
}

fn default_reflection_enabled() -> bool {
    true
}

fn default_sleep_capacity_limit() -> usize {
    2000
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
    /// Tavily Search API key.  When set, enables the Tavily search provider
    /// which returns AI-optimized search results with extracted content.
    /// Can also be set via `TAVILY_API_KEY` env var (env takes precedence).
    #[serde(default)]
    pub tavily_api_key: String,
    /// SearXNG instance base URL (e.g. "http://localhost:8080").
    /// When set, enables SearXNG as a self-hosted search provider.
    /// Can also be set via `SEARXNG_BASE_URL` env var (env takes precedence).
    #[serde(default)]
    pub searxng_base_url: String,
    /// Serper (Google Search) API key.
    /// Can also be set via `SERPER_API_KEY` env var (env takes precedence).
    #[serde(default)]
    pub serper_api_key: String,
    /// Exa.ai API key for semantic/neural search.
    /// Can also be set via `EXA_API_KEY` env var (env takes precedence).
    #[serde(default)]
    pub exa_api_key: String,
    /// Preferred search provider ordering.  The agent tries providers in order
    /// and uses the first one that has valid credentials configured.
    /// Options: "brave", "tavily", "serper", "exa", "searxng", "duckduckgo".
    /// Default: ["brave", "tavily", "serper", "exa", "searxng", "duckduckgo"]
    #[serde(default = "default_search_providers")]
    pub search_providers: Vec<String>,
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
    /// Maximum number of tool-call rounds before forcing a text response.
    /// Prevents infinite tool-calling loops.  Default: 5.
    #[serde(default = "default_max_tool_rounds")]
    pub max_tool_rounds: usize,
    /// Maximum byte length of a shell command string passed to `run_shell`.
    /// Commands exceeding this limit are rejected before execution.
    /// Default: 8192.
    #[serde(default = "default_max_shell_command_bytes")]
    pub max_shell_command_bytes: usize,
    /// Maximum byte length of combined stdout+stderr captured from `run_shell`.
    /// Output beyond this limit is truncated.  Default: 32768.
    #[serde(default = "default_max_shell_output_bytes")]
    pub max_shell_output_bytes: usize,
    /// Default maximum bytes to read when `read_file` is called without an
    /// explicit `max_bytes` argument.  Default: 65536.
    #[serde(default = "default_max_file_read_bytes")]
    pub max_file_read_bytes: usize,
    /// Agent-written WASM modules subsystem configuration.
    #[serde(default, alias = "skills")]
    pub modules: ModulesConfig,
}

/// Configuration for the agent-written WASM modules subsystem.
///
/// Modules are WASM tools that the agent writes for itself and hot-loads
/// from a dedicated directory.  Distinct from built-in tools (compiled into
/// the daemon) and from thinker-driven prompt workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModulesConfig {
    /// Master switch — when `false` the modules directory is never scanned.
    pub enabled: bool,
    /// Path to the modules directory, relative to the workspace root.
    #[serde(default = "default_modules_dir", alias = "skills_dir")]
    pub modules_dir: String,
    /// Automatically reload modules when `ReloadConfig` is triggered.
    pub auto_reload: bool,
    /// Hard cap on the number of dynamic modules that may be loaded.
    /// Acts as a safety guard.  `0` means unlimited.
    #[serde(alias = "max_skills")]
    pub max_modules: usize,
}

impl Default for ModulesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            modules_dir: default_modules_dir(),
            auto_reload: true,
            max_modules: 64,
        }
    }
}

fn default_modules_dir() -> String {
    "extensions/modules".to_string()
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            approval_mode: ApprovalMode::Balanced,
            brave_api_key: String::new(),
            tavily_api_key: String::new(),
            searxng_base_url: String::new(),
            serper_api_key: String::new(),
            exa_api_key: String::new(),
            search_providers: default_search_providers(),
            git_auto_commit: false,
            sandbox_enabled: true,
            max_tool_rounds: 5,
            max_shell_command_bytes: default_max_shell_command_bytes(),
            max_shell_output_bytes: default_max_shell_output_bytes(),
            max_file_read_bytes: default_max_file_read_bytes(),
            modules: ModulesConfig::default(),
        }
    }
}

fn default_search_providers() -> Vec<String> {
    vec![
        "brave".to_string(),
        "tavily".to_string(),
        "serper".to_string(),
        "exa".to_string(),
        "searxng".to_string(),
        "duckduckgo".to_string(),
    ]
}

fn default_sandbox_enabled() -> bool {
    true
}

fn default_max_tool_rounds() -> usize {
    5
}

fn default_max_shell_command_bytes() -> usize {
    8192
}

fn default_max_shell_output_bytes() -> usize {
    32768
}

fn default_max_file_read_bytes() -> usize {
    65536
}

// ── Safety config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SafetyConfig {
    pub approval_required: bool,
    pub allow_shell: bool,
    pub allow_wasm: bool,
    /// Explicit allow-list of tool names (supports `@group` syntax, e.g.
    /// `@filesystem`).  Empty (the default) means all tools are allowed,
    /// subject to `allow_shell` / `allow_wasm` capability gates.
    pub tool_allowlist: Vec<String>,
    /// Explicit deny-list of tool names (supports `@group` syntax).
    /// Takes precedence over `tool_allowlist`.
    pub tool_denylist: Vec<String>,
    /// Tools that bypass the interactive approval flow even when
    /// `approval_required = true`.  Defaults to the four safe built-in tools.
    pub approval_exempt_tools: Vec<String>,
    /// Maximum security level allowed for any tool execution.
    /// Tools whose metadata declares a higher level are blocked outright.
    /// Values: "low", "medium", "high".  Default: "high" (no restriction).
    #[serde(default = "default_max_security_level")]
    pub max_security_level: String,
    /// Per-tool policy overrides keyed by tool name.
    /// Maximum number of times any single tool may be called per session.
    /// `0` means unlimited.  Default: 25.
    #[serde(default)]
    pub max_calls_per_tool: Option<usize>,
    #[serde(default)]
    pub tool_overrides: std::collections::HashMap<String, ToolPolicyOverride>,
}

/// Per-tool policy overrides — allows fine-grained control over individual
/// tools without changing the global safety settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolPolicyOverride {
    /// Override the global approval mode for this tool.
    /// Values: "safer", "balanced", "autonomous".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_mode: Option<ApprovalMode>,
    /// When `true`, this tool is completely blocked regardless of other settings.
    #[serde(default)]
    pub denied: bool,
    /// Override the maximum security level for this specific tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_security_level: Option<String>,
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
                "browse_page".to_string(),
            ],
            max_security_level: "high".to_string(),
            max_calls_per_tool: None,
            tool_overrides: std::collections::HashMap::new(),
        }
    }
}

fn default_max_security_level() -> String {
    "high".to_string()
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

// ── Inference config (hybrid routing + Candle local models) ───────────────────

/// Configuration for inference routing and local model execution.
///
/// Controls the hybrid router that decides which backend handles a request:
/// Candle (local, fast, private) vs Ollama/OpenRouter (powerful, cloud).
///
/// Maps to the `[inference]` section of `default.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// Enable the Candle local inference backend.
    /// Requires the `candle` feature flag to be compiled in.
    pub candle_enabled: bool,
    /// HuggingFace model repo for GGUF download
    /// (e.g. `"Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"`).
    pub candle_model_repo: String,
    /// Specific GGUF filename inside the repo
    /// (e.g. `"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"`).
    pub candle_model_file: String,
    /// Path to a local GGUF file (overrides HuggingFace download when set).
    pub candle_model_path: String,
    /// Directory for cached model downloads.
    pub candle_local_models_dir: String,
    /// Maximum sequence length for Candle inference.
    pub candle_max_seq_len: usize,
    /// Temperature for sampling (0.0 = greedy).
    pub candle_temperature: f64,
    /// Top-p nucleus sampling threshold.
    pub candle_top_p: f64,
    /// Repeat penalty.
    pub candle_repeat_penalty: f32,
    /// Device selection: `"cpu"`, `"cuda"`, `"metal"`.
    pub candle_device: String,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            candle_enabled: false,
            candle_model_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF".to_string(),
            candle_model_file: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string(),
            candle_model_path: String::new(),
            candle_local_models_dir: "~/.cache/aigent/models".to_string(),
            candle_max_seq_len: 4096,
            candle_temperature: 0.7,
            candle_top_p: 0.9,
            candle_repeat_penalty: 1.1,
            candle_device: "cpu".to_string(),
        }
    }
}

// ── Learning config ───────────────────────────────────────────────────────────

/// Learning rate parameters for the confidence-based memory system.
///
/// All deltas listed under "Contradiction deltas" are stored as **positive**
/// values and applied as negative signals by the caller.
///
/// Maps to `[memory.learning]` in `config/default.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LearningConfig {
    // ── Initial confidence by belief kind × source ─────────────────────────
    pub empirical_tool_success: f32,
    pub empirical_tool_failure_transient: f32,
    pub empirical_tool_failure_arch: f32,
    pub empirical_human: f32,
    pub procedural_tool_success: f32,
    pub procedural_tool_failure_transient: f32,
    pub procedural_tool_failure_arch: f32,
    pub procedural_human: f32,
    pub self_model_tool_success: f32,
    pub self_model_tool_failure_transient: f32,
    pub self_model_tool_failure_arch: f32,
    pub self_model_human: f32,
    pub opinion_human: f32,
    /// Initial confidence for opinions proposed by the sleep pipeline (Pass 5).
    pub opinion_sleep_synthesis: f32,
    // ── Confirmation deltas ────────────────────────────────────────────────
    /// Tool confirmation delta for Empirical beliefs.
    pub confirm_tool_empirical: f32,
    /// Tool confirmation delta for Procedural beliefs.
    pub confirm_tool_procedural: f32,
    /// Tool confirmation delta for SelfModel beliefs.
    pub confirm_tool_self_model: f32,
    /// Confirmation from the same source type as the original belief.
    pub confirm_same_source: f32,
    /// Confirmation from a different source type (cross-source corroboration).
    pub confirm_cross_source: f32,
    /// Explicit user confirmation (e.g. "yes, that's right").
    pub confirm_user_explicit: f32,
    /// Confirmation via nightly sleep synthesis.
    pub confirm_sleep_synthesis: f32,
    // ── Contradiction deltas (positive values; applied as negatives) ───────
    /// Tool success that directly contradicts a prior failure belief (Pass 3).
    pub contradict_tool_success_vs_failure: f32,
    /// Explicit user correction.
    pub contradict_user_explicit: f32,
    /// Tool returns a fact that contradicts an existing belief.
    pub contradict_tool_fact: f32,
    /// Tool failure — transient error (e.g. network timeout).
    pub contradict_tool_failure_transient: f32,
    /// Tool failure — configuration error (e.g. missing API key).
    pub contradict_tool_failure_config: f32,
    /// Tool failure — architectural limit (e.g. model has no access to tool).
    pub contradict_tool_failure_arch: f32,
    // ── Stale decay per nightly sleep cycle (negative; applied as-is) ─────
    pub decay_empirical: f32,
    pub decay_procedural: f32,
    pub decay_self_model: f32,
    pub decay_opinion: f32,
    // ── Confidence tier rendering thresholds ───────────────────────────────
    /// "You know that …" threshold.
    pub tier_know_threshold: f32,
    /// "You have come to believe …" threshold.
    pub tier_believe_threshold: f32,
    /// "From experience, you suspect …" threshold.
    pub tier_suspect_threshold: f32,
    /// "You once observed …" (below this: excluded from prompt).
    pub tier_observed_threshold: f32,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            empirical_tool_success: 0.45,
            empirical_tool_failure_transient: 0.20,
            empirical_tool_failure_arch: 0.30,
            empirical_human: 0.55,
            procedural_tool_success: 0.40,
            procedural_tool_failure_transient: 0.15,
            procedural_tool_failure_arch: 0.25,
            procedural_human: 0.55,
            self_model_tool_success: 0.50,
            self_model_tool_failure_transient: 0.10,
            self_model_tool_failure_arch: 0.30,
            self_model_human: 0.50,
            opinion_human: 0.45,
            opinion_sleep_synthesis: 0.25,
            confirm_tool_empirical: 0.08,
            confirm_tool_procedural: 0.10,
            confirm_tool_self_model: 0.12,
            confirm_same_source: 0.08,
            confirm_cross_source: 0.15,
            confirm_user_explicit: 0.20,
            confirm_sleep_synthesis: 0.05,
            contradict_tool_success_vs_failure: 0.35,
            contradict_user_explicit: 0.35,
            contradict_tool_fact: 0.15,
            contradict_tool_failure_transient: 0.05,
            contradict_tool_failure_config: 0.08,
            contradict_tool_failure_arch: 0.20,
            decay_empirical: 0.03,
            decay_procedural: 0.01,
            decay_self_model: 0.02,
            decay_opinion: 0.005,
            tier_know_threshold: 0.70,
            tier_believe_threshold: 0.50,
            tier_suspect_threshold: 0.35,
            tier_observed_threshold: 0.10,
        }
    }
}

/// Sleep-cycle tuning parameters that do not already appear as flat fields on
/// [`MemoryConfig`].
///
/// Maps to `[memory.sleep]` in `config/default.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemorySleepConfig {
    /// How long (in hours) episodic memories stay in the "hot window" before
    /// they become eligible for Pass 2 consolidation.  Default: 24.
    pub hot_window_hours: u32,
    /// Minimum distinct episodic observations required for Pass 5 to propose a
    /// new Opinion entry.  The hard lower bound is 5 and this value may not be
    /// set below it at runtime.  Default: 5.
    pub opinion_min_observations: usize,
    /// Nightly quiet-window start in "HH:MM" 24-hour local time.
    /// The confidence sleep cycle runs during this window when ≥ 22 h have
    /// elapsed since the last run.  Default: "22:00".
    pub nightly_window_start: String,
    /// Nightly quiet-window end in "HH:MM" 24-hour local time.  Default: "06:00".
    pub nightly_window_end: String,
    /// Maximum number of Active entries allowed in the working memory store.
    /// When the count exceeds 90 % of this limit the confidence sleep cycle
    /// triggers immediately.  Default: 2000.
    pub capacity_limit: usize,
    /// Episodic memories older than this many days without consolidation
    /// transition to `NodeState::Archived`.  `0` disables archival.
    /// Default: 30.
    pub archival_age_days: u32,
    /// Hard timeout in minutes for the nightly multi-agent LLM consolidation
    /// pipeline.  With 5 specialists × N batches + synthesis, a full cycle on
    /// local Ollama hardware typically takes 25–45 minutes.  Set to 0 to
    /// disable the cap entirely (not recommended).  Default: 60.
    pub multi_agent_timeout_mins: u64,
}

impl Default for MemorySleepConfig {
    fn default() -> Self {
        Self {
            hot_window_hours: 24,
            opinion_min_observations: 5,
            nightly_window_start: "22:00".to_string(),
            nightly_window_end: "06:00".to_string(),
            capacity_limit: 2000,
            archival_age_days: 30,
            multi_agent_timeout_mins: 60,
        }
    }
}

// ── Subagents ─────────────────────────────────────────────────────────────────
/// Configuration for the parallel subagent reasoning system.
///
/// When enabled, complex queries are evaluated by multiple specialist agents
/// (Researcher, Planner, Critic) in parallel via `tokio::join!`, and their
/// outputs are synthesised by the main "Captain" agent before tool execution.
///
/// Requires `OLLAMA_NUM_PARALLEL >= 3` on the Ollama server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SubagentsConfig {
    /// Master switch.  When `false`, the subagent pipeline is skipped entirely.
    pub enabled: bool,

    /// Ollama model used for the three specialist subagents.
    ///
    /// Set this to a smaller, faster model (e.g. `"qwen3:8b"`) while the
    /// Captain uses a larger model (e.g. `"qwen3:35b"`).  This unlocks
    /// better capability at lower latency when `OLLAMA_NUM_PARALLEL >= 3`.
    ///
    /// When empty (the default), subagents use the same model as the Captain
    /// (`[llm].ollama_model`).
    pub ollama_model: String,

    /// OpenRouter model used for subagents.  Empty = use captain's model.
    pub openrouter_model: String,

    /// Researcher role system prompt.
    pub researcher_prompt: String,
    /// Planner role system prompt.
    pub planner_prompt: String,
    /// Critic role system prompt.
    pub critic_prompt: String,

    /// Maximum tool-call rounds each specialist may take in its thinking loop.
    ///
    /// Keep this low (2–3) — specialists are pre-turn advisors, not full agents.
    /// 0 means "use default (3)".
    #[serde(default = "default_subagent_max_rounds")]
    pub max_rounds: usize,
}

fn default_subagent_max_rounds() -> usize {
    3
}

impl Default for SubagentsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ollama_model: String::new(),
            openrouter_model: String::new(),
            researcher_prompt: "You are a meticulous researcher. Examine the conversation history and memory context deeply. Identify key facts, missing information, and relevant background knowledge that could help answer the user's request. Output your findings as structured analysis.".to_string(),
            planner_prompt: "You are a strategic planner. Given the conversation and context, devise a clear step-by-step plan for how to best respond to or accomplish the user's request. Consider which tools might be needed and in what order. Output your plan as structured analysis.".to_string(),
            critic_prompt: "You are a rigorous critic. Examine the conversation for potential pitfalls: hallucination risks, incorrect assumptions, edge cases, security concerns, and logical gaps. Challenge the obvious approach. Output your critique as structured analysis.".to_string(),
            max_rounds: default_subagent_max_rounds(),
        }
    }
}

/// Runtime debug and observability settings exposed in `[debug]`.
///
/// All options default to sensible production values (minimal overhead).
/// Change them in `config/default.toml` or at runtime without recompiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugConfig {
    /// Master log level for the tracing subscriber.
    /// Accepted values: `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"`.
    /// This takes effect **after** config is loaded — overriding the
    /// bootstrap `RUST_LOG` if one was set.
    /// Default: `"info"`.
    pub log_level: String,

    /// Write a copy of log output to `.aigent/runtime/debug.log` (rotated
    /// per daemon restart).  Useful for post-hoc inspection without needing
    /// `RUST_LOG` or a terminal attached.
    /// Default: `true`.
    pub log_to_file: bool,

    /// When `true`, emit per-turn timing breakdowns as `info!` logs:
    /// prompt build time, router classify time, LLM response time,
    /// memory write time, and total wall-clock time.
    /// Default: `true`.
    pub timing: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            log_to_file: true,
            timing: true,
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
    pub inference: InferenceConfig,
    pub subagents: SubagentsConfig,
    pub debug: DebugConfig,
}

/// Default config file path, overridable via `AIGENT_CONFIG` env var.
pub const DEFAULT_CONFIG_PATH: &str = "config/default.toml";

impl AppConfig {
    /// Resolve the config file path: honours `AIGENT_CONFIG` env var, falls
    /// back to [`DEFAULT_CONFIG_PATH`].
    pub fn config_path() -> String {
        std::env::var("AIGENT_CONFIG")
            .unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string())
    }

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

        // ── Secret env-var overrides (take precedence over config file) ──
        //
        // API keys and credentials should be set via environment variables
        // rather than stored in the config file.  Each field listed below
        // can be overridden by the corresponding env var.
        type SecretOverride = (&'static str, fn(&mut AppConfig, String));
        let secret_overrides: &[SecretOverride] = &[
            ("BRAVE_API_KEY",   |c, v| c.tools.brave_api_key   = v),
            ("TAVILY_API_KEY",  |c, v| c.tools.tavily_api_key  = v),
            ("SERPER_API_KEY",  |c, v| c.tools.serper_api_key  = v),
            ("EXA_API_KEY",     |c, v| c.tools.exa_api_key     = v),
            ("SEARXNG_BASE_URL",|c, v| c.tools.searxng_base_url = v),
        ];
        for (var, setter) in secret_overrides {
            if let Ok(val) = env::var(var) {
                if !val.is_empty() {
                    setter(&mut config, val);
                }
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
        if self.llm.provider.eq_ignore_ascii_case("candle") {
            &self.inference.candle_model_repo
        } else if self.llm.provider.eq_ignore_ascii_case("openrouter") {
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
        assert_eq!(safety.approval_exempt_tools.len(), 5);
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
        assert_eq!(mem.max_relational_in_prompt, 20);
        assert_eq!(mem.proactive_cooldown_minutes, 5);
    }

    // ── InferenceConfig defaults ───────────────────────────────────────────

    #[test]
    fn inference_config_defaults() {
        let inf = InferenceConfig::default();
        assert!(!inf.candle_enabled);
        assert_eq!(inf.candle_model_repo, "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF");
        assert_eq!(inf.candle_model_file, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        assert!(inf.candle_model_path.is_empty());
        assert_eq!(inf.candle_max_seq_len, 4096);
        assert!((inf.candle_temperature - 0.7).abs() < f64::EPSILON);
        assert!((inf.candle_top_p - 0.9).abs() < f64::EPSILON);
        assert!((inf.candle_repeat_penalty - 1.1_f32).abs() < f32::EPSILON);
        assert_eq!(inf.candle_device, "cpu");
    }

    #[test]
    fn active_model_returns_candle_repo_when_candle_provider() {
        let mut cfg = AppConfig::default();
        cfg.llm.provider = "candle".to_string();
        cfg.inference.candle_model_repo = "test/model-gguf".to_string();
        assert_eq!(cfg.active_model(), "test/model-gguf");
    }

    #[test]
    fn active_model_returns_ollama_for_default_provider() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.active_model(), &cfg.llm.ollama_model);
    }

    // ── Phase 7: LearningConfig / MemorySleepConfig defaults ─────────────

    #[test]
    fn learning_config_default_values_match_spec() {
        let lc = LearningConfig::default();
        // Initial confidence table
        assert!((lc.empirical_tool_success - 0.45).abs() < f32::EPSILON);
        assert!((lc.empirical_tool_failure_transient - 0.20).abs() < f32::EPSILON);
        assert!((lc.empirical_tool_failure_arch - 0.30).abs() < f32::EPSILON);
        assert!((lc.empirical_human - 0.55).abs() < f32::EPSILON);
        assert!((lc.procedural_tool_success - 0.40).abs() < f32::EPSILON);
        assert!((lc.procedural_tool_failure_transient - 0.15).abs() < f32::EPSILON);
        assert!((lc.procedural_tool_failure_arch - 0.25).abs() < f32::EPSILON);
        assert!((lc.procedural_human - 0.55).abs() < f32::EPSILON);
        assert!((lc.self_model_tool_success - 0.50).abs() < f32::EPSILON);
        assert!((lc.self_model_tool_failure_transient - 0.10).abs() < f32::EPSILON);
        assert!((lc.self_model_tool_failure_arch - 0.30).abs() < f32::EPSILON);
        assert!((lc.self_model_human - 0.50).abs() < f32::EPSILON);
        assert!((lc.opinion_human - 0.45).abs() < f32::EPSILON);
        assert!((lc.opinion_sleep_synthesis - 0.25).abs() < f32::EPSILON);
        // Confirmation deltas
        assert!((lc.confirm_tool_empirical - 0.08).abs() < f32::EPSILON);
        assert!((lc.confirm_tool_procedural - 0.10).abs() < f32::EPSILON);
        assert!((lc.confirm_tool_self_model - 0.12).abs() < f32::EPSILON);
        // Contradiction deltas
        assert!((lc.contradict_tool_success_vs_failure - 0.35).abs() < f32::EPSILON);
        assert!((lc.contradict_tool_failure_transient - 0.05).abs() < f32::EPSILON);
        assert!((lc.contradict_tool_failure_config - 0.08).abs() < f32::EPSILON);
        assert!((lc.contradict_tool_failure_arch - 0.20).abs() < f32::EPSILON);
        // Decay deltas
        assert!((lc.decay_empirical - 0.03).abs() < f32::EPSILON);
        assert!((lc.decay_procedural - 0.01).abs() < f32::EPSILON);
        assert!((lc.decay_self_model - 0.02).abs() < f32::EPSILON);
        assert!((lc.decay_opinion - 0.005).abs() < f32::EPSILON);
        // Tier thresholds
        assert!((lc.tier_know_threshold - 0.70).abs() < f32::EPSILON);
        assert!((lc.tier_believe_threshold - 0.50).abs() < f32::EPSILON);
        assert!((lc.tier_suspect_threshold - 0.35).abs() < f32::EPSILON);
        assert!((lc.tier_observed_threshold - 0.10).abs() < f32::EPSILON);
    }

    #[test]
    fn memory_sleep_config_defaults() {
        let sc = MemorySleepConfig::default();
        assert_eq!(sc.hot_window_hours, 24);
        assert_eq!(sc.opinion_min_observations, 5);
    }

    #[test]
    fn default_toml_round_trips_learning_config() {
        // Parse the real default.toml and ensure learning + sleep sections
        // deserialise without error and produce the expected hot_window_hours.
        let toml_str = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../config/default.toml"),
        )
        .expect("default.toml must exist");
        let cfg: AppConfig = toml::from_str(&toml_str)
            .expect("default.toml must deserialise cleanly");
        assert_eq!(cfg.memory.sleep.hot_window_hours, 24);
        assert_eq!(cfg.memory.sleep.opinion_min_observations, 5);
        assert!((cfg.memory.learning.empirical_tool_success - 0.45).abs() < f32::EPSILON);
        assert!((cfg.memory.learning.tier_know_threshold - 0.70).abs() < f32::EPSILON);
    }

}
