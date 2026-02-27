pub mod git;
pub mod sandbox;

#[cfg(feature = "wasm")]
pub mod wasm;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use aigent_config::{ApprovalMode, ToolPolicyOverride};
use aigent_tools::{SecurityLevel, ToolMetadata, ToolOutput, ToolRegistry};

// ── Execution Policy ─────────────────────────────────────────────────────────

/// Built from `SafetyConfig` + `ToolsConfig` in aigent-config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    /// Coarse approval mode — governs the default approval behaviour.
    /// `approval_required` is retained for backward compatibility but is
    /// ignored when `approval_mode` is `Autonomous`.
    pub approval_mode: ApprovalMode,
    /// Legacy flag — kept for config backward compat.  Prefer `approval_mode`.
    pub approval_required: bool,
    pub allow_shell: bool,
    pub allow_wasm: bool,
    pub workspace_root: PathBuf,
    /// Explicit allow-list of tool names.  Empty = all tools are eligible
    /// (subject to the capability gates above).
    pub tool_allowlist: Vec<String>,
    /// Explicit deny-list of tool names.  Takes precedence over `tool_allowlist`.
    pub tool_denylist: Vec<String>,
    /// Tools that bypass interactive approval regardless of `approval_mode`.
    /// In `Balanced` mode these are automatically added to the read-only set.
    pub approval_exempt_tools: Vec<String>,
    /// When `true` the executor calls `git add -A && git commit` after every
    /// successful `write_file` or `run_shell` invocation.
    pub git_auto_commit: bool,
    /// Apply platform sandbox to shell children when `true` (default).
    /// Requires the `sandbox` Cargo feature to be compiled in, otherwise
    /// this field has no effect.
    pub sandbox_enabled: bool,
    /// Maximum security level allowed for tool execution.
    /// Tools with `metadata.security_level` above this threshold are denied.
    /// Default: `SecurityLevel::High` (permits everything).
    pub max_security_level: SecurityLevel,
    /// Per-tool policy overrides keyed by tool name.
    #[serde(default)]
    pub tool_overrides: HashMap<String, ToolPolicyOverride>,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            approval_mode: ApprovalMode::Balanced,
            approval_required: true,
            allow_shell: false,
            allow_wasm: false,
            workspace_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            tool_allowlist: vec![],
            tool_denylist: vec![],
            approval_exempt_tools: vec![
                "calendar_add_event".to_string(),
                "remind_me".to_string(),
                "draft_email".to_string(),
                "web_search".to_string(),
            ],
            git_auto_commit: false,
            sandbox_enabled: true,
            max_security_level: SecurityLevel::High,
            tool_overrides: HashMap::new(),
        }
    }
}

// ── Approval Flow ────────────────────────────────────────────────────────────

/// A request sent to the user for approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub tool_name: String,
    pub args: HashMap<String, String>,
    pub risk_summary: String,
}

/// The user's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approve,
    Deny,
}

/// Channel-based approval gate.
/// The executor sends an `ApprovalRequest`, and the UI/Telegram side responds
/// with an `ApprovalDecision` via the oneshot.
pub type ApprovalSender = mpsc::Sender<(ApprovalRequest, oneshot::Sender<ApprovalDecision>)>;
pub type ApprovalReceiver = mpsc::Receiver<(ApprovalRequest, oneshot::Sender<ApprovalDecision>)>;

/// Create a new approval channel pair.
pub fn approval_channel() -> (ApprovalSender, ApprovalReceiver) {
    mpsc::channel(16)
}

// ── Tool Executor ────────────────────────────────────────────────────────────

/// Orchestrates tool invocation with safety checks and approval flow.
pub struct ToolExecutor {
    policy: ExecutionPolicy,
    approval_tx: Option<ApprovalSender>,
}

impl ToolExecutor {
    pub fn new(policy: ExecutionPolicy) -> Self {
        Self {
            policy,
            approval_tx: None,
        }
    }

    /// Attach an approval channel for interactive approval flow.
    pub fn with_approval(mut self, tx: ApprovalSender) -> Self {
        self.approval_tx = Some(tx);
        self
    }

    /// Execute a tool by name from the registry, applying safety policy.
    pub async fn execute(
        &self,
        registry: &ToolRegistry,
        tool_name: &str,
        args: &HashMap<String, String>,
    ) -> Result<ToolOutput> {
        // 1. Check if the tool exists
        let tool = registry
            .get(tool_name)
            .ok_or_else(|| anyhow::anyhow!("unknown tool: {tool_name}"))?;

        let metadata = tool.spec().metadata;

        // 2. Enforce capability gates (deny-list / allow-list / shell guard / security level)
        self.check_capability(registry, tool_name, &metadata)?;

        // 3. Approval flow — governed by ApprovalMode + metadata
        if self.requires_approval(tool_name, &metadata) {
            let approved = self.request_approval(tool_name, args, &metadata).await?;
            if !approved {
                info!(tool = tool_name, "tool execution denied by user");
                return Ok(ToolOutput {
                    success: false,
                    output: format!("execution of '{tool_name}' denied by user"),
                });
            }
        }

        // 4. Run the tool
        info!(tool = tool_name, "executing tool");

        // For run_shell with the `sandbox` feature active we spawn the child
        // with a pre-exec sandbox hook instead of delegating to the tool impl.
        // Respects `policy.sandbox_enabled` so operators can opt out at runtime.
        #[cfg(all(feature = "sandbox", unix))]
        if tool_name == "run_shell" && self.policy.sandbox_enabled {
            let result = self.run_shell_sandboxed(args).await?;
            if result.success && self.policy.git_auto_commit && !metadata.read_only {
                let detail = args
                    .get("command")
                    .map(|s| s.as_str())
                    .unwrap_or("(unknown)");
                if let Err(err) = git::git_auto_commit(
                    &self.policy.workspace_root,
                    tool_name,
                    detail,
                )
                .await
                {
                    warn!(?err, tool = tool_name, "git auto-commit failed (non-fatal)");
                }
            }
            return Ok(result);
        }

        let result = tool.run(args).await?;

        // 5. Git auto-commit after successful write operations (metadata-driven)
        if result.success && self.policy.git_auto_commit && !metadata.read_only {
            let detail = args
                .get("path")
                .or_else(|| args.get("command"))
                .map(|s| s.as_str())
                .unwrap_or("(unknown)");
            if let Err(err) = git::git_auto_commit(
                &self.policy.workspace_root,
                tool_name,
                detail,
            )
            .await
            {
                warn!(?err, tool = tool_name, "git auto-commit failed (non-fatal)");
            }
        }

        Ok(result)
    }

    /// Returns `true` when this tool invocation needs interactive approval
    /// based on the configured `ApprovalMode` and tool metadata.
    ///
    /// | Mode         | Needs approval                                          |
    /// |--------------|---------------------------------------------------------|
    /// | `Autonomous` | Never                                                   |
    /// | `Balanced`   | Non-read-only tools or `SecurityLevel::High`            |
    /// | `Safer`      | Every tool (unless explicitly exempt)                   |
    fn requires_approval(&self, tool_name: &str, metadata: &ToolMetadata) -> bool {
        // Explicit exempt list always short-circuits.
        if self.policy
            .approval_exempt_tools
            .contains(&tool_name.to_string())
        {
            return false;
        }

        // Per-tool override can override the global approval mode.
        let effective_mode = self.policy.tool_overrides.get(tool_name)
            .and_then(|ovr| ovr.approval_mode.as_ref())
            .unwrap_or(&self.policy.approval_mode);

        match effective_mode {
            ApprovalMode::Autonomous => false,
            ApprovalMode::Balanced => {
                // Read-only + non-High tools skip approval.
                // Anything that writes or has High security level requires approval.
                !metadata.read_only || metadata.security_level == SecurityLevel::High
            }
            ApprovalMode::Safer => true,
        }
    }

    fn check_capability(
        &self,
        registry: &ToolRegistry,
        tool_name: &str,
        metadata: &ToolMetadata,
    ) -> Result<()> {
        // Per-tool override: if explicitly denied, short-circuit.
        if let Some(ovr) = self.policy.tool_overrides.get(tool_name) {
            if ovr.denied {
                bail!("tool '{}' is denied by per-tool policy override", tool_name);
            }
        }

        // Shell is only available when explicitly enabled in config.
        if tool_name == "run_shell" && !self.policy.allow_shell {
            bail!("shell execution is disabled by safety policy (set allow_shell = true)");
        }

        // Security level ceiling — check per-tool override first, then global.
        let effective_max = self.policy.tool_overrides.get(tool_name)
            .and_then(|ovr| ovr.max_security_level.as_deref())
            .map(|s| parse_security_level_str(s))
            .unwrap_or(self.policy.max_security_level);
        if !security_level_permitted(metadata.security_level, effective_max) {
            bail!(
                "tool '{}' has security level {:?} which exceeds the policy maximum {:?}",
                tool_name,
                metadata.security_level,
                effective_max,
            );
        }

        // Per-tool deny-list (supports `@group` expansion, takes precedence over allow-list).
        let expanded_deny = registry.expand_groups(&self.policy.tool_denylist);
        if expanded_deny.contains(&tool_name.to_string()) {
            bail!("tool '{}' is blocked by policy (tool_denylist)", tool_name);
        }
        // Per-tool allow-list (supports `@group` expansion, empty = all permitted).
        if !self.policy.tool_allowlist.is_empty() {
            let expanded_allow = registry.expand_groups(&self.policy.tool_allowlist);
            if !expanded_allow.contains(&tool_name.to_string()) {
                bail!("tool '{}' is not in the tool_allowlist", tool_name);
            }
        }
        Ok(())
    }

    /// Run `run_shell` with a sandbox pre-exec hook on supported platforms.
    ///
    /// Mirrors the logic in `RunShellTool::run()` but inserts
    /// `sandbox::apply_to_child` into the child process before the shell
    /// binary executes.  Only compiled when the `sandbox` feature is active.
    #[cfg(all(feature = "sandbox", unix))]
    async fn run_shell_sandboxed(
        &self,
        args: &HashMap<String, String>,
    ) -> Result<aigent_tools::ToolOutput> {
        use std::os::unix::process::CommandExt as _;

        let command = args
            .get("command")
            .ok_or_else(|| anyhow::anyhow!("missing required param: command"))?
            .clone();
        let timeout_secs: u64 = args
            .get("timeout_secs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let workspace_root = self.policy.workspace_root.clone();
        let workspace_str = workspace_root.display().to_string();

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(&command)
            .current_dir(&workspace_root);

        // SAFETY: `apply_to_child` is designed to be called between fork and
        // exec and only makes async-signal-safe syscalls (prctl, seccomp,
        // sandbox_init).
        unsafe {
            let ws = workspace_str.clone();
            cmd.as_std_mut().pre_exec(move || {
                // SAFETY: called between fork and exec; only async-signal-safe calls.
                sandbox::apply_to_child(&ws)
            });
        }

        let output_result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            cmd.output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("command timed out after {}s", timeout_secs))??;

        let stdout = String::from_utf8_lossy(&output_result.stdout);
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else {
            format!("{stdout}\n[stderr] {stderr}")
        };
        let max_output = 32768;
        let result = if combined.len() > max_output {
            let mut end = max_output;
            while end > 0 && !combined.is_char_boundary(end) { end -= 1; }
            format!("{}…[truncated]", &combined[..end])
        } else {
            combined
        };

        Ok(aigent_tools::ToolOutput {
            success: output_result.status.success(),
            output: result,
        })
    }

    async fn request_approval(
        &self,
        tool_name: &str,
        args: &HashMap<String, String>,
        metadata: &ToolMetadata,
    ) -> Result<bool> {
        // Approval-exempt tools are auto-approved regardless of policy.
        if self.policy.approval_exempt_tools.contains(&tool_name.to_string()) {
            info!(tool = tool_name, "tool is approval-exempt; auto-approving");
            return Ok(true);
        }

        let Some(tx) = &self.approval_tx else {
            // No approval channel → auto-deny when policy says approval required.
            warn!(
                tool = tool_name,
                "approval required but no approval channel configured; denying"
            );
            return Ok(false);
        };

        // Build a metadata-driven risk summary.
        let level_tag = match metadata.security_level {
            SecurityLevel::Low => "LOW",
            SecurityLevel::Medium => "MEDIUM",
            SecurityLevel::High => "HIGH",
        };
        let rw_tag = if metadata.read_only { "read-only" } else { "READ-WRITE" };
        let group_tag = if metadata.group.is_empty() {
            String::new()
        } else {
            format!(" [{}]", metadata.group)
        };

        // Tool-specific detail for common tools.
        let detail = match tool_name {
            "run_shell" => format!(
                "command: {}",
                args.get("command").unwrap_or(&"(unknown)".to_string())
            ),
            "write_file" => format!(
                "path: {}",
                args.get("path").unwrap_or(&"(unknown)".to_string())
            ),
            "read_file" => format!(
                "path: {}",
                args.get("path").unwrap_or(&"(unknown)".to_string())
            ),
            "git_rollback" => format!(
                "commit: {}",
                args.get("commit_hash").unwrap_or(&"(unknown)".to_string())
            ),
            _ => {
                // Generic: show up to 2 args as key=value
                let mut parts: Vec<String> = args
                    .iter()
                    .take(2)
                    .map(|(k, v)| {
                        let truncated = if v.len() > 60 { &v[..60] } else { v };
                        format!("{k}={truncated}")
                    })
                    .collect();
                if args.len() > 2 {
                    parts.push(format!("(+{} more)", args.len() - 2));
                }
                parts.join(", ")
            }
        };

        let risk = format!("[{level_tag}/{rw_tag}]{group_tag} {tool_name}: {detail}");

        let request = ApprovalRequest {
            tool_name: tool_name.to_string(),
            args: args.clone(),
            risk_summary: risk,
        };

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send((request, reply_tx))
            .await
            .map_err(|_| anyhow::anyhow!("approval channel closed"))?;

        let decision = reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("approval response channel dropped"))?;

        Ok(decision == ApprovalDecision::Approve)
    }
}

// ── Security level ordering ──────────────────────────────────────────────────

/// Returns `true` when `tool_level` is at or below `max_level`.
///
/// Ordering: `Low < Medium < High`.
fn security_level_permitted(tool_level: SecurityLevel, max_level: SecurityLevel) -> bool {
    fn rank(l: SecurityLevel) -> u8 {
        match l {
            SecurityLevel::Low => 0,
            SecurityLevel::Medium => 1,
            SecurityLevel::High => 2,
        }
    }
    rank(tool_level) <= rank(max_level)
}

/// Parse a string into a `SecurityLevel`, defaulting to `High`.
fn parse_security_level_str(s: &str) -> SecurityLevel {
    match s.to_lowercase().as_str() {
        "low" => SecurityLevel::Low,
        "medium" => SecurityLevel::Medium,
        _ => SecurityLevel::High,
    }
}

// ── Convenience: create a default registry with built-in tools ───────────────

pub fn default_registry(
    workspace_root: PathBuf,
    agent_data_dir: PathBuf,
    brave_api_key: Option<String>,
    tavily_api_key: Option<String>,
    searxng_base_url: Option<String>,
    search_providers: Vec<String>,
) -> ToolRegistry {
    use aigent_tools::builtins::{
        CalendarAddEventTool, DraftEmailTool, GitRollbackTool, ReadFileTool, RemindMeTool,
        RunShellTool, WebSearchTool, WriteFileTool, FetchPageTool,
    };

    let mut registry = ToolRegistry::default();

    // ── Step 1: WASM-first ─────────────────────────────────────────────────
    // Register compiled WASM guests before native tools.  `ToolRegistry::get`
    // uses `.find()` (first-match wins) so WASM tools always take precedence
    // over any native fallback registered below.
    #[cfg(feature = "wasm")]
    let wasm_names: std::collections::HashSet<String> = {
        let extensions_dir = workspace_root.join("extensions");
        let tools = wasm::load_wasm_tools_from_dir(&extensions_dir, Some(&workspace_root));
        let names: std::collections::HashSet<String> =
            tools.iter().map(|t| t.spec().name.clone()).collect();
        if !names.is_empty() {
            info!(count = names.len(), "wasm: {} guest tool(s) active", names.len());
        }
        for tool in tools {
            registry.register(tool);
        }
        names
    };
    #[cfg(not(feature = "wasm"))]
    let wasm_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    // ── Step 2: Native fallbacks (only for names not covered by WASM) ──────
    // Build the candidate list.  Each entry is (canonical-name, boxed-tool).
    // We consume each Box exactly once so we shadow the outer variables after
    // construction to avoid partial-move issues.
    let native_candidates: Vec<(&str, Box<dyn aigent_tools::Tool>)> = vec![
        (
            "read_file",
            Box::new(ReadFileTool { workspace_root: workspace_root.clone() }),
        ),
        (
            "write_file",
            Box::new(WriteFileTool { workspace_root: workspace_root.clone() }),
        ),
        (
            "run_shell",
            Box::new(RunShellTool { workspace_root: workspace_root.clone() }),
        ),
        (
            "calendar_add_event",
            Box::new(CalendarAddEventTool { data_dir: agent_data_dir.clone() }),
        ),
        (
            "web_search",
            Box::new(WebSearchTool {
                brave_api_key: brave_api_key.clone(),
                tavily_api_key: tavily_api_key.clone(),
                searxng_base_url: searxng_base_url.clone(),
                search_providers: search_providers.clone(),
            }),
        ),
        (
            "fetch_page",
            Box::new(FetchPageTool),
        ),
        (
            "draft_email",
            Box::new(DraftEmailTool { data_dir: agent_data_dir.clone() }),
        ),
        (
            "remind_me",
            Box::new(RemindMeTool { data_dir: agent_data_dir }),
        ),
        (
            "git_rollback",
            Box::new(GitRollbackTool { workspace_root: workspace_root.clone() }),
        ),
    ];

    let mut native_active: Vec<&str> = Vec::new();
    for (name, tool) in native_candidates {
        if !wasm_names.contains(name) {
            native_active.push(name);
            registry.register(tool);
        }
    }
    if !native_active.is_empty() {
        info!(
            tools = ?native_active,
            "native Rust fallback active for {} tool(s) — no WASM binary built yet \
             — run `aigent tools build` to activate WASM mode",
            native_active.len()
        );
    }

    registry
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;

    use crate::{ExecutionPolicy, ToolExecutor, default_registry};
    use aigent_tools::{SecurityLevel, ToolMetadata};

    // Helper metadata for common tool categories used in tests.
    fn read_only_low() -> ToolMetadata {
        ToolMetadata { security_level: SecurityLevel::Low, read_only: true, group: "filesystem".into(), ..Default::default() }
    }
    fn read_only_high() -> ToolMetadata {
        ToolMetadata { security_level: SecurityLevel::High, read_only: true, group: "web".into(), ..Default::default() }
    }
    fn write_medium() -> ToolMetadata {
        ToolMetadata { security_level: SecurityLevel::Medium, read_only: false, group: "filesystem".into(), ..Default::default() }
    }
    fn write_high() -> ToolMetadata {
        ToolMetadata { security_level: SecurityLevel::High, read_only: false, group: "shell".into(), ..Default::default() }
    }



    // ── requires_approval tests ────────────────────────────────────────────

    #[test]
    fn autonomous_never_requires_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Autonomous,
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(!executor.requires_approval("run_shell", &write_high()));
        assert!(!executor.requires_approval("write_file", &write_medium()));
        assert!(!executor.requires_approval("read_file", &read_only_low()));
    }

    #[test]
    fn safer_always_requires_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Safer,
            approval_exempt_tools: vec![], // clear exemptions
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.requires_approval("read_file", &read_only_low()));
        assert!(executor.requires_approval("write_file", &write_medium()));
        assert!(executor.requires_approval("run_shell", &write_high()));
    }

    #[test]
    fn balanced_read_only_low_no_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Balanced,
            approval_exempt_tools: vec![],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        // read_only + Low → no approval
        assert!(!executor.requires_approval("read_file", &read_only_low()));
    }

    #[test]
    fn balanced_read_only_high_needs_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Balanced,
            approval_exempt_tools: vec![],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        // read_only + High → still needs approval (High triggers it)
        assert!(executor.requires_approval("web_search", &read_only_high()));
    }

    #[test]
    fn balanced_write_tools_need_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Balanced,
            approval_exempt_tools: vec![],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.requires_approval("write_file", &write_medium()));
        assert!(executor.requires_approval("run_shell", &write_high()));
    }

    #[test]
    fn exempt_tools_bypass_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Safer,
            approval_exempt_tools: vec!["run_shell".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(!executor.requires_approval("run_shell", &write_high()));
    }

    // ── check_capability tests ─────────────────────────────────────────────

    /// Make an empty registry (no tools) — sufficient for testing capability
    /// checks since `check_capability` only uses the registry for group
    /// expansion when allow/deny lists contain `@group` entries.
    fn empty_reg() -> aigent_tools::ToolRegistry {
        aigent_tools::ToolRegistry::default()
    }

    #[test]
    fn denylist_blocks_tool() {
        let policy = ExecutionPolicy {
            tool_denylist: vec!["write_file".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability(&empty_reg(), "write_file", &write_medium()).is_err());
    }

    #[test]
    fn allowlist_blocks_unlisted_tool() {
        let policy = ExecutionPolicy {
            tool_allowlist: vec!["read_file".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability(&empty_reg(), "read_file", &read_only_low()).is_ok());
        assert!(executor.check_capability(&empty_reg(), "write_file", &write_medium()).is_err());
    }

    #[test]
    fn empty_allowlist_permits_all() {
        let policy = ExecutionPolicy {
            tool_allowlist: vec![],
            tool_denylist: vec![],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability(&empty_reg(), "read_file", &read_only_low()).is_ok());
        assert!(executor.check_capability(&empty_reg(), "write_file", &write_medium()).is_ok());
        assert!(executor.check_capability(&empty_reg(), "run_shell", &write_high()).is_err()); // shell blocked by allow_shell=false
    }

    #[test]
    fn denylist_overrides_allowlist() {
        let policy = ExecutionPolicy {
            tool_allowlist: vec!["write_file".to_string()],
            tool_denylist: vec!["write_file".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability(&empty_reg(), "write_file", &write_medium()).is_err());
    }

    #[test]
    fn security_level_ceiling_blocks_high_when_max_medium() {
        let policy = ExecutionPolicy {
            max_security_level: SecurityLevel::Medium,
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability(&empty_reg(), "run_shell", &write_high()).is_err());
        assert!(executor.check_capability(&empty_reg(), "write_file", &write_medium()).is_ok());
        assert!(executor.check_capability(&empty_reg(), "read_file", &read_only_low()).is_ok());
    }

    #[test]
    fn security_level_ceiling_low_blocks_medium() {
        let policy = ExecutionPolicy {
            max_security_level: SecurityLevel::Low,
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability(&empty_reg(), "write_file", &write_medium()).is_err());
        assert!(executor.check_capability(&empty_reg(), "read_file", &read_only_low()).is_ok());
    }

    // ── Integration tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn shell_blocked_when_capability_disabled() -> anyhow::Result<()> {
        let workspace = std::env::temp_dir().join("aigent-exec-shell-test");
        fs::create_dir_all(&workspace)?;

        let policy = ExecutionPolicy {
            allow_shell: false,
            approval_required: false,
            ..ExecutionPolicy::default()
        };

        let executor = ToolExecutor::new(policy);
        let registry =
            default_registry(workspace, std::env::temp_dir().join("aigent-exec-shell-data"), None, None, None, vec![]);

        let mut args = HashMap::new();
        args.insert("command".to_string(), "echo hi".to_string());

        let result = executor.execute(&registry, "run_shell", &args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
        Ok(())
    }

    #[tokio::test]
    async fn read_file_within_workspace() -> anyhow::Result<()> {
        let workspace = std::env::temp_dir().join("aigent-exec-read-test");
        fs::create_dir_all(&workspace)?;
        fs::write(workspace.join("hello.txt"), "Hello, world!")?;

        let policy = ExecutionPolicy {
            allow_shell: true,
            allow_wasm: true,
            approval_mode: aigent_config::ApprovalMode::Autonomous,
            approval_required: false,
            workspace_root: workspace.clone(),
            ..ExecutionPolicy::default()
        };

        let executor = ToolExecutor::new(policy);
        let registry =
            default_registry(workspace, std::env::temp_dir().join("aigent-exec-read-data"), None, None, None, vec![]);

        let mut args = HashMap::new();
        args.insert("path".to_string(), "hello.txt".to_string());

        let result = executor.execute(&registry, "read_file", &args).await?;
        assert!(result.success);
        assert_eq!(result.output, "Hello, world!");
        Ok(())
    }

    #[tokio::test]
    async fn unknown_tool_returns_error() -> anyhow::Result<()> {
        let workspace = std::env::temp_dir().join("aigent-exec-unknown-test");
        fs::create_dir_all(&workspace)?;

        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Autonomous,
            ..ExecutionPolicy::default()
        };

        let executor = ToolExecutor::new(policy);
        let registry =
            default_registry(workspace, std::env::temp_dir().join("aigent-exec-unknown-data"), None, None, None, vec![]);

        let result = executor
            .execute(&registry, "nonexistent_tool", &HashMap::new())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown tool"));
        Ok(())
    }

    // ── Default policy tests ───────────────────────────────────────────────

    #[test]
    fn default_policy_shell_disabled() {
        let p = ExecutionPolicy::default();
        assert!(!p.allow_shell);
    }

    #[test]
    fn default_policy_balanced_mode() {
        let p = ExecutionPolicy::default();
        assert!(matches!(p.approval_mode, aigent_config::ApprovalMode::Balanced));
    }

    #[test]
    fn default_policy_has_exempt_tools() {
        let p = ExecutionPolicy::default();
        assert!(!p.approval_exempt_tools.is_empty());
    }
}
