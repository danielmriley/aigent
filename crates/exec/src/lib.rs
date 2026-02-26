pub mod gait;
pub mod git;
pub mod sandbox;

#[cfg(feature = "wasm")]
pub mod wasm;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use aigent_config::ApprovalMode;
use aigent_tools::{ToolOutput, ToolRegistry};

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

        // 2. Enforce capability gates (deny-list / allow-list / shell guard)
        self.check_capability(tool_name)?;

        // 3. Approval flow — governed by ApprovalMode
        if self.requires_approval(tool_name) {
            let approved = self.request_approval(tool_name, args).await?;
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
            if result.success && self.policy.git_auto_commit {
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

        // 5. Git auto-commit after successful write operations
        if result.success && self.policy.git_auto_commit {
            const WRITE_TOOLS: &[&str] = &["write_file", "run_shell"];
            if WRITE_TOOLS.contains(&tool_name) {
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
        }

        Ok(result)
    }

    /// Returns `true` when this tool invocation needs interactive approval
    /// based on the configured `ApprovalMode`.
    ///
    /// | Mode         | Needs approval                                          |
    /// |--------------|---------------------------------------------------------|
    /// | `Autonomous` | Never                                                   |
    /// | `Balanced`   | Write / shell tools and anything not read-only          |
    /// | `Safer`      | Every tool (unless explicitly exempt)                   |
    fn requires_approval(&self, tool_name: &str) -> bool {
        // Explicit exempt list always short-circuits.
        if self.policy
            .approval_exempt_tools
            .contains(&tool_name.to_string())
        {
            return false;
        }
        match &self.policy.approval_mode {
            ApprovalMode::Autonomous => false,
            ApprovalMode::Balanced => {
                // Read-only tools don't need approval.
                const READ_ONLY: &[&str] = &[
                    "read_file",
                    "web_search",
                    "perform_gait",
                ];
                !READ_ONLY.contains(&tool_name)
            }
            ApprovalMode::Safer => true,
        }
    }

    fn check_capability(&self, tool_name: &str) -> Result<()> {
        // Shell is only available when explicitly enabled in config.
        if tool_name == "run_shell" && !self.policy.allow_shell {
            bail!("shell execution is disabled by safety policy (set allow_shell = true)");
        }
        // Per-tool deny-list (takes precedence over allow-list).
        if self.policy.tool_denylist.contains(&tool_name.to_string()) {
            bail!("tool '{}' is blocked by policy (tool_denylist)", tool_name);
        }
        // Per-tool allow-list (empty = all permitted).
        if !self.policy.tool_allowlist.is_empty()
            && !self.policy.tool_allowlist.contains(&tool_name.to_string())
        {
            bail!("tool '{}' is not in the tool_allowlist", tool_name);
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

    async fn request_approval(        &self,
        tool_name: &str,
        args: &HashMap<String, String>,
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

        let risk = match tool_name {
            "run_shell" => format!(
                "Execute shell command: {}",
                args.get("command").unwrap_or(&"(unknown)".to_string())
            ),
            "write_file" => format!(
                "Write to file: {}",
                args.get("path").unwrap_or(&"(unknown)".to_string())
            ),
            "read_file" => format!(
                "Read file: {}",
                args.get("path").unwrap_or(&"(unknown)".to_string())
            ),
            _ => format!("Execute tool: {tool_name}"),
        };

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

// ── Workspace boundary helper ────────────────────────────────────────────────

pub fn ensure_within_workspace(workspace_root: &Path, target: &Path) -> Result<PathBuf> {
    let canonical_root = workspace_root.canonicalize()?;
    let joined = if target.is_absolute() {
        target.to_path_buf()
    } else {
        canonical_root.join(target)
    };
    let canonical_target = joined.canonicalize()?;

    if !canonical_target.starts_with(&canonical_root) {
        bail!(
            "path escapes workspace boundary: {}",
            canonical_target.display()
        );
    }

    Ok(canonical_target)
}

// ── Convenience: create a default registry with built-in tools ───────────────

pub fn default_registry(
    workspace_root: PathBuf,
    agent_data_dir: PathBuf,
    brave_api_key: Option<String>,
) -> ToolRegistry {
    use aigent_tools::builtins::{
        CalendarAddEventTool, DraftEmailTool, GitRollbackTool, ReadFileTool, RemindMeTool,
        RunShellTool, WebSearchTool, WriteFileTool,
    };

    let mut registry = ToolRegistry::default();

    // ── Step 0: Native gait (git) tool ─────────────────────────────────────
    // Registered first so it appears before other tools in listings.
    // We build a GaitPolicy from a minimal AppConfig using the workspace root.
    {
        let mut app_config = aigent_config::AppConfig::default();
        app_config.agent.workspace_path = workspace_root.display().to_string();
        let policy = gait::GaitPolicy::from_config(&app_config);
        let gait_tool = gait::GaitTool { policy };
        registry.register(Box::new(gait_tool));
    }

    // ── Step 1: WASM-first ─────────────────────────────────────────────────
    // Register compiled WASM guests before native tools.  `ToolRegistry::get`
    // uses `.find()` (first-match wins) so WASM tools always take precedence
    // over any native fallback registered below.
    #[cfg(feature = "wasm")]
    let wasm_names: std::collections::HashSet<String> = {
        let extensions_dir = workspace_root.join("extensions");
        let tools = wasm::load_wasm_tools_from_dir(&extensions_dir);
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
            Box::new(WebSearchTool { brave_api_key: brave_api_key.clone() }),
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
    use std::path::PathBuf;

    use crate::{ExecutionPolicy, ToolExecutor, default_registry, ensure_within_workspace};

    #[test]
    fn workspace_guard_rejects_escape() -> anyhow::Result<()> {
        let base = std::env::temp_dir().join("aigent-exec-workspace-test");
        let child = base.join("safe");
        fs::create_dir_all(&child)?;

        let escaped = ensure_within_workspace(&base, &PathBuf::from("../"));
        assert!(escaped.is_err());
        Ok(())
    }

    #[test]
    fn workspace_guard_accepts_child_path() -> anyhow::Result<()> {
        let base = std::env::temp_dir().join("aigent-exec-ws-accept-test");
        let child = base.join("subdir");
        fs::create_dir_all(&child)?;
        let result = ensure_within_workspace(&base, &PathBuf::from("subdir"));
        assert!(result.is_ok());
        Ok(())
    }

    // ── requires_approval tests ────────────────────────────────────────────

    #[test]
    fn autonomous_never_requires_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Autonomous,
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(!executor.requires_approval("run_shell"));
        assert!(!executor.requires_approval("write_file"));
        assert!(!executor.requires_approval("read_file"));
    }

    #[test]
    fn safer_always_requires_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Safer,
            approval_exempt_tools: vec![], // clear exemptions
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.requires_approval("read_file"));
        assert!(executor.requires_approval("write_file"));
        assert!(executor.requires_approval("run_shell"));
    }

    #[test]
    fn balanced_read_only_no_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Balanced,
            approval_exempt_tools: vec![],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(!executor.requires_approval("read_file"));
        assert!(!executor.requires_approval("web_search"));
    }

    #[test]
    fn balanced_write_tools_need_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Balanced,
            approval_exempt_tools: vec![],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.requires_approval("write_file"));
        assert!(executor.requires_approval("run_shell"));
        assert!(executor.requires_approval("git_rollback"));
        assert!(executor.requires_approval("remind_me"));
        assert!(executor.requires_approval("calendar_add_event"));
    }

    #[test]
    fn exempt_tools_bypass_approval() {
        let policy = ExecutionPolicy {
            approval_mode: aigent_config::ApprovalMode::Safer,
            approval_exempt_tools: vec!["run_shell".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(!executor.requires_approval("run_shell"));
    }

    // ── check_capability tests ─────────────────────────────────────────────

    #[test]
    fn denylist_blocks_tool() {
        let policy = ExecutionPolicy {
            tool_denylist: vec!["write_file".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability("write_file").is_err());
    }

    #[test]
    fn allowlist_blocks_unlisted_tool() {
        let policy = ExecutionPolicy {
            tool_allowlist: vec!["read_file".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability("read_file").is_ok());
        assert!(executor.check_capability("write_file").is_err());
    }

    #[test]
    fn empty_allowlist_permits_all() {
        let policy = ExecutionPolicy {
            tool_allowlist: vec![],
            tool_denylist: vec![],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability("read_file").is_ok());
        assert!(executor.check_capability("write_file").is_ok());
        assert!(executor.check_capability("run_shell").is_err()); // shell blocked by allow_shell=false
    }

    #[test]
    fn denylist_overrides_allowlist() {
        let policy = ExecutionPolicy {
            tool_allowlist: vec!["write_file".to_string()],
            tool_denylist: vec!["write_file".to_string()],
            ..ExecutionPolicy::default()
        };
        let executor = ToolExecutor::new(policy);
        assert!(executor.check_capability("write_file").is_err());
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
            default_registry(workspace, std::env::temp_dir().join("aigent-exec-shell-data"), None);

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
            default_registry(workspace, std::env::temp_dir().join("aigent-exec-read-data"), None);

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
            default_registry(workspace, std::env::temp_dir().join("aigent-exec-unknown-data"), None);

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
