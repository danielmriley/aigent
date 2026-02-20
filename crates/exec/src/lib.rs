use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use aigent_tools::{ToolOutput, ToolRegistry};

// ── Execution Policy ─────────────────────────────────────────────────────────

/// Built from `SafetyConfig` in aigent-config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    pub approval_required: bool,
    pub allow_shell: bool,
    pub allow_wasm: bool,
    pub workspace_root: PathBuf,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            approval_required: true,
            allow_shell: false,
            allow_wasm: false,
            workspace_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
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

        // 2. Enforce capability gates
        self.check_capability(tool_name)?;

        // 3. Approval flow (if required)
        if self.policy.approval_required {
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
        tool.run(args).await
    }

    fn check_capability(&self, tool_name: &str) -> Result<()> {
        match tool_name {
            "run_shell" if !self.policy.allow_shell => {
                bail!("shell execution is disabled by safety policy")
            }
            "read_file" | "write_file" if !self.policy.allow_shell && !self.policy.allow_wasm => {
                bail!("file access is disabled by safety policy (requires allow_shell or allow_wasm)")
            }
            _ => Ok(()),
        }
    }

    async fn request_approval(
        &self,
        tool_name: &str,
        args: &HashMap<String, String>,
    ) -> Result<bool> {
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

pub fn default_registry(workspace_root: PathBuf) -> ToolRegistry {
    use aigent_tools::builtins::{ReadFileTool, RunShellTool, WriteFileTool};

    let mut registry = ToolRegistry::default();
    registry.register(Box::new(ReadFileTool {
        workspace_root: workspace_root.clone(),
    }));
    registry.register(Box::new(WriteFileTool {
        workspace_root: workspace_root.clone(),
    }));
    registry.register(Box::new(RunShellTool { workspace_root }));
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
        let registry = default_registry(workspace);

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
            approval_required: false,
            workspace_root: workspace.clone(),
        };

        let executor = ToolExecutor::new(policy);
        let registry = default_registry(workspace);

        let mut args = HashMap::new();
        args.insert("path".to_string(), "hello.txt".to_string());

        let result = executor.execute(&registry, "read_file", &args).await?;
        assert!(result.success);
        assert_eq!(result.output, "Hello, world!");
        Ok(())
    }
}

