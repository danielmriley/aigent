use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

#[derive(Debug, Clone)]
pub struct ShellExecution {
    pub command: String,
    pub approved: bool,
    pub requested_workdir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ExecutionPolicy {
    pub approval_required: bool,
    pub allow_shell: bool,
    pub workspace_root: PathBuf,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            approval_required: true,
            allow_shell: false,
            workspace_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

pub fn execute_shell(request: &ShellExecution, policy: &ExecutionPolicy) -> Result<String> {
    if !policy.allow_shell {
        return Ok("blocked: shell capability disabled by policy".to_string());
    }

    if policy.approval_required && !request.approved {
        return Ok("blocked: approval required".to_string());
    }

    if let Some(workdir) = &request.requested_workdir {
        let _ = ensure_within_workspace(&policy.workspace_root, workdir)?;
    }

    Ok(format!("shell placeholder executed: {}", request.command))
}

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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use anyhow::Result;

    use crate::{ExecutionPolicy, ShellExecution, ensure_within_workspace, execute_shell};

    #[test]
    fn shell_blocked_when_capability_disabled() -> Result<()> {
        let policy = ExecutionPolicy {
            allow_shell: false,
            ..ExecutionPolicy::default()
        };
        let request = ShellExecution {
            command: "echo hi".to_string(),
            approved: true,
            requested_workdir: None,
        };

        let result = execute_shell(&request, &policy)?;
        assert!(result.contains("capability disabled"));
        Ok(())
    }

    #[test]
    fn workspace_guard_rejects_escape() -> Result<()> {
        let base = std::env::temp_dir().join("aigent-exec-workspace-test");
        let child = base.join("safe");
        fs::create_dir_all(&child)?;

        let escaped = ensure_within_workspace(&base, &PathBuf::from("../"));
        assert!(escaped.is_err());
        Ok(())
    }
}
