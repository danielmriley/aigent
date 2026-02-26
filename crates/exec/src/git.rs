//! Git integration helpers for the tool executor.
//!
//! All functions are best-effort: if `git` is not installed, or the workspace
//! is not a git repository, functions return `Ok(…)` rather than propagating
//! an error.  Only genuine git errors (e.g., merge conflicts) are surfaced as
//! `Err`.
//!
//! # Auto-commit lifecycle
//!
//! When `[tools] git_auto_commit = true`:
//! 1. `ToolExecutor::execute` calls `git_auto_commit` after a successful write.
//! 2. `git_auto_commit` stages all changes and commits with
//!    `"Aigent tool: <name> — <detail>"`.
//! 3. The agent (or the user via CLI) can call `git_rollback_last` to revert
//!    the most recent auto-commit.

use std::path::Path;

use anyhow::Result;
use tracing::{info, warn};

// ── git_init_if_needed ────────────────────────────────────────────────────────

/// Initialises a git repository at `workspace_root` unless one already exists.
///
/// Returns `true` when a new repository was created, `false` when one was
/// already present.  Silently skips when `git` is not in `$PATH`.
pub async fn git_init_if_needed(workspace_root: &Path) -> Result<bool> {
    if workspace_root.join(".git").exists() {
        return Ok(false); // already a repo
    }

    let out = tokio::process::Command::new("git")
        .args(["init"])
        .current_dir(workspace_root)
        .output()
        .await;

    match out {
        Ok(o) if o.status.success() => {
            info!(workspace = ?workspace_root, "aigent: initialised git repository");
            Ok(true)
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            warn!(%stderr, "git init failed (non-fatal)");
            Ok(false)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // git not installed — skip silently
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}

// ── git_auto_commit ───────────────────────────────────────────────────────────

/// Stages all changes in `workspace_root` and creates a commit with a
/// standardised message.  Returns `Ok(())` without committing when:
///
/// - `git` is not installed.
/// - `workspace_root` is not a git repository.
/// - There are no staged changes after `git add -A`.
pub async fn git_auto_commit(workspace_root: &Path, tool_name: &str, detail: &str) -> Result<()> {
    // Silently skip if the workspace is not a git repo.
    if !workspace_root.join(".git").exists() {
        return Ok(());
    }

    // Stage everything.
    let add = tokio::process::Command::new("git")
        .args(["add", "-A"])
        .current_dir(workspace_root)
        .output()
        .await;

    match add {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()), // no git
        Err(e) => return Err(e.into()),
        Ok(o) if !o.status.success() => {
            warn!(
                stderr = %String::from_utf8_lossy(&o.stderr),
                "git add -A failed (non-fatal)"
            );
            return Ok(());
        }
        Ok(_) => {}
    }

    // Check whether there's actually anything staged.
    // `git diff --cached --quiet` exits 0 when nothing is staged.
    let dirty = tokio::process::Command::new("git")
        .args(["diff", "--cached", "--quiet"])
        .current_dir(workspace_root)
        .status()
        .await?;
    if dirty.success() {
        return Ok(()); // nothing to commit
    }

    // Truncate the detail to keep commit messages readable (UTF-8 safe).
    let max_detail = 72;
    let detail_end = if detail.len() > max_detail {
        let mut end = max_detail;
        while end > 0 && !detail.is_char_boundary(end) { end -= 1; }
        end
    } else {
        detail.len()
    };
    let detail_short = &detail[..detail_end];
    let msg = format!("Aigent tool: {tool_name} — {detail_short}");

    let commit = tokio::process::Command::new("git")
        .args(["commit", "-m", &msg, "--no-verify"])
        .env("GIT_AUTHOR_NAME", "Aigent")
        .env("GIT_AUTHOR_EMAIL", "aigent@localhost")
        .env("GIT_COMMITTER_NAME", "Aigent")
        .env("GIT_COMMITTER_EMAIL", "aigent@localhost")
        .current_dir(workspace_root)
        .output()
        .await;

    match commit {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            info!(tool = tool_name, commit_msg = %msg, "auto-committed workspace changes");
            let _ = stdout;
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            warn!(%stderr, "git commit failed (non-fatal)");
        }
        Err(e) => {
            warn!(?e, "git commit I/O error (non-fatal)");
        }
    }

    Ok(())
}

// ── git_rollback_last ─────────────────────────────────────────────────────────

/// Reverts the most recent commit in `workspace_root` via `git revert HEAD`.
///
/// Returns a human-readable summary on success.  Returns an error when:
///
/// - `git` is not installed.
/// - The workspace is not a git repository.
/// - The revert fails (e.g., merge conflict, no commits to revert).
pub async fn git_rollback_last(workspace_root: &Path) -> Result<String> {
    if !workspace_root.join(".git").exists() {
        anyhow::bail!("workspace is not a git repository; cannot roll back");
    }

    let out = tokio::process::Command::new("git")
        .args(["revert", "HEAD", "--no-edit"])
        .env("GIT_AUTHOR_NAME", "Aigent")
        .env("GIT_AUTHOR_EMAIL", "aigent@localhost")
        .env("GIT_COMMITTER_NAME", "Aigent")
        .env("GIT_COMMITTER_EMAIL", "aigent@localhost")
        .current_dir(workspace_root)
        .output()
        .await?;

    if out.status.success() {
        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        info!("git_rollback_last succeeded");
        Ok(if stdout.is_empty() {
            "Last commit reverted successfully.".to_string()
        } else {
            stdout
        })
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        anyhow::bail!("git revert failed: {}", stderr);
    }
}

// ── git_log_last ──────────────────────────────────────────────────────────────

/// Returns a one-line summary of the most recent commit, or `None` when the
/// workspace has no commits.
pub async fn git_log_last(workspace_root: &Path) -> Option<String> {
    if !workspace_root.join(".git").exists() {
        return None;
    }
    let out = tokio::process::Command::new("git")
        .args(["log", "-1", "--oneline"])
        .current_dir(workspace_root)
        .output()
        .await
        .ok()?;
    if out.status.success() {
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if line.is_empty() { None } else { Some(line) }
    } else {
        None
    }
}
