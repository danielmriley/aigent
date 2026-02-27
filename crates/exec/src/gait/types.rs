//! **gait** — Git Agent Integration Tool.
//!
//! Safe, in-process git operations using libgit2.  Designed to be the
//! *preferred* git interface for the agent, replacing ad-hoc `run_shell git …`
//! for all normal operations.
//!
//! ## Security Model
//!
//! | Classification | Operations | Rule |
//! |----------------|-----------|------|
//! | **WRITE** | commit, checkout, merge, reset, pull, push, clone, branch (create/delete), tag | Path MUST be inside `trusted_write_paths`. |
//! | **READ** | status, log, diff, show, blame, ls-remote, fetch (metadata only) | When `allow_system_read = true`, any accessible path. Otherwise same as WRITE. |
//!
//! `clone` is **always WRITE** — it requires an explicit `target_dir` that
//! resolves inside `trusted_write_paths`.
//!
//! `trusted_write_paths` always auto-includes `workspace_path` and the
//! detected Aigent source directory.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use tokio::fs;

// ── Public types ─────────────────────────────────────────────────────────────

/// A single git operation request.  This mirrors the WIT `git-operation` record
/// and is the sole entry-point for all gait operations.
#[derive(Debug, Clone)]
pub struct GitOperation {
    /// One of: status, log, diff, pull, fetch, commit, checkout, branch,
    /// reset, clone, ls-remote, show, blame, push, tag, stash.
    pub action: String,
    /// Target repository:
    /// - `"workspace"` — agent workspace directory.
    /// - `"self"` — Aigent source directory (auto-detected).
    /// - An absolute local path.
    /// - A remote URL (used by `clone` and `ls-remote`).
    pub repo: String,
    /// Required for `clone`; must resolve inside `trusted_write_paths`.
    pub target_dir: Option<String>,
    pub branch: Option<String>,
    pub message: Option<String>,
    pub paths: Option<Vec<String>>,
    pub force: Option<bool>,
}

/// Runtime-resolved trust boundaries built from config + automatic paths.
#[derive(Debug, Clone)]
pub struct GaitPolicy {
    /// Canonical absolute paths where write ops are allowed.
    pub trusted_write_paths: Vec<PathBuf>,
    /// Trusted remote URLs for identity / push.
    pub trusted_repos: HashSet<String>,
    /// When true, read-only ops can target any path; when false, reads are
    /// also restricted to `trusted_write_paths`.
    pub allow_system_read: bool,
    /// Resolved workspace root (canonical).
    pub workspace_root: PathBuf,
    /// Resolved Aigent source directory (canonical), if detectable.
    pub self_repo_path: Option<PathBuf>,
}

// ── Action classification ────────────────────────────────────────────────────

/// Returns `true` for actions that mutate the repository (write operations).
pub fn is_write_action(action: &str) -> bool {
    matches!(
        action,
        "commit"
            | "checkout"
            | "merge"
            | "reset"
            | "pull"
            | "push"
            | "clone"
            | "branch"
            | "tag"
            | "stash"
    )
}

/// Fine-grained mutation check — like [`is_write_action`] but accounts for
/// dual-purpose actions (`branch` and `tag`) that are *read-only* when no
/// name parameter is supplied (list mode).
pub fn is_mutating_op(op: &GitOperation) -> bool {
    match op.action.as_str() {
        // branch/tag without a name just lists — read-only.
        "branch" | "tag" => op.branch.is_some(),
        other => is_write_action(other),
    }
}

/// Same as [`is_mutating_op`] but works with the string args map from the
/// tool invocation (for use in the approval flow).
pub fn is_mutating_call(action: &str, args: &std::collections::HashMap<String, String>) -> bool {
    match action {
        "branch" | "tag" => args.get("branch").is_some_and(|v| !v.is_empty()),
        other => is_write_action(other),
    }
}

// ── Policy construction ──────────────────────────────────────────────────────

impl GaitPolicy {
    /// Build from `AppConfig`.  Automatically adds workspace root and Aigent
    /// source directory to the trusted write paths.
    pub async fn from_config(config: &aigent_config::AppConfig) -> Self {
        let workspace_root = fs::canonicalize(&config.agent.workspace_path).await
            .unwrap_or_else(|_| PathBuf::from(&config.agent.workspace_path));

        // Detect Aigent source directory: first try AIGENT_SOURCE_DIR env, then
        // the directory containing the running executable.
        let self_repo_path = std::env::var("AIGENT_SOURCE_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::current_exe().ok().and_then(|exe| {
                    // Walk up from the binary to find a Cargo.toml with
                    // workspace members containing "crates/exec".
                    let mut dir = exe.parent()?.to_path_buf();
                    for _ in 0..5 {
                        if dir.join("crates/exec/src/gait.rs").exists() {
                            return Some(dir);
                        }
                        dir = dir.parent()?.to_path_buf();
                    }
                    None
                })
            })
            .and_then(|p| std::fs::canonicalize(p).ok()); // sync: runs once at startup

        let mut trusted_write_paths: Vec<PathBuf> = Vec::new();
        for p in &config.git.trusted_write_paths {
            if let Ok(c) = fs::canonicalize(p).await {
                trusted_write_paths.push(c);
            }
        }

        // Always include workspace root.
        if let Ok(canonical_ws) = fs::canonicalize(&config.agent.workspace_path).await {
            if !trusted_write_paths.contains(&canonical_ws) {
                trusted_write_paths.push(canonical_ws);
            }
        }
        // NOTE: self_repo_path is intentionally NOT added to
        // trusted_write_paths — the agent can read its own source code
        // but must not modify it.

        let trusted_repos: HashSet<String> =
            config.git.trusted_repos.iter().cloned().collect();

        // In "safer" approval mode, override allow_system_read to false.
        let allow_system_read = if config.tools.approval_mode == aigent_config::ApprovalMode::Safer
        {
            false
        } else {
            config.git.allow_system_read
        };

        Self {
            trusted_write_paths,
            trusted_repos,
            allow_system_read,
            workspace_root,
            self_repo_path,
        }
    }
}

// ── Path resolution & authorisation ──────────────────────────────────────────

/// Resolve the `repo` field of a `GitOperation` to a canonical local path.
/// Returns `Err` for remote URLs that aren't being cloned (those go through
/// `ls-remote` / `clone` directly).
pub(super) async fn resolve_repo_path(op: &GitOperation, policy: &GaitPolicy) -> Result<PathBuf> {
    let raw = op.repo.trim();
    if raw.eq_ignore_ascii_case("workspace") {
        // The workspace directory is auto-initialised as a git repo on
        // daemon startup (see `git_init_if_needed`).  Never walk upward —
        // that would escape the sandbox boundary.
        return Ok(policy.workspace_root.clone());
    }
    if raw.eq_ignore_ascii_case("self") {
        return policy
            .self_repo_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!(
                "cannot resolve \"self\" — Aigent source directory not detected. \
                 Set AIGENT_SOURCE_DIR or run from inside the source tree."
            ));
    }
    // Remote URL — only valid for clone and ls-remote
    if raw.starts_with("https://") || raw.starts_with("http://") || raw.starts_with("git@") {
        bail!(
            "remote URL '{}' can only be used with 'clone' or 'ls-remote' actions",
            raw
        );
    }
    // Treat as a local path.
    fs::canonicalize(raw).await
        .map_err(|e| anyhow::anyhow!("cannot resolve repo path '{}': {}", raw, e))
}

/// Check that `target` is inside one of the trusted write paths.
pub(super) fn assert_inside_trusted(target: &Path, policy: &GaitPolicy) -> Result<()> {
    for trusted in &policy.trusted_write_paths {
        if target.starts_with(trusted) {
            return Ok(());
        }
    }
    bail!(
        "path '{}' is outside all trusted_write_paths.\n\
         Trusted: {:?}",
        target.display(),
        policy
            .trusted_write_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
    );
}

/// Authorise the operation according to the security policy.
pub(super) fn authorise(op: &GitOperation, repo_path: &Path, policy: &GaitPolicy) -> Result<()> {
    // The agent's own source repo is strictly read-only.
    if let Some(ref self_path) = policy.self_repo_path {
        if repo_path.starts_with(self_path) && is_mutating_op(op) {
            bail!(
                "write operations on the Aigent source repository are not allowed. \
                 Use repo=\"workspace\" for your own work."
            );
        }
    }

    if is_mutating_op(op) {
        assert_inside_trusted(repo_path, policy)
    } else if policy.allow_system_read {
        Ok(()) // Reads allowed anywhere
    } else {
        assert_inside_trusted(repo_path, policy)
    }
}

