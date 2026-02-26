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
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use tracing::info;

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
    pub fn from_config(config: &aigent_config::AppConfig) -> Self {
        let workspace_root = std::fs::canonicalize(&config.agent.workspace_path)
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
            .and_then(|p| std::fs::canonicalize(p).ok());

        let mut trusted_write_paths: Vec<PathBuf> = config
            .git
            .trusted_write_paths
            .iter()
            .filter_map(|p| std::fs::canonicalize(p).ok())
            .collect();

        // Always include workspace root.
        if let Ok(canonical_ws) = std::fs::canonicalize(&config.agent.workspace_path) {
            if !trusted_write_paths.contains(&canonical_ws) {
                trusted_write_paths.push(canonical_ws);
            }
        }
        // Always include self-repo path.
        if let Some(ref self_path) = self_repo_path {
            if !trusted_write_paths.contains(self_path) {
                trusted_write_paths.push(self_path.clone());
            }
        }

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
fn resolve_repo_path(op: &GitOperation, policy: &GaitPolicy) -> Result<PathBuf> {
    let raw = op.repo.trim();
    if raw.eq_ignore_ascii_case("workspace") {
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
    std::fs::canonicalize(raw)
        .map_err(|e| anyhow::anyhow!("cannot resolve repo path '{}': {}", raw, e))
}

/// Check that `target` is inside one of the trusted write paths.
fn assert_inside_trusted(target: &Path, policy: &GaitPolicy) -> Result<()> {
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
fn authorise(op: &GitOperation, repo_path: &Path, policy: &GaitPolicy) -> Result<()> {
    if is_mutating_op(op) {
        assert_inside_trusted(repo_path, policy)
    } else if policy.allow_system_read {
        Ok(()) // Reads allowed anywhere
    } else {
        assert_inside_trusted(repo_path, policy)
    }
}

// ── Public entry-point ───────────────────────────────────────────────────────

/// Execute a git operation.  Returns a rich, formatted string suitable for
/// LLM reflection on success, or an error on failure.
pub async fn perform_gait(op: GitOperation, policy: &GaitPolicy) -> Result<String> {
    let action = op.action.to_lowercase();
    info!(action = %action, repo = %op.repo, "gait: performing git operation");

    match action.as_str() {
        "clone" => do_clone(&op, policy).await,
        "ls-remote" => do_ls_remote(&op, policy).await,
        // All other actions need a local repo path
        _ => {
            let repo_path = resolve_repo_path(&op, policy)?;
            authorise(&op, &repo_path, policy)?;
            match action.as_str() {
                "status" => do_status(&repo_path),
                "log" => do_log(&repo_path, &op),
                "diff" => do_diff(&repo_path, &op),
                "commit" => do_commit(&repo_path, &op).await,
                "checkout" => do_checkout(&repo_path, &op),
                "branch" => do_branch(&repo_path, &op),
                "reset" => do_reset(&repo_path, &op),
                "show" => do_show(&repo_path, &op),
                "pull" => do_pull(&repo_path, &op).await,
                "push" => do_push(&repo_path, &op).await,
                "fetch" => do_fetch(&repo_path, &op).await,
                "stash" => do_stash(&repo_path, &op),
                "blame" => do_blame(&repo_path, &op).await,
                "tag" => do_tag(&repo_path, &op),
                other => bail!("unsupported gait action: '{other}'"),
            }
        }
    }
}

// ── Individual action implementations ────────────────────────────────────────
//
// Where possible we use libgit2 (via `git2` crate) for safety and performance.
// For operations that are complex to implement purely with libgit2 (pull, push,
// fetch, blame), we delegate to `git` CLI commands while still enforcing all
// path trust checks.

fn open_repo(path: &Path) -> Result<git2::Repository> {
    git2::Repository::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open git repository at '{}': {}", path.display(), e))
}

// ── status ───────────────────────────────────────────────────────────────────

fn do_status(repo_path: &Path) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let statuses = repo.statuses(Some(
        git2::StatusOptions::new()
            .include_untracked(true)
            .recurse_untracked_dirs(true),
    ))?;

    if statuses.is_empty() {
        return Ok("Working tree clean — nothing to commit.".to_string());
    }

    let mut out = String::from("Git status:\n");
    for entry in statuses.iter() {
        let path = entry.path().unwrap_or("?");
        let st = entry.status();
        let label = if st.contains(git2::Status::WT_NEW) {
            "untracked"
        } else if st.contains(git2::Status::WT_MODIFIED) | st.contains(git2::Status::INDEX_MODIFIED)
        {
            "modified"
        } else if st.contains(git2::Status::WT_DELETED) | st.contains(git2::Status::INDEX_DELETED)
        {
            "deleted"
        } else if st.contains(git2::Status::INDEX_NEW) {
            "staged (new)"
        } else if st.contains(git2::Status::WT_RENAMED) | st.contains(git2::Status::INDEX_RENAMED)
        {
            "renamed"
        } else {
            "changed"
        };
        let _ = writeln!(out, "  {label}: {path}");
    }
    Ok(out)
}

// ── log ──────────────────────────────────────────────────────────────────────

fn do_log(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;
    revwalk.set_sorting(git2::Sort::TIME)?;

    let max_entries = 15;
    let mut out = String::from("Recent commits:\n");
    let mut count = 0;

    for oid in revwalk {
        if count >= max_entries {
            let _ = writeln!(out, "  … (showing first {max_entries})");
            break;
        }
        let oid = oid?;
        let commit = repo.find_commit(oid)?;
        let short = &oid.to_string()[..7];
        let summary = commit.summary().unwrap_or("(no message)");
        let time = commit.time();
        let ts = chrono::DateTime::from_timestamp(time.seconds(), 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "?".to_string());
        let author = commit.author();
        let name = author.name().unwrap_or("?");
        let _ = writeln!(out, "  {short} {ts} ({name}) {summary}");
        count += 1;
    }

    if count == 0 {
        return Ok("No commits found.".to_string());
    }
    let _ = op.branch; // reserved for future branch-specific log
    Ok(out)
}

// ── diff ─────────────────────────────────────────────────────────────────────

fn do_diff(repo_path: &Path, _op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let diff = repo.diff_index_to_workdir(None, None)?;
    let stats = diff.stats()?;

    if stats.files_changed() == 0 {
        return Ok("No differences found.".to_string());
    }

    let mut out = format!(
        "Diff: {} file(s) changed, {} insertion(s), {} deletion(s)\n",
        stats.files_changed(),
        stats.insertions(),
        stats.deletions(),
    );

    // Collect patch text (capped at 8 KiB to keep LLM context manageable).
    let mut patch_buf = String::new();
    diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
        if patch_buf.len() < 8192 {
            let origin = line.origin();
            if origin == '+' || origin == '-' || origin == ' ' {
                patch_buf.push(origin);
            }
            if let Ok(content) = std::str::from_utf8(line.content()) {
                patch_buf.push_str(content);
            }
        }
        true
    })?;

    if patch_buf.len() >= 8192 {
        patch_buf.push_str("\n…[patch truncated at 8 KiB]");
    }
    out.push_str(&patch_buf);
    Ok(out)
}

// ── commit ───────────────────────────────────────────────────────────────────

async fn do_commit(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let msg = op
        .message
        .as_deref()
        .unwrap_or("Aigent: gait auto-commit");

    let repo = open_repo(repo_path)?;

    // Stage all changes (mirrors `git add -A`).  If `paths` is provided,
    // stage only those paths.
    let mut index = repo.index()?;
    if let Some(ref paths) = op.paths {
        for p in paths {
            index.add_path(Path::new(p))?;
        }
    } else {
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
    }
    index.write()?;
    let tree_oid = index.write_tree()?;
    let tree = repo.find_tree(tree_oid)?;

    let sig = repo.signature().unwrap_or_else(|_| {
        git2::Signature::now("Aigent", "aigent@localhost").expect("valid signature")
    });

    let parent = repo
        .head()
        .ok()
        .and_then(|h| h.peel_to_commit().ok());
    let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();

    let oid = repo.commit(Some("HEAD"), &sig, &sig, msg, &tree, &parents)?;
    let short = &oid.to_string()[..7];

    info!(sha = %short, path = %repo_path.display(), "gait: committed");
    Ok(format!("Committed {short}: {msg}"))
}

// ── checkout ─────────────────────────────────────────────────────────────────

fn do_checkout(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let branch_name = op
        .branch
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("checkout requires a branch name"))?;

    let repo = open_repo(repo_path)?;

    // Try as local branch first, then as remote tracking branch.
    let refname = format!("refs/heads/{branch_name}");
    let obj = repo
        .revparse_single(&refname)
        .or_else(|_| repo.revparse_single(&format!("refs/remotes/origin/{branch_name}")))?;

    repo.checkout_tree(&obj, Some(git2::build::CheckoutBuilder::new().force()))?;
    repo.set_head(&refname)
        .or_else(|_| repo.set_head_detached(obj.id()))?;

    Ok(format!("Checked out '{branch_name}'."))
}

// ── branch ───────────────────────────────────────────────────────────────────

fn do_branch(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;

    if let Some(ref name) = op.branch {
        // Create new branch at HEAD.
        let head = repo.head()?.peel_to_commit()?;
        let force = op.force.unwrap_or(false);
        repo.branch(name, &head, force)?;
        Ok(format!("Created branch '{name}'."))
    } else {
        // List branches.
        let branches = repo.branches(Some(git2::BranchType::Local))?;
        let mut out = String::from("Branches:\n");
        for b in branches {
            let (branch, _) = b?;
            let name = branch.name()?.unwrap_or("?");
            let marker = if branch.is_head() { " *" } else { "" };
            let _ = writeln!(out, "  {name}{marker}");
        }
        Ok(out)
    }
}

// ── reset ────────────────────────────────────────────────────────────────────

fn do_reset(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let target = op.branch.as_deref().unwrap_or("HEAD~1");
    let obj = repo.revparse_single(target)?;
    let commit = obj
        .peel_to_commit()
        .map_err(|e| anyhow::anyhow!("cannot resolve '{}' to a commit: {}", target, e))?;

    let reset_type = if op.force.unwrap_or(false) {
        git2::ResetType::Hard
    } else {
        git2::ResetType::Mixed
    };

    repo.reset(commit.as_object(), reset_type, None)?;

    let type_label = if op.force.unwrap_or(false) {
        "hard"
    } else {
        "mixed"
    };
    Ok(format!(
        "Reset ({type_label}) to '{target}' ({})",
        &commit.id().to_string()[..7]
    ))
}

// ── show ─────────────────────────────────────────────────────────────────────

fn do_show(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;
    let rev = op.branch.as_deref().unwrap_or("HEAD");
    let obj = repo.revparse_single(rev)?;
    let commit = obj.peel_to_commit()?;

    let author = commit.author();
    let time = commit.time();
    let ts = chrono::DateTime::from_timestamp(time.seconds(), 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| "?".to_string());

    let mut out = format!(
        "commit {}\nAuthor: {} <{}>\nDate:   {}\n\n    {}\n",
        commit.id(),
        author.name().unwrap_or("?"),
        author.email().unwrap_or("?"),
        ts,
        commit.message().unwrap_or("(no message)"),
    );

    // Show diff stats for this commit.
    if let Ok(parent) = commit.parent(0) {
        let a = parent.tree()?;
        let b = commit.tree()?;
        let diff = repo.diff_tree_to_tree(Some(&a), Some(&b), None)?;
        let stats = diff.stats()?;
        let _ = writeln!(
            out,
            " {} file(s) changed, {} insertion(s), {} deletion(s)",
            stats.files_changed(),
            stats.insertions(),
            stats.deletions(),
        );
    }
    Ok(out)
}

// ── clone ────────────────────────────────────────────────────────────────────

async fn do_clone(op: &GitOperation, policy: &GaitPolicy) -> Result<String> {
    let url = op.repo.trim();
    let target_dir = op
        .target_dir
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!(
            "clone requires 'target_dir' — specifies where to put the cloned repo. \
             Must be inside trusted_write_paths."
        ))?;

    // Resolve target_dir: if relative, resolve against workspace root.
    let target = if Path::new(target_dir).is_absolute() {
        PathBuf::from(target_dir)
    } else {
        policy.workspace_root.join(target_dir)
    };

    // SECURITY: trust check BEFORE creating any directories.
    // We canonicalize the existing portion of the path (its nearest existing
    // ancestor) to perform a reliable prefix check.
    let canonical_target = {
        // Walk up to find the nearest existing ancestor, canonicalize it,
        // then re-append the remaining components.
        let mut existing = target.clone();
        let mut suffix_parts: Vec<std::ffi::OsString> = Vec::new();
        while !existing.exists() {
            if let Some(name) = existing.file_name() {
                suffix_parts.push(name.to_os_string());
            } else {
                break;
            }
            existing = match existing.parent() {
                Some(p) => p.to_path_buf(),
                None => break,
            };
        }
        let canonical_base = std::fs::canonicalize(&existing)
            .map_err(|e| anyhow::anyhow!("cannot resolve clone target base '{}': {}", existing.display(), e))?;
        let mut result = canonical_base;
        for part in suffix_parts.into_iter().rev() {
            result.push(part);
        }
        result
    };

    assert_inside_trusted(&canonical_target, policy)?;

    // When allow_system_read is false, also verify local clone sources.
    if !url.starts_with("https://") && !url.starts_with("http://") && !url.starts_with("git@")
        && !policy.allow_system_read
    {
        let source = std::fs::canonicalize(url)
            .map_err(|e| anyhow::anyhow!("cannot resolve clone source '{}': {}", url, e))?;
        assert_inside_trusted(&source, policy)?;
    }

    // Now that the trust check passed, create parent directories.
    if let Some(parent) = canonical_target.parent() {
        std::fs::create_dir_all(parent)?;
    }

    info!(url = %url, target = %canonical_target.display(), "gait: cloning");

    // Use git CLI for clone — libgit2 clone with TLS/SSH auth is complex
    // and the CLI handles credential helpers and SSH agent natively.
    let output = tokio::process::Command::new("git")
        .args(["clone", "--progress", url])
        .arg(&canonical_target)
        .output()
        .await?;

    if output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!(
            "Cloned '{}' → '{}'\n{}",
            url,
            canonical_target.display(),
            stderr.trim()
        ))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git clone failed: {}", stderr.trim());
    }
}

// ── ls-remote ────────────────────────────────────────────────────────────────

async fn do_ls_remote(op: &GitOperation, policy: &GaitPolicy) -> Result<String> {
    let url = op.repo.trim();
    // ls-remote is read-only but hits the network.  For local paths, check
    // access according to policy.
    if !url.starts_with("https://") && !url.starts_with("http://") && !url.starts_with("git@") {
        let path = std::fs::canonicalize(url)?;
        if !policy.allow_system_read {
            assert_inside_trusted(&path, policy)?;
        }
    }

    let output = tokio::process::Command::new("git")
        .args(["ls-remote", "--heads", "--tags", url])
        .output()
        .await?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(format!("Refs for '{url}':\n{}", stdout.trim()))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git ls-remote failed: {}", stderr.trim());
    }
}

// ── pull ─────────────────────────────────────────────────────────────────────

async fn do_pull(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let mut args = vec!["pull"];
    if let Some(ref branch) = op.branch {
        args.push("origin");
        args.push(branch);
    }

    let output = tokio::process::Command::new("git")
        .args(&args)
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!("{}{}", stdout.trim(), if stderr.is_empty() { String::new() } else { format!("\n{}", stderr.trim()) }))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git pull failed: {}", stderr.trim());
    }
}

// ── push ─────────────────────────────────────────────────────────────────────

async fn do_push(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let mut args = vec!["push"];
    if op.force.unwrap_or(false) {
        args.push("--force-with-lease");
    }
    if let Some(ref branch) = op.branch {
        args.push("origin");
        args.push(branch);
    }

    let output = tokio::process::Command::new("git")
        .args(&args)
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!("Pushed successfully.\n{}", stderr.trim()))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git push failed: {}", stderr.trim());
    }
}

// ── fetch ────────────────────────────────────────────────────────────────────

async fn do_fetch(repo_path: &Path, _op: &GitOperation) -> Result<String> {
    let output = tokio::process::Command::new("git")
        .args(["fetch", "--all", "--prune"])
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!(
            "Fetched all remotes.\n{}",
            if stderr.is_empty() {
                "(no new data)"
            } else {
                stderr.trim()
            }
        ))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git fetch failed: {}", stderr.trim());
    }
}

// ── stash ────────────────────────────────────────────────────────────────────

fn do_stash(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let mut repo = open_repo(repo_path)?;
    let sig = repo.signature().unwrap_or_else(|_| {
        git2::Signature::now("Aigent", "aigent@localhost").expect("valid signature")
    });

    let msg = op.message.as_deref().unwrap_or("gait stash");
    let _oid = repo.stash_save(&sig, msg, None)?;
    Ok(format!("Stashed working directory: '{msg}'"))
}

// ── blame ────────────────────────────────────────────────────────────────────

async fn do_blame(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let file = op
        .paths
        .as_ref()
        .and_then(|p| p.first())
        .ok_or_else(|| anyhow::anyhow!("blame requires a file path in 'paths'"))?;

    let output = tokio::process::Command::new("git")
        .args(["blame", "--line-porcelain", file])
        .current_dir(repo_path)
        .output()
        .await?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Compact the porcelain output for the LLM.
        let compact = compact_blame(&stdout);
        Ok(compact)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git blame failed: {}", stderr.trim());
    }
}

/// Compact git blame porcelain output into a readable summary.
fn compact_blame(raw: &str) -> String {
    let mut out = String::from("Blame:\n");
    let mut lineno = 1u32;
    let mut current_author = String::new();
    for line in raw.lines() {
        if let Some(author) = line.strip_prefix("author ") {
            current_author = author.to_string();
        }
        if let Some(code) = line.strip_prefix('\t') {
            let _ = writeln!(out, "  {lineno:>4} ({current_author}) {code}");
            lineno += 1;
        }
        // Cap output at 4 KiB.
        if out.len() > 4096 {
            out.push_str("\n…[blame truncated]");
            break;
        }
    }
    out
}

// ── tag ──────────────────────────────────────────────────────────────────────

fn do_tag(repo_path: &Path, op: &GitOperation) -> Result<String> {
    let repo = open_repo(repo_path)?;

    if let Some(ref tag_name) = op.branch {
        // Create a lightweight tag at HEAD.
        let head = repo.head()?.peel(git2::ObjectType::Commit)?;
        let force = op.force.unwrap_or(false);
        repo.tag_lightweight(tag_name, &head, force)?;
        Ok(format!("Created tag '{tag_name}'."))
    } else {
        // List tags.
        let tags = repo.tag_names(None)?;
        let mut out = String::from("Tags:\n");
        for i in 0..tags.len() {
            if let Some(name) = tags.get(i) {
                let _ = writeln!(out, "  {name}");
            }
        }
        if tags.is_empty() {
            out.push_str("  (none)");
        }
        Ok(out)
    }
}

// ── Tool trait implementation ────────────────────────────────────────────────
//
// A `Tool` wrapper so `perform_gait` can be registered in the standard
// `ToolRegistry` alongside other built-in tools.

use aigent_tools::{Tool, ToolOutput, ToolParam, ToolSpec};
use async_trait::async_trait;
use std::collections::HashMap;

pub struct GaitTool {
    pub policy: GaitPolicy,
}

#[async_trait]
impl Tool for GaitTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "perform_gait".to_string(),
            description: "Preferred git tool — safe, powerful, and recommended for all git \
                          operations including self-updates. All writes (including clone) are \
                          restricted to trusted_write_paths only."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "action".to_string(),
                    description: "Git action: status, log, diff, commit, checkout, branch, \
                                  reset, clone, pull, push, fetch, ls-remote, show, blame, \
                                  tag, stash"
                        .to_string(),
                    required: true,
                },
                ToolParam {
                    name: "repo".to_string(),
                    description: "Target repository: \"workspace\", \"self\", an absolute path, \
                                  or a remote URL (for clone/ls-remote)"
                        .to_string(),
                    required: true,
                },
                ToolParam {
                    name: "target_dir".to_string(),
                    description: "Required for clone: destination directory (must be inside \
                                  trusted_write_paths)"
                        .to_string(),
                    required: false,
                },
                ToolParam {
                    name: "branch".to_string(),
                    description: "Branch name (for checkout, branch, pull, push, reset, tag, \
                                  or revision for show/log)"
                        .to_string(),
                    required: false,
                },
                ToolParam {
                    name: "message".to_string(),
                    description: "Commit/stash message".to_string(),
                    required: false,
                },
                ToolParam {
                    name: "paths".to_string(),
                    description: "Comma-separated file paths (for selective commit or blame)"
                        .to_string(),
                    required: false,
                },
                ToolParam {
                    name: "force".to_string(),
                    description: "Force operation (true/false) — for push (--force-with-lease), \
                                  reset (--hard), branch (overwrite)"
                        .to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let action = args
            .get("action")
            .ok_or_else(|| anyhow::anyhow!("missing required param: action"))?
            .clone();
        let repo = args
            .get("repo")
            .ok_or_else(|| anyhow::anyhow!("missing required param: repo"))?
            .clone();

        let paths = args.get("paths").map(|p| {
            p.split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>()
        });

        let op = GitOperation {
            action,
            repo,
            target_dir: args.get("target_dir").cloned(),
            branch: args.get("branch").cloned(),
            message: args.get("message").cloned(),
            paths,
            force: args.get("force").and_then(|v| v.parse().ok()),
        };

        match perform_gait(op, &self.policy).await {
            Ok(output) => Ok(ToolOutput {
                success: true,
                output,
            }),
            Err(err) => Ok(ToolOutput {
                success: false,
                output: format!("gait error: {err}"),
            }),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn test_policy(workspace: &Path) -> GaitPolicy {
        let canonical = fs::canonicalize(workspace).unwrap();
        GaitPolicy {
            trusted_write_paths: vec![canonical.clone()],
            trusted_repos: HashSet::new(),
            allow_system_read: true,
            workspace_root: canonical,
            self_repo_path: None,
        }
    }

    fn init_test_repo(path: &Path) -> git2::Repository {
        let repo = git2::Repository::init(path).unwrap();
        // Create an initial commit so HEAD exists.
        let sig = git2::Signature::now("Test", "test@test.com").unwrap();
        let tree_oid = {
            let mut index = repo.index().unwrap();
            index.write_tree().unwrap()
        };
        let tree = repo.find_tree(tree_oid).unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .unwrap();
        // Drop tree borrow before returning repo.
        drop(tree);
        repo
    }

    #[test]
    fn write_actions_classified_correctly() {
        assert!(is_write_action("commit"));
        assert!(is_write_action("clone"));
        assert!(is_write_action("push"));
        assert!(is_write_action("checkout"));
        assert!(!is_write_action("status"));
        assert!(!is_write_action("log"));
        assert!(!is_write_action("diff"));
        assert!(!is_write_action("blame"));
        assert!(!is_write_action("ls-remote"));
    }

    #[test]
    fn branch_list_is_not_mutating() {
        let op = GitOperation {
            action: "branch".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None, // list mode
            message: None,
            paths: None,
            force: None,
        };
        assert!(!is_mutating_op(&op));
    }

    #[test]
    fn branch_create_is_mutating() {
        let op = GitOperation {
            action: "branch".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("new-feature".to_string()),
            message: None,
            paths: None,
            force: None,
        };
        assert!(is_mutating_op(&op));
    }

    #[test]
    fn tag_list_is_not_mutating() {
        let op = GitOperation {
            action: "tag".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None, // list mode
            message: None,
            paths: None,
            force: None,
        };
        assert!(!is_mutating_op(&op));
    }

    #[test]
    fn tag_create_is_mutating() {
        let op = GitOperation {
            action: "tag".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("v1.0".to_string()),
            message: None,
            paths: None,
            force: None,
        };
        assert!(is_mutating_op(&op));
    }

    #[test]
    fn assert_inside_trusted_accepts_child() {
        let dir = TempDir::new().unwrap();
        let child = dir.path().join("sub");
        fs::create_dir_all(&child).unwrap();
        let policy = test_policy(dir.path());
        let canonical_child = fs::canonicalize(&child).unwrap();
        assert!(assert_inside_trusted(&canonical_child, &policy).is_ok());
    }

    #[test]
    fn assert_inside_trusted_rejects_outside() {
        let dir = TempDir::new().unwrap();
        let policy = test_policy(dir.path());
        // Use /etc which should not be inside a temp dir.
        let unrelated = PathBuf::from("/etc");
        if let Ok(canonical) = fs::canonicalize(&unrelated) {
            assert!(assert_inside_trusted(&canonical, &policy).is_err());
        }
    }

    #[tokio::test]
    async fn status_on_clean_repo() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let result = do_status(dir.path()).unwrap();
        assert!(result.contains("clean") || result.contains("nothing to commit"));
    }

    #[tokio::test]
    async fn commit_and_log() {
        let dir = TempDir::new().unwrap();
        let _repo = init_test_repo(dir.path());
        // Create a file to commit.
        fs::write(dir.path().join("test.txt"), "hello").unwrap();

        let policy = test_policy(dir.path());
        let op = GitOperation {
            action: "commit".to_string(),
            repo: dir.path().display().to_string(),
            target_dir: None,
            branch: None,
            message: Some("test commit".to_string()),
            paths: None,
            force: None,
        };
        let result = perform_gait(op, &policy).await.unwrap();
        assert!(result.contains("Committed"));
        assert!(result.contains("test commit"));

        // Log should show the commit.
        let log_op = GitOperation {
            action: "log".to_string(),
            repo: dir.path().display().to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let log = perform_gait(log_op, &policy).await.unwrap();
        assert!(log.contains("test commit"));

        // Verify the file is actually in the committed tree
        let repo = open_repo(dir.path()).unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let tree = head.tree().unwrap();
        assert!(
            tree.get_name("test.txt").is_some(),
            "committed tree must contain test.txt"
        );
    }

    #[tokio::test]
    async fn clone_rejected_outside_trusted() {
        let dir = TempDir::new().unwrap();
        let policy = test_policy(dir.path());
        let op = GitOperation {
            action: "clone".to_string(),
            repo: "https://example.com/test.git".to_string(),
            target_dir: Some("/etc/evil".to_string()),
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let result = perform_gait(op, &policy).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("outside") || err_msg.contains("trusted"));
    }

    /// Regression test: local clone source must be checked against trusted
    /// paths when allow_system_read is false. This was a real bug we fixed.
    #[tokio::test]
    async fn clone_local_source_rejected_when_system_read_disabled() {
        let source_dir = TempDir::new().unwrap();
        init_test_repo(source_dir.path());

        let workspace_dir = TempDir::new().unwrap();
        fs::create_dir_all(workspace_dir.path().join("target")).unwrap();

        let canonical_ws = fs::canonicalize(workspace_dir.path()).unwrap();
        let policy = GaitPolicy {
            trusted_write_paths: vec![canonical_ws.clone()],
            trusted_repos: HashSet::new(),
            allow_system_read: false, // <-- the critical flag
            workspace_root: canonical_ws.clone(),
            self_repo_path: None,
        };

        let op = GitOperation {
            action: "clone".to_string(),
            repo: source_dir.path().display().to_string(), // local path outside trusted
            target_dir: Some("target/cloned".to_string()),
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let result = perform_gait(op, &policy).await;
        assert!(
            result.is_err(),
            "local clone from untrusted path should fail when allow_system_read=false"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("outside") || err_msg.contains("trusted"),
            "error should mention trust boundary, got: {err_msg}"
        );
    }

    // ── diff ───────────────────────────────────────────────────────────────

    #[test]
    fn diff_on_clean_repo_shows_no_differences() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let op = GitOperation {
            action: "diff".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let result = do_diff(dir.path(), &op).unwrap();
        assert!(result.contains("No differences"));
    }

    #[test]
    fn diff_shows_changes_after_edit() {
        let dir = TempDir::new().unwrap();
        let repo = init_test_repo(dir.path());
        // Create and stage a file, then commit
        fs::write(dir.path().join("a.txt"), "hello").unwrap();
        {
            let mut idx = repo.index().unwrap();
            idx.add_path(Path::new("a.txt")).unwrap();
            idx.write().unwrap();
            let tree_oid = idx.write_tree().unwrap();
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("Test", "t@t").unwrap();
            let head = repo.head().unwrap().peel_to_commit().unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "add a", &tree, &[&head]).unwrap();
        }
        // Modify the file in working tree
        fs::write(dir.path().join("a.txt"), "world").unwrap();

        let op = GitOperation {
            action: "diff".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let result = do_diff(dir.path(), &op).unwrap();
        assert!(result.contains("1 file(s) changed"));
    }

    // ── branch ─────────────────────────────────────────────────────────────

    #[test]
    fn branch_list_shows_main() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let op = GitOperation {
            action: "branch".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let result = do_branch(dir.path(), &op).unwrap();
        // Should list branches; the default branch after init is usually "main" or "master"
        assert!(result.contains("Branches:"));
    }

    #[test]
    fn branch_create_and_list() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());

        // Create branch
        let create_op = GitOperation {
            action: "branch".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("feature-x".to_string()),
            message: None,
            paths: None,
            force: None,
        };
        let result = do_branch(dir.path(), &create_op).unwrap();
        assert!(result.contains("Created branch 'feature-x'"));

        // List should now show the new branch
        let list_op = GitOperation {
            action: "branch".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let list = do_branch(dir.path(), &list_op).unwrap();
        assert!(list.contains("feature-x"));
    }

    // ── tag ────────────────────────────────────────────────────────────────

    #[test]
    fn tag_list_empty() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let op = GitOperation {
            action: "tag".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let result = do_tag(dir.path(), &op).unwrap();
        assert!(result.contains("(none)"));
    }

    #[test]
    fn tag_create_and_list() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());

        // Create tag
        let create_op = GitOperation {
            action: "tag".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("v0.1.0".to_string()),
            message: None,
            paths: None,
            force: None,
        };
        let result = do_tag(dir.path(), &create_op).unwrap();
        assert!(result.contains("Created tag 'v0.1.0'"));

        // List should show it
        let list_op = GitOperation {
            action: "tag".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let list = do_tag(dir.path(), &list_op).unwrap();
        assert!(list.contains("v0.1.0"));
    }

    // ── show ───────────────────────────────────────────────────────────────

    #[test]
    fn show_head_commit() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let op = GitOperation {
            action: "show".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None, // defaults to HEAD
            message: None,
            paths: None,
            force: None,
        };
        let result = do_show(dir.path(), &op).unwrap();
        assert!(result.contains("commit "));
        assert!(result.contains("Initial commit"));
        assert!(result.contains("Author:"));
    }

    // ── reset ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn reset_mixed_to_previous_commit() {
        let dir = TempDir::new().unwrap();
        let _repo = init_test_repo(dir.path());
        // Create a second commit
        fs::write(dir.path().join("file.txt"), "content").unwrap();
        let policy = test_policy(dir.path());
        let commit_op = GitOperation {
            action: "commit".to_string(),
            repo: dir.path().display().to_string(),
            target_dir: None,
            branch: None,
            message: Some("second commit".to_string()),
            paths: None,
            force: None,
        };
        perform_gait(commit_op, &policy).await.unwrap();

        // Remember HEAD oid before reset
        let repo = open_repo(dir.path()).unwrap();
        let head_before = repo.head().unwrap().peel_to_commit().unwrap().id();
        let parent_oid = repo.head().unwrap().peel_to_commit().unwrap().parent(0).unwrap().id();
        drop(repo);

        // Reset to HEAD~1
        let reset_op = GitOperation {
            action: "reset".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("HEAD~1".to_string()),
            message: None,
            paths: None,
            force: None,
        };
        let result = do_reset(dir.path(), &reset_op).unwrap();
        assert!(result.contains("Reset (mixed)"));

        // Verify HEAD actually moved to the parent commit
        let repo = open_repo(dir.path()).unwrap();
        let head_after = repo.head().unwrap().peel_to_commit().unwrap().id();
        assert_ne!(head_after, head_before, "HEAD should have moved");
        assert_eq!(head_after, parent_oid, "HEAD should point to parent");
    }

    #[test]
    fn reset_hard_cleans_working_tree() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        // Create and commit a file, then modify it
        fs::write(dir.path().join("f.txt"), "original").unwrap();
        {
            let repo = open_repo(dir.path()).unwrap();
            let mut idx = repo.index().unwrap();
            idx.add_path(Path::new("f.txt")).unwrap();
            idx.write().unwrap();
            let tree_oid = idx.write_tree().unwrap();
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("Test", "t@t").unwrap();
            let head = repo.head().unwrap().peel_to_commit().unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "add f", &tree, &[&head]).unwrap();
        }
        fs::write(dir.path().join("f.txt"), "dirty").unwrap();

        let op = GitOperation {
            action: "reset".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("HEAD".to_string()),
            message: None,
            paths: None,
            force: Some(true), // hard reset
        };
        let result = do_reset(dir.path(), &op).unwrap();
        assert!(result.contains("Reset (hard)"));

        // Verify working tree was actually restored
        let contents = fs::read_to_string(dir.path().join("f.txt")).unwrap();
        assert_eq!(contents, "original", "hard reset should restore file contents");
    }

    // ── checkout ───────────────────────────────────────────────────────────

    #[test]
    fn checkout_new_branch() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        // First create the branch
        let create_op = GitOperation {
            action: "branch".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("dev".to_string()),
            message: None,
            paths: None,
            force: None,
        };
        do_branch(dir.path(), &create_op).unwrap();

        // Checkout it
        let checkout_op = GitOperation {
            action: "checkout".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: Some("dev".to_string()),
            message: None,
            paths: None,
            force: None,
        };
        let result = do_checkout(dir.path(), &checkout_op).unwrap();
        assert!(result.contains("Checked out 'dev'"));

        // Verify HEAD actually points to the new branch
        let repo = open_repo(dir.path()).unwrap();
        let head_ref = repo.head().unwrap();
        assert!(head_ref.is_branch(), "HEAD should be a branch reference");
        assert_eq!(
            head_ref.shorthand().unwrap(),
            "dev",
            "HEAD should point to 'dev' branch"
        );
    }

    #[test]
    fn checkout_requires_branch_name() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let op = GitOperation {
            action: "checkout".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        assert!(do_checkout(dir.path(), &op).is_err());
    }

    // ── stash ──────────────────────────────────────────────────────────────

    #[test]
    fn stash_with_changes() {
        let dir = TempDir::new().unwrap();
        let repo = init_test_repo(dir.path());
        // Create and stage a file first (stash needs at least index content)
        fs::write(dir.path().join("stash_me.txt"), "content").unwrap();
        {
            let mut idx = repo.index().unwrap();
            idx.add_path(Path::new("stash_me.txt")).unwrap();
            idx.write().unwrap();
        }

        let op = GitOperation {
            action: "stash".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: Some("wip save".to_string()),
            paths: None,
            force: None,
        };
        let result = do_stash(dir.path(), &op).unwrap();
        assert!(result.contains("Stashed"));
        assert!(result.contains("wip save"));

        // Verify index was actually cleaned by the stash
        let statuses = repo.statuses(Some(
            git2::StatusOptions::new()
                .include_untracked(true)
                .recurse_untracked_dirs(true),
        )).unwrap();
        assert!(
            statuses.is_empty(),
            "working tree should be clean after stash, but has {} entries",
            statuses.len()
        );
    }

    // ── unsupported action ─────────────────────────────────────────────────

    #[tokio::test]
    async fn unsupported_action_returns_error() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let policy = test_policy(dir.path());
        let op = GitOperation {
            action: "rebase".to_string(),
            repo: dir.path().display().to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let result = perform_gait(op, &policy).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported"));
    }

    // ── GaitTool::spec ─────────────────────────────────────────────────────

    #[test]
    fn gait_tool_spec_has_correct_name_and_params() {
        let dir = TempDir::new().unwrap();
        let policy = test_policy(dir.path());
        let tool = GaitTool { policy };
        let spec = tool.spec();
        assert_eq!(spec.name, "perform_gait");
        assert!(spec.params.len() >= 7);
        let param_names: Vec<&str> = spec.params.iter().map(|p| p.name.as_str()).collect();
        assert!(param_names.contains(&"action"));
        assert!(param_names.contains(&"repo"));
        assert!(param_names.contains(&"target_dir"));
        assert!(param_names.contains(&"branch"));
        assert!(param_names.contains(&"message"));
        assert!(param_names.contains(&"paths"));
        assert!(param_names.contains(&"force"));
    }

    // ── GaitTool::run ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn gait_tool_run_status() {
        let dir = TempDir::new().unwrap();
        init_test_repo(dir.path());
        let policy = test_policy(dir.path());
        let tool = GaitTool { policy };
        let mut args = HashMap::new();
        args.insert("action".to_string(), "status".to_string());
        args.insert("repo".to_string(), dir.path().display().to_string());
        let output = tool.run(&args).await.unwrap();
        assert!(output.success);
        assert!(output.output.contains("clean") || output.output.contains("nothing"));
    }

    #[tokio::test]
    async fn gait_tool_run_missing_action() {
        let dir = TempDir::new().unwrap();
        let policy = test_policy(dir.path());
        let tool = GaitTool { policy };
        let args = HashMap::new(); // missing "action"
        let result = tool.run(&args).await;
        assert!(result.is_err());
    }

    // ── GaitPolicy::from_config ────────────────────────────────────────────

    #[test]
    fn gait_policy_from_default_config() {
        let mut config = aigent_config::AppConfig::default();
        // Point workspace to a temp dir so canonicalize works
        let dir = TempDir::new().unwrap();
        config.agent.workspace_path = dir.path().display().to_string();

        let policy = GaitPolicy::from_config(&config);
        assert!(policy.allow_system_read);
        assert!(policy.trusted_repos.contains("https://github.com/danielmriley/aigent"));
        // Workspace root should be in trusted write paths
        let canonical_ws = fs::canonicalize(dir.path()).unwrap();
        assert!(policy.trusted_write_paths.contains(&canonical_ws));
    }

    #[test]
    fn gait_policy_safer_mode_disables_system_read() {
        let mut config = aigent_config::AppConfig::default();
        let dir = TempDir::new().unwrap();
        config.agent.workspace_path = dir.path().display().to_string();
        config.tools.approval_mode = aigent_config::ApprovalMode::Safer;
        config.git.allow_system_read = true; // would be true, but safer overrides

        let policy = GaitPolicy::from_config(&config);
        assert!(!policy.allow_system_read);
    }

    // ── is_mutating_call helper ────────────────────────────────────────────

    #[test]
    fn is_mutating_call_branch_with_name() {
        let mut args = HashMap::new();
        args.insert("branch".to_string(), "feature".to_string());
        assert!(is_mutating_call("branch", &args));
    }

    #[test]
    fn is_mutating_call_branch_without_name() {
        let args = HashMap::new();
        assert!(!is_mutating_call("branch", &args));
    }

    #[test]
    fn is_mutating_call_branch_empty_name() {
        let mut args = HashMap::new();
        args.insert("branch".to_string(), String::new());
        assert!(!is_mutating_call("branch", &args));
    }

    #[test]
    fn is_mutating_call_regular_write_action() {
        let args = HashMap::new();
        assert!(is_mutating_call("commit", &args));
        assert!(is_mutating_call("push", &args));
        assert!(!is_mutating_call("log", &args));
    }

    // ── authorise ──────────────────────────────────────────────────────────

    #[test]
    fn authorise_read_with_system_read_disabled() {
        let dir = TempDir::new().unwrap();
        let mut policy = test_policy(dir.path());
        policy.allow_system_read = false;

        // A read op on a path outside trusted should fail
        let op = GitOperation {
            action: "log".to_string(),
            repo: "workspace".to_string(),
            target_dir: None,
            branch: None,
            message: None,
            paths: None,
            force: None,
        };
        let foreign_path = PathBuf::from("/etc");
        if let Ok(canonical) = fs::canonicalize(&foreign_path) {
            assert!(authorise(&op, &canonical, &policy).is_err());
        }

        // Same read op inside trusted should work
        let workspace = fs::canonicalize(dir.path()).unwrap();
        assert!(authorise(&op, &workspace, &policy).is_ok());
    }

    // ── compact_blame ──────────────────────────────────────────────────────

    #[test]
    fn compact_blame_formats_correctly() {
        let raw = "abc123\nauthor Test User\n\tline one content\ndef456\nauthor Another\n\tline two content\n";
        let result = compact_blame(raw);
        assert!(result.contains("1") && result.contains("Test User") && result.contains("line one content"));
        assert!(result.contains("2") && result.contains("Another") && result.contains("line two content"));
    }
}
