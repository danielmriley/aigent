//! **gait** — Git Agent Integration Tool.
//!
//! Safe, in-process git operations using libgit2.  Designed to be the
//! *preferred* git interface for the agent.

pub mod types;
mod read;
mod write;

use std::path::Path;

use anyhow::{Result, bail};
use tracing::info;
use async_trait::async_trait;
use aigent_tools::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata};
use std::collections::HashMap;

pub use types::{GitOperation, GaitPolicy, is_write_action, is_mutating_op, is_mutating_call};

pub(super) fn open_repo(path: &Path) -> Result<git2::Repository> {
    git2::Repository::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open git repository at '{}': {}", path.display(), e))
}


/// Execute a git operation.  Returns a rich, formatted string suitable for
/// LLM reflection on success, or an error on failure.
pub async fn perform_gait(op: GitOperation, policy: &GaitPolicy) -> Result<String> {
    let action = op.action.to_lowercase();
    info!(action = %action, repo = %op.repo, "gait: performing git operation");

    match action.as_str() {
        "clone" => write::do_clone(&op, policy).await,
        "ls-remote" => read::do_ls_remote(&op, policy).await,
        // All other actions need a local repo path
        _ => {
            let repo_path = types::resolve_repo_path(&op, policy).await?;
            types::authorise(&op, &repo_path, policy)?;
            match action.as_str() {
                // Network operations — already async (use git CLI under the hood).
                "pull" => write::do_pull(&repo_path, &op).await,
                "push" => write::do_push(&repo_path, &op).await,
                "fetch" => read::do_fetch(&repo_path, &op).await,
                "blame" => read::do_blame(&repo_path, &op).await,
                // libgit2 operations — synchronous and potentially CPU/IO-heavy.
                // Offload to a blocking thread to avoid stalling the executor.
                _ => {
                    let repo_path = repo_path.clone();
                    let op = op.clone();
                    let action = action.clone();
                    tokio::task::spawn_blocking(move || {
                        match action.as_str() {
                            "status" => read::do_status(&repo_path),
                            "log" => read::do_log(&repo_path, &op),
                            "diff" => read::do_diff(&repo_path, &op),
                            "commit" => write::do_commit_sync(&repo_path, &op),
                            "checkout" => write::do_checkout(&repo_path, &op),
                            "branch" => write::do_branch(&repo_path, &op),
                            "reset" => write::do_reset(&repo_path, &op),
                            "show" => read::do_show(&repo_path, &op),
                            "stash" => write::do_stash(&repo_path, &op),
                            "tag" => write::do_tag(&repo_path, &op),
                            other => bail!("unsupported gait action: '{other}'"),
                        }
                    }).await?
                }
            }
        }
    }
}


pub struct GaitTool {
    pub policy: GaitPolicy,
}

#[async_trait]
impl Tool for GaitTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "perform_gait".to_string(),
            description: "Preferred git tool — safe, powerful, and recommended for all git \
                          operations. Use repo=\"workspace\" for the sandboxed workspace \
                          where you create and edit files on behalf of the user. Use \
                          repo=\"self\" to READ the Aigent source code (read-only — \
                          writes are blocked). The workspace has its own git repository \
                          (auto-initialised); never confuse it with the Aigent project \
                          repo. All writes are restricted to trusted_write_paths only."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "action".to_string(),
                    description: "Git action: status, log, diff, commit, checkout, branch, \
                                  reset, clone, pull, push, fetch, ls-remote, show, blame, \
                                  tag, stash"
                        .to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "repo".to_string(),
                    description: "Target repository. \"workspace\" = the sandboxed directory \
                                  where you work on user tasks (file edits, projects). \
                                  \"self\" = the Aigent source code (read-only: status, log, \
                                  diff, show, blame only). You may also pass an absolute \
                                  path or a remote URL (clone/ls-remote only)."
                        .to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "target_dir".to_string(),
                    description: "Required for clone: destination directory (must be inside \
                                  trusted_write_paths)"
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "branch".to_string(),
                    description: "Branch name (for checkout, branch, pull, push, reset, tag, \
                                  or revision for show/log)"
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "message".to_string(),
                    description: "Commit/stash message".to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "paths".to_string(),
                    description: "Comma-separated file paths (for selective commit or blame)"
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "force".to_string(),
                    description: "Force operation (true/false) — for push (--force-with-lease), \
                                  reset (--hard), branch (overwrite)"
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata::default(),
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
    use super::types::{assert_inside_trusted, authorise};
    use super::read::compact_blame;
    use super::read::{do_status, do_diff, do_show};
    use super::write::{do_checkout, do_branch, do_reset, do_stash, do_tag};
    use std::collections::HashSet;
    use std::path::PathBuf;
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

    #[tokio::test]
    async fn gait_policy_from_default_config() {
        let mut config = aigent_config::AppConfig::default();
        // Point workspace to a temp dir so canonicalize works
        let dir = TempDir::new().unwrap();
        config.agent.workspace_path = dir.path().display().to_string();

        let policy = GaitPolicy::from_config(&config).await;
        assert!(policy.allow_system_read);
        assert!(policy.trusted_repos.contains("https://github.com/danielmriley/aigent"));
        // Workspace root should be in trusted write paths
        let canonical_ws = fs::canonicalize(dir.path()).unwrap();
        assert!(policy.trusted_write_paths.contains(&canonical_ws));
    }

    #[tokio::test]
    async fn gait_policy_safer_mode_disables_system_read() {
        let mut config = aigent_config::AppConfig::default();
        let dir = TempDir::new().unwrap();
        config.agent.workspace_path = dir.path().display().to_string();
        config.tools.approval_mode = aigent_config::ApprovalMode::Safer;
        config.git.allow_system_read = true; // would be true, but safer overrides

        let policy = GaitPolicy::from_config(&config).await;
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
