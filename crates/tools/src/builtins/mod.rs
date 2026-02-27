//! Built-in tool implementations.

mod fs;
mod shell;
mod web;
mod calendar;
mod web_browse;

pub use fs::{ReadFileTool, WriteFileTool};
pub use shell::RunShellTool;
pub use web::{WebSearchTool, FetchPageTool};
pub use calendar::CalendarAddEventTool;
pub use web_browse::WebBrowseTool;

use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use fs2::FileExt;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};

/// Saves an email draft to `{data_dir}/drafts/` as a plain-text file.
pub struct DraftEmailTool {
    pub data_dir: PathBuf,
}

#[async_trait]
impl Tool for DraftEmailTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "draft_email".to_string(),
            description: "Save an email draft to the agent's drafts folder.".to_string(),
            params: vec![
                ToolParam {
                    name: "to".to_string(),
                    description: "Recipient email address or name".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "subject".to_string(),
                    description: "Email subject line".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "body".to_string(),
                    description: "Email body text".to_string(),
                    required: true,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Medium,
                read_only: false,
                group: "email".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let to = args
            .get("to")
            .ok_or_else(|| anyhow::anyhow!("missing required param: to"))?;
        let subject = args
            .get("subject")
            .ok_or_else(|| anyhow::anyhow!("missing required param: subject"))?;
        let body = args
            .get("body")
            .ok_or_else(|| anyhow::anyhow!("missing required param: body"))?;

        let drafts_dir = self.data_dir.join("drafts");
        std::fs::create_dir_all(&drafts_dir)?;

        // Build a filesystem-safe filename from timestamp + subject.
        let safe_subject: String = subject
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .take(40)
            .collect();
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("{timestamp}_{safe_subject}.txt");
        let draft_path = drafts_dir.join(&filename);

        let content = format!(
            "To: {to}\nSubject: {subject}\nDate: {}\n\n{body}",
            chrono::Utc::now().to_rfc2822()
        );
        std::fs::write(&draft_path, &content)?;

        Ok(ToolOutput {
            success: true,
            output: format!(
                "draft saved to .aigent/drafts/{} ({} bytes)",
                filename,
                content.len()
            ),
        })
    }
}

/// Appends a reminder to `{data_dir}/reminders.json` (a JSON array).
/// The proactive background task can read this file to surface reminders.
pub struct RemindMeTool {
    pub data_dir: PathBuf,
}

#[async_trait]
impl Tool for RemindMeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "remind_me".to_string(),
            description: "Add a reminder that the agent will surface proactively.".to_string(),
            params: vec![
                ToolParam {
                    name: "text".to_string(),
                    description: "Reminder text".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "when".to_string(),
                    description: "When to surface the reminder (natural language, optional)".to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: false,
                group: "calendar".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let text = args
            .get("text")
            .ok_or_else(|| anyhow::anyhow!("missing required param: text"))?;

        std::fs::create_dir_all(&self.data_dir)?;
        let reminders_path = self.data_dir.join("reminders.json");

        // Open (or create) and lock the file to prevent concurrent corruption.
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&reminders_path)?;
        file.lock_exclusive()?;

        let mut raw = String::new();
        file.read_to_string(&mut raw)?;
        let mut reminders: Vec<serde_json::Value> =
            if raw.trim().is_empty() { Vec::new() }
            else { serde_json::from_str(&raw).unwrap_or_default() };

        let reminder = serde_json::json!({
            "text": text,
            "when": args.get("when").cloned().unwrap_or_default(),
            "added_at": chrono::Utc::now().to_rfc3339(),
            "surfaced": false,
        });
        reminders.push(reminder);

        let rendered = serde_json::to_string_pretty(&reminders)?;
        file.set_len(0)?;
        file.seek(std::io::SeekFrom::Start(0))?;
        file.write_all(rendered.as_bytes())?;
        file.unlock()?;

        let when_note = args.get("when").filter(|s| !s.is_empty())
            .map(|w| format!(" (when: {w})"))
            .unwrap_or_default();
        Ok(ToolOutput {
            success: true,
            output: format!("reminder added: '{text}'{when_note}"),
        })
    }
}

/// Reverts the most recent commit in the workspace using `git revert HEAD`.
///
/// Safe to call after any `write_file` or `run_shell` auto-commit to undo
/// an accidental change.  Requires git to be installed and the workspace to
/// be a git repository.
pub struct GitRollbackTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for GitRollbackTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git_rollback".to_string(),
            description: "Revert the last automated git commit in the workspace (undo the most recent write_file or run_shell change). Requires git.".to_string(),
            params: vec![],
            metadata: ToolMetadata {
                security_level: SecurityLevel::High,
                read_only: false,
                group: "git".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, _args: &HashMap<String, String>) -> Result<ToolOutput> {
        if !self.workspace_root.join(".git").exists() {
            return Ok(ToolOutput {
                success: false,
                output: "workspace is not a git repository; cannot roll back".to_string(),
            });
        }

        let out = tokio::process::Command::new("git")
            .args(["revert", "HEAD", "--no-edit"])
            .env("GIT_AUTHOR_NAME", "Aigent")
            .env("GIT_AUTHOR_EMAIL", "aigent@localhost")
            .env("GIT_COMMITTER_NAME", "Aigent")
            .env("GIT_COMMITTER_EMAIL", "aigent@localhost")
            .current_dir(&self.workspace_root)
            .output()
            .await?;

        if out.status.success() {
            let msg = String::from_utf8_lossy(&out.stdout).trim().to_string();
            Ok(ToolOutput {
                success: true,
                output: if msg.is_empty() {
                    "Last commit reverted successfully.".to_string()
                } else {
                    msg
                },
            })
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            Ok(ToolOutput {
                success: false,
                output: format!("git revert failed: {stderr}"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    
        use crate::builtins::fs::truncate_byte_boundary;
        use crate::builtins::web::html_to_text;

    // ── truncate_byte_boundary ──────────────────────────────────────────────

    #[test]
    fn truncate_within_ascii() {
        assert_eq!(truncate_byte_boundary("abcdef", 3), 3);
    }

    #[test]
    fn truncate_beyond_string_len() {
        assert_eq!(truncate_byte_boundary("abc", 100), 3);
    }

    #[test]
    fn truncate_at_zero() {
        assert_eq!(truncate_byte_boundary("anything", 0), 0);
    }

    #[test]
    fn truncate_multibyte_char_boundary() {
        // "café" = c(1) a(1) f(1) é(2) = 5 bytes
        let s = "café";
        // max=4 lands inside the 2-byte é; should back up to 3.
        assert_eq!(truncate_byte_boundary(s, 4), 3);
        // max=5 lands at end.
        assert_eq!(truncate_byte_boundary(s, 5), 5);
    }

    #[test]
    fn truncate_emoji_boundary() {
        // "hi🎉" = h(1) i(1) 🎉(4) = 6 bytes
        let s = "hi🎉";
        for mid in 3..6 {
            // All should back up to byte 2 (after 'i').
            assert_eq!(truncate_byte_boundary(s, mid), 2, "mid={mid}");
        }
        assert_eq!(truncate_byte_boundary(s, 6), 6);
    }

    #[test]
    fn truncate_empty_string() {
        assert_eq!(truncate_byte_boundary("", 10), 0);
    }

    // ── html_to_text ────────────────────────────────────────────────────────

    #[test]
    fn html_strips_tags() {
        let out = html_to_text("<p>hello</p>", 1000);
        assert!(out.contains("hello"), "got: {out}");
        assert!(!out.contains("<p>"));
    }

    #[test]
    fn html_strips_script_blocks() {
        let out = html_to_text(
            "<p>before</p><script>alert('xss');</script><p>after</p>",
            1000,
        );
        assert!(out.contains("before"));
        assert!(out.contains("after"));
        assert!(!out.contains("alert"));
    }

    #[test]
    fn html_strips_style_blocks() {
        let out = html_to_text(
            "<style>body{color:red}</style><p>text</p>",
            1000,
        );
        assert!(out.contains("text"));
        assert!(!out.contains("color:red"));
    }

    #[test]
    fn html_decodes_entities() {
        let out = html_to_text("&amp; &lt; &gt; &quot; &#39; &nbsp;", 1000);
        assert!(out.contains("&"), "got: {out}");
        assert!(out.contains("<"), "got: {out}");
        assert!(out.contains(">"), "got: {out}");
    }

    #[test]
    fn html_collapses_whitespace() {
        let out = html_to_text("<p>  lots   of   spaces  </p>", 1000);
        // Should not contain runs of multiple spaces.
        assert!(!out.contains("  "), "got: {out}");
    }

    #[test]
    fn html_respects_max_chars() {
        let big = "<p>".to_owned() + &"a".repeat(500) + "</p>";
        let out = html_to_text(&big, 100);
        // Output should be ≤ 100 chars + trailing ellipsis.
        assert!(out.len() <= 104, "len={}: {}", out.len(), out);
    }

    #[test]
    fn html_handles_non_ascii_content() {
        let out = html_to_text("<p>café résumé naïve</p>", 1000);
        assert!(out.contains("café"), "got: {out}");
        assert!(out.contains("résumé"), "got: {out}");
        assert!(out.contains("naïve"), "got: {out}");
    }

    #[test]
    fn html_handles_cjk_content() {
        let out = html_to_text("<div>日本語テスト</div>", 1000);
        assert!(out.contains("日本語テスト"), "got: {out}");
    }

    #[test]
    fn html_handles_emoji_content() {
        let out = html_to_text("<p>hello 🌍🎉</p>", 1000);
        assert!(out.contains("🌍"), "got: {out}");
        assert!(out.contains("🎉"), "got: {out}");
    }

    #[test]
    fn html_empty_input() {
        let out = html_to_text("", 1000);
        assert!(out.is_empty());
    }

    #[test]
    fn html_plain_text_passthrough() {
        let out = html_to_text("just plain text", 1000);
        assert_eq!(out, "just plain text");
    }
}
