use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::{Tool, ToolOutput, ToolParam, ToolSpec};

// ── read_file ────────────────────────────────────────────────────────────────

pub struct ReadFileTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".to_string(),
            description: "Read the contents of a file within the workspace.".to_string(),
            params: vec![
                ToolParam {
                    name: "path".to_string(),
                    description: "Relative path from workspace root".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "max_bytes".to_string(),
                    description: "Maximum bytes to read (default: 65536)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel_path = args
            .get("path")
            .ok_or_else(|| anyhow::anyhow!("missing required param: path"))?;

        let full = self.workspace_root.join(rel_path);
        let canonical = full
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("cannot resolve path '{}': {}", rel_path, e))?;

        let root_canonical = self.workspace_root.canonicalize()?;
        if !canonical.starts_with(&root_canonical) {
            bail!(
                "path escapes workspace boundary: {}",
                canonical.display()
            );
        }

        let max_bytes: usize = args
            .get("max_bytes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(65536);

        let content = std::fs::read_to_string(&canonical)?;
        let truncated = if content.len() > max_bytes {
            format!("{}…[truncated at {} bytes]", &content[..max_bytes], max_bytes)
        } else {
            content
        };

        Ok(ToolOutput {
            success: true,
            output: truncated,
        })
    }
}

// ── write_file ───────────────────────────────────────────────────────────────

pub struct WriteFileTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".to_string(),
            description: "Write content to a file within the workspace (creates or overwrites)."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "path".to_string(),
                    description: "Relative path from workspace root".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "content".to_string(),
                    description: "File content to write".to_string(),
                    required: true,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let rel_path = args
            .get("path")
            .ok_or_else(|| anyhow::anyhow!("missing required param: path"))?;
        let content = args
            .get("content")
            .ok_or_else(|| anyhow::anyhow!("missing required param: content"))?;

        let full = self.workspace_root.join(rel_path);

        // Prevent escaping workspace even before file exists (can't canonicalize yet)
        let root_canonical = self.workspace_root.canonicalize()?;
        if let Ok(canonical) = full.canonicalize() {
            if !canonical.starts_with(&root_canonical) {
                bail!(
                    "path escapes workspace boundary: {}",
                    canonical.display()
                );
            }
        } else {
            // File doesn't exist yet; check parent
            let parent = full
                .parent()
                .ok_or_else(|| anyhow::anyhow!("invalid path"))?;
            std::fs::create_dir_all(parent)?;
            let parent_canonical = parent.canonicalize()?;
            if !parent_canonical.starts_with(&root_canonical) {
                bail!(
                    "parent escapes workspace boundary: {}",
                    parent_canonical.display()
                );
            }
        }

        std::fs::write(&full, content)?;
        Ok(ToolOutput {
            success: true,
            output: format!("wrote {} bytes to {}", content.len(), rel_path),
        })
    }
}

// ── run_shell ────────────────────────────────────────────────────────────────

pub struct RunShellTool {
    pub workspace_root: PathBuf,
}

#[async_trait]
impl Tool for RunShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_shell".to_string(),
            description: "Execute a shell command within the workspace directory.".to_string(),
            params: vec![
                ToolParam {
                    name: "command".to_string(),
                    description: "Shell command to execute".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "timeout_secs".to_string(),
                    description: "Max execution time in seconds (default: 30)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let command = args
            .get("command")
            .ok_or_else(|| anyhow::anyhow!("missing required param: command"))?;
        let timeout_secs: u64 = args
            .get("timeout_secs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&self.workspace_root)
                .output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("command timed out after {}s", timeout_secs))??;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else {
            format!("{stdout}\n[stderr] {stderr}")
        };

        // Truncate output to prevent context explosion
        let max_output = 32768;
        let result = if combined.len() > max_output {
            format!(
                "{}…[truncated at {} bytes]",
                &combined[..max_output],
                max_output
            )
        } else {
            combined
        };

        Ok(ToolOutput {
            success: output.status.success(),
            output: result,
        })
    }
}

// ── calendar_add_event ───────────────────────────────────────────────────────

/// Appends an event object to `{data_dir}/calendar.json` (a JSON array).
/// Creates the file if it does not exist.
pub struct CalendarAddEventTool {
    pub data_dir: PathBuf,
}

#[async_trait]
impl Tool for CalendarAddEventTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "calendar_add_event".to_string(),
            description: "Add an event to the agent's local calendar store.".to_string(),
            params: vec![
                ToolParam {
                    name: "title".to_string(),
                    description: "Event title".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "date".to_string(),
                    description: "Event date (natural language or ISO-8601)".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "time".to_string(),
                    description: "Event time (e.g. '14:00' or '2pm')".to_string(),
                    required: false,
                },
                ToolParam {
                    name: "description".to_string(),
                    description: "Optional description or notes".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let title = args
            .get("title")
            .ok_or_else(|| anyhow::anyhow!("missing required param: title"))?;
        let date = args
            .get("date")
            .ok_or_else(|| anyhow::anyhow!("missing required param: date"))?;

        std::fs::create_dir_all(&self.data_dir)?;
        let calendar_path = self.data_dir.join("calendar.json");

        // Load existing events array (or start fresh).
        let mut events: Vec<serde_json::Value> = if calendar_path.exists() {
            let raw = std::fs::read_to_string(&calendar_path)?;
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            Vec::new()
        };

        let event = serde_json::json!({
            "title": title,
            "date": date,
            "time": args.get("time").cloned().unwrap_or_default(),
            "description": args.get("description").cloned().unwrap_or_default(),
            "added_at": chrono::Utc::now().to_rfc3339(),
        });
        events.push(event);

        let rendered = serde_json::to_string_pretty(&events)?;
        std::fs::write(&calendar_path, rendered)?;

        Ok(ToolOutput {
            success: true,
            output: format!("event '{}' added for {}", title, date),
        })
    }
}

// ── web_search ───────────────────────────────────────────────────────────────

/// Queries the DuckDuckGo Instant Answer API (no key required) and returns
/// the abstract text plus the top related topics.
pub struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_search".to_string(),
            description: "Search the web using DuckDuckGo Instant Answers (no API key required).".to_string(),
            params: vec![
                ToolParam {
                    name: "query".to_string(),
                    description: "Search query string".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "max_results".to_string(),
                    description: "Maximum related topics to include (default: 5)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let query = args
            .get("query")
            .ok_or_else(|| anyhow::anyhow!("missing required param: query"))?;
        let max_results: usize = args
            .get("max_results")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent("aigent/0.1 (https://github.com/danielmriley/aigent)")
            .build()?;

        let resp = client
            .get("https://api.duckduckgo.com/")
            .query(&[
                ("q", query.as_str()),
                ("format", "json"),
                ("no_html", "1"),
                ("skip_disambig", "1"),
            ])
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;

        let abstract_text = json["AbstractText"].as_str().unwrap_or("").trim().to_string();
        let abstract_source = json["AbstractSource"].as_str().unwrap_or("").trim().to_string();

        let mut parts: Vec<String> = Vec::new();
        if !abstract_text.is_empty() {
            if abstract_source.is_empty() {
                parts.push(abstract_text);
            } else {
                parts.push(format!("{abstract_text} (source: {abstract_source})"));
            }
        }

        if let Some(topics) = json["RelatedTopics"].as_array() {
            for topic in topics.iter().take(max_results) {
                let text = topic["Text"].as_str().unwrap_or("").trim();
                if !text.is_empty() {
                    parts.push(format!("• {text}"));
                }
            }
        }

        if parts.is_empty() {
            Ok(ToolOutput {
                success: true,
                output: format!("No instant-answer results found for: {query}"),
            })
        } else {
            Ok(ToolOutput {
                success: true,
                output: parts.join("\n"),
            })
        }
    }
}

// ── draft_email ──────────────────────────────────────────────────────────────

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
                },
                ToolParam {
                    name: "subject".to_string(),
                    description: "Email subject line".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "body".to_string(),
                    description: "Email body text".to_string(),
                    required: true,
                },
            ],
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

// ── remind_me ────────────────────────────────────────────────────────────────

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
                },
                ToolParam {
                    name: "when".to_string(),
                    description: "When to surface the reminder (natural language, optional)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let text = args
            .get("text")
            .ok_or_else(|| anyhow::anyhow!("missing required param: text"))?;

        std::fs::create_dir_all(&self.data_dir)?;
        let reminders_path = self.data_dir.join("reminders.json");

        let mut reminders: Vec<serde_json::Value> = if reminders_path.exists() {
            let raw = std::fs::read_to_string(&reminders_path)?;
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            Vec::new()
        };

        let reminder = serde_json::json!({
            "text": text,
            "when": args.get("when").cloned().unwrap_or_default(),
            "added_at": chrono::Utc::now().to_rfc3339(),
            "surfaced": false,
        });
        reminders.push(reminder);

        let rendered = serde_json::to_string_pretty(&reminders)?;
        std::fs::write(&reminders_path, rendered)?;

        let when_note = args.get("when").filter(|s| !s.is_empty())
            .map(|w| format!(" (when: {w})"))
            .unwrap_or_default();
        Ok(ToolOutput {
            success: true,
            output: format!("reminder added: '{text}'{when_note}"),
        })
    }
}
