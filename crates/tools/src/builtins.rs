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

/// Searches the web and returns results.
///
/// When `brave_api_key` is set (or the `BRAVE_API_KEY` env var is non-empty)
/// the [Brave Search API](https://api.search.brave.com/app/documentation/web-search)
/// is used, providing higher-quality results.  Otherwise the tool falls back
/// to the DuckDuckGo Instant Answers API (no key required).
pub struct WebSearchTool {
    /// Optional Brave Search API key.  Takes precedence over the env var
    /// when both are set.  Set to `None` to always use DuckDuckGo.
    pub brave_api_key: Option<String>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_search".to_string(),
            description: "Search the web (Brave API when configured, DuckDuckGo otherwise).".to_string(),
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

        // Resolve the Brave API key: explicit field > env var > fallback to DDG.
        let brave_key: Option<String> = self
            .brave_api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| std::env::var("BRAVE_API_KEY").ok().filter(|k| !k.trim().is_empty()));

        if let Some(ref key) = brave_key {
            self.search_brave(query, max_results, key).await
        } else {
            self.search_duckduckgo(query, max_results).await
        }
    }
}

impl WebSearchTool {
    async fn search_brave(
        &self,
        query: &str,
        max_results: usize,
        api_key: &str,
    ) -> Result<ToolOutput> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent("aigent/0.1 (https://github.com/danielmriley/aigent)")
            .build()?;

        let resp = client
            .get("https://api.search.brave.com/res/v1/web/search")
            .query(&[("q", query), ("count", &max_results.to_string())])
            .header("Accept", "application/json")
            .header("X-Subscription-Token", api_key)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Brave Search API error {}: {}", status, body);
        }

        let json: serde_json::Value = resp.json().await?;

        let mut parts: Vec<String> = Vec::new();
        let mut top_url: Option<String> = None;
        if let Some(results) = json["web"]["results"].as_array() {
            for item in results.iter().take(max_results) {
                let title = item["title"].as_str().unwrap_or("").trim();
                let url = item["url"].as_str().unwrap_or("").trim();
                let desc = item["description"].as_str().unwrap_or("").trim();
                if !title.is_empty() {
                    if top_url.is_none() && !url.is_empty() {
                        top_url = Some(url.to_string());
                    }
                    parts.push(format!("{title}\n  {url}\n  {desc}"));
                }
            }
        }

        // Fetch the top result page to extract actual content (not just
        // search snippets).  This prevents the LLM from hallucinating data
        // that wasn't in the search description (e.g. stock prices, scores).
        if let Some(url) = top_url {
            if let Some(excerpt) = fetch_page_excerpt(&client, &url, 2000).await {
                parts.push(format!("\n--- Page content from {url} ---\n{excerpt}"));
            }
        }

        if parts.is_empty() {
            Ok(ToolOutput {
                success: true,
                output: format!("No Brave Search results for: {query}"),
            })
        } else {
            Ok(ToolOutput {
                success: true,
                output: parts.join("\n\n"),
            })
        }
    }

    async fn search_duckduckgo(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<ToolOutput> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent("aigent/0.1 (https://github.com/danielmriley/aigent)")
            .build()?;

        let resp = client
            .get("https://api.duckduckgo.com/")
            .query(&[
                ("q", query),
                ("format", "json"),
                ("no_html", "1"),
                ("skip_disambig", "1"),
            ])
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;

        let abstract_text = json["AbstractText"].as_str().unwrap_or("").trim().to_string();
        let abstract_source = json["AbstractSource"].as_str().unwrap_or("").trim().to_string();
        let abstract_url = json["AbstractURL"].as_str().unwrap_or("").trim().to_string();

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

        // Fetch the abstract source page for real content when available.
        if !abstract_url.is_empty() {
            if let Some(excerpt) = fetch_page_excerpt(&client, &abstract_url, 2000).await {
                parts.push(format!("\n--- Page content from {abstract_url} ---\n{excerpt}"));
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

/// Fetch a web page and extract a plain-text excerpt by stripping HTML tags.
///
/// Returns `None` on any error (timeout, non-HTML response, etc.) so the
/// caller can fall back gracefully to the search snippet data.
///
/// `max_chars` limits the returned excerpt to prevent context explosion.
async fn fetch_page_excerpt(
    client: &reqwest::Client,
    url: &str,
    max_chars: usize,
) -> Option<String> {
    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(8))
        .header("Accept", "text/html")
        .send()
        .await
        .ok()?;

    // Only process HTML responses.
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.contains("text/html") && !content_type.contains("text/plain") {
        return None;
    }

    // Limit download to 256 KB to avoid pulling huge pages.
    let body = resp.text().await.ok()?;
    let body = if body.len() > 256_000 { &body[..256_000] } else { &body };

    Some(html_to_text(body, max_chars))
}

/// Minimal HTML-to-text extraction.  Strips tags, collapses whitespace, and
/// drops `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>` blocks.
///
/// This is intentionally simple (no third-party HTML parser dependency).
/// It produces "good enough" text for the LLM to extract facts from.
fn html_to_text(html: &str, max_chars: usize) -> String {
    // Remove script/style/nav/header/footer blocks (case-insensitive via lowering the tag scan).
    let mut cleaned = String::with_capacity(html.len());
    let mut skip_depth: usize = 0;
    let mut i = 0;
    let bytes = html.as_bytes();
    let len = bytes.len();

    while i < len {
        if bytes[i] == b'<' {
            // Peek at the tag name.
            let tag_start = i;
            i += 1;
            let is_close = i < len && bytes[i] == b'/';
            if is_close { i += 1; }

            let name_start = i;
            while i < len && bytes[i] != b'>' && bytes[i] != b' ' && bytes[i] != b'/' {
                i += 1;
            }
            let tag_name = std::str::from_utf8(&bytes[name_start..i])
                .unwrap_or("")
                .to_ascii_lowercase();

            // Skip to end of tag.
            while i < len && bytes[i] != b'>' {
                i += 1;
            }
            if i < len { i += 1; } // skip '>'

            let strip_tags = ["script", "style", "nav", "header", "footer", "noscript", "svg"];
            if strip_tags.contains(&tag_name.as_str()) {
                if is_close {
                    skip_depth = skip_depth.saturating_sub(1);
                } else {
                    skip_depth += 1;
                }
                continue;
            }

            if skip_depth > 0 {
                continue;
            }

            // Block-level tags emit a newline to preserve structure.
            let block_tags = ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
                              "li", "tr", "td", "th", "article", "section", "main"];
            if block_tags.contains(&tag_name.as_str()) {
                cleaned.push('\n');
            }

            // Drop the tag itself (no output).
            let _ = tag_start; // consumed
        } else {
            if skip_depth == 0 {
                cleaned.push(bytes[i] as char);
            }
            i += 1;
        }
    }

    // Decode common HTML entities.
    let cleaned = cleaned
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");

    // Collapse runs of whitespace into single space, trim blank lines.
    let mut result = String::with_capacity(cleaned.len().min(max_chars + 64));
    let mut prev_was_space = true;
    let mut consecutive_newlines = 0u32;
    for ch in cleaned.chars() {
        if ch == '\n' {
            consecutive_newlines += 1;
            if consecutive_newlines <= 2 {
                result.push('\n');
            }
            prev_was_space = true;
        } else if ch.is_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
            consecutive_newlines = 0;
        } else {
            result.push(ch);
            prev_was_space = false;
            consecutive_newlines = 0;
        }
        if result.len() >= max_chars {
            break;
        }
    }

    let trimmed = result.trim().to_string();
    if trimmed.len() > max_chars {
        // Truncate to a word boundary.
        let end = trimmed[..max_chars].rfind(' ').unwrap_or(max_chars);
        format!("{}…", &trimmed[..end])
    } else {
        trimmed
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

// ── git_rollback ──────────────────────────────────────────────────────────────

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
    use super::*;

    #[test]
    fn html_to_text_strips_tags() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>";
        let text = html_to_text(html, 200);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<h1>"));
        assert!(!text.contains("<b>"));
    }

    #[test]
    fn html_to_text_strips_scripts() {
        let html = "<p>Before</p><script>var x = 1;</script><p>After</p>";
        let text = html_to_text(html, 200);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn html_to_text_decodes_entities() {
        let html = "<p>A &amp; B &lt; C &gt; D</p>";
        let text = html_to_text(html, 200);
        assert!(text.contains("A & B < C > D"));
    }

    #[test]
    fn html_to_text_respects_max_chars() {
        let html = "<p>".to_string() + &"a".repeat(5000) + "</p>";
        let text = html_to_text(&html, 100);
        assert!(text.len() <= 110); // small tolerance for trailing ellipsis
    }

    #[test]
    fn html_to_text_collapses_whitespace() {
        let html = "<p>  lots   of    spaces  </p>";
        let text = html_to_text(html, 200);
        // Should not have runs of multiple spaces.
        assert!(!text.contains("  "));
    }
}
