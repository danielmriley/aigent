//! Web search, page fetching, and HTML extraction.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde_json;
use reqwest;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput};
use super::fs::truncate_byte_boundary;


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
        let mut page_urls: Vec<String> = Vec::new();
        if let Some(results) = json["web"]["results"].as_array() {
            for item in results.iter().take(max_results) {
                let title = item["title"].as_str().unwrap_or("").trim();
                let url = item["url"].as_str().unwrap_or("").trim();
                let desc = item["description"].as_str().unwrap_or("").trim();
                if !title.is_empty() {
                    if page_urls.len() < 3 && !url.is_empty() {
                        page_urls.push(url.to_string());
                    }
                    parts.push(format!("{title}\n  {url}\n  {desc}"));
                }
            }
        }

        // Fetch the top result pages to extract actual content (not just
        // search snippets).  Multiple pages give the LLM cross-references
        // for factual queries (stock prices, scores, etc.).
        // Structured data (JSON-LD, meta tags) is extracted first since it
        // survives JavaScript-heavy single-page apps.
        for url in &page_urls {
            if let Some(excerpt) = fetch_page_excerpt(&client, url, 4000).await {
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
            if let Some(excerpt) = fetch_page_excerpt(&client, &abstract_url, 4000).await {
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
    let body = if body.len() > 256_000 {
        let end = truncate_byte_boundary(&body, 256_000);
        &body[..end]
    } else {
        &body
    };

    // Extract structured data first (JSON-LD, meta tags, title) — these
    // survive JS-heavy SPAs where the body text is empty / boilerplate.
    let structured = extract_structured_data(body);
    let plain = html_to_text(body, max_chars);

    if structured.is_empty() {
        Some(plain)
    } else if plain.is_empty() {
        Some(structured)
    } else {
        // Budget: give structured data up to 1/3 of max_chars, rest to plain text.
        let struct_budget = max_chars / 3;
        let struct_part = if structured.len() > struct_budget {
            let end = truncate_byte_boundary(&structured, struct_budget);
            format!("{}…", &structured[..end])
        } else {
            structured
        };
        Some(format!("{struct_part}\n\n{plain}"))
    }
}

/// Minimal HTML-to-text extraction.  Strips tags, collapses whitespace, and
/// drops `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>` blocks.
///
/// This is intentionally simple (no third-party HTML parser dependency).
/// It produces "good enough" text for the LLM to extract facts from.
pub(super) fn html_to_text(html: &str, max_chars: usize) -> String {
    // Remove script/style/nav/header/footer blocks (case-insensitive via lowering the tag scan).
    let mut cleaned = String::with_capacity(html.len());
    let mut skip_depth: usize = 0;
    let mut chars = html.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '<' {
            // Peek at the tag name.
            let mut tag_chars = Vec::new();
            let is_close = chars.peek() == Some(&'/');
            if is_close { chars.next(); }

            // Collect tag name chars until '>', ' ', or '/'
            while let Some(&c) = chars.peek() {
                if c == '>' || c == ' ' || c == '/' { break; }
                tag_chars.push(c);
                chars.next();
            }
            let tag_name: String = tag_chars.into_iter().collect::<String>().to_ascii_lowercase();

            // Skip to end of tag.
            while let Some(&c) = chars.peek() {
                if c == '>' { chars.next(); break; }
                chars.next();
            }

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
        } else {
            if skip_depth == 0 {
                cleaned.push(ch);
            }
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
        // Truncate to a word boundary (safely, respecting char boundaries).
        let safe_end = truncate_byte_boundary(&trimmed, max_chars);
        let end = trimmed[..safe_end].rfind(' ').unwrap_or(safe_end);
        format!("{}…", &trimmed[..end])
    } else {
        trimmed
    }
}

/// Extract structured data from HTML that survives JavaScript-heavy pages.
///
/// Pulls out:
///   - `<title>` — almost always present even in SPAs
///   - `<meta>` tags: `og:title`, `og:description`, `description`, plus any
///     tag whose `name` or `property` contains "price", "amount", or "stock"
///   - `<script type="application/ld+json">` — structured data used by Google
///     (financial sites often embed stock quotes here)
///
/// Returns a compact multi-line summary.  Empty string when nothing is found.
fn extract_structured_data(html: &str) -> String {
    let mut lines: Vec<String> = Vec::new();

    // ── <title> ────────────────────────────────────────────────────────────
    if let Some(start) = html.to_ascii_lowercase().find("<title") {
        if let Some(gt) = html[start..].find('>') {
            let after = start + gt + 1;
            if let Some(end) = html[after..].find("</") {
                let title = html[after..after + end].trim();
                if !title.is_empty() && title.len() < 500 {
                    lines.push(format!("Title: {title}"));
                }
            }
        }
    }

    // ── <meta> tags ────────────────────────────────────────────────────────
    // We scan for `<meta` and extract `name=`/`property=` and `content=`.
    let lower = html.to_ascii_lowercase();
    let interesting_attrs = [
        "og:title", "og:description", "og:type",
        "description", "twitter:title", "twitter:description",
    ];
    let price_keywords = ["price", "amount", "stock", "ticker", "quote"];

    let mut search_from = 0;
    while let Some(pos) = lower[search_from..].find("<meta") {
        let abs_pos = search_from + pos;
        let tag_end = match html[abs_pos..].find('>') {
            Some(e) => abs_pos + e,
            None => break,
        };
        let tag = &html[abs_pos..=tag_end];
        let tag_lower = &lower[abs_pos..=tag_end];

        // Extract attribute values from the tag.
        let attr_val = |attr: &str| -> Option<&str> {
            let needle = format!("{attr}=\"");
            tag_lower.find(&needle).and_then(|i| {
                let start = i + needle.len();
                tag[start..].find('"').map(|end| tag[start..start + end].trim())
            })
        };

        let name = attr_val("name").or_else(|| attr_val("property")).unwrap_or("");
        let content = attr_val("content").unwrap_or("");

        if !content.is_empty() && content.len() < 500 {
            let name_lower = name.to_ascii_lowercase();
            let is_interesting = interesting_attrs.iter().any(|a| name_lower == *a)
                || price_keywords.iter().any(|kw| name_lower.contains(kw));
            if is_interesting {
                lines.push(format!("meta[{name}]: {content}"));
            }
        }

        search_from = tag_end + 1;
    }

    // ── <script type="application/ld+json"> ────────────────────────────────
    let ld_marker = "application/ld+json";
    let mut ld_from = 0;
    while let Some(pos) = lower[ld_from..].find(ld_marker) {
        let abs_pos = ld_from + pos;
        // Find the '>' that closes this <script> tag.
        let script_body_start = match html[abs_pos..].find('>') {
            Some(e) => abs_pos + e + 1,
            None => break,
        };
        // Find closing </script>.
        let script_body_end = match lower[script_body_start..].find("</script") {
            Some(e) => script_body_start + e,
            None => break,
        };
        let json_str = html[script_body_start..script_body_end].trim();
        if !json_str.is_empty() && json_str.len() < 8000 {
            // Try to parse and extract a compact summary.
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let summary = summarise_ld_json(&val);
                if !summary.is_empty() {
                    lines.push(format!("LD+JSON: {summary}"));
                }
            }
        }
        ld_from = script_body_end + 1;
    }

    lines.join("\n")
}

/// Produce a compact one-line summary of a JSON-LD object, pulling out the
/// most useful fields for factual queries.
fn summarise_ld_json(val: &serde_json::Value) -> String {
    // Handle @graph arrays (common wrapper).
    if let Some(graph) = val.get("@graph").and_then(|g| g.as_array()) {
        let summaries: Vec<String> = graph.iter()
            .filter_map(|item| {
                let s = summarise_ld_json(item);
                if s.is_empty() { None } else { Some(s) }
            })
            .take(3)
            .collect();
        return summaries.join(" | ");
    }

    let mut parts: Vec<String> = Vec::new();
    let type_val = val.get("@type")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if !type_val.is_empty() {
        parts.push(format!("type={type_val}"));
    }
    // Pull common fields.
    for key in &["name", "headline", "description", "tickerSymbol",
                 "price", "priceCurrency", "lowPrice", "highPrice",
                 "url", "exchange", "currentPrice", "previousClose",
                 "openPrice", "dayLow", "dayHigh", "52WeekLow", "52WeekHigh"] {
        if let Some(v) = val.get(*key) {
            let text = match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if !text.is_empty() && text.len() < 300 {
                parts.push(format!("{key}={text}"));
            }
        }
    }
    // Nested "offers" (e-commerce / financial).
    if let Some(offers) = val.get("offers") {
        let offer_summary = summarise_ld_json(offers);
        if !offer_summary.is_empty() {
            parts.push(format!("offers({offer_summary})"));
        }
    }
    if parts.len() <= 1 {
        // Only had @type or nothing — not useful.
        return String::new();
    }
    parts.join("; ")
}

