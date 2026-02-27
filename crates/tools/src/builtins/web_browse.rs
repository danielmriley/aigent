//! `web_browse` — a general-purpose browsing tool that handles JS-rendered
//! and real-time content by routing through Jina Reader with a direct-fetch
//! fallback.  When the target is a natural-language query rather than a URL
//! the tool resolves it via DuckDuckGo first.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use percent_encoding::percent_decode_str;
use scraper::{Html, Selector};

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};
use super::fs::truncate_byte_boundary;

// ─── Constants ───────────────────────────────────────────────────────────────

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/// Jina Reader prefix — renders JS and returns clean Markdown.
const JINA_READER_PREFIX: &str = "https://r.jina.ai/";

/// Maximum characters in the final output sent to the LLM.
const MAX_OUTPUT_CHARS: usize = 16_000;

/// Maximum raw bytes we'll download from a direct fetch.
const MAX_DOWNLOAD_BYTES: usize = 256_000;

/// HTTP timeout shared across all requests.
const TIMEOUT: Duration = Duration::from_secs(15);

// ═════════════════════════════════════════════════════════════════════════════
//  WebBrowseTool
// ═════════════════════════════════════════════════════════════════════════════

/// Browse any URL or topic and return the most current, clean, LLM-friendly
/// content available.  Automatically handles dynamic/JS-rendered pages and
/// time-sensitive information.
pub struct WebBrowseTool;

#[async_trait]
impl Tool for WebBrowseTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_browse".to_string(),
            description: "Browse any URL or topic and return the most current, \
                clean, LLM-friendly content available. Automatically handles \
                dynamic/JS-rendered pages and time-sensitive information. \
                Preferred tool whenever up-to-date or real-time data is needed."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "target".to_string(),
                    description: "A full URL (e.g. \"https://example.com/page\") \
                        or a short query/topic (e.g. \"current TSLA stock price\", \
                        \"latest FSU women's tennis roster 2026\")."
                        .to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "instructions".to_string(),
                    description: "Optional guidance on what to extract or \
                        summarise (e.g. \"Extract only the current price, \
                        change percent, and day range\")."
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::High,
                read_only: true,
                group: "web".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let target = args
            .get("target")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("missing required param: target"))?;

        let instructions = args
            .get("instructions")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // Decide: URL or search query?
        let urls = if looks_like_url(&target) {
            vec![target.clone()]
        } else {
            // Resolve the query to URLs via DuckDuckGo.
            match search_for_urls(&target, 3).await {
                Ok(urls) if !urls.is_empty() => urls,
                Ok(_) => {
                    return Ok(ToolOutput {
                        success: false,
                        output: format!(
                            "No relevant search results found for: {target}"
                        ),
                    });
                }
                Err(e) => {
                    return Ok(ToolOutput {
                        success: false,
                        output: format!(
                            "Search failed while resolving \"{target}\": {e}"
                        ),
                    });
                }
            }
        };

        // Try fetching each URL — Jina first, direct fallback.
        let mut sections: Vec<String> = Vec::new();
        let mut last_err = String::new();

        for url in &urls {
            match fetch_with_jina_fallback(url).await {
                Ok(content) if !content.trim().is_empty() => {
                    sections.push(content);
                    // For a direct URL target we only need one result;
                    // for search-resolved targets, take the first 2 successes.
                    if looks_like_url(&target) || sections.len() >= 2 {
                        break;
                    }
                }
                Ok(_) => {
                    last_err = format!("page returned no extractable content: {url}");
                }
                Err(e) => {
                    last_err = format!("could not fetch {url}: {e}");
                }
            }
        }

        if sections.is_empty() {
            return Ok(ToolOutput {
                success: false,
                output: format!(
                    "Could not retrieve live data — {last_err}"
                ),
            });
        }

        let mut output = sections.join("\n\n---\n\n");

        // Append extraction instructions for the LLM when provided.
        if let Some(ref instr) = instructions {
            output.push_str("\n\n[Extraction guidance: ");
            output.push_str(instr);
            output.push(']');
        }

        // Trim to budget.
        if output.len() > MAX_OUTPUT_CHARS {
            let safe = truncate_byte_boundary(&output, MAX_OUTPUT_CHARS);
            let end = output[..safe].rfind('\n').unwrap_or(safe);
            output.truncate(end);
            output.push_str("\n…(truncated)");
        }

        Ok(ToolOutput {
            success: true,
            output,
        })
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  URL detection
// ═════════════════════════════════════════════════════════════════════════════

/// Heuristic: does the target look like a URL?
fn looks_like_url(s: &str) -> bool {
    s.starts_with("http://")
        || s.starts_with("https://")
        || (s.contains('.') && !s.contains(' ') && s.len() < 2048)
}

// ═════════════════════════════════════════════════════════════════════════════
//  DuckDuckGo search — resolve a query to ranked URLs
// ═════════════════════════════════════════════════════════════════════════════

/// Search DuckDuckGo HTML and return up to `max` result URLs.
async fn search_for_urls(query: &str, max: usize) -> Result<Vec<String>> {
    let client = build_client()?;
    let resp = client
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query)])
        .header("Accept", "text/html")
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!("DuckDuckGo search error: {}", resp.status());
    }

    let body = resp.text().await?;

    // Html is !Send, scope it.
    let urls = {
        let doc = Html::parse_document(&body);
        let result_sel = Selector::parse(".result").unwrap();
        let link_sel = Selector::parse("a.result__a").unwrap();

        let mut urls = Vec::new();
        for result in doc.select(&result_sel).take(max + 2) {
            if let Some(href) = result
                .select(&link_sel)
                .next()
                .and_then(|el| el.value().attr("href"))
            {
                let url = extract_ddg_url(href);
                if url.starts_with("http") {
                    urls.push(url);
                }
            }
            if urls.len() >= max {
                break;
            }
        }
        urls
    };

    Ok(urls)
}

/// DDG wraps result URLs in redirect links; extract and decode the real URL.
fn extract_ddg_url(href: &str) -> String {
    if let Some(pos) = href.find("uddg=") {
        let start = pos + 5;
        let end = href[start..]
            .find('&')
            .map(|i| start + i)
            .unwrap_or(href.len());
        let encoded = &href[start..end];
        if !encoded.is_empty() {
            return percent_decode_str(encoded)
                .decode_utf8_lossy()
                .into_owned();
        }
    }
    href.to_string()
}

// ═════════════════════════════════════════════════════════════════════════════
//  Content fetching — Jina Reader primary, direct reqwest fallback
// ═════════════════════════════════════════════════════════════════════════════

/// Try Jina Reader first; on any failure fall back to a direct fetch with
/// the project's HTML-to-markdown pipeline.
async fn fetch_with_jina_fallback(url: &str) -> Result<String> {
    // ── Primary: Jina Reader ────────────────────────────────────────────
    match fetch_via_jina(url).await {
        Ok(content) if content.trim().len() >= 80 => return Ok(content),
        Ok(_) => {} // too short — try direct
        Err(_) => {}
    }

    // ── Fallback: direct fetch + local HTML extraction ──────────────────
    fetch_direct(url).await
}

/// Fetch rendered content via Jina Reader (`r.jina.ai/{url}`).
///
/// Returns Markdown with title/URL/date metadata prepended by the service.
async fn fetch_via_jina(url: &str) -> Result<String> {
    let jina_url = format!("{JINA_READER_PREFIX}{url}");
    let client = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()?;

    let resp = client
        .get(&jina_url)
        .header("Accept", "text/markdown")
        .header("User-Agent", USER_AGENT)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        // Jina returns JSON errors — extract the message if possible.
        let msg = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("readableMessage")?.as_str().map(String::from))
            .unwrap_or_else(|| format!("HTTP {status}"));
        anyhow::bail!("Jina Reader error: {msg}");
    }

    let text = resp.text().await?;
    if text.trim().is_empty() {
        anyhow::bail!("Jina Reader returned empty content");
    }

    // Trim to budget.
    if text.len() > MAX_OUTPUT_CHARS {
        let safe = truncate_byte_boundary(&text, MAX_OUTPUT_CHARS);
        let end = text[..safe].rfind('\n').unwrap_or(safe);
        Ok(format!("{}…", &text[..end]))
    } else {
        Ok(text)
    }
}

/// Direct reqwest fetch → local HTML extraction (plain text + structured data).
///
/// This mirrors the approach in `web.rs::fetch_page_content` but is
/// self-contained so `web_browse` does not depend on private functions there.
async fn fetch_direct(url: &str) -> Result<String> {
    let client = build_client()?;
    let resp = client
        .get(url)
        .timeout(TIMEOUT)
        .header("Accept", "text/html")
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                anyhow::anyhow!("request timed out")
            } else if e.is_connect() {
                anyhow::anyhow!("connection failed: {e}")
            } else {
                anyhow::anyhow!("request error: {e}")
            }
        })?;

    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status}");
    }

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.contains("text/html") && !content_type.contains("text/plain") {
        anyhow::bail!("unsupported content type: {content_type}");
    }

    let body = resp.text().await?;
    let body = if body.len() > MAX_DOWNLOAD_BYTES {
        let end = truncate_byte_boundary(&body, MAX_DOWNLOAD_BYTES);
        &body[..end]
    } else {
        &body
    };

    // Extract structured metadata (title, description, JSON-LD).
    let structured = extract_structured_data(body);

    // Extract page body text.
    let text = extract_body_text(body, MAX_OUTPUT_CHARS);

    if structured.is_empty() && text.is_empty() {
        anyhow::bail!("no extractable content");
    }

    let mut output = String::new();
    if !structured.is_empty() {
        output.push_str(&structured);
    }
    if !text.is_empty() {
        if !output.is_empty() {
            output.push_str("\n\n");
        }
        output.push_str(&text);
    }

    // Append source URL.
    output.push_str("\n\nSource: ");
    output.push_str(url);

    Ok(output)
}

// ═════════════════════════════════════════════════════════════════════════════
//  HTML helpers  (self-contained — mirrors web.rs utilities)
// ═════════════════════════════════════════════════════════════════════════════

fn build_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(TIMEOUT)
        .user_agent(USER_AGENT)
        .build()?)
}

/// Extract structured metadata from HTML: `<title>`, `<meta>` description/OG,
/// and JSON-LD `@type`/`name`/`description` blobs.
fn extract_structured_data(html: &str) -> String {
    let doc = Html::parse_document(html);
    let mut parts: Vec<String> = Vec::new();

    // Title.
    if let Ok(sel) = Selector::parse("title") {
        if let Some(el) = doc.select(&sel).next() {
            let t: String = el.text().collect();
            let t = t.trim();
            if !t.is_empty() {
                parts.push(format!("Title: {t}"));
            }
        }
    }

    // Meta description.
    if let Ok(sel) = Selector::parse("meta[name=\"description\"]") {
        if let Some(el) = doc.select(&sel).next() {
            if let Some(c) = el.value().attr("content") {
                let c = c.trim();
                if !c.is_empty() {
                    parts.push(format!("Description: {c}"));
                }
            }
        }
    }

    // OG title / description.
    for (prop, label) in &[
        ("og:title", "OG Title"),
        ("og:description", "OG Description"),
    ] {
        let selector_str = format!("meta[property=\"{prop}\"]");
        if let Ok(sel) = Selector::parse(&selector_str) {
            if let Some(el) = doc.select(&sel).next() {
                if let Some(c) = el.value().attr("content") {
                    let c = c.trim();
                    if !c.is_empty() {
                        parts.push(format!("{label}: {c}"));
                    }
                }
            }
        }
    }

    // JSON-LD (first script block with meaningful data).
    if let Ok(sel) = Selector::parse("script[type=\"application/ld+json\"]") {
        for el in doc.select(&sel).take(3) {
            let raw: String = el.text().collect();
            let raw = raw.trim();
            if raw.len() < 20 {
                continue;
            }
            // Limit how much JSON-LD we include.
            let budget = 1500;
            if raw.len() > budget {
                let safe = truncate_byte_boundary(raw, budget);
                parts.push(format!("JSON-LD: {}…", &raw[..safe]));
            } else {
                parts.push(format!("JSON-LD: {raw}"));
            }
            break; // one JSON-LD block is enough
        }
    }

    parts.join("\n")
}

/// Extract readable body text from HTML, targeting article/main/body regions.
fn extract_body_text(html: &str, max_chars: usize) -> String {
    let doc = Html::parse_document(html);

    // Try focused content regions first.
    let selectors = ["article", "main", "[role=\"main\"]", ".post-content", ".entry-content"];
    for sel_str in &selectors {
        if let Ok(sel) = Selector::parse(sel_str) {
            if let Some(el) = doc.select(&sel).next() {
                let text: String = el.text().collect();
                let text = collapse_whitespace(&text);
                if text.len() >= 80 {
                    return truncate_text(&text, max_chars);
                }
            }
        }
    }

    // Fall back to body with noise stripped.
    if let Ok(body_sel) = Selector::parse("body") {
        if let Some(body) = doc.select(&body_sel).next() {
            let raw: String = body.text().collect();
            let text = collapse_whitespace(&raw);
            return truncate_text(&text, max_chars);
        }
    }

    String::new()
}

/// Collapse runs of whitespace into single spaces and trim.
fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Truncate text to a char budget, preferring a word boundary.
fn truncate_text(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let safe = truncate_byte_boundary(s, max);
    let end = s[..safe].rfind(' ').unwrap_or(safe);
    format!("{}…", &s[..end])
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_is_correct() {
        let tool = WebBrowseTool;
        let spec = tool.spec();
        assert_eq!(spec.name, "web_browse");
        assert!(spec.params.iter().any(|p| p.name == "target" && p.required));
        assert!(spec.params.iter().any(|p| p.name == "instructions" && !p.required));
        assert!(spec.metadata.read_only);
    }

    #[test]
    fn looks_like_url_tests() {
        assert!(looks_like_url("https://example.com"));
        assert!(looks_like_url("http://foo.bar/page"));
        assert!(looks_like_url("example.com"));
        assert!(!looks_like_url("current TSLA stock price"));
        assert!(!looks_like_url("latest news about rust"));
    }

    #[test]
    fn collapse_whitespace_works() {
        assert_eq!(collapse_whitespace("  hello   world  "), "hello world");
        assert_eq!(collapse_whitespace("no\nnewlines\there"), "no newlines here");
    }

    #[test]
    fn truncate_text_within_budget() {
        let s = "short";
        assert_eq!(truncate_text(s, 100), "short");
    }

    #[test]
    fn truncate_text_over_budget() {
        let s = "hello world this is a long string";
        let t = truncate_text(s, 15);
        assert!(t.len() <= 20); // with ellipsis
        assert!(t.ends_with('…'));
    }

    #[test]
    fn extract_structured_data_title() {
        let html = "<html><head><title>Test Page</title></head><body></body></html>";
        let result = extract_structured_data(html);
        assert!(result.contains("Title: Test Page"));
    }

    #[test]
    fn extract_structured_data_empty() {
        let result = extract_structured_data("<html><body></body></html>");
        assert!(result.is_empty());
    }

    #[test]
    fn extract_body_text_from_article() {
        let html = r#"<html><body>
            <nav>Navigation stuff</nav>
            <article>This is the main article content that should be extracted for the test.</article>
        </body></html>"#;
        let text = extract_body_text(html, 1000);
        assert!(text.contains("main article content"));
    }

    #[tokio::test]
    async fn missing_target_errors() {
        let tool = WebBrowseTool;
        let result = tool.run(&HashMap::new()).await;
        assert!(result.is_err() || !result.unwrap().success);
    }

    #[tokio::test]
    async fn empty_target_errors() {
        let tool = WebBrowseTool;
        let mut args = HashMap::new();
        args.insert("target".to_string(), "".to_string());
        let result = tool.run(&args).await;
        assert!(result.is_err() || !result.unwrap().success);
    }

    #[tokio::test]
    #[ignore] // network — run with `cargo test --ignored`
    async fn fetch_wikipedia() {
        let tool = WebBrowseTool;
        let mut args = HashMap::new();
        args.insert(
            "target".to_string(),
            "https://en.wikipedia.org/wiki/Rust_(programming_language)".to_string(),
        );
        let result = tool.run(&args).await.expect("should not error");
        assert!(result.success, "output: {}", result.output);
        assert!(result.output.contains("Rust"), "output: {}", result.output);
    }
}
