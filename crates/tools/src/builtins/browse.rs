//! `browse_page` — unified page-fetching / browsing tool.
//!
//! This tool merges the former `fetch_page` and `web_browse` tools into a
//! single, best-of-both-worlds implementation:
//!
//! * Jina Reader primary (handles JS-rendered pages, returns Markdown).
//! * Direct reqwest + scraper fallback (for when Jina is slow or blocked).
//! * Firecrawl API fallback (when `FIRECRAWL_API_KEY` is set).
//! * Batch URL support — up to 5 concurrent fetches.
//! * Output mode: `"markdown"` (default), `"text"`, `"structured"`.
//! * Private IP blocking for security.
//! * Rich structured output with title, content, word_count, published_date.

use std::collections::HashMap;
use std::net::{IpAddr, ToSocketAddrs};
use std::time::Duration;

use super::cache::TtlCache;

use anyhow::Result;
use async_trait::async_trait;
use scraper::{Html, Selector};

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel, ParamType};
use super::fs::truncate_byte_boundary;

// ─── Constants ───────────────────────────────────────────────────────────────

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/// Jina Reader prefix — renders JS and returns clean Markdown.
const JINA_READER_PREFIX: &str = "https://r.jina.ai/";

/// Maximum characters in the final output sent to the LLM.
const DEFAULT_MAX_CHARS: usize = 16_000;

/// Maximum raw bytes we'll download from a direct fetch.
const MAX_DOWNLOAD_BYTES: usize = 512_000;

/// HTTP timeout shared across all requests.
const TIMEOUT: Duration = Duration::from_secs(15);

/// Maximum number of URLs we'll fetch in a single batch call.
const MAX_BATCH_URLS: usize = 5;

/// Page fetch cache — avoids re-fetching the same URL within the TTL window.
static PAGE_CACHE: std::sync::LazyLock<TtlCache> =
    std::sync::LazyLock::new(|| TtlCache::new(128, Duration::from_secs(600)));

// ═════════════════════════════════════════════════════════════════════════════
//  BrowsePageTool
// ═════════════════════════════════════════════════════════════════════════════

/// Browse one or more URLs and return clean, LLM-friendly content.
///
/// Combines the best of the old `fetch_page` (direct + scraper) and
/// `web_browse` (Jina Reader + natural-language query resolution) tools.
/// Supports batch fetching, output mode selection, and rich metadata.
pub struct BrowsePageTool;

#[async_trait]
impl Tool for BrowsePageTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "browse_page".to_string(),
            description: "Fetch one or more web pages and return clean, \
                LLM-friendly content. Handles JS-rendered pages automatically. \
                Supports batch URLs (comma-separated), output mode selection, \
                and returns rich metadata (title, word count, dates). \
                Use after web_search to read full article text, documentation, \
                or any specific page content."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "url".to_string(),
                    description: "One or more URLs to fetch, comma-separated. \
                        Example: \"https://example.com/a, https://example.com/b\""
                        .to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "mode".to_string(),
                    description: "Output format: \"markdown\" (default, best for \
                        articles/docs), \"text\" (plain text, smaller), or \
                        \"structured\" (metadata + key extractions only)."
                        .to_string(),
                    required: false,
                    enum_values: vec![
                        "markdown".to_string(),
                        "text".to_string(),
                        "structured".to_string(),
                    ],
                    default: Some("markdown".to_string()),
                    ..Default::default()
                },
                ToolParam {
                    name: "max_chars".to_string(),
                    description: "Maximum characters to return per URL \
                        (default: 16000). Reduce for quick scans."
                        .to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("16000".to_string()),
                    ..Default::default()
                },
                ToolParam {
                    name: "instructions".to_string(),
                    description: "Optional extraction guidance, e.g. \
                        \"Extract only the pricing table\" or \
                        \"Focus on the API reference section\"."
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::High,
                read_only: true,
                group: "web".to_string(),
                cost_estimate: Some(0.0),
                examples: vec![
                    "browse_page url=\"https://docs.rs/tokio/latest/tokio/\" mode=\"markdown\"".to_string(),
                    "browse_page url=\"https://news.ycombinator.com, https://lobste.rs\" mode=\"text\" max_chars=4000".to_string(),
                ],
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let raw_url = args
            .get("url")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("missing required param: url"))?;

        let mode = args
            .get("mode")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_else(|| "markdown".to_string());

        let max_chars: usize = args
            .get("max_chars")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_CHARS)
            .min(50_000); // hard cap

        let instructions = args
            .get("instructions")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // Parse comma-separated URLs.
        let urls: Vec<String> = raw_url
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.starts_with("http"))
            .take(MAX_BATCH_URLS)
            .collect();

        if urls.is_empty() {
            return Ok(ToolOutput {
                success: false,
                output: format!(
                    "No valid URLs found in: \"{raw_url}\". URLs must start with http:// or https://."
                ),
            });
        }

        // Security: block private/reserved IPs.
        let mut safe_urls: Vec<String> = Vec::with_capacity(urls.len());
        for url in &urls {
            if is_private_url(url) {
                return Ok(ToolOutput {
                    success: false,
                    output: format!(
                        "Blocked: {url} resolves to a private/reserved IP address. \
                         Only public URLs are allowed."
                    ),
                });
            }
            safe_urls.push(url.clone());
        }

        // Split URLs into cached hits and those that need fetching.
        let mut sections: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        let mut to_fetch: Vec<(usize, String)> = Vec::new(); // (index, url)

        for (i, url) in safe_urls.iter().enumerate() {
            let cache_key = format!("{url}:{mode}");
            if let Some(cached) = PAGE_CACHE.get(&cache_key) {
                // Use cached content directly.
                if safe_urls.len() > 1 {
                    sections.push(format!("## [{url}]\n\n{}", cached.output));
                } else {
                    sections.push(cached.output.clone());
                }
            } else {
                to_fetch.push((i, url.clone()));
            }
        }

        // Fetch uncached URLs in parallel.
        if !to_fetch.is_empty() {
            let futs: Vec<_> = to_fetch
                .iter()
                .map(|(_, url)| {
                    let url = url.clone();
                    let mode = mode.clone();
                    let max = max_chars;
                    tokio::spawn(async move {
                        fetch_single_url(&url, &mode, max).await
                    })
                })
                .collect();

            let results = futures::future::join_all(futs).await;

            for (j, result) in results.into_iter().enumerate() {
                let url = &to_fetch[j].1;
                match result {
                    Ok(Ok(content)) if !content.trim().is_empty() => {
                        // Cache the successful fetch.
                        let cache_key = format!("{url}:{mode}");
                        PAGE_CACHE.insert(cache_key, ToolOutput {
                            success: true,
                            output: content.clone(),
                        });
                        if safe_urls.len() > 1 {
                            sections.push(format!("## [{url}]\n\n{content}"));
                        } else {
                            sections.push(content);
                        }
                    }
                    Ok(Ok(_)) => {
                        errors.push(format!("{url}: no extractable content"));
                    }
                    Ok(Err(e)) => {
                        errors.push(format!("{url}: {e}"));
                    }
                    Err(e) => {
                        errors.push(format!("{url}: task failed: {e}"));
                    }
                }
            }
        }

        if sections.is_empty() {
            return Ok(ToolOutput {
                success: false,
                output: format!(
                    "Could not fetch any pages:\n{}",
                    errors.join("\n")
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

        // Append error summary if some URLs failed in a batch.
        if !errors.is_empty() && safe_urls.len() > 1 {
            output.push_str("\n\n[Failed URLs: ");
            output.push_str(&errors.join("; "));
            output.push(']');
        }

        // Final length trim.
        let total_budget = max_chars * safe_urls.len().min(3);
        if output.len() > total_budget {
            let safe = truncate_byte_boundary(&output, total_budget);
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
//  Single URL fetch — mode-aware
// ═════════════════════════════════════════════════════════════════════════════

/// Fetch a single URL, trying providers in order:
/// 1. Jina Reader (best for markdown mode)
/// 2. Firecrawl API (if key available)
/// 3. Direct reqwest + scraper fallback
async fn fetch_single_url(url: &str, mode: &str, max_chars: usize) -> Result<String> {
    // For structured mode, skip Jina and go straight to direct fetch.
    if mode == "structured" {
        return fetch_structured_only(url).await;
    }

    // Try Jina Reader first (returns Markdown).
    match fetch_via_jina(url).await {
        Ok(content) if content.trim().len() >= 80 => {
            let content = if mode == "text" {
                strip_markdown_formatting(&content)
            } else {
                content
            };
            return Ok(truncate_content(&content, max_chars));
        }
        _ => {} // fall through
    }

    // Try Firecrawl if API key is available.
    if let Some(ref key) = firecrawl_key() {
        match fetch_via_firecrawl(url, key, mode).await {
            Ok(content) if content.trim().len() >= 80 => {
                return Ok(truncate_content(&content, max_chars));
            }
            _ => {} // fall through
        }
    }

    // Final fallback: direct fetch + local extraction.
    let content = fetch_direct(url, mode, max_chars).await?;
    Ok(content)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Jina Reader
// ═════════════════════════════════════════════════════════════════════════════

/// Fetch rendered content via Jina Reader (`r.jina.ai/{url}`).
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

    Ok(text)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Firecrawl API
// ═════════════════════════════════════════════════════════════════════════════

/// Try the Firecrawl API to scrape a URL.
async fn fetch_via_firecrawl(url: &str, api_key: &str, mode: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    let format = match mode {
        "text" => "text",
        _ => "markdown",
    };

    let body = serde_json::json!({
        "url": url,
        "formats": [format],
    });

    let resp = client
        .post("https://api.firecrawl.dev/v1/scrape")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Firecrawl API error {status}: {text}");
    }

    let json: serde_json::Value = resp.json().await?;
    let content = json
        .get("data")
        .and_then(|d| {
            d.get(format)
                .or_else(|| d.get("markdown"))
                .or_else(|| d.get("text"))
        })
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if content.is_empty() {
        anyhow::bail!("Firecrawl returned no content");
    }

    // Build rich output with metadata.
    let mut output = String::new();
    if let Some(title) = json["data"]["metadata"]["title"].as_str() {
        output.push_str(&format!("# {title}\n\n"));
    }
    if let Some(desc) = json["data"]["metadata"]["description"].as_str() {
        if !desc.is_empty() {
            output.push_str(&format!("> {desc}\n\n"));
        }
    }
    output.push_str(content);
    output.push_str(&format!("\n\nSource: {url}"));

    Ok(output)
}

/// Get Firecrawl API key from environment.
fn firecrawl_key() -> Option<String> {
    std::env::var("FIRECRAWL_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
}

// ═════════════════════════════════════════════════════════════════════════════
//  Direct fetch + local HTML extraction
// ═════════════════════════════════════════════════════════════════════════════

/// Direct reqwest fetch with local HTML extraction.
async fn fetch_direct(url: &str, mode: &str, max_chars: usize) -> Result<String> {
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

    match mode {
        "structured" => fetch_structured_from_html(body, url),
        "text" => {
            let text = extract_body_text(body, max_chars);
            if text.is_empty() {
                anyhow::bail!("no extractable text");
            }
            Ok(format!("{text}\n\nSource: {url}"))
        }
        _ => {
            // Markdown mode from direct fetch: extract structured + body text.
            let structured = extract_structured_data(body);
            let text = extract_body_text(body, max_chars);

            if structured.is_empty() && text.is_empty() {
                anyhow::bail!("no extractable content");
            }

            let mut output = String::new();
            if !structured.is_empty() {
                output.push_str(&structured);
                output.push_str("\n\n");
            }
            output.push_str(&text);
            output.push_str(&format!("\n\nSource: {url}"));
            Ok(output)
        }
    }
}

/// Structured-only mode: extract metadata without reading full body.
async fn fetch_structured_only(url: &str) -> Result<String> {
    let client = build_client()?;
    let resp = client
        .get(url)
        .timeout(TIMEOUT)
        .header("Accept", "text/html")
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }

    let body = resp.text().await?;
    let body = if body.len() > MAX_DOWNLOAD_BYTES {
        let end = truncate_byte_boundary(&body, MAX_DOWNLOAD_BYTES);
        &body[..end]
    } else {
        &body
    };

    fetch_structured_from_html(body, url)
}

/// Extract structured metadata from HTML and format it.
fn fetch_structured_from_html(html: &str, url: &str) -> Result<String> {
    let structured = extract_structured_data(html);
    if structured.is_empty() {
        anyhow::bail!("no structured data found");
    }

    // Also compute word count from body text.
    let text = extract_body_text(html, 50_000);
    let word_count = text.split_whitespace().count();

    let mut output = structured;
    output.push_str(&format!("\nWord count: {word_count}"));
    output.push_str(&format!("\nSource: {url}"));
    Ok(output)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Security: Private IP blocking
// ═════════════════════════════════════════════════════════════════════════════

/// Check if a URL resolves to a private/reserved IP address.
///
/// This prevents SSRF attacks where an LLM could be tricked into accessing
/// internal services (localhost, 192.168.x.x, 10.x.x.x, etc.).
fn is_private_url(url: &str) -> bool {
    let host = match url::Url::parse(url) {
        Ok(u) => match u.host_str() {
            Some(h) => h.to_string(),
            None => return true, // no host = suspicious
        },
        Err(_) => return false, // let the HTTP client handle parse errors
    };

    // Check common private hostnames.
    let host_lower = host.to_lowercase();
    if host_lower == "localhost"
        || host_lower == "0.0.0.0"
        || host_lower.ends_with(".local")
        || host_lower.ends_with(".internal")
    {
        return true;
    }

    // Try DNS resolution and check if any address is private.
    let port = 80; // dummy port for resolution
    if let Ok(addrs) = format!("{host}:{port}").to_socket_addrs() {
        for addr in addrs {
            if is_private_ip(addr.ip()) {
                return true;
            }
        }
    }

    false
}

/// Returns true if the IP is in a private/reserved range.
fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()             // 127.0.0.0/8
                || v4.is_private()       // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                || v4.is_link_local()    // 169.254.0.0/16
                || v4.is_broadcast()     // 255.255.255.255
                || v4.is_unspecified()   // 0.0.0.0
                || v4.octets()[0] == 100 && (v4.octets()[1] & 0xC0) == 64 // 100.64.0.0/10 (CGNAT)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()       // ::1
                || v6.is_unspecified() // ::
                // ULA (fc00::/7)
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // Link-local (fe80::/10)
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  HTML helpers
// ═════════════════════════════════════════════════════════════════════════════

fn build_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(TIMEOUT)
        .user_agent(USER_AGENT)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?)
}

/// Extract structured metadata from HTML: `<title>`, `<meta>` description/OG,
/// JSON-LD, published dates.
fn extract_structured_data(html: &str) -> String {
    let doc = Html::parse_document(html);
    let mut parts: Vec<String> = Vec::new();

    // Title.
    if let Ok(sel) = Selector::parse("title") {
        if let Some(el) = doc.select(&sel).next() {
            let t: String = el.text().collect();
            let t = t.trim();
            if !t.is_empty() && t.len() < 500 {
                parts.push(format!("Title: {t}"));
            }
        }
    }

    // Meta description.
    if let Ok(sel) = Selector::parse("meta[name=\"description\"]") {
        if let Some(el) = doc.select(&sel).next() {
            if let Some(c) = el.value().attr("content") {
                let c = c.trim();
                if !c.is_empty() && c.len() < 500 {
                    parts.push(format!("Description: {c}"));
                }
            }
        }
    }

    // OG / Twitter meta tags.
    for (prop, label) in &[
        ("og:title", "OG Title"),
        ("og:description", "OG Description"),
        ("og:type", "OG Type"),
        ("og:published_time", "Published"),
        ("article:published_time", "Published"),
        ("article:modified_time", "Modified"),
    ] {
        let selector_str = format!("meta[property=\"{prop}\"]");
        if let Ok(sel) = Selector::parse(&selector_str) {
            if let Some(el) = doc.select(&sel).next() {
                if let Some(c) = el.value().attr("content") {
                    let c = c.trim();
                    if !c.is_empty() && c.len() < 500 {
                        parts.push(format!("{label}: {c}"));
                    }
                }
            }
        }
    }

    // Published date from various meta tags.
    for selector_str in &[
        "meta[name=\"date\"]",
        "meta[name=\"pubdate\"]",
        "meta[name=\"publish_date\"]",
        "meta[name=\"DC.date.issued\"]",
        "time[datetime]",
    ] {
        if let Ok(sel) = Selector::parse(selector_str) {
            if let Some(el) = doc.select(&sel).next() {
                let date = el.value().attr("content")
                    .or_else(|| el.value().attr("datetime"))
                    .unwrap_or("");
                if !date.is_empty() && date.len() < 50 {
                    // Avoid duplicate "Published:" entries.
                    let already = parts.iter().any(|p| p.starts_with("Published:"));
                    if !already {
                        parts.push(format!("Published: {date}"));
                    }
                    break;
                }
            }
        }
    }

    // Price-related meta tags.
    let price_keywords = ["price", "amount", "stock", "ticker", "quote"];
    if let Ok(sel) = Selector::parse("meta") {
        for el in doc.select(&sel) {
            let name = el.value().attr("name")
                .or_else(|| el.value().attr("property"))
                .unwrap_or("");
            let content = el.value().attr("content").unwrap_or("");
            if !content.is_empty() && content.len() < 500 {
                let name_lower = name.to_ascii_lowercase();
                let is_price = price_keywords.iter().any(|kw| name_lower.contains(kw));
                if is_price {
                    parts.push(format!("meta[{name}]: {content}"));
                }
            }
        }
    }

    // JSON-LD.
    if let Ok(sel) = Selector::parse("script[type=\"application/ld+json\"]") {
        for el in doc.select(&sel).take(3) {
            let raw: String = el.text().collect();
            let raw = raw.trim();
            if raw.len() < 20 {
                continue;
            }
            // Try to parse and summarise rather than dumping raw JSON.
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw) {
                let summary = summarise_ld_json(&val);
                if !summary.is_empty() {
                    parts.push(format!("JSON-LD: {summary}"));
                }
            } else {
                // Fallback: include truncated raw JSON.
                let budget = 1500;
                if raw.len() > budget {
                    let safe = truncate_byte_boundary(raw, budget);
                    parts.push(format!("JSON-LD: {}…", &raw[..safe]));
                } else {
                    parts.push(format!("JSON-LD: {raw}"));
                }
            }
            break; // one block is enough
        }
    }

    parts.join("\n")
}

/// Produce a compact one-line summary of a JSON-LD object.
fn summarise_ld_json(val: &serde_json::Value) -> String {
    // Handle @graph arrays.
    if let Some(graph) = val.get("@graph").and_then(|g| g.as_array()) {
        let summaries: Vec<String> = graph
            .iter()
            .filter_map(|item| {
                let s = summarise_ld_json(item);
                if s.is_empty() { None } else { Some(s) }
            })
            .take(3)
            .collect();
        return summaries.join(" | ");
    }

    let mut parts: Vec<String> = Vec::new();
    let type_val = val
        .get("@type")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if !type_val.is_empty() {
        parts.push(format!("type={type_val}"));
    }
    for key in &[
        "name", "headline", "description", "tickerSymbol",
        "price", "priceCurrency", "lowPrice", "highPrice",
        "url", "exchange", "currentPrice", "previousClose",
        "openPrice", "dayLow", "dayHigh", "52WeekLow", "52WeekHigh",
        "datePublished", "dateModified", "author",
    ] {
        if let Some(v) = val.get(*key) {
            let text = match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Object(obj) => {
                    // author: { name: "..." }
                    obj.get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string()
                }
                _ => continue,
            };
            if !text.is_empty() && text.len() < 300 {
                parts.push(format!("{key}={text}"));
            }
        }
    }
    if let Some(offers) = val.get("offers") {
        let offer_summary = summarise_ld_json(offers);
        if !offer_summary.is_empty() {
            parts.push(format!("offers({offer_summary})"));
        }
    }
    if parts.len() <= 1 {
        return String::new();
    }
    parts.join("; ")
}

/// Extract readable body text from HTML, targeting article/main/body regions.
fn extract_body_text(html: &str, max_chars: usize) -> String {
    let doc = Html::parse_document(html);

    // Try focused content regions first.
    let selectors = [
        "article", "main", "[role=\"main\"]",
        ".post-content", ".entry-content", ".article-body",
        // Weather / data-heavy sites
        ".forecast", ".current-conditions", ".weather-detail",
        "#forecast", "#current", ".ten-day", ".daily-forecast",
        ".region-content-main", ".content-module",
        "[data-testid=\"forecast\"]",
        // Generic content wrappers
        "#content", ".content", "#main-content", ".page-content",
    ];
    for sel_str in &selectors {
        if let Ok(sel) = Selector::parse(sel_str) {
            if let Some(el) = doc.select(&sel).next() {
                let text = extract_text_from_element(&el, max_chars);
                if text.len() >= 80 {
                    return text;
                }
            }
        }
    }

    // Fall back to body with noise stripped.
    if let Ok(body_sel) = Selector::parse("body") {
        if let Some(body) = doc.select(&body_sel).next() {
            return extract_text_from_element(&body, max_chars);
        }
    }

    String::new()
}

/// Recursively extract text from an element, skipping noisy subtrees.
fn extract_text_from_element(el: &scraper::ElementRef<'_>, max_chars: usize) -> String {
    let skip_tags: &[&str] = &[
        "script", "style", "nav", "header", "footer", "noscript", "svg",
        "aside", "form", "iframe", "button",
    ];
    // Also skip elements by class name (cookie banners, ads, navigation chrome)
    let skip_classes: &[&str] = &[
        "cookie", "consent", "banner", "modal", "popup", "overlay",
        "ad-", "advert", "sidebar", "social-share", "newsletter",
        "privacy", "gdpr", "onetrust", "nav-", "menu",
    ];
    let block_tags: &[&str] = &[
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "td", "th", "article", "section", "main",
        "blockquote", "pre", "figcaption", "dt", "dd",
    ];

    let mut buf = String::with_capacity(max_chars + 256);
    collect_text(el, &mut buf, skip_tags, skip_classes, block_tags, max_chars);

    // Decode common HTML entities.
    let decoded = buf
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");

    collapse_whitespace(&decoded, max_chars)
}

fn collect_text(
    node: &scraper::ElementRef<'_>,
    buf: &mut String,
    skip_tags: &[&str],
    skip_classes: &[&str],
    block_tags: &[&str],
    max_chars: usize,
) {
    use scraper::Node;

    for child in node.children() {
        if buf.len() >= max_chars {
            return;
        }
        match child.value() {
            Node::Text(text) => {
                buf.push_str(text);
            }
            Node::Element(el) => {
                let tag = el.name();
                if skip_tags.contains(&tag) {
                    continue;
                }
                // Skip elements whose class contains noise keywords.
                if let Some(class_attr) = el.attr("class") {
                    let class_lower = class_attr.to_ascii_lowercase();
                    if skip_classes.iter().any(|c| class_lower.contains(c)) {
                        continue;
                    }
                }
                // Extract alt text from images (useful for weather icons, etc.)
                if tag == "img" {
                    if let Some(alt) = el.attr("alt") {
                        let alt = alt.trim();
                        if !alt.is_empty() && alt.len() < 200 {
                            buf.push_str(&format!(" [{alt}] "));
                        }
                    }
                    continue;
                }
                if block_tags.contains(&tag) {
                    buf.push('\n');
                }
                if let Some(child_ref) = scraper::ElementRef::wrap(child) {
                    collect_text(&child_ref, buf, skip_tags, skip_classes, block_tags, max_chars);
                }
            }
            _ => {}
        }
    }
}

/// Collapse runs of whitespace into single spaces / double-newlines and
/// truncate to `max_chars`.
fn collapse_whitespace(text: &str, max_chars: usize) -> String {
    let mut result = String::with_capacity(text.len().min(max_chars + 64));
    let mut prev_was_space = true;
    let mut consecutive_newlines = 0u32;

    for ch in text.chars() {
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
        let safe_end = truncate_byte_boundary(&trimmed, max_chars);
        let end = trimmed[..safe_end].rfind(' ').unwrap_or(safe_end);
        format!("{}…", &trimmed[..end])
    } else {
        trimmed
    }
}

/// Strip basic Markdown formatting to produce plain text.
fn strip_markdown_formatting(md: &str) -> String {
    let mut result = String::with_capacity(md.len());
    for line in md.lines() {
        let trimmed = line.trim();
        // Strip heading markers.
        let line = if trimmed.starts_with('#') {
            trimmed.trim_start_matches('#').trim_start()
        } else {
            trimmed
        };
        // Strip bold/italic markers.
        let line = line.replace("**", "").replace("__", "").replace('*', "").replace('_', " ");
        // Strip link syntax [text](url) → text.
        let line = strip_markdown_links(&line);
        result.push_str(&line);
        result.push('\n');
    }
    result
}

/// Convert `[text](url)` to just `text`.
fn strip_markdown_links(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '[' {
            // Collect link text until ']'.
            let mut link_text = String::new();
            let mut found_close = false;
            for inner in chars.by_ref() {
                if inner == ']' {
                    found_close = true;
                    break;
                }
                link_text.push(inner);
            }
            if found_close {
                // Skip (url) part if present.
                if chars.peek() == Some(&'(') {
                    chars.next(); // consume '('
                    for inner in chars.by_ref() {
                        if inner == ')' {
                            break;
                        }
                    }
                }
                result.push_str(&link_text);
            } else {
                result.push('[');
                result.push_str(&link_text);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Truncate content to a character budget, preferring a newline boundary.
fn truncate_content(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        return content.to_string();
    }
    let safe = truncate_byte_boundary(content, max_chars);
    let end = content[..safe].rfind('\n').unwrap_or(safe);
    format!("{}…(truncated)", &content[..end])
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_is_correct() {
        let tool = BrowsePageTool;
        let spec = tool.spec();
        assert_eq!(spec.name, "browse_page");
        assert!(spec.params.iter().any(|p| p.name == "url" && p.required));
        assert!(spec.params.iter().any(|p| p.name == "mode" && !p.required));
        assert!(spec.params.iter().any(|p| p.name == "max_chars" && !p.required));
        assert!(spec.params.iter().any(|p| p.name == "instructions" && !p.required));
        assert!(spec.metadata.read_only);
        assert_eq!(spec.metadata.group, "web");
    }

    #[test]
    fn is_private_ip_localhost() {
        assert!(is_private_ip("127.0.0.1".parse().unwrap()));
        assert!(is_private_ip("::1".parse().unwrap()));
    }

    #[test]
    fn is_private_ip_rfc1918() {
        assert!(is_private_ip("10.0.0.1".parse().unwrap()));
        assert!(is_private_ip("172.16.0.1".parse().unwrap()));
        assert!(is_private_ip("192.168.1.1".parse().unwrap()));
    }

    #[test]
    fn is_private_ip_public() {
        assert!(!is_private_ip("8.8.8.8".parse().unwrap()));
        assert!(!is_private_ip("1.1.1.1".parse().unwrap()));
    }

    #[test]
    fn private_url_detection() {
        assert!(is_private_url("http://localhost:8080/admin"));
        assert!(is_private_url("http://0.0.0.0/secret"));
        assert!(is_private_url("http://foo.local/api"));
        // Public URLs should pass.
        assert!(!is_private_url("https://example.com/page"));
    }

    #[test]
    fn extract_structured_data_title() {
        let html = "<html><head><title>Test Page</title></head><body></body></html>";
        let result = extract_structured_data(html);
        assert!(result.contains("Title: Test Page"));
    }

    #[test]
    fn extract_structured_data_og_tags() {
        let html = r#"<html><head>
            <meta property="og:title" content="My Article">
            <meta property="og:description" content="A great article">
        </head><body></body></html>"#;
        let result = extract_structured_data(html);
        assert!(result.contains("OG Title: My Article"));
        assert!(result.contains("OG Description: A great article"));
    }

    #[test]
    fn extract_structured_data_published_date() {
        let html = r#"<html><head>
            <title>Post</title>
            <meta property="article:published_time" content="2025-01-15T10:00:00Z">
        </head><body></body></html>"#;
        let result = extract_structured_data(html);
        assert!(result.contains("Published: 2025-01-15"));
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

    #[test]
    fn collapse_whitespace_works() {
        let result = collapse_whitespace("  hello   world  ", 1000);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn strip_markdown_links_works() {
        assert_eq!(
            strip_markdown_links("See [this link](https://example.com) for details"),
            "See this link for details"
        );
    }

    #[test]
    fn strip_markdown_formatting_works() {
        let md = "# Hello\n\n**Bold** and *italic* [link](url)";
        let text = strip_markdown_formatting(md);
        assert!(text.contains("Hello"));
        assert!(text.contains("Bold"));
        assert!(!text.contains("**"));
        assert!(!text.contains("](url)"));
    }

    #[test]
    fn truncate_content_short() {
        assert_eq!(truncate_content("short", 100), "short");
    }

    #[test]
    fn truncate_content_long() {
        let long = "a".repeat(200);
        let result = truncate_content(&long, 50);
        assert!(result.len() < 70); // 50 + "(truncated)"
        assert!(result.contains("truncated"));
    }

    #[test]
    fn url_parsing_comma_separated() {
        let raw = "https://example.com/a, https://example.com/b, not-a-url";
        let urls: Vec<String> = raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.starts_with("http"))
            .take(MAX_BATCH_URLS)
            .collect();
        assert_eq!(urls.len(), 2);
    }

    #[tokio::test]
    async fn missing_url_errors() {
        let tool = BrowsePageTool;
        let result = tool.run(&HashMap::new()).await;
        assert!(result.is_err() || !result.unwrap().success);
    }

    #[tokio::test]
    async fn empty_url_errors() {
        let tool = BrowsePageTool;
        let mut args = HashMap::new();
        args.insert("url".to_string(), "".to_string());
        let result = tool.run(&args).await;
        assert!(result.is_err() || !result.unwrap().success);
    }

    #[tokio::test]
    async fn non_http_url_rejected() {
        let tool = BrowsePageTool;
        let mut args = HashMap::new();
        args.insert("url".to_string(), "ftp://example.com".to_string());
        let result = tool.run(&args).await.unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    #[ignore] // network — run with `cargo test -- --ignored`
    async fn fetch_wikipedia() {
        let tool = BrowsePageTool;
        let mut args = HashMap::new();
        args.insert(
            "url".to_string(),
            "https://en.wikipedia.org/wiki/Rust_(programming_language)".to_string(),
        );
        args.insert("mode".to_string(), "markdown".to_string());
        let result = tool.run(&args).await.expect("should not error");
        assert!(result.success, "output: {}", result.output);
        assert!(result.output.contains("Rust"), "output: {}", result.output);
    }
}
