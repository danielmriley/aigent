//! Web search, page fetching, and HTML extraction.
//!
//! Two tools are exposed:
//!
//! * **`web_search`** — returns titles, URLs, and search-engine snippets.
//!   Uses Brave Search when an API key is available, otherwise scrapes the
//!   DuckDuckGo HTML search page.
//! * **`fetch_page`** — fetches a single URL and returns the extracted
//!   plain-text content, using `scraper` for robust HTML parsing.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use percent_encoding::percent_decode_str;
use scraper::{Html, Selector};

use crate::{Tool, ToolSpec, ToolParam, ToolOutput};
use super::fs::truncate_byte_boundary;

// ─── Constants ───────────────────────────────────────────────────────────────

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

const MAX_DOWNLOAD_BYTES: usize = 256_000;

// ═══════════════════════════════════════════════════════════════════════════
//  WebSearchTool — search-only, no automatic page fetching
// ═══════════════════════════════════════════════════════════════════════════

/// Searches the web and returns titles, URLs, and snippets.
///
/// When `brave_api_key` is set (or the `BRAVE_API_KEY` env var is non-empty)
/// the [Brave Search API](https://api.search.brave.com/app/documentation/web-search)
/// is used, providing higher-quality results.  Otherwise the tool falls back
/// to scraping DuckDuckGo HTML search.
pub struct WebSearchTool {
    pub brave_api_key: Option<String>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_search".to_string(),
            description: "Search the web and return result titles, URLs, snippets, and \
                structured data extracted from top results (meta tags, JSON-LD). \
                This often includes live prices, stats, and other facts directly. \
                Use `fetch_page` only when you need the full body text of a specific page."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "query".to_string(),
                    description: "Search query string".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "max_results".to_string(),
                    description: "Maximum results to return (default: 5)".to_string(),
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

        let brave_key: Option<String> = self
            .brave_api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| std::env::var("BRAVE_API_KEY").ok().filter(|k| !k.trim().is_empty()));

        if let Some(ref key) = brave_key {
            search_brave(query, max_results, key).await
        } else {
            search_duckduckgo(query, max_results).await
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  FetchPageTool — fetch a single URL and extract text
// ═══════════════════════════════════════════════════════════════════════════

/// Fetches a web page and returns extracted plain-text content.
pub struct FetchPageTool;

#[async_trait]
impl Tool for FetchPageTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "fetch_page".to_string(),
            description: "Fetch a web page URL and return its full extracted text content. \
                Use this when you need the complete body text of a specific page \
                (e.g. reading an article, documentation, or detailed content). \
                For quick facts like prices or stats, web_search already includes \
                structured data from top results."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "url".to_string(),
                    description: "The URL to fetch".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "max_chars".to_string(),
                    description: "Maximum characters to return (default: 8000)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let url = args
            .get("url")
            .ok_or_else(|| anyhow::anyhow!("missing required param: url"))?;
        let max_chars: usize = args
            .get("max_chars")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8000);

        let client = build_client()?;
        match fetch_page_text(&client, url, max_chars).await {
            Ok(text) if !text.is_empty() => Ok(ToolOutput {
                success: true,
                output: text,
            }),
            Ok(_) => Ok(ToolOutput {
                success: false,
                output: format!("Page returned no extractable text: {url}"),
            }),
            Err(reason) => Ok(ToolOutput {
                success: false,
                output: format!("Could not fetch {url}: {reason}"),
            }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Brave Search
// ═══════════════════════════════════════════════════════════════════════════

async fn search_brave(
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
    let mut urls: Vec<String> = Vec::new();

    // ── Infobox / knowledge panel ──────────────────────────────────────
    // Brave returns these for stock tickers, company lookups, etc.
    if let Some(ib) = json.get("infobox") {
        if let Some(results) = ib.get("results").and_then(|r| r.as_array()) {
            for item in results.iter().take(2) {
                let mut ib_parts: Vec<String> = Vec::new();
                if let Some(t) = item["title"].as_str() {
                    ib_parts.push(format!("[Infobox] {t}"));
                }
                if let Some(d) = item["description"].as_str() {
                    ib_parts.push(format!("  {d}"));
                }
                if let Some(ld) = item["long_desc"].as_str() {
                    if ld.len() < 300 {
                        ib_parts.push(format!("  {ld}"));
                    }
                }
                // Key-value attributes (stock price, market cap, etc.)
                if let Some(attrs) = item.get("attributes").and_then(|a| a.as_array()) {
                    for attr in attrs {
                        if let Some(pair) = attr.as_array() {
                            let key = pair.first().and_then(|v| v.as_str()).unwrap_or("");
                            let val = pair.get(1).and_then(|v| v.as_str()).unwrap_or("");
                            if !key.is_empty() && !val.is_empty() {
                                ib_parts.push(format!("  {key}: {val}"));
                            }
                        }
                    }
                }
                // Key-value data (alt format Brave sometimes uses)
                if let Some(data) = item.get("data").and_then(|d| d.as_object()) {
                    for (k, v) in data {
                        if let Some(s) = v.as_str() {
                            ib_parts.push(format!("  {k}: {s}"));
                        }
                    }
                }
                if !ib_parts.is_empty() {
                    parts.push(ib_parts.join("\n"));
                }
            }
        }
    }

    // ── Web results ────────────────────────────────────────────────────
    if let Some(results) = json["web"]["results"].as_array() {
        for item in results.iter().take(max_results) {
            let title = item["title"].as_str().unwrap_or("").trim();
            let url = item["url"].as_str().unwrap_or("").trim();
            let desc = item["description"].as_str().unwrap_or("").trim();
            if !title.is_empty() {
                parts.push(format!("{title}\n  {url}\n  {desc}"));
                if url.starts_with("http") {
                    urls.push(url.to_string());
                }
            }
        }
    }

    if parts.is_empty() {
        return Ok(ToolOutput {
            success: true,
            output: format!("No Brave Search results for: {query}"),
        });
    }

    // Enrich with structured data from top result pages.
    let enrichment = enrich_with_structured_data(&urls, 3).await;
    let mut output = parts.join("\n\n");
    if !enrichment.is_empty() {
        output.push_str("\n\n--- Structured data from top results ---\n");
        output.push_str(&enrichment);
    }
    Ok(ToolOutput {
        success: true,
        output,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  DuckDuckGo HTML Search (replaces Instant Answers API)
// ═══════════════════════════════════════════════════════════════════════════

async fn search_duckduckgo(
    query: &str,
    max_results: usize,
) -> Result<ToolOutput> {
    let client = build_client()?;

    // POST to the DuckDuckGo HTML search endpoint.
    let resp = client
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query)])
        .header("Accept", "text/html")
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        anyhow::bail!("DuckDuckGo HTML search error: {}", status);
    }

    let body = resp.text().await?;

    // Parse the HTML and extract search results synchronously.
    // The scraper `Html` type is !Send, so it must not live across an await.
    let (parts, urls) = {
        let doc = Html::parse_document(&body);

        let result_sel = Selector::parse(".result").unwrap();
        let link_sel = Selector::parse("a.result__a").unwrap();
        let snippet_sel = Selector::parse("a.result__snippet, .result__snippet").unwrap();

        let mut parts: Vec<String> = Vec::new();
        let mut urls: Vec<String> = Vec::new();
        for result in doc.select(&result_sel).take(max_results) {
            let title = result
                .select(&link_sel)
                .next()
                .map(|el| el.text().collect::<String>())
                .unwrap_or_default();
            let title = title.trim();

            let url = result
                .select(&link_sel)
                .next()
                .and_then(|el| el.value().attr("href"))
                .unwrap_or("");

            let url = extract_ddg_url(url);

            let snippet = result
                .select(&snippet_sel)
                .next()
                .map(|el| el.text().collect::<String>())
                .unwrap_or_default();
            let snippet = snippet.trim();

            if !title.is_empty() {
                if url.starts_with("http") {
                    urls.push(url.clone());
                }
                parts.push(format!("{title}\n  {url}\n  {snippet}"));
            }
        }
        (parts, urls)
    }; // `doc` dropped here — before any .await

    if parts.is_empty() {
        return Ok(ToolOutput {
            success: true,
            output: format!("No DuckDuckGo results for: {query}"),
        });
    }

    // Enrich with structured data from top result pages.
    let enrichment = enrich_with_structured_data(&urls, 3).await;
    let mut output = parts.join("\n\n");
    if !enrichment.is_empty() {
        output.push_str("\n\n--- Structured data from top results ---\n");
        output.push_str(&enrichment);
    }
    Ok(ToolOutput {
        success: true,
        output,
    })
}

/// DDG sometimes wraps result URLs in redirect links like
/// `//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...`.
/// Extract and percent-decode the actual destination URL.
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

// ═══════════════════════════════════════════════════════════════════════════
//  Search result enrichment — fetch structured data from top URLs
// ═══════════════════════════════════════════════════════════════════════════

/// Fetch the top `max_urls` search result pages in parallel and extract only
/// structured data (JSON-LD, meta tags, title).  This is fast and works even
/// when the page is a JS SPA, because the structured metadata is embedded in
/// the `<head>` before any client-side rendering.
///
/// Returns a single string with one section per successfully enriched URL.
async fn enrich_with_structured_data(urls: &[String], max_urls: usize) -> String {
    const ENRICH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(4);
    const ENRICH_MAX_BYTES: usize = 128_000; // only need <head> section

    let client = match reqwest::Client::builder()
        .timeout(ENRICH_TIMEOUT)
        .user_agent(USER_AGENT)
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return String::new(),
    };

    let fetches = urls.iter().take(max_urls).map(|url| {
        let client = client.clone();
        let url = url.clone();
        tokio::spawn(async move {
            let resp = client
                .get(&url)
                .header("Accept", "text/html")
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            let ct = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if !ct.contains("text/html") {
                return None;
            }
            let body = resp.text().await.ok()?;
            let body_slice: &str = if body.len() > ENRICH_MAX_BYTES {
                &body[..truncate_byte_boundary(&body, ENRICH_MAX_BYTES)]
            } else {
                &body
            };
            let structured = extract_structured_data(body_slice);
            if structured.is_empty() {
                None
            } else {
                Some((url, structured))
            }
        })
    })
    .collect::<Vec<_>>();

    let mut sections: Vec<String> = Vec::new();
    for handle in fetches {
        if let Ok(Some((url, data))) = handle.await {
            sections.push(format!("[{url}]\n{data}"));
        }
    }
    sections.join("\n\n")
}

// ═══════════════════════════════════════════════════════════════════════════
//  Page Fetching & Content Extraction (scraper-based)
// ═══════════════════════════════════════════════════════════════════════════

fn build_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent(USER_AGENT)
        .build()?)
}

/// Fetch a URL and return extracted plain-text content.
///
/// Returns `Err(reason)` with a human-readable failure reason so the LLM
/// can decide whether to retry or move on.
async fn fetch_page_text(
    client: &reqwest::Client,
    url: &str,
    max_chars: usize,
) -> std::result::Result<String, String> {
    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(8))
        .header("Accept", "text/html")
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                "request timed out after 8s".to_string()
            } else if e.is_connect() {
                format!("connection failed: {e}")
            } else {
                format!("request error: {e}")
            }
        })?;

    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.contains("text/html") && !content_type.contains("text/plain") {
        return Err(format!("unsupported content type: {content_type}"));
    }

    let body = resp.text().await.map_err(|e| format!("failed to read body: {e}"))?;
    let body = if body.len() > MAX_DOWNLOAD_BYTES {
        let end = truncate_byte_boundary(&body, MAX_DOWNLOAD_BYTES);
        &body[..end]
    } else {
        &body
    };

    // Extract structured data first (survives JS-heavy SPAs).
    let structured = extract_structured_data(body);
    let plain = html_to_text(body, max_chars);

    if structured.is_empty() && plain.is_empty() {
        return Err("page contained no extractable text".to_string());
    }

    if structured.is_empty() {
        Ok(plain)
    } else if plain.is_empty() {
        Ok(structured)
    } else {
        // Budget: give structured data up to 1/3 of max_chars, rest to plain text.
        let struct_budget = max_chars / 3;
        let struct_part = if structured.len() > struct_budget {
            let end = truncate_byte_boundary(&structured, struct_budget);
            format!("{}…", &structured[..end])
        } else {
            structured
        };
        Ok(format!("{struct_part}\n\n{plain}"))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  HTML → text via `scraper`
// ═══════════════════════════════════════════════════════════════════════════

/// Parse HTML and extract readable text content.
///
/// Strategy:
/// 1. Try `<article>` or `<main>` for the content region.
/// 2. Fall back to `<body>` with `<script>`, `<style>`, `<nav>`, `<header>`,
///    `<footer>`, `<noscript>`, `<svg>` subtrees stripped.
/// 3. Collapse whitespace and truncate to `max_chars`.
pub(super) fn html_to_text(html: &str, max_chars: usize) -> String {
    let doc = Html::parse_document(html);

    // Try focused content regions first.
    let content_selectors = ["article", "main", "[role=\"main\"]"];
    for sel_str in &content_selectors {
        if let Ok(sel) = Selector::parse(sel_str) {
            if let Some(el) = doc.select(&sel).next() {
                let text = extract_text_from_element(&el, max_chars);
                if text.len() >= 100 {
                    return text;
                }
            }
        }
    }

    // Fall back to <body>.
    if let Ok(body_sel) = Selector::parse("body") {
        if let Some(body) = doc.select(&body_sel).next() {
            return extract_text_from_element(&body, max_chars);
        }
    }

    // Last resort: extract all text from the document.
    let raw: String = doc.root_element().text().collect();
    collapse_whitespace(&raw, max_chars)
}

/// Recursively extract text from an element, skipping noisy subtrees.
fn extract_text_from_element(
    el: &scraper::ElementRef<'_>,
    max_chars: usize,
) -> String {
    let skip_tags: &[&str] = &[
        "script", "style", "nav", "header", "footer", "noscript", "svg",
        "aside", "form", "iframe",
    ];
    let block_tags: &[&str] = &[
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "td", "th", "article", "section", "main",
        "blockquote", "pre", "figcaption", "dt", "dd",
    ];

    let mut buf = String::with_capacity(max_chars + 256);
    collect_text(el, &mut buf, skip_tags, block_tags, max_chars);

    // Decode common HTML entities that survive parsing.
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
                if block_tags.contains(&tag) {
                    buf.push('\n');
                }
                if let Some(child_ref) = scraper::ElementRef::wrap(child) {
                    collect_text(&child_ref, buf, skip_tags, block_tags, max_chars);
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

// ═══════════════════════════════════════════════════════════════════════════
//  Structured data extraction (JSON-LD, meta tags, title)
// ═══════════════════════════════════════════════════════════════════════════

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
    let doc = Html::parse_document(html);
    let mut lines: Vec<String> = Vec::new();

    // ── <title> ────────────────────────────────────────────────────────────
    if let Ok(sel) = Selector::parse("title") {
        if let Some(el) = doc.select(&sel).next() {
            let title: String = el.text().collect();
            let title = title.trim();
            if !title.is_empty() && title.len() < 500 {
                lines.push(format!("Title: {title}"));
            }
        }
    }

    // ── <meta> tags ────────────────────────────────────────────────────────
    let interesting_attrs = [
        "og:title", "og:description", "og:type",
        "description", "twitter:title", "twitter:description",
    ];
    let price_keywords = ["price", "amount", "stock", "ticker", "quote"];

    if let Ok(sel) = Selector::parse("meta") {
        for el in doc.select(&sel) {
            let name = el.value().attr("name")
                .or_else(|| el.value().attr("property"))
                .unwrap_or("");
            let content = el.value().attr("content").unwrap_or("");

            if !content.is_empty() && content.len() < 500 {
                let name_lower = name.to_ascii_lowercase();
                let is_interesting = interesting_attrs.iter().any(|a| name_lower == *a)
                    || price_keywords.iter().any(|kw| name_lower.contains(kw));
                if is_interesting {
                    lines.push(format!("meta[{name}]: {content}"));
                }
            }
        }
    }

    // ── <script type="application/ld+json"> ────────────────────────────────
    if let Ok(sel) = Selector::parse("script[type=\"application/ld+json\"]") {
        for el in doc.select(&sel) {
            let json_str: String = el.text().collect();
            let json_str = json_str.trim();
            if !json_str.is_empty() && json_str.len() < 8000 {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    let summary = summarise_ld_json(&val);
                    if !summary.is_empty() {
                        lines.push(format!("LD+JSON: {summary}"));
                    }
                }
            }
        }
    }

    lines.join("\n")
}

/// Produce a compact one-line summary of a JSON-LD object.
fn summarise_ld_json(val: &serde_json::Value) -> String {
    // Handle @graph arrays (common wrapper).
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
    ] {
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

