//! Web search tool.
//!
//! Exposes **`web_search`** — returns titles, URLs, and search-engine snippets.
//! Supports pluggable providers: Brave, Tavily, SearXNG, DuckDuckGo.
//!
//! Search results are cached with a configurable TTL to avoid redundant API calls.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use percent_encoding::percent_decode_str;
use scraper::{Html, Selector};

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};
use super::browse::{build_client, extract_body_text as html_to_text, extract_structured_data};
use super::fs::truncate_byte_boundary;
use super::cache::TtlCache;

// ─── Constants ───────────────────────────────────────────────────────────────

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

/// Default TTL for cached search results (5 minutes).
const CACHE_TTL: Duration = Duration::from_secs(300);
/// Maximum number of cached search entries.
const CACHE_MAX_ENTRIES: usize = 64;

/// Global search result cache.
static SEARCH_CACHE: std::sync::LazyLock<TtlCache> =
    std::sync::LazyLock::new(|| TtlCache::new(CACHE_MAX_ENTRIES, CACHE_TTL));

/// Cache key: combine query + max_results for uniqueness.
fn cache_key(query: &str, max_results: usize) -> String {
    format!("search:{}:{}", query.to_lowercase().trim(), max_results)
}

// ═══════════════════════════════════════════════════════════════════════════
//  WebSearchTool — search-only, no automatic page fetching
// ═══════════════════════════════════════════════════════════════════════════

/// Searches the web and returns titles, URLs, and snippets.
///
/// Supports multiple search providers in a configurable priority chain:
/// Brave Search, Tavily, SearXNG, and DuckDuckGo (fallback).
/// The tool tries providers in order and uses the first one with valid credentials.
pub struct WebSearchTool {
    pub brave_api_key: Option<String>,
    /// Tavily API key.  When set, enables the Tavily search provider.
    pub tavily_api_key: Option<String>,
    /// SearXNG instance base URL (e.g. "http://localhost:8080").
    pub searxng_base_url: Option<String>,
    /// Serper (Google Search) API key.
    pub serper_api_key: Option<String>,
    /// Exa.ai API key for semantic/neural search.
    pub exa_api_key: Option<String>,
    /// Provider priority order.  Default: ["brave", "tavily", "searxng", "duckduckgo"]
    pub search_providers: Vec<String>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_search".to_string(),
            description: "Search the web and return result titles, URLs, snippets, and \
                structured data extracted from top results (meta tags, JSON-LD). \
                This often includes live prices, stats, and other facts directly. \
                Use `browse_page` only when you need the full body text of a specific page."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "query".to_string(),
                    description: "Search query string".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "max_results".to_string(),
                    description: "Maximum results to return (default: 5)".to_string(),
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
        let query = args
            .get("query")
            .ok_or_else(|| anyhow::anyhow!("missing required param: query"))?;
        let max_results: usize = args
            .get("max_results")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5)
            .clamp(1, 20);

        // Check cache first.
        let key = cache_key(query, max_results);
        if let Some(cached) = SEARCH_CACHE.get(&key) {
            return Ok(cached);
        }

        // Resolve credentials from struct fields (config) or env vars.
        let brave_key: Option<String> = self
            .brave_api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| std::env::var("BRAVE_API_KEY").ok().filter(|k| !k.trim().is_empty()));
        let tavily_key: Option<String> = self
            .tavily_api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| std::env::var("TAVILY_API_KEY").ok().filter(|k| !k.trim().is_empty()));
        let searxng_url: Option<String> = self
            .searxng_base_url
            .clone()
            .filter(|u| !u.trim().is_empty())
            .or_else(|| std::env::var("SEARXNG_BASE_URL").ok().filter(|u| !u.trim().is_empty()));
        let serper_key: Option<String> = self
            .serper_api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| std::env::var("SERPER_API_KEY").ok().filter(|k| !k.trim().is_empty()));
        let exa_key: Option<String> = self
            .exa_api_key
            .clone()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| std::env::var("EXA_API_KEY").ok().filter(|k| !k.trim().is_empty()));

        // Try providers in configured priority order.
        let providers = if self.search_providers.is_empty() {
            vec!["brave".to_string(), "tavily".to_string(), "serper".to_string(), "exa".to_string(), "searxng".to_string(), "duckduckgo".to_string()]
        } else {
            self.search_providers.clone()
        };

        let result = 'provider: {
            for provider in &providers {
                match provider.as_str() {
                    "brave" => {
                        if let Some(ref key) = brave_key {
                            break 'provider search_brave(query, max_results, key).await;
                        }
                    }
                    "tavily" => {
                        if let Some(ref key) = tavily_key {
                            break 'provider search_tavily(query, max_results, key).await;
                        }
                    }
                    "serper" => {
                        if let Some(ref key) = serper_key {
                            break 'provider search_serper(query, max_results, key).await;
                        }
                    }
                    "exa" => {
                        if let Some(ref key) = exa_key {
                            break 'provider search_exa(query, max_results, key).await;
                        }
                    }
                    "searxng" => {
                        if let Some(ref url) = searxng_url {
                            break 'provider search_searxng(query, max_results, url).await;
                        }
                    }
                    "duckduckgo" => {
                        break 'provider search_duckduckgo(query, max_results).await;
                    }
                    _ => {} // Unknown provider — skip
                }
            }
            // Final fallback: DuckDuckGo (always available, no API key needed)
            search_duckduckgo(query, max_results).await
        };

        // Cache successful results.
        if let Ok(ref output) = result {
            if output.success {
                SEARCH_CACHE.insert(key, output.clone());
            }
        }

        result
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

    // Enrich with structured data + body text from top result pages.
    let enrichment = enrich_with_structured_data(&urls, 3).await;
    let mut output = parts.join("\n\n");
    if !enrichment.is_empty() {
        output.push_str("\n\n--- Data extracted from top results ---\n");
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

    // Enrich with structured data + body text from top result pages.
    let enrichment = enrich_with_structured_data(&urls, 3).await;
    let mut output = parts.join("\n\n");
    if !enrichment.is_empty() {
        output.push_str("\n\n--- Data extracted from top results ---\n");
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
//  Tavily Search — AI-optimized search with extracted content
// ═══════════════════════════════════════════════════════════════════════════

async fn search_tavily(
    query: &str,
    max_results: usize,
    api_key: &str,
) -> Result<ToolOutput> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let payload = serde_json::json!({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": true,
        "include_raw_content": false,
        "search_depth": "basic"
    });

    let resp = client
        .post("https://api.tavily.com/search")
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Tavily Search API error {}: {}", status, body);
    }

    let json: serde_json::Value = resp.json().await?;
    let mut parts: Vec<String> = Vec::new();

    // Tavily's AI-generated answer (when available) is highly valuable.
    if let Some(answer) = json.get("answer").and_then(|v| v.as_str()) {
        if !answer.is_empty() {
            parts.push(format!("[Tavily AI Answer] {answer}"));
        }
    }

    // Individual search results with content snippets.
    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        for item in results.iter().take(max_results) {
            let title = item["title"].as_str().unwrap_or("").trim();
            let url = item["url"].as_str().unwrap_or("").trim();
            let content = item["content"].as_str().unwrap_or("").trim();
            let score = item["score"].as_f64().unwrap_or(0.0);
            if !title.is_empty() {
                parts.push(format!("{title} (relevance: {score:.2})\n  {url}\n  {content}"));
            }
        }
    }

    if parts.is_empty() {
        return Ok(ToolOutput {
            success: true,
            output: format!("No Tavily results for: {query}"),
        });
    }

    Ok(ToolOutput {
        success: true,
        output: parts.join("\n\n"),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  SearXNG Search — self-hosted metasearch
// ═══════════════════════════════════════════════════════════════════════════

async fn search_searxng(
    query: &str,
    max_results: usize,
    base_url: &str,
) -> Result<ToolOutput> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent(USER_AGENT)
        .build()?;

    let endpoint = format!("{}/search", base_url.trim_end_matches('/'));
    let resp = client
        .get(&endpoint)
        .query(&[
            ("q", query),
            ("format", "json"),
            ("categories", "general"),
        ])
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("SearXNG error {}: {}", status, body);
    }

    let json: serde_json::Value = resp.json().await?;
    let mut parts: Vec<String> = Vec::new();
    let mut urls: Vec<String> = Vec::new();

    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        for item in results.iter().take(max_results) {
            let title = item["title"].as_str().unwrap_or("").trim();
            let url = item["url"].as_str().unwrap_or("").trim();
            let content = item["content"].as_str().unwrap_or("").trim();
            let engine = item["engine"].as_str().unwrap_or("unknown");
            if !title.is_empty() {
                parts.push(format!("{title} (via {engine})\n  {url}\n  {content}"));
                if url.starts_with("http") {
                    urls.push(url.to_string());
                }
            }
        }
    }

    // SearXNG infobox (knowledge graph)
    if let Some(infoboxes) = json.get("infoboxes").and_then(|v| v.as_array()) {
        for ib in infoboxes.iter().take(1) {
            let title = ib["infobox"].as_str().unwrap_or("");
            let content = ib["content"].as_str().unwrap_or("");
            if !title.is_empty() && !content.is_empty() {
                parts.insert(0, format!("[Infobox] {title}\n  {content}"));
            }
        }
    }

    if parts.is_empty() {
        return Ok(ToolOutput {
            success: true,
            output: format!("No SearXNG results for: {query}"),
        });
    }

    // Enrich top results with body text (same as Brave/DDG path).
    let enrichment = enrich_with_structured_data(&urls, 2).await;
    let mut output = parts.join("\n\n");
    if !enrichment.is_empty() {
        output.push_str("\n\n--- Data extracted from top results ---\n");
        output.push_str(&enrichment);
    }

    Ok(ToolOutput {
        success: true,
        output,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Serper (Google Search) — via serper.dev API
// ═══════════════════════════════════════════════════════════════════════════

async fn search_serper(
    query: &str,
    max_results: usize,
    api_key: &str,
) -> Result<ToolOutput> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let payload = serde_json::json!({
        "q": query,
        "num": max_results
    });

    let resp = client
        .post("https://google.serper.dev/search")
        .header("X-API-KEY", api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Serper API error {}: {}", status, body);
    }

    let json: serde_json::Value = resp.json().await?;
    let mut parts: Vec<String> = Vec::new();
    let mut urls: Vec<String> = Vec::new();

    // Answer box (direct answer from Google).
    if let Some(ab) = json.get("answerBox") {
        let answer = ab.get("answer")
            .or_else(|| ab.get("snippet"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let title = ab.get("title").and_then(|v| v.as_str()).unwrap_or("");
        if !answer.is_empty() {
            parts.push(format!("[Answer Box] {title}\n  {answer}"));
        }
    }

    // Knowledge graph.
    if let Some(kg) = json.get("knowledgeGraph") {
        let title = kg.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let desc = kg.get("description").and_then(|v| v.as_str()).unwrap_or("");
        if !title.is_empty() && !desc.is_empty() {
            let mut kg_text = format!("[Knowledge Graph] {title}\n  {desc}");
            if let Some(attrs) = kg.get("attributes").and_then(|v| v.as_object()) {
                for (k, v) in attrs.iter().take(5) {
                    if let Some(val) = v.as_str() {
                        kg_text.push_str(&format!("\n  {k}: {val}"));
                    }
                }
            }
            parts.push(kg_text);
        }
    }

    // Organic results.
    if let Some(results) = json.get("organic").and_then(|v| v.as_array()) {
        for item in results.iter().take(max_results) {
            let title = item["title"].as_str().unwrap_or("").trim();
            let link = item["link"].as_str().unwrap_or("").trim();
            let snippet = item["snippet"].as_str().unwrap_or("").trim();
            let position = item["position"].as_u64().unwrap_or(0);
            if !title.is_empty() {
                parts.push(format!("{title} (#{position})\n  {link}\n  {snippet}"));
                if link.starts_with("http") {
                    urls.push(link.to_string());
                }
            }
        }
    }

    if parts.is_empty() {
        return Ok(ToolOutput {
            success: true,
            output: format!("No Serper results for: {query}"),
        });
    }

    // Enrich top results with structured data.
    let enrichment = enrich_with_structured_data(&urls, 2).await;
    let mut output = parts.join("\n\n");
    if !enrichment.is_empty() {
        output.push_str("\n\n--- Data extracted from top results ---\n");
        output.push_str(&enrichment);
    }

    Ok(ToolOutput {
        success: true,
        output,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exa.ai — neural / semantic web search
// ═══════════════════════════════════════════════════════════════════════════

async fn search_exa(
    query: &str,
    max_results: usize,
    api_key: &str,
) -> Result<ToolOutput> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let payload = serde_json::json!({
        "query": query,
        "numResults": max_results,
        "type": "auto",
        "useAutoprompt": true,
        "contents": {
            "text": {
                "maxCharacters": 1000
            }
        }
    });

    let resp = client
        .post("https://api.exa.ai/search")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Exa.ai API error {}: {}", status, body);
    }

    let json: serde_json::Value = resp.json().await?;
    let mut parts: Vec<String> = Vec::new();

    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        for item in results.iter().take(max_results) {
            let title = item["title"].as_str().unwrap_or("").trim();
            let url = item["url"].as_str().unwrap_or("").trim();
            let text = item["text"].as_str().unwrap_or("").trim();
            let score = item["score"].as_f64().unwrap_or(0.0);
            let published = item["publishedDate"].as_str().unwrap_or("");
            if !title.is_empty() {
                let mut entry = format!("{title} (relevance: {score:.2})");
                if !published.is_empty() {
                    entry.push_str(&format!(" [published: {published}]"));
                }
                entry.push_str(&format!("\n  {url}"));
                if !text.is_empty() {
                    // Truncate text to ~500 chars for conciseness.
                    let snippet = if text.len() > 500 { &text[..500] } else { text };
                    entry.push_str(&format!("\n  {snippet}"));
                }
                parts.push(entry);
            }
        }
    }

    if parts.is_empty() {
        return Ok(ToolOutput {
            success: true,
            output: format!("No Exa results for: {query}"),
        });
    }

    Ok(ToolOutput {
        success: true,
        output: parts.join("\n\n"),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Search result enrichment — fetch structured data from top URLs
// ═══════════════════════════════════════════════════════════════════════════

/// Fetch the top `max_urls` search result pages in parallel and extract
/// structured data (JSON-LD, meta tags, title) **plus** body text from the
/// top results.  This ensures the LLM sees actual page content — not just
/// metadata — so it can quote real prices, temperatures, scores, etc.
///
/// Returns a single string with one section per successfully enriched URL.
async fn enrich_with_structured_data(urls: &[String], max_urls: usize) -> String {
    const ENRICH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(6);
    const ENRICH_MAX_BYTES: usize = 256_000;
    /// Per-page character budget for extracted body text.
    const BODY_TEXT_BUDGET: usize = 4000;

    let client = match reqwest::Client::builder()
        .timeout(ENRICH_TIMEOUT)
        .user_agent(USER_AGENT)
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return String::new(),
    };

    let fetches = urls.iter().take(max_urls).enumerate().map(|(idx, url)| {
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

            // Also extract body text from the top results so the LLM gets
            // real data (prices, temperatures, stats) not just metadata.
            let page_text = if idx < 2 {
                let text = html_to_text(body_slice, BODY_TEXT_BUDGET);
                if text.len() >= 40 { Some(text) } else { None }
            } else {
                None
            };

            if structured.is_empty() && page_text.is_none() {
                None
            } else {
                Some((url, structured, page_text))
            }
        })
    })
    .collect::<Vec<_>>();

    let mut sections: Vec<String> = Vec::new();
    for handle in fetches {
        if let Ok(Some((url, data, page_text))) = handle.await {
            let mut section = format!("[{url}]");
            if !data.is_empty() {
                section.push('\n');
                section.push_str(&data);
            }
            if let Some(text) = page_text {
                section.push_str("\n--- Page content ---\n");
                section.push_str(&text);
            }
            sections.push(section);
        }
    }
    sections.join("\n\n")
}

