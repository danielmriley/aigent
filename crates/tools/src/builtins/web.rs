//! Web search, page fetching, and HTML extraction.
//!
//! Three tools are exposed:
//!
//! * **`web_search`** — returns titles, URLs, and search-engine snippets.
//!   Supports pluggable providers: Brave, Tavily, SearXNG, DuckDuckGo.
//! * **`fetch_page`** — fetches one or more URLs, returning extracted content
//!   in plain text or markdown (with headings, links, code blocks, tables).
//!
//! Search results are cached with a configurable TTL to avoid redundant API calls.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use percent_encoding::percent_decode_str;
use scraper::{Html, Selector};

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};
use super::fs::truncate_byte_boundary;

// ─── Constants ───────────────────────────────────────────────────────────────

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

const MAX_DOWNLOAD_BYTES: usize = 256_000;

/// Default TTL for cached search results (5 minutes).
const CACHE_TTL: Duration = Duration::from_secs(300);
/// Maximum number of cached search entries.
const CACHE_MAX_ENTRIES: usize = 64;

// ═══════════════════════════════════════════════════════════════════════════
//  Search Result Cache — LRU with TTL
// ═══════════════════════════════════════════════════════════════════════════

struct CacheEntry {
    output: ToolOutput,
    inserted: Instant,
}

/// A simple in-memory LRU cache with TTL for search results.
///
/// Thread-safe via `Mutex`.  Eviction on insert: removes expired entries first,
/// then evicts the oldest if still over capacity.
struct SearchCache {
    entries: HashMap<String, CacheEntry>,
    order: VecDeque<String>,  // oldest first
    max_entries: usize,
    ttl: Duration,
}

impl SearchCache {
    fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
            ttl,
        }
    }

    fn get(&mut self, key: &str) -> Option<ToolOutput> {
        let expired = self.entries
            .get(key)
            .map(|e| e.inserted.elapsed() >= self.ttl)
            .unwrap_or(false);

        if expired {
            self.entries.remove(key);
            self.order.retain(|k| k != key);
            return None;
        }

        self.entries.get(key).map(|e| e.output.clone())
    }

    fn insert(&mut self, key: String, output: ToolOutput) {
        // Remove expired entries.
        while let Some(oldest_key) = self.order.front() {
            if let Some(entry) = self.entries.get(oldest_key) {
                if entry.inserted.elapsed() >= self.ttl {
                    let k = self.order.pop_front().unwrap();
                    self.entries.remove(&k);
                    continue;
                }
            }
            break;
        }

        // Evict oldest if over capacity.
        while self.entries.len() >= self.max_entries {
            if let Some(k) = self.order.pop_front() {
                self.entries.remove(&k);
            } else {
                break;
            }
        }

        // Remove existing entry if present (to refresh position).
        if self.entries.contains_key(&key) {
            self.order.retain(|k| k != &key);
        }

        self.order.push_back(key.clone());
        self.entries.insert(key, CacheEntry {
            output,
            inserted: Instant::now(),
        });
    }
}

/// Global search result cache.
static SEARCH_CACHE: std::sync::LazyLock<Mutex<SearchCache>> =
    std::sync::LazyLock::new(|| Mutex::new(SearchCache::new(CACHE_MAX_ENTRIES, CACHE_TTL)));

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
                Use `fetch_page` only when you need the full body text of a specific page."
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
            .unwrap_or(5);

        // Check cache first.
        let key = cache_key(query, max_results);
        if let Ok(mut cache) = SEARCH_CACHE.lock() {
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.clone());
            }
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

        // Try providers in configured priority order.
        let providers = if self.search_providers.is_empty() {
            vec!["brave".to_string(), "tavily".to_string(), "searxng".to_string(), "duckduckgo".to_string()]
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
                if let Ok(mut cache) = SEARCH_CACHE.lock() {
                    cache.insert(key, output.clone());
                }
            }
        }

        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  FetchPageTool — fetch URLs and extract content (text or markdown)
// ═══════════════════════════════════════════════════════════════════════════

/// Fetches one or more web pages and returns extracted content.
///
/// Supports plain-text and markdown output formats, CSS-selector targeting,
/// and parallel batch fetching of multiple URLs.
pub struct FetchPageTool;

#[async_trait]
impl Tool for FetchPageTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "fetch_page".to_string(),
            description: "Fetch one or more web page URLs and return extracted content. \
                Supports markdown output (preserves headings, links, lists, code blocks) \
                and plain text. Use 'urls' for batch parallel fetching. \
                Use 'selector' to target specific CSS selectors on the page. \
                For quick facts like prices or stats, web_search already includes \
                structured data from top results."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "url".to_string(),
                    description: "A single URL to fetch (use this OR 'urls', not both)".to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "urls".to_string(),
                    description: "JSON array of URLs to fetch in parallel, e.g. [\"https://a.com\", \"https://b.com\"]. \
                        Max 5 URLs per call."
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "format".to_string(),
                    description: "Output format: 'markdown' (default, preserves structure) or 'text' (plain text)"
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "selector".to_string(),
                    description: "CSS selector to target specific content, e.g. 'article', '.main-content', '#readme'. \
                        If omitted, uses readability heuristics (article > main > body)."
                        .to_string(),
                    required: false,
                    ..Default::default()
                },
                ToolParam {
                    name: "max_chars".to_string(),
                    description: "Maximum characters to return per URL (default: 12000)".to_string(),
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
        let format = args.get("format").map(|s| s.as_str()).unwrap_or("markdown");
        let selector = args.get("selector").map(|s| s.as_str());
        let max_chars: usize = args
            .get("max_chars")
            .and_then(|v| v.parse().ok())
            .unwrap_or(12000);

        // Collect URLs from either "url" or "urls" param.
        let mut urls: Vec<String> = Vec::new();
        if let Some(url) = args.get("url") {
            if !url.is_empty() {
                urls.push(url.clone());
            }
        }
        if let Some(urls_json) = args.get("urls") {
            if let Ok(parsed) = serde_json::from_str::<Vec<String>>(urls_json) {
                urls.extend(parsed);
            } else {
                // Try comma-separated fallback.
                for u in urls_json.split(',') {
                    let u = u.trim().trim_matches('"').trim();
                    if !u.is_empty() {
                        urls.push(u.to_string());
                    }
                }
            }
        }

        if urls.is_empty() {
            return Ok(ToolOutput {
                success: false,
                output: "missing required param: provide 'url' or 'urls'".to_string(),
            });
        }

        // Cap at 5 URLs.
        urls.truncate(5);
        let use_markdown = format == "markdown" || format == "md";

        let client = build_client()?;
        let selector_owned = selector.map(|s| s.to_string());

        if urls.len() == 1 {
            // Single URL — straightforward fetch.
            let result = fetch_page_content(
                &client, &urls[0], max_chars, use_markdown, selector_owned.as_deref(),
            ).await;
            match result {
                Ok(text) if !text.is_empty() => Ok(ToolOutput {
                    success: true,
                    output: text,
                }),
                Ok(_) => Ok(ToolOutput {
                    success: false,
                    output: format!("Page returned no extractable content: {}", urls[0]),
                }),
                Err(reason) => Ok(ToolOutput {
                    success: false,
                    output: format!("Could not fetch {}: {reason}", urls[0]),
                }),
            }
        } else {
            // Batch fetch — parallel.
            let fetches: Vec<_> = urls.iter().map(|url| {
                let client = client.clone();
                let url = url.clone();
                let sel = selector_owned.clone();
                tokio::spawn(async move {
                    let result = fetch_page_content(
                        &client, &url, max_chars, use_markdown, sel.as_deref(),
                    ).await;
                    (url, result)
                })
            }).collect();

            let mut sections: Vec<String> = Vec::new();
            let mut any_success = false;
            for handle in fetches {
                if let Ok((url, result)) = handle.await {
                    match result {
                        Ok(text) if !text.is_empty() => {
                            any_success = true;
                            if use_markdown {
                                sections.push(format!("## {url}\n\n{text}"));
                            } else {
                                sections.push(format!("=== {url} ===\n{text}"));
                            }
                        }
                        Ok(_) => sections.push(format!("[{url}] No extractable content")),
                        Err(reason) => sections.push(format!("[{url}] Error: {reason}")),
                    }
                }
            }

            Ok(ToolOutput {
                success: any_success,
                output: sections.join("\n\n---\n\n"),
            })
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

// ═══════════════════════════════════════════════════════════════════════════
//  Enhanced page fetching — text or markdown with optional CSS selector
// ═══════════════════════════════════════════════════════════════════════════

/// Fetch a URL and return extracted content as plain text or markdown.
///
/// When `use_markdown` is true, the output preserves headings, links, lists,
/// code blocks, and emphasis using markdown syntax.
///
/// An optional `css_selector` can target a specific DOM region (e.g. "article",
/// ".main-content", "#readme").
async fn fetch_page_content(
    client: &reqwest::Client,
    url: &str,
    max_chars: usize,
    use_markdown: bool,
    css_selector: Option<&str>,
) -> std::result::Result<String, String> {
    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(10))
        .header("Accept", "text/html")
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                "request timed out after 10s".to_string()
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

    let doc = Html::parse_document(body);

    // Resolve the target element: CSS selector → readability heuristics.
    let target: Option<scraper::ElementRef<'_>> = if let Some(sel_str) = css_selector {
        Selector::parse(sel_str)
            .ok()
            .and_then(|sel| doc.select(&sel).next())
    } else {
        None
    };

    // If no explicit selector matched, use readability heuristics.
    let target = target.or_else(|| {
        let selectors = ["article", "main", "[role=\"main\"]", ".post-content", ".entry-content"];
        for sel_str in &selectors {
            if let Ok(sel) = Selector::parse(sel_str) {
                if let Some(el) = doc.select(&sel).next() {
                    // Only use if it has meaningful content.
                    let text: String = el.text().take(200).collect();
                    if text.trim().len() >= 80 {
                        return Some(el);
                    }
                }
            }
        }
        // Fall back to body.
        Selector::parse("body").ok().and_then(|sel| doc.select(&sel).next())
    });

    let target = match target {
        Some(t) => t,
        None => return Err("no content element found".to_string()),
    };

    // Extract structured data header (title, meta, JSON-LD) — always useful.
    let structured = extract_structured_data(body);

    let content = if use_markdown {
        html_to_markdown(&target, max_chars)
    } else {
        extract_text_from_element(&target, max_chars)
    };

    if structured.is_empty() && content.is_empty() {
        return Err("page contained no extractable content".to_string());
    }

    if structured.is_empty() {
        Ok(content)
    } else if content.is_empty() {
        Ok(structured)
    } else {
        // Give structured data a header, then the main content.
        let struct_budget = max_chars / 5;
        let struct_part = if structured.len() > struct_budget {
            let end = truncate_byte_boundary(&structured, struct_budget);
            format!("{}…", &structured[..end])
        } else {
            structured
        };
        Ok(format!("{struct_part}\n\n{content}"))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  HTML → Markdown conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an HTML element subtree to clean Markdown.
///
/// Preserves:
///  - Headings (h1–h6 → `#`–`######`)
///  - Links (`<a>` → `[text](href)`)
///  - Emphasis (`<em>`/`<i>` → `*text*`, `<strong>`/`<b>` → `**text**`)
///  - Inline code (`<code>` → `` `code` ``)
///  - Code blocks (`<pre>` → fenced code blocks)
///  - Lists (`<ul>/<ol>/<li>` → `-` / `1.`)
///  - Blockquotes (`<blockquote>` → `>`)
///  - Paragraphs and line breaks
///
/// Skips: script, style, nav, header, footer, noscript, svg, aside, form, iframe.
fn html_to_markdown(el: &scraper::ElementRef<'_>, max_chars: usize) -> String {
    let mut buf = String::with_capacity(max_chars + 512);
    md_collect(el, &mut buf, max_chars, &MdContext::default());

    // Clean up excessive blank lines.
    let mut cleaned = String::with_capacity(buf.len());
    let mut consecutive_blanks = 0u32;
    for line in buf.lines() {
        if line.trim().is_empty() {
            consecutive_blanks += 1;
            if consecutive_blanks <= 2 {
                cleaned.push('\n');
            }
        } else {
            consecutive_blanks = 0;
            cleaned.push_str(line);
            cleaned.push('\n');
        }
        if cleaned.len() >= max_chars {
            break;
        }
    }

    let trimmed = cleaned.trim().to_string();
    if trimmed.len() > max_chars {
        let safe_end = truncate_byte_boundary(&trimmed, max_chars);
        let end = trimmed[..safe_end].rfind('\n').unwrap_or(safe_end);
        format!("{}…", &trimmed[..end])
    } else {
        trimmed
    }
}

#[derive(Default, Clone)]
struct MdContext {
    list_depth: u32,
    ordered_counter: Option<u32>,
    in_pre: bool,
    in_blockquote: bool,
}

const MD_SKIP_TAGS: &[&str] = &[
    "script", "style", "nav", "header", "footer", "noscript", "svg",
    "aside", "form", "iframe",
];

fn md_collect(
    node: &scraper::ElementRef<'_>,
    buf: &mut String,
    max_chars: usize,
    ctx: &MdContext,
) {
    use scraper::Node;

    for child in node.children() {
        if buf.len() >= max_chars {
            return;
        }
        match child.value() {
            Node::Text(text) => {
                if ctx.in_pre {
                    buf.push_str(text);
                } else {
                    // Collapse whitespace in normal flow.
                    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !collapsed.is_empty() {
                        buf.push_str(&collapsed);
                    }
                }
            }
            Node::Element(el) => {
                let tag = el.name();
                if MD_SKIP_TAGS.contains(&tag) {
                    continue;
                }
                let Some(child_ref) = scraper::ElementRef::wrap(child) else {
                    continue;
                };

                match tag {
                    // ── Headings ────────────────────────────────────
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                        let level = tag[1..].parse::<usize>().unwrap_or(1);
                        let hashes = "#".repeat(level);
                        let text = child_ref.text().collect::<String>();
                        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
                        if !text.is_empty() {
                            ensure_blank_line(buf);
                            buf.push_str(&hashes);
                            buf.push(' ');
                            buf.push_str(&text);
                            buf.push_str("\n\n");
                        }
                    }

                    // ── Links ──────────────────────────────────────
                    "a" => {
                        let href = el.attr("href").unwrap_or("");
                        let text = child_ref.text().collect::<String>();
                        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
                        if !text.is_empty() && !href.is_empty()
                            && (href.starts_with("http") || href.starts_with('/'))
                        {
                            buf.push('[');
                            buf.push_str(&text);
                            buf.push_str("](");
                            buf.push_str(href);
                            buf.push(')');
                        } else if !text.is_empty() {
                            buf.push_str(&text);
                        }
                    }

                    // ── Bold / Strong ──────────────────────────────
                    "strong" | "b" => {
                        let text = child_ref.text().collect::<String>();
                        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
                        if !text.is_empty() {
                            buf.push_str("**");
                            buf.push_str(&text);
                            buf.push_str("**");
                        }
                    }

                    // ── Italic / Emphasis ──────────────────────────
                    "em" | "i" => {
                        let text = child_ref.text().collect::<String>();
                        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
                        if !text.is_empty() {
                            buf.push('*');
                            buf.push_str(&text);
                            buf.push('*');
                        }
                    }

                    // ── Inline code ────────────────────────────────
                    "code" if !ctx.in_pre => {
                        let text = child_ref.text().collect::<String>();
                        if !text.is_empty() {
                            buf.push('`');
                            buf.push_str(&text);
                            buf.push('`');
                        }
                    }

                    // ── Code blocks ────────────────────────────────
                    "pre" => {
                        ensure_blank_line(buf);
                        // Try to detect language from nested <code> class.
                        let lang = child_ref
                            .select(&Selector::parse("code").unwrap())
                            .next()
                            .and_then(|code| {
                                code.value().attr("class").and_then(|cls| {
                                    cls.split_whitespace()
                                        .find(|c| c.starts_with("language-") || c.starts_with("lang-"))
                                        .map(|c| c.trim_start_matches("language-").trim_start_matches("lang-"))
                                })
                            })
                            .unwrap_or("");
                        buf.push_str("```");
                        buf.push_str(lang);
                        buf.push('\n');
                        let mut pre_ctx = ctx.clone();
                        pre_ctx.in_pre = true;
                        md_collect(&child_ref, buf, max_chars, &pre_ctx);
                        if !buf.ends_with('\n') {
                            buf.push('\n');
                        }
                        buf.push_str("```\n\n");
                    }

                    // ── Unordered list ─────────────────────────────
                    "ul" => {
                        ensure_blank_line(buf);
                        let mut list_ctx = ctx.clone();
                        list_ctx.list_depth += 1;
                        list_ctx.ordered_counter = None;
                        md_collect(&child_ref, buf, max_chars, &list_ctx);
                        buf.push('\n');
                    }

                    // ── Ordered list ───────────────────────────────
                    "ol" => {
                        ensure_blank_line(buf);
                        let mut list_ctx = ctx.clone();
                        list_ctx.list_depth += 1;
                        list_ctx.ordered_counter = Some(0);
                        md_collect(&child_ref, buf, max_chars, &list_ctx);
                        buf.push('\n');
                    }

                    // ── List item ──────────────────────────────────
                    "li" => {
                        let indent = "  ".repeat(ctx.list_depth.saturating_sub(1) as usize);
                        let bullet = if let Some(ref _counter) = ctx.ordered_counter {
                            // We can't easily track mutable counter through
                            // recursion, so use "-" for simplicity.
                            format!("{indent}- ")
                        } else {
                            format!("{indent}- ")
                        };
                        buf.push_str(&bullet);
                        md_collect(&child_ref, buf, max_chars, ctx);
                        if !buf.ends_with('\n') {
                            buf.push('\n');
                        }
                    }

                    // ── Blockquote ─────────────────────────────────
                    "blockquote" => {
                        ensure_blank_line(buf);
                        let mut bq_ctx = ctx.clone();
                        bq_ctx.in_blockquote = true;
                        let mut inner = String::new();
                        md_collect(&child_ref, &mut inner, max_chars, &bq_ctx);
                        for line in inner.lines() {
                            buf.push_str("> ");
                            buf.push_str(line);
                            buf.push('\n');
                        }
                        buf.push('\n');
                    }

                    // ── Horizontal rule ────────────────────────────
                    "hr" => {
                        ensure_blank_line(buf);
                        buf.push_str("---\n\n");
                    }

                    // ── Line break ─────────────────────────────────
                    "br" => {
                        buf.push('\n');
                    }

                    // ── Paragraphs and block elements ──────────────
                    "p" | "div" | "section" | "article" | "main" | "figure"
                    | "figcaption" | "details" | "summary" => {
                        ensure_blank_line(buf);
                        md_collect(&child_ref, buf, max_chars, ctx);
                        buf.push_str("\n\n");
                    }

                    // ── Images ─────────────────────────────────────
                    "img" => {
                        let alt = el.attr("alt").unwrap_or("");
                        let src = el.attr("src").unwrap_or("");
                        if !alt.is_empty() && !src.is_empty() && src.starts_with("http") {
                            buf.push_str(&format!("![{alt}]({src})"));
                        } else if !alt.is_empty() {
                            buf.push_str(&format!("[Image: {alt}]"));
                        }
                    }

                    // ── Table ──────────────────────────────────────
                    "table" => {
                        ensure_blank_line(buf);
                        md_collect_table(&child_ref, buf, max_chars);
                        buf.push('\n');
                    }

                    // ── Definition list ────────────────────────────
                    "dt" => {
                        ensure_blank_line(buf);
                        buf.push_str("**");
                        md_collect(&child_ref, buf, max_chars, ctx);
                        buf.push_str("**\n");
                    }
                    "dd" => {
                        buf.push_str(": ");
                        md_collect(&child_ref, buf, max_chars, ctx);
                        buf.push('\n');
                    }

                    // ── Everything else: recurse ───────────────────
                    _ => {
                        md_collect(&child_ref, buf, max_chars, ctx);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Ensure the buffer ends with at least one blank line (for block-level separation).
fn ensure_blank_line(buf: &mut String) {
    if buf.is_empty() {
        return;
    }
    if !buf.ends_with('\n') {
        buf.push('\n');
    }
    if !buf.ends_with("\n\n") {
        buf.push('\n');
    }
}

/// Convert an HTML `<table>` to a markdown table.
fn md_collect_table(table: &scraper::ElementRef<'_>, buf: &mut String, max_chars: usize) {
    let row_sel = Selector::parse("tr").unwrap();
    let th_sel = Selector::parse("th").unwrap();
    let td_sel = Selector::parse("td").unwrap();

    let mut rows: Vec<Vec<String>> = Vec::new();
    for row in table.select(&row_sel) {
        if buf.len() >= max_chars {
            break;
        }
        let cells: Vec<String> = row.select(&th_sel)
            .chain(row.select(&td_sel))
            .map(|cell| {
                cell.text().collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        if !cells.is_empty() {
            rows.push(cells);
        }
    }

    if rows.is_empty() {
        return;
    }

    // Determine column count from the widest row.
    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if cols == 0 {
        return;
    }

    // Emit header row.
    let first = &rows[0];
    buf.push('|');
    for i in 0..cols {
        let cell = first.get(i).map(|s| s.as_str()).unwrap_or("");
        buf.push_str(&format!(" {cell} |"));
    }
    buf.push('\n');

    // Separator.
    buf.push('|');
    for _ in 0..cols {
        buf.push_str(" --- |");
    }
    buf.push('\n');

    // Data rows.
    for row in rows.iter().skip(1) {
        buf.push('|');
        for i in 0..cols {
            let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
            buf.push_str(&format!(" {cell} |"));
        }
        buf.push('\n');
    }
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

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_to_markdown_headings() {
        let html = "<html><body><h1>Title</h1><h2>Subtitle</h2><p>Paragraph text.</p></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("# Title"), "md was: {md}");
        assert!(md.contains("## Subtitle"), "md was: {md}");
        assert!(md.contains("Paragraph text."), "md was: {md}");
    }

    #[test]
    fn test_html_to_markdown_links() {
        let html = r#"<html><body><p>Visit <a href="https://example.com">Example</a> now.</p></body></html>"#;
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("[Example](https://example.com)"), "md was: {md}");
    }

    #[test]
    fn test_html_to_markdown_emphasis() {
        let html = "<html><body><p><strong>Bold</strong> and <em>italic</em> text.</p></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("**Bold**"), "md was: {md}");
        assert!(md.contains("*italic*"), "md was: {md}");
    }

    #[test]
    fn test_html_to_markdown_code() {
        let html = "<html><body><p>Use <code>println!</code> to print.</p><pre><code class=\"language-rust\">fn main() {\n    println!(\"hello\");\n}</code></pre></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("`println!`"), "md was: {md}");
        assert!(md.contains("```rust"), "md was: {md}");
        assert!(md.contains("fn main()"), "md was: {md}");
    }

    #[test]
    fn test_html_to_markdown_list() {
        let html = "<html><body><ul><li>First</li><li>Second</li><li>Third</li></ul></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("- First"), "md was: {md}");
        assert!(md.contains("- Second"), "md was: {md}");
        assert!(md.contains("- Third"), "md was: {md}");
    }

    #[test]
    fn test_html_to_markdown_blockquote() {
        let html = "<html><body><blockquote>Quoted text here.</blockquote></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("> Quoted text here."), "md was: {md}");
    }

    #[test]
    fn test_html_to_markdown_table() {
        let html = "<html><body><table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("| Name | Age |"), "md was: {md}");
        assert!(md.contains("| --- | --- |"), "md was: {md}");
        assert!(md.contains("| Alice | 30 |"), "md was: {md}");
    }

    #[test]
    fn test_html_to_markdown_skips_script_style() {
        let html = "<html><body><script>alert('xss')</script><style>.x{color:red}</style><p>Visible content.</p></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 10000);
        assert!(md.contains("Visible content."), "md was: {md}");
        assert!(!md.contains("alert"), "should not contain script: {md}");
        assert!(!md.contains("color:red"), "should not contain style: {md}");
    }

    #[test]
    fn test_html_to_markdown_truncation() {
        let html = "<html><body><p>A very long paragraph that should be truncated at the specified limit.</p></body></html>";
        let doc = Html::parse_document(html);
        let body = Selector::parse("body").unwrap();
        let el = doc.select(&body).next().unwrap();
        let md = html_to_markdown(&el, 30);
        assert!(md.len() <= 35, "should be truncated, got len={}: {md}", md.len());
    }
}

