//! Live financial market data via the Yahoo Finance Chart API.
//!
//! Provides a single tool:
//!
//! * **`finance_quote`** — returns real-time (or latest-close) price data
//!   for a given ticker symbol.  Uses the free Yahoo Finance v8 Chart API
//!   which returns JSON and does not require an API key.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;

use crate::{Tool, ToolSpec, ToolParam, ToolOutput, ToolMetadata, SecurityLevel};

// ─── Constants ───────────────────────────────────────────────────────────────

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

const CHART_API_BASE: &str = "https://query2.finance.yahoo.com/v8/finance/chart";

// ═══════════════════════════════════════════════════════════════════════════
//  FinanceQuoteTool
// ═══════════════════════════════════════════════════════════════════════════

/// Returns a real-time (or latest-close) financial quote for a ticker symbol.
///
/// Hits the Yahoo Finance v8 Chart API, which returns structured JSON and
/// does not require JavaScript execution.  No API key needed — only a
/// standard browser User-Agent header.
pub struct FinanceQuoteTool;

#[async_trait]
impl Tool for FinanceQuoteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "finance_quote".to_string(),
            description: "Get a real-time (or latest-close) stock / ETF / crypto \
                quote. Returns current price, previous close, day range, 52-week \
                range, and basic info. Much more reliable and current than scraping \
                web pages for financial data."
                .to_string(),
            params: vec![
                ToolParam {
                    name: "ticker".to_string(),
                    description: "Ticker symbol, e.g. TSLA, AAPL, BTC-USD, ^GSPC".to_string(),
                    required: true,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                security_level: SecurityLevel::Low,
                read_only: true,
                group: "finance".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let ticker = args
            .get("ticker")
            .ok_or_else(|| anyhow::anyhow!("missing required param: ticker"))?
            .trim()
            .to_uppercase();

        if ticker.is_empty() {
            return Ok(ToolOutput {
                success: false,
                output: "ticker symbol cannot be empty".to_string(),
            });
        }

        fetch_quote(&ticker).await
    }
}

// ─── Yahoo Finance Chart API ─────────────────────────────────────────────────

async fn fetch_quote(ticker: &str) -> Result<ToolOutput> {
    let url = format!("{CHART_API_BASE}/{ticker}?interval=1d&range=1d");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent(USER_AGENT)
        .build()?;

    let resp = client.get(&url).send().await;

    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            return Ok(ToolOutput {
                success: false,
                output: format!("Network error fetching quote for {ticker}: {e}"),
            });
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Ok(ToolOutput {
            success: false,
            output: format!(
                "Yahoo Finance returned HTTP {status} for ticker '{ticker}'. \
                 The symbol may be invalid or the service may be temporarily unavailable.\n{body}"
            ),
        });
    }

    let json: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return Ok(ToolOutput {
                success: false,
                output: format!("Failed to parse Yahoo Finance response for {ticker}: {e}"),
            });
        }
    };

    // Navigate: chart.result[0].meta
    let meta = json
        .get("chart")
        .and_then(|c| c.get("result"))
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .and_then(|first| first.get("meta"));

    let meta = match meta {
        Some(m) => m,
        None => {
            // Check for an error message from Yahoo
            let err_msg = json
                .get("chart")
                .and_then(|c| c.get("error"))
                .and_then(|e| e.get("description"))
                .and_then(|d| d.as_str())
                .unwrap_or("No data returned");
            return Ok(ToolOutput {
                success: false,
                output: format!("No quote data for '{ticker}': {err_msg}"),
            });
        }
    };

    // Extract fields, falling back to "N/A" for missing ones.
    let symbol = meta_str(meta, "symbol");
    let long_name = meta_str_opt(meta, "longName")
        .or_else(|| meta_str_opt(meta, "shortName"))
        .unwrap_or_else(|| symbol.clone());
    let currency = meta_str(meta, "currency");
    let exchange = meta_str(meta, "fullExchangeName");

    let price = meta_f64(meta, "regularMarketPrice");
    let prev_close = meta_f64(meta, "chartPreviousClose");
    let day_low = meta_f64(meta, "regularMarketDayLow");
    let day_high = meta_f64(meta, "regularMarketDayHigh");
    let week52_low = meta_f64(meta, "fiftyTwoWeekLow");
    let week52_high = meta_f64(meta, "fiftyTwoWeekHigh");

    // Compute change from previous close.
    let (change_str, pct_str) = match (price, prev_close) {
        (Some(p), Some(pc)) if pc > 0.0 => {
            let change = p - pc;
            let pct = (change / pc) * 100.0;
            let sign = if change >= 0.0 { "+" } else { "" };
            (
                format!("{sign}{change:.2}"),
                format!("{sign}{pct:.2}%"),
            )
        }
        _ => ("N/A".to_string(), "N/A".to_string()),
    };

    // Format the market timestamp.
    let market_time = meta
        .get("regularMarketTime")
        .and_then(|v| v.as_i64())
        .map(|ts| {
            chrono::DateTime::from_timestamp(ts, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                .unwrap_or_else(|| ts.to_string())
        })
        .unwrap_or_else(|| "N/A".to_string());

    // Build the output.
    let output = format!(
        "{symbol} — {long_name}\n\
         Exchange: {exchange} | Currency: {currency}\n\
         \n\
         Price:          {price}\n\
         Change:         {change} ({pct})\n\
         Previous Close: {prev}\n\
         Day Range:      {day_low} – {day_high}\n\
         52-Week Range:  {w52_low} – {w52_high}\n\
         Market Time:    {market_time}",
        symbol = symbol,
        long_name = long_name,
        exchange = exchange,
        currency = currency,
        price = fmt_f64(price),
        change = change_str,
        pct = pct_str,
        prev = fmt_f64(prev_close),
        day_low = fmt_f64(day_low),
        day_high = fmt_f64(day_high),
        w52_low = fmt_f64(week52_low),
        w52_high = fmt_f64(week52_high),
        market_time = market_time,
    );

    Ok(ToolOutput {
        success: true,
        output,
    })
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn meta_str(meta: &serde_json::Value, key: &str) -> String {
    meta.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("N/A")
        .to_string()
}

fn meta_str_opt(meta: &serde_json::Value, key: &str) -> Option<String> {
    meta.get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn meta_f64(meta: &serde_json::Value, key: &str) -> Option<f64> {
    meta.get(key).and_then(|v| v.as_f64())
}

fn fmt_f64(val: Option<f64>) -> String {
    match val {
        Some(v) => format!("{v:.2}"),
        None => "N/A".to_string(),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies spec() returns the expected tool name and params.
    #[test]
    fn spec_is_correct() {
        let tool = FinanceQuoteTool;
        let spec = tool.spec();
        assert_eq!(spec.name, "finance_quote");
        assert_eq!(spec.params.len(), 1);
        assert_eq!(spec.params[0].name, "ticker");
        assert!(spec.params[0].required);
        assert!(spec.metadata.read_only);
    }

    /// Empty ticker should fail gracefully.
    #[tokio::test]
    async fn empty_ticker_fails() {
        let tool = FinanceQuoteTool;
        let mut args = HashMap::new();
        args.insert("ticker".to_string(), "".to_string());
        let result = tool.run(&args).await.unwrap();
        assert!(!result.success);
        assert!(result.output.contains("empty"));
    }

    /// Missing ticker param should error.
    #[tokio::test]
    async fn missing_ticker_errors() {
        let tool = FinanceQuoteTool;
        let args = HashMap::new();
        let result = tool.run(&args).await;
        assert!(result.is_err());
    }

    /// Integration test: fetch a real quote (TSLA).
    /// Ignored by default to avoid network calls in CI.
    #[tokio::test]
    #[ignore]
    async fn fetch_tsla_quote() {
        let tool = FinanceQuoteTool;
        let mut args = HashMap::new();
        args.insert("ticker".to_string(), "TSLA".to_string());
        let result = tool.run(&args).await.unwrap();
        assert!(result.success, "output: {}", result.output);
        assert!(result.output.contains("TSLA"));
        assert!(result.output.contains("Price:"));
    }
}
