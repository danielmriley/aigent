//! Memory tools — lets the agent actively query and write its long-term memory
//! during a conversation or background tick.
//!
//! Because the `MemoryManager` lives inside the daemon state (behind `Arc<Mutex>`),
//! these tools communicate through boxed async closures injected at startup.
//! The closures capture a clone of the daemon state handle and perform the actual
//! work while the tools themselves stay crate-independent.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::{Tool, ToolMetadata, ToolOutput, ToolParam, ToolSpec, ParamType};

// ── Types shared with the wiring layer ───────────────────────────────────────

/// A single result from a memory search.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub content: String,
    pub tier: String,
    pub score: f32,
    pub source: String,
    pub created_at: String,
}

/// Async callback that runs a ranked semantic search over the agent's memory.
///
/// Signature: `(query: String, limit: usize) -> Vec<MemorySearchResult>`
///
/// The runtime constructs this closure once (capturing a handle to the daemon
/// state) and passes it into `SearchMemoryTool` during registry setup.
pub type MemoryQueryFn = Arc<
    dyn Fn(String, usize) -> Pin<Box<dyn Future<Output = Vec<MemorySearchResult>> + Send>>
        + Send
        + Sync,
>;

// ── Tool implementation ──────────────────────────────────────────────────────

/// Tool that allows the agent to search its own memory store at runtime.
pub struct SearchMemoryTool {
    pub query_fn: MemoryQueryFn,
}

#[async_trait]
impl Tool for SearchMemoryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "search_memory".to_string(),
            description:
                "Actively search your long-term memory, core beliefs, and past episodic \
                 conversations. Use this whenever you need more context about the user, \
                 past projects, previous discussions, or your own identity than what is \
                 already provided in your system prompt. Returns the most relevant \
                 memory entries ranked by semantic similarity."
                    .to_string(),
            params: vec![
                ToolParam::required(
                    "query",
                    "Semantic search query — describe what you are looking for \
                     (e.g. 'user's preferred programming language', \
                     'conversation about vacation plans last week')",
                ),
                ToolParam {
                    name: "limit".to_string(),
                    description: "Maximum number of memory entries to return (default: 10)"
                        .to_string(),
                    required: false,
                    param_type: ParamType::Integer,
                    default: Some("10".to_string()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                read_only: true,
                group: "memory".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let query = args
            .get("query")
            .ok_or_else(|| anyhow::anyhow!("missing required param: query"))?;

        let limit: usize = args
            .get("limit")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10)
            .min(25); // hard cap to keep context reasonable

        if query.trim().is_empty() {
            return Ok(ToolOutput {
                success: false,
                output: "query must not be empty".to_string(),
            });
        }

        let results = (self.query_fn)(query.clone(), limit).await;

        if results.is_empty() {
            return Ok(ToolOutput {
                success: true,
                output: "No matching memories found for that query.".to_string(),
            });
        }

        // Format results as markdown for the LLM.
        let mut out = format!("Found {} matching memories:\n\n", results.len());
        for (i, r) in results.iter().enumerate() {
            out.push_str(&format!(
                "### Memory {} (tier: {}, score: {:.2}, source: {})\n{}\n— _{}_\n\n",
                i + 1,
                r.tier,
                r.score,
                r.source,
                r.content,
                r.created_at,
            ));
        }

        Ok(ToolOutput {
            success: true,
            output: out,
        })
    }
}

// ── WriteMemoryTool ───────────────────────────────────────────────────────────

/// Async callback that records a new entry into the agent's memory store.
///
/// Signature: `(content: String, tier: String) -> Result<String, String>`
/// Returns the ID of the newly created memory entry, or an error message.
pub type MemoryWriteFn = Arc<
    dyn Fn(String, String) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

/// Tool that lets the agent explicitly record a fact or insight to its memory.
///
/// Distinct from the automatic recording that happens at turn-end: this lets
/// the agent decide *right now* that something is important enough to persist,
/// at a tier it chooses.
pub struct WriteMemoryTool {
    pub write_fn: MemoryWriteFn,
}

#[async_trait]
impl Tool for WriteMemoryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_memory".to_string(),
            description:
                "Explicitly record a fact, insight, or note to your long-term memory. \
                 Use this when you learn something important mid-conversation that you \
                 want to ensure survives the current session, or when you want to leave \
                 a note for your future self. For general observations prefer 'episodic'; \
                 for stable learned facts prefer 'semantic'; for plans or intentions use \
                 'reflective'."
                    .to_string(),
            params: vec![
                ToolParam::required(
                    "content",
                    "The text to store — write it as a self-contained sentence or short \
                     paragraph, as if briefing your future self",
                ),
                ToolParam {
                    name: "tier".to_string(),
                    description: "Memory tier: 'episodic' (recent event), 'semantic' (stable \
                                  fact), or 'reflective' (insight / plan)"
                        .to_string(),
                    required: false,
                    enum_values: vec![
                        "episodic".to_string(),
                        "semantic".to_string(),
                        "reflective".to_string(),
                    ],
                    default: Some("episodic".to_string()),
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata {
                read_only: false,
                group: "memory".to_string(),
                ..Default::default()
            },
        }
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let content = args
            .get("content")
            .ok_or_else(|| anyhow::anyhow!("missing required param: content"))?;

        if content.trim().is_empty() {
            return Ok(ToolOutput {
                success: false,
                output: "content must not be empty".to_string(),
            });
        }

        let tier = args
            .get("tier")
            .map(|s| s.as_str())
            .unwrap_or("episodic")
            .to_string();

        if !matches!(tier.as_str(), "episodic" | "semantic" | "reflective") {
            return Ok(ToolOutput {
                success: false,
                output: format!(
                    "invalid tier '{}'. Must be one of: episodic, semantic, reflective",
                    tier
                ),
            });
        }

        match (self.write_fn)(content.clone(), tier.clone()).await {
            Ok(id) => Ok(ToolOutput {
                success: true,
                output: format!("Memory recorded to {} tier (id: {})", tier, id),
            }),
            Err(e) => Ok(ToolOutput {
                success: false,
                output: format!("Failed to record memory: {}", e),
            }),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_query_fn(results: Vec<MemorySearchResult>) -> MemoryQueryFn {
        Arc::new(move |_query, limit| {
            let res = results.clone();
            Box::pin(async move {
                res.into_iter().take(limit).collect()
            })
        })
    }

    #[tokio::test]
    async fn empty_query_fails() {
        let tool = SearchMemoryTool {
            query_fn: make_dummy_query_fn(vec![]),
        };
        let mut args = HashMap::new();
        args.insert("query".to_string(), "".to_string());
        let out = tool.run(&args).await.unwrap();
        assert!(!out.success);
    }

    #[tokio::test]
    async fn no_results_returns_success() {
        let tool = SearchMemoryTool {
            query_fn: make_dummy_query_fn(vec![]),
        };
        let mut args = HashMap::new();
        args.insert("query".to_string(), "something obscure".to_string());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("No matching memories"));
    }

    #[tokio::test]
    async fn results_formatted_as_markdown() {
        let results = vec![
            MemorySearchResult {
                content: "User loves Rust".to_string(),
                tier: "Core".to_string(),
                score: 0.95,
                source: "user-input".to_string(),
                created_at: "2026-01-15T10:00:00Z".to_string(),
            },
            MemorySearchResult {
                content: "Built an agent framework together".to_string(),
                tier: "Episodic".to_string(),
                score: 0.72,
                source: "assistant-reply".to_string(),
                created_at: "2026-02-20T14:30:00Z".to_string(),
            },
        ];
        let tool = SearchMemoryTool {
            query_fn: make_dummy_query_fn(results),
        };
        let mut args = HashMap::new();
        args.insert("query".to_string(), "rust programming".to_string());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("Found 2 matching memories"));
        assert!(out.output.contains("User loves Rust"));
        assert!(out.output.contains("tier: Core"));
    }

    #[tokio::test]
    async fn limit_is_respected() {
        let results = (0..20)
            .map(|i| MemorySearchResult {
                content: format!("Memory {i}"),
                tier: "Episodic".to_string(),
                score: 0.5,
                source: "test".to_string(),
                created_at: "2026-01-01T00:00:00Z".to_string(),
            })
            .collect();
        let tool = SearchMemoryTool {
            query_fn: make_dummy_query_fn(results),
        };
        let mut args = HashMap::new();
        args.insert("query".to_string(), "anything".to_string());
        args.insert("limit".to_string(), "3".to_string());
        let out = tool.run(&args).await.unwrap();
        assert!(out.success);
        assert!(out.output.contains("Found 3 matching memories"));
    }

    #[tokio::test]
    async fn missing_query_param_returns_error() {
        let tool = SearchMemoryTool {
            query_fn: make_dummy_query_fn(vec![]),
        };
        let args = HashMap::new();
        let result = tool.run(&args).await;
        assert!(result.is_err());
    }
}
