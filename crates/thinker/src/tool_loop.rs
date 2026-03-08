//! Tool loop types and utilities shared by the external thinking loop.

use std::collections::HashMap;

use aigent_llm::Provider;
use aigent_tools::ToolSpec;

use crate::events::ThinkerEvent;

/// Result of the structured tool loop: the final assistant text content
/// plus any tool results that were gathered along the way.
#[derive(Debug, Clone)]
pub struct ToolLoopResult {
    /// Provider that handled the final response.
    pub provider: Provider,
    /// Full assistant text response (accumulated across streaming).
    pub content: String,
    /// All tool executions that happened during the loop, in order.
    pub tool_executions: Vec<ToolExecution>,
    /// Chain-of-thought strings extracted from each reasoning step's `thought`
    /// field.  Populated only by the external thinking loop.  When
    /// `memory.store_reasoning_traces` is enabled, the caller persists these
    /// to `MemoryTier::Reflective` after the turn completes.
    pub reasoning_traces: Vec<String>,
}

/// Record of a single tool execution within the loop.
#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub tool_name: String,
    pub args: HashMap<String, serde_json::Value>,
    pub success: bool,
    pub output: String,
    /// Wall-clock duration of the tool invocation in milliseconds.
    pub duration_ms: u64,
}

/// Callback for thinker progress events (tool start/end).
///
/// Callers provide this to bridge thinker events into their own event
/// system (e.g. `BackendEvent` in the runtime crate).
pub type EventSink = dyn Fn(ThinkerEvent) + Send + Sync;

/// Build the OpenAI-compatible tools JSON array from tool specs.
///
/// This is the array passed as the `tools` parameter to the LLM API.
pub fn build_tools_json(specs: &[ToolSpec]) -> serde_json::Value {
    aigent_tools::specs_to_openai_tools(specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tools_json_produces_array() {
        let specs = vec![aigent_tools::ToolSpec {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            params: vec![aigent_tools::ToolParam::required("query", "Search query")],
            metadata: Default::default(),
        }];
        let tools = build_tools_json(&specs);
        assert!(tools.is_array());
        let arr = tools.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["function"]["name"], "web_search");
    }
}
