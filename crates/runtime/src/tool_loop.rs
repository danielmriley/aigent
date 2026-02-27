//! Structured tool calling loop.
//!
//! Sends chat messages with tool definitions to the LLM, parses any
//! `tool_calls` from the response, executes them (in parallel when
//! multiple are requested), feeds results back as tool-role messages,
//! and repeats until the model stops requesting tools.
//!
//! Falls back to the legacy text-based `maybe_tool_call` path when
//! the model doesn't support native tool calling.

use std::collections::HashMap;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use aigent_exec::ToolExecutor;
use aigent_llm::{
    ChatMessage, ChatResponse, LlmRouter, Provider, ToolCall,
};
use aigent_tools::{ToolRegistry, ToolSpec};

use crate::events::{ToolCallInfo, ToolResult as ToolResultEvent};
use crate::BackendEvent;

/// Maximum number of tool-call → result → re-prompt iterations before we
/// force the model to produce a final text answer (prevents infinite loops).
const MAX_TOOL_ROUNDS: usize = 5;

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
}

/// Record of a single tool execution within the loop.
#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub tool_name: String,
    pub args: HashMap<String, serde_json::Value>,
    pub success: bool,
    pub output: String,
}

/// Run the structured tool calling loop.
///
/// 1. Sends `messages` + `tools` schema to the LLM.
/// 2. If the response contains `tool_calls`, execute them (in parallel if >1).
/// 3. Append tool results as tool-role messages and re-send.
/// 4. Repeat until the model returns a normal text response or we hit `MAX_TOOL_ROUNDS`.
///
/// Text tokens are streamed via `token_tx` as they arrive.
/// Tool lifecycle events are sent via `event_tx` for UI display.
pub async fn run_tool_loop(
    llm: &LlmRouter,
    primary: Provider,
    ollama_model: &str,
    openrouter_model: &str,
    messages: &mut Vec<ChatMessage>,
    tools_json: Option<&serde_json::Value>,
    tool_registry: &ToolRegistry,
    tool_executor: &ToolExecutor,
    token_tx: mpsc::Sender<String>,
    event_tx: Option<&tokio::sync::broadcast::Sender<BackendEvent>>,
) -> Result<ToolLoopResult> {
    let mut all_executions: Vec<ToolExecution> = Vec::new();
    let mut final_content = String::new();
    let mut final_provider = primary;

    for round in 0..MAX_TOOL_ROUNDS {
        debug!(round, msg_count = messages.len(), "tool loop iteration");

        // On the last allowed round, omit tools to force a text answer.
        let effective_tools = if round < MAX_TOOL_ROUNDS - 1 {
            tools_json
        } else {
            warn!("tool loop hit max rounds ({MAX_TOOL_ROUNDS}), forcing text response");
            None
        };

        let response: ChatResponse = llm
            .chat_messages_stream(
                primary,
                ollama_model,
                openrouter_model,
                messages,
                effective_tools,
                token_tx.clone(),
            )
            .await?;

        final_provider = response.provider;

        // No tool calls → done, return the text response
        if response.tool_calls.is_empty() {
            final_content = response.content;
            break;
        }

        // Model wants tool calls — execute them
        info!(
            round,
            count = response.tool_calls.len(),
            "LLM requested tool calls"
        );

        // Append the assistant's tool-call message to the conversation
        messages.push(ChatMessage::assistant_tool_calls(response.tool_calls.clone()));

        // Execute tool calls (in parallel when >1)
        let executions = execute_tool_calls(
            &response.tool_calls,
            tool_registry,
            tool_executor,
            event_tx,
        )
        .await;

        // Append each tool result as a tool-role message
        for (call, exec) in response.tool_calls.iter().zip(executions.iter()) {
            messages.push(ChatMessage::tool_result(
                &call.id,
                &exec.output,
            ));
        }

        all_executions.extend(executions);

        // If this was the last round and we forced no tools, the loop will
        // exit via the break above on the next iteration.  But if the model
        // somehow returned tool_calls on the final round (should be impossible
        // since we passed None), we'll fall through and return what we have.
    }

    // If the loop exhausted without a final text response, build one from
    // the tool results so the user sees *something*.
    if final_content.is_empty() && !all_executions.is_empty() {
        final_content = all_executions
            .iter()
            .map(|e| format!("[{}]: {}", e.tool_name, &e.output[..e.output.len().min(500)]))
            .collect::<Vec<_>>()
            .join("\n\n");
    }

    Ok(ToolLoopResult {
        provider: final_provider,
        content: final_content,
        tool_executions: all_executions,
    })
}

/// Execute a batch of tool calls, running them in parallel when possible.
async fn execute_tool_calls(
    calls: &[ToolCall],
    registry: &ToolRegistry,
    executor: &ToolExecutor,
    event_tx: Option<&tokio::sync::broadcast::Sender<BackendEvent>>,
) -> Vec<ToolExecution> {
    if calls.is_empty() {
        return vec![];
    }

    // Emit ToolCallStart events
    for call in calls {
        let info = ToolCallInfo {
            name: call.function.name.clone(),
            args: call.function.arguments.to_string(),
        };
        if let Some(tx) = event_tx {
            let _ = tx.send(BackendEvent::ToolCallStart(info));
        }
    }

    // Execute all calls in parallel using futures::join_all
    let futs: Vec<_> = calls
        .iter()
        .map(|call| {
            let tool_name = call.function.name.clone();
            let args = call.function.arguments.clone();
            async move {
                let string_args = json_value_to_string_map(&args);
                let result = executor.execute(registry, &tool_name, &string_args).await;
                let (success, output) = match result {
                    Ok(ref o) => (o.success, o.output.clone()),
                    Err(ref e) => (false, e.to_string()),
                };
                ToolExecution {
                    tool_name,
                    args: args
                        .as_object()
                        .map(|o| {
                            o.iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default(),
                    success,
                    output,
                }
            }
        })
        .collect();

    let results = futures::future::join_all(futs).await;

    // Emit ToolCallEnd events
    for exec in &results {
        let result_event = ToolResultEvent {
            name: exec.tool_name.clone(),
            success: exec.success,
            output: exec.output.clone(),
        };
        if let Some(tx) = event_tx {
            let _ = tx.send(BackendEvent::ToolCallEnd(result_event));
        }
    }

    results
}

/// Convert a JSON object value to `HashMap<String, String>` for the tool executor.
fn json_value_to_string_map(val: &serde_json::Value) -> HashMap<String, String> {
    val.as_object()
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| {
                    let s = match v {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        serde_json::Value::Bool(b) => b.to_string(),
                        serde_json::Value::Null => String::new(),
                        other => other.to_string(),
                    };
                    (k.clone(), s)
                })
                .collect()
        })
        .unwrap_or_default()
}

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
    fn json_value_to_string_map_handles_types() {
        let val = serde_json::json!({
            "query": "rust programming",
            "count": 5,
            "verbose": true,
            "empty": null
        });
        let map = json_value_to_string_map(&val);
        assert_eq!(map["query"], "rust programming");
        assert_eq!(map["count"], "5");
        assert_eq!(map["verbose"], "true");
        assert_eq!(map["empty"], "");
    }

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
