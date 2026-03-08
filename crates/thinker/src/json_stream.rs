//! JSON stream interceptor for the external thinking mode.
//!
//! When `external_thinking` is active the model outputs structured JSON
//! objects token-by-token via the streaming API:
//!
//! ```json
//! {"type":"final_answer","thought":"...","final_answer":"..."}
//! {"type":"tool_call","thought":"...","tool_call":{"name":"...","args":{...}}}
//! ```
//!
//! Because tokens arrive at arbitrary network boundaries, we cannot simply
//! look for '\n' delimiters.  Instead this module uses a lightweight
//! brace-counting state machine:
//!
//! 1. Track `depth` (number of unmatched `{` minus `}`), respecting JSON
//!    string escaping.
//! 2. When `depth` returns to 0 we have a complete JSON object.
//! 3. Parse with `serde_json`, extract the fields, and dispatch.
//!
//! The interceptor is intentionally a plain synchronous function
//! (`feed` + `take_parsed`) so the async plumbing lives in the caller.

use serde_json::Value;

/// Parsed result of one complete JSON object from the stream.
#[derive(Debug, Clone)]
pub enum AgentStep {
    /// Model wants to call a tool.
    ToolCall {
        thought: String,
        tool_name: String,
        args: serde_json::Map<String, Value>,
    },
    /// Model produced a final user-visible answer.
    FinalAnswer {
        thought: String,
        answer: String,
    },
}

/// Brace-counting JSON stream accumulator.
///
/// Feed tokens into [`Self::feed`], then call [`Self::take_parsed`] to
/// extract any complete `AgentStep` that has been recognised.
pub struct JsonStreamBuffer {
    buf: String,
    /// Current nesting depth (`{` increments, `}` decrements).
    depth: i32,
    /// Whether we are inside a JSON string literal (skip brace counting).
    in_string: bool,
    /// Previous character was a backslash (escape inside string).
    escape_next: bool,
    /// Whether we have seen the opening `{` at all.
    started: bool,
    /// Completed JSON object ready for parsing (if any).
    completed: Option<String>,
}

impl Default for JsonStreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonStreamBuffer {
    pub fn new() -> Self {
        Self {
            buf: String::with_capacity(4096),
            depth: 0,
            in_string: false,
            escape_next: false,
            started: false,
            completed: None,
        }
    }

    /// Feed a chunk of streamed text into the buffer.
    ///
    /// After calling this, check [`Self::take_parsed`] for a completed step.
    pub fn feed(&mut self, chunk: &str) {
        for ch in chunk.chars() {
            // ── String-literal tracking ──────────────────────────────────
            if self.in_string {
                self.buf.push(ch);
                if self.escape_next {
                    self.escape_next = false;
                    continue;
                }
                if ch == '\\' {
                    self.escape_next = true;
                    continue;
                }
                if ch == '"' {
                    self.in_string = false;
                }
                continue;
            }

            // ── Outside string ───────────────────────────────────────────
            match ch {
                '{' => {
                    if !self.started {
                        self.started = true;
                        // Discard any leading garbage (e.g. "JSON" prefix)
                        self.buf.clear();
                    }
                    self.buf.push(ch);
                    self.depth += 1;
                }
                '}' => {
                    self.buf.push(ch);
                    self.depth -= 1;
                    if self.started && self.depth == 0 {
                        // Complete JSON object detected.
                        self.completed = Some(std::mem::take(&mut self.buf));
                        self.started = false;
                        // Stop processing further chars in this chunk —
                        // the caller should drain before feeding the next
                        // chunk.
                        return;
                    }
                }
                '"' => {
                    self.buf.push(ch);
                    if self.started {
                        self.in_string = true;
                    }
                }
                _ => {
                    if self.started {
                        self.buf.push(ch);
                    }
                    // Characters before the first `{` are silently discarded
                    // (handles models that prefix JSON with "JSON" or similar).
                }
            }
        }
    }

    /// Take a completed JSON object (if any) and parse it into an
    /// [`AgentStep`].
    ///
    /// Returns `None` if no complete object has been accumulated yet.
    /// Returns `Some(Err(_))` if the object was complete but didn't parse
    /// as a valid agent step.
    pub fn take_parsed(&mut self) -> Option<Result<AgentStep, String>> {
        let raw = self.completed.take()?;
        Some(parse_agent_step(&raw))
    }

    /// Returns `true` when the buffer has a complete JSON object ready.
    pub fn has_complete(&self) -> bool {
        self.completed.is_some()
    }

    /// Return the raw accumulated buffer (for debugging / fallback).
    pub fn raw_buffer(&self) -> &str {
        &self.buf
    }
}

/// Parse a complete JSON string into an [`AgentStep`].
///
/// The canonical schema is:
/// ```json
/// {"type":"tool_call","thought":"...","tool_call":{"name":"...","args":{...}}}
/// {"type":"final_answer","thought":"...","final_answer":"..."}
/// ```
///
/// However, models frequently produce off-schema variants.  This parser
/// handles the following common deviations:
///
/// 1. Missing `"type"` field — inferred from which keys are present.
/// 2. Flat tool call: `{"tool_call":"read_file","path":"x"}` — `"tool_call"`
///    value is the tool name, remaining keys (minus `"thought"`) are args.
/// 3. `"tool_call"` is a string with a separate `"args"` object.
/// 4. `"name"` at top level instead of nested under `"tool_call"`.
fn parse_agent_step(raw: &str) -> Result<AgentStep, String> {
    let val: Value = serde_json::from_str(raw)
        .map_err(|e| format!("invalid JSON: {e}"))?;

    let obj = val.as_object()
        .ok_or_else(|| "top-level value is not an object".to_string())?;

    let step_type = obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let thought = obj
        .get("thought")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // ── Canonical "type" field present ────────────────────────────────────
    match step_type {
        "final_answer" => {
            // Models commonly use 'content', 'response', or 'answer' instead of
            // 'final_answer' as the value key (e.g. qwen3.5:4b uses 'content'
            // after tool calls and 'answer' after web searches).
            let answer = obj
                .get("final_answer")
                .or_else(|| obj.get("content"))
                .or_else(|| obj.get("answer"))
                .or_else(|| obj.get("response"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            return Ok(AgentStep::FinalAnswer { thought, answer });
        }
        "tool_call" => {
            if let Some(tc) = obj.get("tool_call") {
                if let Some(tc_obj) = tc.as_object() {
                    // Canonical nested object form.
                    let tool_name = tc_obj
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = tc_obj
                        .get("args")
                        .and_then(|v| v.as_object())
                        .cloned()
                        .unwrap_or_default();
                    return Ok(AgentStep::ToolCall { thought, tool_name, args });
                }
            }
            // "type":"tool_call" present but "tool_call" key is missing or
            // not an object — fall through to heuristic recovery below.
        }
        "" => {
            // No "type" field — fall through to heuristic.
        }
        _ => {
            // Unknown "type" value — fall through to heuristic.
        }
    }

    // ── Heuristic recovery for off-schema responses ──────────────────────

    // Case A: "final_answer", "content", "answer", or "response" key present (regardless of "type").
    // qwen3.5:4b and some other models use "content" or "answer" instead of "final_answer".
    let final_answer_val = obj.get("final_answer")
        .or_else(|| obj.get("content"))
        .or_else(|| obj.get("answer"))
        .or_else(|| obj.get("response"));
    if let Some(answer_val) = final_answer_val {
        // Only treat as FinalAnswer if value is a plain string (not nested
        // object/array), to avoid misidentifying tool-call structures.
        if let Some(answer) = answer_val.as_str() {
            return Ok(AgentStep::FinalAnswer { thought, answer: answer.to_string() });
        }
    }

    // Case B: "tool_call" key is a nested object (canonical structure, just
    // missing "type").
    if let Some(tc) = obj.get("tool_call") {
        if let Some(tc_obj) = tc.as_object() {
            let tool_name = tc_obj
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = tc_obj
                .get("args")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            return Ok(AgentStep::ToolCall { thought, tool_name, args });
        }

        // Case C: "tool_call" is a string (the tool name).
        // Args are either in a separate "args" key or flattened at the top level.
        if let Some(name) = tc.as_str() {
            let args = if let Some(args_obj) = obj.get("args").and_then(|v| v.as_object()) {
                // Separate "args" object.
                args_obj.clone()
            } else {
                // Flat form: every key except "tool_call", "thought", "type", "name"
                // is treated as an arg.
                let skip = ["tool_call", "thought", "type", "name"];
                obj.iter()
                    .filter(|(k, _)| !skip.contains(&k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            };
            return Ok(AgentStep::ToolCall {
                thought,
                tool_name: name.to_string(),
                args,
            });
        }
    }

    // Case D: Top-level "name" key (no "tool_call" wrapper at all).
    //         e.g. {"name":"read_file","args":{"path":"x"}}
    if let Some(name_val) = obj.get("name") {
        if let Some(name) = name_val.as_str() {
            let args = obj
                .get("args")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            return Ok(AgentStep::ToolCall {
                thought,
                tool_name: name.to_string(),
                args,
            });
        }
    }

    Err(format!("unrecognised agent step structure: {raw}"))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_answer_single_chunk() {
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"type":"final_answer","thought":"simple","final_answer":"Hello!"}"#);
        let step = buf.take_parsed().expect("should have a result").unwrap();
        match step {
            AgentStep::FinalAnswer { thought, answer } => {
                assert_eq!(thought, "simple");
                assert_eq!(answer, "Hello!");
            }
            _ => panic!("expected FinalAnswer"),
        }
    }

    #[test]
    fn tool_call_single_chunk() {
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"type":"tool_call","thought":"need file","tool_call":{"name":"read_file","args":{"path":"/tmp/x"}}}"#);
        let step = buf.take_parsed().expect("should have a result").unwrap();
        match step {
            AgentStep::ToolCall { thought, tool_name, args } => {
                assert_eq!(thought, "need file");
                assert_eq!(tool_name, "read_file");
                assert_eq!(args["path"], "/tmp/x");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn multi_token_streaming() {
        let mut buf = JsonStreamBuffer::new();
        // Simulate token-by-token streaming
        let tokens = [
            r#"{"type":"#,
            r#""final_answer","#,
            r#""thought":"think"#,
            r#"ing","final_"#,
            r#"answer":"42"}"#,
        ];
        for (i, tok) in tokens.iter().enumerate() {
            buf.feed(tok);
            if i < tokens.len() - 1 {
                assert!(buf.take_parsed().is_none(), "should not be complete yet at token {i}");
            }
        }
        let step = buf.take_parsed().expect("should have a result").unwrap();
        match step {
            AgentStep::FinalAnswer { answer, .. } => assert_eq!(answer, "42"),
            _ => panic!("expected FinalAnswer"),
        }
    }

    #[test]
    fn handles_leading_garbage() {
        let mut buf = JsonStreamBuffer::new();
        // Some models prefix with "JSON" or whitespace
        buf.feed(r#"JSON{"type":"final_answer","thought":"ok","final_answer":"yes"}"#);
        let step = buf.take_parsed().expect("should have a result").unwrap();
        match step {
            AgentStep::FinalAnswer { answer, .. } => assert_eq!(answer, "yes"),
            _ => panic!("expected FinalAnswer"),
        }
    }

    #[test]
    fn handles_escaped_braces_in_strings() {
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"type":"final_answer","thought":"code: {x}","final_answer":"fn main() { println!(\"{}\", 1); }"}"#);
        let step = buf.take_parsed().expect("should have a result").unwrap();
        match step {
            AgentStep::FinalAnswer { thought, answer } => {
                assert!(thought.contains("{x}"));
                assert!(answer.contains("println!"));
            }
            _ => panic!("expected FinalAnswer"),
        }
    }

    #[test]
    fn nested_args_object() {
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"type":"tool_call","thought":"shell","tool_call":{"name":"run_shell","args":{"command":"echo hi","timeout":"30"}}}"#);
        let step = buf.take_parsed().expect("should have a result").unwrap();
        match step {
            AgentStep::ToolCall { tool_name, args, .. } => {
                assert_eq!(tool_name, "run_shell");
                assert_eq!(args["command"], "echo hi");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn char_by_char_streaming() {
        let mut buf = JsonStreamBuffer::new();
        let json = r#"{"type":"final_answer","thought":"t","final_answer":"ok"}"#;
        for (i, ch) in json.chars().enumerate() {
            buf.feed(&ch.to_string());
            if i < json.len() - 1 {
                assert!(buf.take_parsed().is_none());
            }
        }
        let step = buf.take_parsed().expect("should complete").unwrap();
        match step {
            AgentStep::FinalAnswer { answer, .. } => assert_eq!(answer, "ok"),
            _ => panic!("expected FinalAnswer"),
        }
    }

    // ── Off-schema recovery tests ────────────────────────────────────────

    #[test]
    fn flat_tool_call_string_with_flat_args() {
        // Model outputs: {"tool_call": "read_file", "path": "tryout.md"}
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"tool_call": "read_file", "path": "tryout.md"}"#);
        let step = buf.take_parsed().expect("should parse").unwrap();
        match step {
            AgentStep::ToolCall { tool_name, args, .. } => {
                assert_eq!(tool_name, "read_file");
                assert_eq!(args["path"], "tryout.md");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn tool_call_string_with_separate_args() {
        // Model outputs: {"tool_call": "web_search", "args": {"query": "rust"}}
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"tool_call": "web_search", "args": {"query": "rust"}}"#);
        let step = buf.take_parsed().expect("should parse").unwrap();
        match step {
            AgentStep::ToolCall { tool_name, args, .. } => {
                assert_eq!(tool_name, "web_search");
                assert_eq!(args["query"], "rust");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn missing_type_with_final_answer_key() {
        // No "type" field, but "final_answer" present.
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"thought": "easy", "final_answer": "Hello there!"}"#);
        let step = buf.take_parsed().expect("should parse").unwrap();
        match step {
            AgentStep::FinalAnswer { thought, answer } => {
                assert_eq!(thought, "easy");
                assert_eq!(answer, "Hello there!");
            }
            _ => panic!("expected FinalAnswer"),
        }
    }

    #[test]
    fn missing_type_with_tool_call_object() {
        // No "type" field, but canonical "tool_call" object present.
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"thought": "need to read", "tool_call": {"name": "read_file", "args": {"path": "/tmp/x"}}}"#);
        let step = buf.take_parsed().expect("should parse").unwrap();
        match step {
            AgentStep::ToolCall { thought, tool_name, args } => {
                assert_eq!(thought, "need to read");
                assert_eq!(tool_name, "read_file");
                assert_eq!(args["path"], "/tmp/x");
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn top_level_name_and_args() {
        // Model outputs: {"name": "run_shell", "args": {"command": "ls"}}
        let mut buf = JsonStreamBuffer::new();
        buf.feed(r#"{"name": "run_shell", "args": {"command": "ls"}}"#);
        let step = buf.take_parsed().expect("should parse").unwrap();
        match step {
            AgentStep::ToolCall { tool_name, args, .. } => {
                assert_eq!(tool_name, "run_shell");
                assert_eq!(args["command"], "ls");
            }
            _ => panic!("expected ToolCall"),
        }
    }
}
