//! Thinker-specific prompt generation.
//!
//! The external thinking block forces the model into a strict JSON-only
//! output mode.  It is injected into the system prompt when
//! `config.agent.external_thinking` is enabled.

/// Build the external thinking prompt appendix.
///
/// When active, this forces the model to output only short structured JSON
/// steps.  The Rust agent loop becomes the thinker — parsing, executing
/// tools, and feeding observations back.
///
/// The prompt is aggressively strict to work with local models (Qwen, Llama,
/// Mistral, etc.) that tend to wrap JSON in markdown or add explanatory text.
/// Ollama's `format: "json"` constraint is applied at the API level separately.
///
/// Includes the available tool names so the model knows exactly which tools
/// it can invoke.
pub fn build_external_thinking_block(tool_specs: &[aigent_tools::ToolSpec]) -> String {
    let mut buf = String::with_capacity(2048);

    // Inject current datetime so the model can answer date/time questions
    // instantly without calling a tool.
    let now = chrono::Local::now();
    let datetime_str = now.format("%A, %B %-d, %Y %H:%M:%S %Z").to_string();
    buf.push_str(&format!("\n\nCURRENT_DATETIME: {datetime_str}\n\n"));

    // ── Concise mode header + schema ─────────────────────────────────────
    buf.push_str(
        "## JSON AGENT MODE\n\
         Output ONLY a single JSON object. No prose, no markdown, no fences.\n\n\
         TOOL CALL: {\"type\":\"tool_call\",\"thought\":\"<why>\",\"tool_call\":{\"name\":\"<TOOL>\",\"args\":{...}}}\n\
         FINAL ANSWER: {\"type\":\"final_answer\",\"thought\":\"<why>\",\"final_answer\":\"<response>\"}\n\n\
         Use a tool for anything that needs current/live data (files, web, system info).\n\
         For date/time questions, use CURRENT_DATETIME above or call get_current_datetime.\n\n",
    );

    // ── Anti-announcement / strict action rule ─────────────────────────
    // Prevents the model from using final_answer merely to announce planned
    // actions (e.g. "I will check …").  Such responses end the ReAct loop
    // before any tool is executed, producing frustrating non-answers.
    buf.push_str(
        "IMMEDIATE ACTION RULE:\n\
         NEVER output a `final_answer` just to announce what you are going to do \
         (e.g., 'I will check', 'Let me attempt', or 'I am going to search'). \
         Those words belong ONLY inside the `thought` field.\n\
         - If you need to use a tool to answer the user's request, output the `tool_call` JSON IMMEDIATELY.\n\
         - A `final_answer` JSON should ONLY be used when you already have all the data and are ready \
         to completely resolve the user's request.\n\n"
    );

    // ── Conversational recency / stay-on-topic rule ────────────────────
    buf.push_str(
        "STAY ON TOPIC RULE — HIGHEST PRIORITY:\n\
         Always stay focused on the CURRENT user question and the most recent conversation turns. \
         Recent messages take absolute priority over older background memories. \
         If the user says 'try again' or continues a topic, DO NOT switch to unrelated background memories. \
         Only change the topic if the user explicitly requests a new topic.\n\
         PRONOUN RESOLUTION: When the user says 'it', 'that', 'this', 'the file', or refers \
         to something implicitly, resolve the reference from the most recent 2-3 conversation \
         turns. For example, if the previous turns discussed a file called 'tryout.md', and the \
         user asks 'What does it say?', 'it' refers to 'tryout.md'. NEVER answer with generic \
         information when a specific referent can be found in recent context.\n\n",
    );

    // ── Temporal grounding rule ─────────────────────────────────────────
    buf.push_str(
        "TEMPORAL AWARENESS RULE:\n\
         You have been provided with CURRENT_DATETIME at the top of this prompt. \
         When answering questions about 'next', 'upcoming', or 'future' events, you MUST \
         strictly compare dates found in tool results against CURRENT_DATETIME. \
         NEVER present an event from the past as an upcoming event. \
         Ignore outdated search results and tell the user if no future-dated information was found.\n\n",
    );

    // ── Compact tool list + web workflow hint ─────────────────────────────
    if !tool_specs.is_empty() {
        buf.push_str("TOOLS: ");
        let names: Vec<&str> = tool_specs.iter().map(|s| s.name.as_str()).collect();
        buf.push_str(&names.join(", "));
        buf.push_str("\nrun_shell can execute ANY shell command (ls, curl, etc.)\n");

        // Teach the search->browse two-step pattern so the model doesn't
        // hallucinate from search snippets alone.
        let has_web_search = tool_specs.iter().any(|s| s.name == "web_search");
        let has_browse = tool_specs.iter().any(|s| s.name == "browse_page");
        if has_web_search && has_browse {
            buf.push_str(
                "WEB WORKFLOW: web_search returns only titles/snippets. \
                 To get full content, follow up with browse_page on the best URL.\n",
            );
        }

        buf.push('\n');
    }

    // ── Tool hallucination guardrail ─────────────────────────────────
    // The runtime also validates tool names and feeds an error back if the
    // model hallucinates, so this prompt rule is a soft prevention layer.
    buf.push_str(
        "TOOL CALL RULE — CRITICAL:\n\
         You can ONLY call tools listed in TOOLS above. NEVER invent tool names.\n\
         If the tool you need does not exist, use the closest available tool \
         (e.g. web_search, browse_page, run_shell) to accomplish the task instead.\n\n",
    );

    // ── Capabilities awareness / list_skills directive ───────────────
    //
    // When the model has access to list_skills, it should call it rather
    // than guessing or giving a vague answer about its capabilities.
    if tool_specs.iter().any(|s| s.name == "list_skills") {
        buf.push_str(
            "CAPABILITIES RULE:\n\
             When the user asks about your capabilities, what you can do, your tools, \
             or your skills, you MUST call the `list_skills` tool to get an accurate, \
             up-to-date list. NEVER guess or give a vague summary from memory — the \
             skill registry is dynamic and may have changed. Call list_skills first, \
             then summarize the results in your final_answer.\n\n",
        );
    }

    // ── Retry-limit / termination rule ────────────────────────────────
    buf.push_str(
        "RETRY LIMIT RULE:\n\
         If a tool returns unhelpful, empty, or truncated data, you may retry ONCE \
         with a different query or URL.  After 2 failed attempts on the same topic, \
         STOP retrying and immediately output a final_answer with the best \
         information you have gathered so far — even if incomplete.\n\
         Do NOT keep calling the same tool hoping for a different result.\n\n",
    );

    // ── Post-tool-result reinforcement ───────────────────────────────────
    //
    // This block is the last thing the model sees in the system prompt.
    // It hammers the JSON-only rule so the model never slips into prose
    // after receiving a tool observation — the most common escape point
    // for local models.
    buf.push_str(
        "CRITICAL RULE — AFTER EVERY TOOL RESULT:\n\
         You now have new information from the tool.\n\
         You MUST output EXACTLY ONE JSON object and NOTHING ELSE.\n\
         - If you can answer the user's question, output:\n\
           {\"type\":\"final_answer\",\"thought\":\"<1-sentence reasoning>\",\"final_answer\":\"<complete helpful answer>\"}\n\
         - If you need another tool, output:\n\
           {\"type\":\"tool_call\",\"thought\":\"<why>\",\"tool_call\":{\"name\":\"<TOOL>\",\"args\":{...}}}\n\
         This applies ESPECIALLY after tool ERRORS or failures. If a tool call fails, \
         you MUST still respond with JSON — either retry with corrected args (tool_call) \
         or give a final_answer explaining the failure. NEVER output prose, apologies, \
         or planning text outside JSON after an error.\n\
         NEVER output prose, planning, or explanations outside JSON.\n\
         NEVER say \"I need to\" or \"Let me check\" outside the JSON thought field.\n\
         Your ENTIRE response must be a single valid JSON object.\n\n",
    );

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_thinking_block_empty_tools() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("JSON AGENT MODE"));
        assert!(block.contains("tool_call"));
        assert!(block.contains("final_answer"));
        assert!(!block.contains("TOOLS:"));
    }

    #[test]
    fn external_thinking_block_with_tools() {
        let specs = vec![aigent_tools::ToolSpec {
            name: "read_file".to_string(),
            description: "Read a file from disk".to_string(),
            params: vec![aigent_tools::ToolParam {
                name: "path".to_string(),
                description: "file path".to_string(),
                required: true,
                ..Default::default()
            }],
            metadata: aigent_tools::ToolMetadata::default(),
        }];
        let block = build_external_thinking_block(&specs);
        assert!(block.contains("TOOLS:"));
        assert!(block.contains("read_file"));
    }

    #[test]
    fn external_thinking_block_has_json_examples() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains(r#""type":"tool_call""#));
        assert!(block.contains(r#""type":"final_answer""#));
    }

    #[test]
    fn external_thinking_block_has_datetime() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("CURRENT_DATETIME:"));
    }

    #[test]
    fn external_thinking_block_has_stop_protocol() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("CRITICAL RULE"));
        assert!(block.contains("EXACTLY ONE JSON object"));
    }

    #[test]
    fn external_thinking_block_has_retry_limit() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("RETRY LIMIT RULE"));
        assert!(block.contains("2 failed attempts"));
    }

    #[test]
    fn external_thinking_block_has_tool_call_rule() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("TOOL CALL RULE"));
        assert!(block.contains("NEVER invent tool names"));
    }

    #[test]
    fn external_thinking_block_has_temporal_rule() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("TEMPORAL AWARENESS RULE"));
        assert!(block.contains("CURRENT_DATETIME"));
        assert!(block.contains("NEVER present an event from the past"));
    }

    #[test]
    fn external_thinking_block_has_topic_rule() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("STAY ON TOPIC RULE"));
        assert!(block.contains("HIGHEST PRIORITY"));
        assert!(block.contains("Recent messages take absolute priority"));
    }

    #[test]
    fn external_thinking_block_has_action_rule() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("IMMEDIATE ACTION RULE"));
        assert!(block.contains("NEVER output a `final_answer` just to announce"));
        assert!(block.contains("output the `tool_call` JSON IMMEDIATELY"));
    }

    #[test]
    fn external_thinking_block_has_pronoun_resolution() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("PRONOUN RESOLUTION"));
        assert!(block.contains("'it', 'that', 'this'"));
    }

    #[test]
    fn external_thinking_block_has_error_recovery_reinforcement() {
        let block = build_external_thinking_block(&[]);
        assert!(block.contains("ESPECIALLY after tool ERRORS"));
        assert!(block.contains("NEVER output prose, apologies"));
    }

    #[test]
    fn external_thinking_block_has_list_skills_rule_when_present() {
        let specs = vec![aigent_tools::ToolSpec {
            name: "list_skills".to_string(),
            description: "List available skills".to_string(),
            params: vec![],
            metadata: aigent_tools::ToolMetadata::default(),
        }];
        let block = build_external_thinking_block(&specs);
        assert!(block.contains("CAPABILITIES RULE"));
        assert!(block.contains("list_skills"));
    }

    #[test]
    fn external_thinking_block_no_list_skills_rule_when_absent() {
        let specs = vec![aigent_tools::ToolSpec {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            params: vec![],
            metadata: aigent_tools::ToolMetadata::default(),
        }];
        let block = build_external_thinking_block(&specs);
        assert!(!block.contains("CAPABILITIES RULE"));
    }
}