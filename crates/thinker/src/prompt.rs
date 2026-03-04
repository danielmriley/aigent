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
}
