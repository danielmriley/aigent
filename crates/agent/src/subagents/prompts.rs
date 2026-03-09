//! Prompt utilities for the subagent specialist pipeline.

/// Build a concise context summary from the system prompt for injection
/// into subagent prompts.  We truncate to avoid blowing the context window
/// on multiple parallel calls.
pub fn truncate_context(system_prompt: &str, max_chars: usize) -> String {
    if system_prompt.len() <= max_chars {
        system_prompt.to_string()
    } else {
        // Find a char-safe boundary to avoid panicking on multi-byte UTF-8.
        let boundary = system_prompt
            .char_indices()
            .take_while(|(i, _)| *i < max_chars)
            .last()
            .map_or(0, |(i, c)| i + c.len_utf8());
        format!(
            "{}... (truncated, {} total chars)",
            &system_prompt[..boundary],
            system_prompt.len()
        )
    }
}
