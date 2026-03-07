//! Prompt builders for each subagent specialist role.

use super::types::SubagentRole;

/// Build the full prompt for a subagent specialist.
///
/// The prompt includes:
///   1. The role-specific system instruction.
///   2. The conversation context (system prompt + recent turns).
///   3. Strict output-format instructions so the reply can be parsed
///      into a [`SubagentAnalysis`].
pub fn build_subagent_prompt(
    role: SubagentRole,
    role_system_prompt: &str,
    user_message: &str,
    context_summary: &str,
) -> String {
    let output_format = "\
IMPORTANT: Be extremely concise. 1-2 bullet items per section MAX. No prose, no markdown, no explanations.

You MUST respond using EXACTLY this format (one item per line, use NONE if empty):

KEY_FACTS:
- <fact>

PROPOSED_ACTIONS:
- <action>

POTENTIAL_PITFALLS:
- <pitfall>

Do NOT output anything outside this structure. No preamble, no summary, no closing remarks.";

    format!(
        "=== {role} SPECIALIST ===\n\
{role_system_prompt}\n\n\
CONTEXT:\n{context_summary}\n\n\
USER REQUEST:\n{user_message}\n\n\
{output_format}"
    )
}

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
