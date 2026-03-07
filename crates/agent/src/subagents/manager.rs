//! Subagent manager — orchestrates parallel specialist LLM calls.
//!
//! Follows the same `tokio::join!` pattern used by the multi-agent sleep
//! pipeline in `crates/agent/src/runtime/sleep.rs`.

use std::time::Duration;

use tokio::time::timeout;
use tracing::{debug, info, warn};

use aigent_config::SubagentsConfig;
use aigent_llm::{LlmRouter, Provider};

use super::prompts::{build_subagent_prompt, truncate_context};
use super::types::{SubagentAnalysis, SubagentRole};

/// Maximum characters of the system prompt to include in each subagent's
/// context.  Keeps parallel calls fast on smaller context windows.
const SUBAGENT_CONTEXT_LIMIT: usize = 4_000;

/// Maximum output tokens per subagent call.  Keeps responses short (bullet
/// lists, not essays) and reduces Ollama queue time when requests serialise.
const SUBAGENT_MAX_TOKENS: u32 = 300;

/// Orchestrates parallel subagent specialist calls and collects their
/// structured analyses.
pub struct SubagentManager<'a> {
    llm: &'a LlmRouter,
    config: &'a SubagentsConfig,
    ollama_model: &'a str,
    openrouter_model: &'a str,
}

impl<'a> SubagentManager<'a> {
    pub fn new(
        llm: &'a LlmRouter,
        config: &'a SubagentsConfig,
        ollama_model: &'a str,
        openrouter_model: &'a str,
    ) -> Self {
        Self {
            llm,
            config,
            ollama_model,
            openrouter_model,
        }
    }

    /// Run all specialist subagents in parallel and return their analyses.
    ///
    /// The `system_prompt` is the already-built Captain system prompt
    /// (memory context, identity, etc.) — subagents receive a truncated
    /// snapshot of it as read-only context.
    pub async fn run_parallel_team(
        &self,
        primary: Provider,
        user_message: &str,
        system_prompt: &str,
    ) -> Vec<(SubagentRole, SubagentAnalysis)> {
        let context = truncate_context(system_prompt, SUBAGENT_CONTEXT_LIMIT);

        let researcher_prompt = build_subagent_prompt(
            SubagentRole::Researcher,
            &self.config.researcher_prompt,
            user_message,
            &context,
        );
        let planner_prompt = build_subagent_prompt(
            SubagentRole::Planner,
            &self.config.planner_prompt,
            user_message,
            &context,
        );
        let critic_prompt = build_subagent_prompt(
            SubagentRole::Critic,
            &self.config.critic_prompt,
            user_message,
            &context,
        );

        info!("subagents: launching 3 specialists in parallel");

        let (researcher_reply, planner_reply, critic_reply) = tokio::join!(
            self.subagent_llm_call(primary, &researcher_prompt, "Researcher"),
            self.subagent_llm_call(primary, &planner_prompt, "Planner"),
            self.subagent_llm_call(primary, &critic_prompt, "Critic"),
        );

        let mut results = Vec::with_capacity(3);

        if let Some(reply) = researcher_reply {
            debug!(len = reply.len(), "subagents: Researcher reply received");
            results.push((SubagentRole::Researcher, parse_analysis("Researcher", &reply)));
        }
        if let Some(reply) = planner_reply {
            debug!(len = reply.len(), "subagents: Planner reply received");
            results.push((SubagentRole::Planner, parse_analysis("Planner", &reply)));
        }
        if let Some(reply) = critic_reply {
            debug!(len = reply.len(), "subagents: Critic reply received");
            results.push((SubagentRole::Critic, parse_analysis("Critic", &reply)));
        }

        info!(
            succeeded = results.len(),
            "subagents: parallel team complete"
        );
        results
    }

    /// Format the team's analyses into a context block that the Captain
    /// can consume as part of its message history.
    pub fn format_debate_block(results: &[(SubagentRole, SubagentAnalysis)]) -> String {
        if results.is_empty() {
            return String::new();
        }

        let mut block = String::from("<subagent_debate>\n");

        for (role, analysis) in results {
            block.push_str(&format!("=== {} ===\n", role.label()));

            if !analysis.key_facts.is_empty() {
                block.push_str("KEY FACTS:\n");
                for fact in &analysis.key_facts {
                    block.push_str(&format!("- {fact}\n"));
                }
            }
            if !analysis.proposed_actions.is_empty() {
                block.push_str("PROPOSED ACTIONS:\n");
                for action in &analysis.proposed_actions {
                    block.push_str(&format!("- {action}\n"));
                }
            }
            if !analysis.potential_pitfalls.is_empty() {
                block.push_str("POTENTIAL PITFALLS:\n");
                for pitfall in &analysis.potential_pitfalls {
                    block.push_str(&format!("- {pitfall}\n"));
                }
            }
            block.push('\n');
        }

        block.push_str("</subagent_debate>");
        block
    }

    /// Per-subagent timeout — 90 seconds should be generous for a small
    /// local model.  If Ollama serialises the requests (NUM_PARALLEL=1),
    /// the third call may hit the timeout; that's fine — the team degrades
    /// gracefully with fewer results.
    const CALL_TIMEOUT: Duration = Duration::from_secs(90);

    /// Single subagent LLM call with logging and timeout.  Returns `None`
    /// on failure so the team degrades gracefully (same pattern as
    /// `sleep_llm_call`).
    async fn subagent_llm_call(
        &self,
        primary: Provider,
        prompt: &str,
        role_label: &str,
    ) -> Option<String> {
        let fut = self.llm.chat_with_fallback_limited(
            primary,
            self.ollama_model,
            self.openrouter_model,
            prompt,
            SUBAGENT_MAX_TOKENS,
        );

        match timeout(Self::CALL_TIMEOUT, fut).await {
            Ok(Ok((_provider, reply))) => {
                info!(
                    role = role_label,
                    reply_len = reply.len(),
                    "subagent: specialist reply received"
                );
                Some(reply)
            }
            Ok(Err(err)) => {
                warn!(?err, role = role_label, "subagent: LLM call failed");
                None
            }
            Err(_elapsed) => {
                warn!(
                    role = role_label,
                    timeout_secs = Self::CALL_TIMEOUT.as_secs(),
                    "subagent: LLM call timed out"
                );
                None
            }
        }
    }
}

/// Parse a subagent's raw text reply into a [`SubagentAnalysis`].
///
/// Looks for `KEY_FACTS:`, `PROPOSED_ACTIONS:`, and `POTENTIAL_PITFALLS:`
/// section headers, collecting bullet lines (`- ...`) under each.
/// Unrecognised lines are silently ignored — this mirrors the best-effort
/// parsing approach used by `parse_agentic_insights` in the sleep pipeline.
fn parse_analysis(role: &str, raw: &str) -> SubagentAnalysis {
    let mut analysis = SubagentAnalysis {
        role: role.to_string(),
        ..Default::default()
    };

    #[derive(PartialEq)]
    enum Section {
        None,
        KeyFacts,
        ProposedActions,
        PotentialPitfalls,
    }

    let mut current = Section::None;

    for line in raw.lines() {
        let trimmed = line.trim();

        if matches_header(trimmed, "KEY_FACTS") || matches_header(trimmed, "KEY FACTS") {
            current = Section::KeyFacts;
            continue;
        }
        if matches_header(trimmed, "PROPOSED_ACTIONS") || matches_header(trimmed, "PROPOSED ACTIONS") {
            current = Section::ProposedActions;
            continue;
        }
        if matches_header(trimmed, "POTENTIAL_PITFALLS") || matches_header(trimmed, "POTENTIAL PITFALLS") {
            current = Section::PotentialPitfalls;
            continue;
        }

        // Collect bullet items under the current section.
        if let Some(item) = trimmed.strip_prefix("- ") {
            let item = item.trim();
            if item.is_empty() || item.eq_ignore_ascii_case("NONE") {
                continue;
            }
            match current {
                Section::KeyFacts => analysis.key_facts.push(item.to_string()),
                Section::ProposedActions => analysis.proposed_actions.push(item.to_string()),
                Section::PotentialPitfalls => analysis.potential_pitfalls.push(item.to_string()),
                Section::None => {}
            }
        }
    }

    debug!(
        role,
        facts = analysis.key_facts.len(),
        actions = analysis.proposed_actions.len(),
        pitfalls = analysis.potential_pitfalls.len(),
        "subagent: parsed analysis"
    );
    analysis
}

/// Check if `line` is a section header for `header` (e.g. "KEY_FACTS").
/// Accepts `KEY_FACTS:`, `KEY_FACTS`, `**KEY_FACTS:**`, etc.
fn matches_header(line: &str, header: &str) -> bool {
    // Strip common markdown decorations.
    let stripped = line.trim_start_matches('*').trim_start_matches('#').trim();
    stripped == header
        || stripped.starts_with(&format!("{header}:"))
        || stripped.starts_with(&format!("{header} :"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_well_formed_analysis() {
        let raw = "\
KEY_FACTS:
- The user wants to refactor the config module
- There are 14 config structs currently

PROPOSED_ACTIONS:
- Split config into sub-modules by domain
- Add validation on load

POTENTIAL_PITFALLS:
- Breaking existing TOML files
- Migration complexity
";
        let analysis = parse_analysis("Researcher", raw);
        assert_eq!(analysis.key_facts.len(), 2);
        assert_eq!(analysis.proposed_actions.len(), 2);
        assert_eq!(analysis.potential_pitfalls.len(), 2);
        assert_eq!(analysis.role, "Researcher");
    }

    #[test]
    fn parse_with_none_items() {
        let raw = "\
KEY_FACTS:
- NONE

PROPOSED_ACTIONS:
- Write tests first

POTENTIAL_PITFALLS:
- NONE
";
        let analysis = parse_analysis("Critic", raw);
        assert_eq!(analysis.key_facts.len(), 0);
        assert_eq!(analysis.proposed_actions.len(), 1);
        assert_eq!(analysis.potential_pitfalls.len(), 0);
    }

    #[test]
    fn parse_empty_input() {
        let analysis = parse_analysis("Planner", "");
        assert!(analysis.key_facts.is_empty());
        assert!(analysis.proposed_actions.is_empty());
        assert!(analysis.potential_pitfalls.is_empty());
    }

    #[test]
    fn format_debate_block_round_trip() {
        let results = vec![
            (
                SubagentRole::Researcher,
                SubagentAnalysis {
                    role: "Researcher".to_string(),
                    key_facts: vec!["fact1".to_string()],
                    proposed_actions: vec![],
                    potential_pitfalls: vec![],
                },
            ),
            (
                SubagentRole::Critic,
                SubagentAnalysis {
                    role: "Critic".to_string(),
                    key_facts: vec![],
                    proposed_actions: vec![],
                    potential_pitfalls: vec!["risk1".to_string()],
                },
            ),
        ];
        let block = SubagentManager::format_debate_block(&results);
        assert!(block.contains("<subagent_debate>"));
        assert!(block.contains("</subagent_debate>"));
        assert!(block.contains("=== Researcher ==="));
        assert!(block.contains("fact1"));
        assert!(block.contains("risk1"));
    }
}
