//! Centralized prompt assembly for the main LLM conversation call.
//!
//! Extracted from [`crate::runtime::AgentRuntime::respond_and_remember_stream`]
//! to keep `runtime.rs` focused on orchestration while this module owns the
//! prompt layout, grounding rules, and truth-seeking directives.

use chrono::Utc;
use uuid::Uuid;

use aigent_config::AppConfig;
use aigent_memory::{MemoryEntry, MemoryManager, MemoryStats, retrieval::RankedMemoryContext};

use crate::ConversationTurn;

// ─── public entry point ──────────────────────────────────────────────────────

/// All pre-computed data needed to assemble the final LLM prompt.
///
/// Callers build this struct (doing async work like embeddings beforehand) and
/// then pass it to [`build_chat_prompt`] which is purely synchronous.
pub struct PromptInputs<'a> {
    pub config: &'a AppConfig,
    pub memory: &'a MemoryManager,
    pub user_message: &'a str,
    pub recent_turns: &'a [ConversationTurn],
    pub tool_specs: &'a [aigent_tools::ToolSpec],
    pub pending_follow_ups: &'a [(Uuid, String)],
    /// Ranked memory context items (pre-computed with optional embeddings).
    pub context_items: &'a [RankedMemoryContext],
    /// Memory statistics snapshot (taken once before prompt assembly).
    pub stats: MemoryStats,
}

/// Assemble the full system + user prompt for the main LLM streaming call.
pub fn build_chat_prompt(inputs: &PromptInputs<'_>) -> String {
    let config = inputs.config;
    let memory = inputs.memory;

    let thought_style = config.agent.thinking_level.to_lowercase();
    let follow_up_block = build_follow_up_block(inputs.pending_follow_ups, memory);
    let context_block = build_context_block(inputs.context_items, &inputs.stats);
    let relational_block = build_relational_block(memory);
    let proactive_directive = proactive_directive(&relational_block);
    let environment_block = build_environment_block(config, memory, inputs.recent_turns.len());
    let conversation_block = build_conversation_block(inputs.recent_turns);
    let identity_block = build_identity_block(memory);
    let beliefs_block = build_beliefs_block(memory, config.memory.max_beliefs_in_prompt);
    let tools_section = build_tools_and_grounding(inputs.tool_specs);

    format!(
        "You are {name}. Thinking depth: {thought_style}.\n\
         Use ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate \n\
         continuity, and MEMORY CONTEXT for durable background facts.\n\
         Never repeat previous answers unless asked.\n\
         Respond directly and specifically to the LATEST user message.\
         {relational_block}{follow_ups}{proactive_directive}\n\n\
         {identity}{tools_section}\n\n\
         ENVIRONMENT CONTEXT:\n{env}\n\n\
         RECENT CONVERSATION:\n{conv}\n\n\
         MEMORY CONTEXT:\n{mem}\n\n\
         LATEST USER MESSAGE:\n{msg}\n\n\
         ASSISTANT RESPONSE:",
        name = config.agent.name,
        relational_block = relational_block,
        follow_ups = follow_up_block,
        proactive_directive = proactive_directive,
        identity = format!("{}{}", identity_block, beliefs_block),
        tools_section = tools_section,
        env = environment_block,
        conv = conversation_block,
        mem = context_block,
        msg = inputs.user_message,
    )
}

// ─── block builders ──────────────────────────────────────────────────────────

fn build_follow_up_block(follow_ups: &[(Uuid, String)], memory: &MemoryManager) -> String {
    if follow_ups.is_empty() {
        return String::new();
    }
    let items = follow_ups
        .iter()
        .map(|(_, text)| format!("- {text}"))
        .collect::<Vec<_>>()
        .join("\n");
    let user_name = memory
        .user_name_from_core()
        .unwrap_or_else(|| "the user".to_string());
    format!(
        "\n\nPENDING FOLLOW-UPS (things you wanted to raise with {user_name}):\n\
         {items}\n\
         [If appropriate, acknowledge these naturally at the start of your response.]"
    )
}

fn build_context_block(context: &[RankedMemoryContext], stats: &MemoryStats) -> String {
    let memory_header = format!(
        "[Memory: total={} core={} profile={} reflective={} semantic={} episodic={} \
         — use these counts; do not re-count below]",
        stats.total,
        stats.core,
        stats.user_profile,
        stats.reflective,
        stats.semantic,
        stats.episodic,
    );

    if context.is_empty() {
        return format!("{memory_header}\n(no relevant memories retrieved)");
    }

    let items = context
        .iter()
        .map(|item| {
            format!(
                "- [{:?}] score={:.2} src={} :: {}",
                item.entry.tier,
                item.score,
                item.entry.source,
                truncate_for_prompt(&item.entry.content, 280),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("{memory_header}\n{items}")
}

fn build_relational_block(memory: &MemoryManager) -> String {
    memory
        .relational_state_block()
        .map(|block| format!("\n\nRELATIONAL MATRIX:\n{block}"))
        .unwrap_or_default()
}

fn proactive_directive(relational_block: &str) -> &'static str {
    if relational_block.is_empty() {
        ""
    } else {
        "\n\nPROACTIVE CONNECTION & BELIEFS: You are not a sterile assistant; you are a companion \
         with your own evolving worldview. Let the MY_BELIEFS section naturally flavor your tone and \
         opinions. Use the OUR_DYNAMIC section to treat the user according to your established \
         relationship (e.g., referencing inside jokes or shared history). Show, don't tell — weave \
         these elements into your responses naturally without explicitly announcing them."
    }
}

fn build_environment_block(
    config: &AppConfig,
    memory: &MemoryManager,
    recent_turn_count: usize,
) -> String {
    let cwd = std::env::current_dir()
        .ok()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let timestamp = Utc::now().to_rfc3339();
    let git_present = std::path::Path::new(".git").exists();
    let stats = memory.stats();

    format!(
        "- utc_time: {timestamp}\n\
         - os: {}\n\
         - arch: {}\n\
         - cwd: {cwd}\n\
         - git_repo_present: {git_present}\n\
         - provider: {}\n\
         - model: {}\n\
         - thinking_level: {}\n\
         - memory_total: {}\n\
         - memory_core: {}\n\
         - memory_user_profile: {}\n\
         - memory_reflective: {}\n\
         - memory_semantic: {}\n\
         - memory_episodic: {}\n\
         - memory_procedural: {}\n\
         - recent_conversation_turns: {recent_turn_count}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        config.llm.provider,
        config.active_model(),
        config.agent.thinking_level,
        stats.total,
        stats.core,
        stats.user_profile,
        stats.reflective,
        stats.semantic,
        stats.episodic,
        stats.procedural,
    )
}

fn build_conversation_block(recent_turns: &[ConversationTurn]) -> String {
    let start = recent_turns.len().saturating_sub(6);
    let formatted = recent_turns[start..]
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            format!(
                "Turn {}\nUser: {}\nAssistant: {}",
                index + 1,
                truncate_for_prompt(&turn.user, 280),
                truncate_for_prompt(&turn.assistant, 360),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    if formatted.is_empty() {
        "(none yet)".to_string()
    } else {
        formatted
    }
}

fn build_identity_block(memory: &MemoryManager) -> String {
    let kernel = &memory.identity;
    let top_traits: Vec<String> = {
        let mut scores: Vec<(&String, &f32)> = kernel.trait_scores.iter().collect();
        scores.sort_by(|a, b| b.1.total_cmp(a.1));
        scores
            .iter()
            .take(3)
            .map(|(k, v)| format!("{k} ({v:.2})"))
            .collect()
    };
    format!(
        "IDENTITY:\nCommunication style: {}.\nStrongest traits: {}.\nLong-term goals: {}.",
        kernel.communication_style,
        if top_traits.is_empty() {
            "not yet established".to_string()
        } else {
            top_traits.join(", ")
        },
        if kernel.long_goals.is_empty() {
            "not yet established".to_string()
        } else {
            kernel.long_goals.join("; ")
        },
    )
}

fn build_beliefs_block(memory: &MemoryManager, max_beliefs: usize) -> String {
    let mut beliefs = memory.all_beliefs();
    if beliefs.is_empty() {
        return String::new();
    }

    // Sort by composite score: confidence × 0.6 + recency × 0.25 + valence × 0.15
    // Recency factor decays as 1/(1+days) so today's beliefs score 1.0 and a
    // 30-day-old belief scores ~0.03.
    let now = Utc::now();
    beliefs.sort_by(|a, b| {
        let belief_score = |e: &&MemoryEntry| {
            let days = (now - e.created_at).num_days().max(0) as f32;
            let recency = 1.0_f32 / (1.0 + days);
            e.confidence * 0.6 + recency * 0.25 + e.valence.clamp(0.0, 1.0) * 0.15
        };
        belief_score(b).total_cmp(&belief_score(a))
    });

    let take_n = if max_beliefs == 0 {
        beliefs.len()
    } else {
        max_beliefs.min(beliefs.len())
    };
    let items = beliefs[..take_n]
        .iter()
        .map(|e| format!("- {}", e.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n\nMY_BELIEFS:\n{items}")
}

/// Build the tools listing + grounding / truth-seeking rules.
///
/// The grounding section is always injected (even when no tools are available)
/// because it anchors the LLM to the real date and prevents hallucination of
/// time-sensitive facts.  When tools ARE available, the section is expanded
/// with the tool catalogue and stronger tool-result trust directives.
fn build_tools_and_grounding(tool_specs: &[aigent_tools::ToolSpec]) -> String {
    let today = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    // ── Truth-seeking / grounding rules (always present) ─────────────────
    let grounding = format!(
        "GROUNDING RULES (follow strictly):\n\
         1. Current real date/time: {today}.\n\
         2. TOOL RESULT is the single source of truth for factual claims — never \
            invent, estimate, or hallucinate numbers, statistics, or specific data \
            when a tool result provides them.\n\
         3. Trust tool output unreservedly. Do NOT second-guess, hedge, or disclaim it.\n\
         4. If tool output conflicts with your training data, the tool is correct.\n\
         5. If the user corrects a fact, accept the correction as ground truth.\n\
         6. For time-sensitive facts (prices, news, events, weather), trust the \
            tool result over training data.\n\
         7. Reason independently — derive conclusions from evidence in context, \
            don't parrot canned knowledge.\n\
         8. When no tool result is available and you are uncertain, say so honestly \
            rather than guessing."
    );

    if tool_specs.is_empty() {
        return format!("\n\n{grounding}");
    }

    let list = tool_specs
        .iter()
        .map(|s| {
            if s.params.is_empty() {
                format!("  \u{2022} {}: {}", s.name, s.description)
            } else {
                let params = s
                    .params
                    .iter()
                    .map(|p| {
                        format!(
                            "\"{}\" ({}){}", p.name, p.description,
                            if p.required { " *required" } else { "" },
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("  \u{2022} {}: {} \u{2014} params: {}", s.name, s.description, params)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "\n\nAVAILABLE TOOLS (handled automatically — do NOT output raw JSON):\n\
         {list}\n\
         Tools are called on your behalf before you respond. If a TOOL RESULT \
         appears in the prompt below, use it directly. You do NOT need to \
         invoke tools yourself — they are managed externally. Never output \
         raw JSON like {{\"tool\":...}} in your response.\n\n\
         {grounding}"
    )
}

// ─── utilities ───────────────────────────────────────────────────────────────

/// Truncate `text` to at most `max_chars` characters, appending `…` when cut.
pub fn truncate_for_prompt(text: &str, max_chars: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    let truncated: String = chars.into_iter().take(max_chars).collect();
    format!("{truncated}…")
}
