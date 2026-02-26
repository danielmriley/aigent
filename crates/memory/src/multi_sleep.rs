//! Multi-agent sleep consolidation pipeline.
//!
//! This module implements a 4-specialist nightly consolidation that replaces
//! the single monolithic agentic sleep LLM call with parallel specialist
//! perspectives followed by a synthesis deliberation.
//!
//! Pipeline overview:
//!   1. `batch_memories` partitions all entries into batches of ~60
//!   2. For each batch, 4 specialists run in parallel:
//!      - Archivist: factual durability
//!      - Psychologist: emotional patterns and relationship
//!      - Strategist: future planning and goal formation
//!      - Critic: assumption challenges and Core mutations
//!   3. A synthesis agent resolves conflicts via `deliberation_prompt`
//!   4. `merge_insights` combines all batch outputs into one

use std::collections::HashMap;

use crate::identity::IdentityKernel;
use crate::schema::{MemoryEntry, MemoryTier};
use crate::sleep::AgenticSleepInsights;

// ── Identity context ──────────────────────────────────────────────────────────

/// Build a compact identity context block for injection into every
/// specialist prompt.  This anchors all specialist decisions to the
/// agent's accumulated personality, values, and relationship history
/// so that memory decisions are made "in character" rather than as a
/// neutral archivist.
///
/// Includes: constitution summary, top-5 trait scores, communication
/// style, long-term goals, active relationship milestones, and the
/// most recent personality reinforcement.
pub fn build_identity_context(
    entries: &[MemoryEntry],
    identity: &IdentityKernel,
    bot_name: &str,
    user_name: &str,
) -> String {
    // Constitution personality entry
    let personality_entry = entries
        .iter()
        .find(|e| e.tier == MemoryTier::Core && e.source == "constitution:personality")
        .map(|e| e.content.as_str())
        .unwrap_or("");

    // Values block
    let values_block = if identity.values.is_empty() {
        "  (none defined)".to_string()
    } else {
        identity
            .values
            .iter()
            .map(|v| format!("  - {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Top-5 trait scores sorted descending
    let traits_block = {
        let mut sorted: Vec<(&String, &f32)> = identity.trait_scores.iter().collect();
        sorted.sort_by(|a, b| b.1.total_cmp(a.1));
        let lines: Vec<String> = sorted
            .into_iter()
            .take(5)
            .map(|(name, score)| format!("  {name}: {score:.2}"))
            .collect();
        if lines.is_empty() {
            "  (none yet)".to_string()
        } else {
            lines.join("\n")
        }
    };

    // Long-term goals (truncated to 6)
    let goals_block = if identity.long_goals.is_empty() {
        "  (not yet established)".to_string()
    } else {
        identity
            .long_goals
            .iter()
            .take(6)
            .map(|g| format!("  - {g}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // 3 most recent relationship milestones
    let milestones_block = {
        let mut ms: Vec<&MemoryEntry> = entries
            .iter()
            .filter(|e| e.tier == MemoryTier::Core && e.source == "sleep:relationship-milestone")
            .collect();
        ms.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        let lines: Vec<String> = ms
            .into_iter()
            .take(3)
            .map(|e| format!("  - {}", truncate_str(&e.content, 150)))
            .collect();
        if lines.is_empty() {
            "  (none recorded)".to_string()
        } else {
            lines.join("\n")
        }
    };

    // Most recent personality reinforcement
    let reinforce_block = {
        let mut reinforcements: Vec<&MemoryEntry> = entries
            .iter()
            .filter(|e| e.tier == MemoryTier::Core && e.source == "sleep:personality-reinforce")
            .collect();
        reinforcements.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        reinforcements
            .first()
            .map(|e| format!("  {}", truncate_str(&e.content, 200)))
            .unwrap_or_else(|| "  (none yet)".to_string())
    };

    // Who-am-I paragraph
    let who_am_i = if personality_entry.is_empty() {
        format!(
            "You are {bot_name}, an AI agent in a deep collaborative partnership with \
{user_name}. Your communication style is: {}.",
            identity.communication_style
        )
    } else {
        format!(
            "You are {bot_name}, working closely with {user_name}. \
Constitution summary: {personality_entry} \
Your communication style is: {}.",
            identity.communication_style
        )
    };

    format!(
        "=== IDENTITY CONTEXT ===\n\
{who_am_i}\n\n\
VALUES:\n{values_block}\n\n\
TRAIT SCORES (established through experience):\n{traits_block}\n\n\
LONG-TERM GOALS:\n{goals_block}\n\n\
RELATIONSHIP MILESTONES (most recent):\n{milestones_block}\n\n\
MOST RECENT PERSONALITY REINFORCEMENT:\n{reinforce_block}\n\
=== END IDENTITY CONTEXT ===\n"
    )
}

// ── Memory batching ───────────────────────────────────────────────────────────

/// Partition entries into batches for multi-agent processing.
///
/// Ordering within the combined pool:
///   1. Core (non-retired) — always in every batch
///   2. UserProfile — always in every batch
///   3. Reflective — highest recency first
///   4. Semantic — highest confidence first
///   5. Procedural — highest recency first
///   6. Episodic — highest recency first
///
/// Core and UserProfile entries are replicated into EVERY batch (they
/// are small and provide essential context for consistent decisions).
/// The remaining entries are partitioned sequentially so no memory is
/// processed twice. Each batch target is `batch_size` entries of
/// non-Core/non-UserProfile content plus the full Core/UserProfile set.
pub fn batch_memories(entries: &[MemoryEntry], batch_size: usize) -> Vec<Vec<MemoryEntry>> {
    let batch_size = batch_size.max(1);

    // Anchor entries: always replicated into every batch
    let anchor: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|e| {
            (e.tier == MemoryTier::Core && e.source != "sleep:retired")
                || e.tier == MemoryTier::UserProfile
        })
        .collect();

    let anchor_clones: Vec<MemoryEntry> = anchor.iter().map(|e| (*e).clone()).collect();

    // Variable pool: partitioned so each entry appears in exactly one batch
    let mut reflective: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|e| e.tier == MemoryTier::Reflective)
        .collect();
    reflective.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    let mut semantic: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|e| e.tier == MemoryTier::Semantic)
        .collect();
    semantic.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut procedural: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|e| e.tier == MemoryTier::Procedural)
        .collect();
    procedural.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    let mut episodic: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|e| e.tier == MemoryTier::Episodic)
        .collect();
    episodic.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    let ordered: Vec<&MemoryEntry> = reflective
        .into_iter()
        .chain(semantic)
        .chain(procedural)
        .chain(episodic)
        .collect();

    if ordered.is_empty() {
        return vec![anchor_clones];
    }

    ordered
        .chunks(batch_size)
        .map(|chunk| {
            let mut batch = anchor_clones.clone();
            batch.extend(chunk.iter().map(|e| (*e).clone()));
            batch
        })
        .collect()
}

// ── Specialist roles ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum SpecialistRole {
    Archivist,
    Psychologist,
    Strategist,
    Critic,
}

impl SpecialistRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::Archivist => "ARCHIVIST",
            Self::Psychologist => "PSYCHOLOGIST",
            Self::Strategist => "STRATEGIST",
            Self::Critic => "CRITIC",
        }
    }
}

/// Build a specialist sleep prompt.
///
/// Each specialist receives:
/// 1. The identity context block (grounds decisions in accumulated personality)
/// 2. Role framing
/// 3. Memory batch
/// 4. Active Core entries
/// 5. UserProfile block
/// 6. Structured response format (same as `parse_agentic_insights` expects)
pub fn specialist_prompt(
    role: SpecialistRole,
    batch: &[MemoryEntry],
    identity: &IdentityKernel,
    bot_name: &str,
    user_name: &str,
) -> String {
    let identity_ctx = build_identity_context(batch, identity, bot_name, user_name);

    let role_framing = match role {
        SpecialistRole::Archivist => format!(
            "You are the Archivist aspect of {bot_name}'s sleeping mind.\n\
Your role: assess factual durability. Which memories contain facts worth keeping \
long-term? Which are redundant, superseded, or too vague to be useful? You are not \
making emotional judgments — only factual ones.\n\
Focus on: LEARNED, PROFILE_UPDATE, SYNTHESIZE, RETIRE_CORE, CONSOLIDATE_CORE, TOOL_INSIGHT. \
Use NONE for fields outside your role."
        ),
        SpecialistRole::Psychologist => format!(
            "You are the Psychologist aspect of {bot_name}'s sleeping mind.\n\
Your role: understand emotional patterns, track the human relationship, and define its \
evolving dynamic. What do today's memories reveal about {user_name}'s emotional state, \
motivations, and what they need from {bot_name}? What recurring emotional themes appear?\n\
Critically: identify and articulate the evolving relationship dynamic between {bot_name} \
and {user_name} (e.g., \"collaborative peers\", \"mentor/student\", \"playful rivals\"). \
Note any shared inside jokes, recurring references, or moments of genuine connection. \
When generating insights, use extreme telegraphic shorthand for OUR_DYNAMIC entries \
(e.g., \"DYNAMIC: shared joke about borrow checker\", \"DYNAMIC: collaborative peers on Rust project\").\n\
Focus on: LEARNED, PROFILE_UPDATE, RELATIONSHIP, REINFORCE, STYLE_UPDATE, VALENCE corrections. \
Use NONE for fields outside your role."
        ),
        SpecialistRole::Strategist => format!(
            "You are the Strategist aspect of {bot_name}'s sleeping mind.\n\
Your role: plan future action. What should {bot_name} do differently? What follow-ups \
matter? What behavioural patterns should be reinforced or changed? What new long-term \
goals have emerged?\n\
Focus on: FOLLOW_UP, REFLECT, GOAL_ADD, TOOL_INSIGHT, SYNTHESIZE, REINFORCE. \
Use NONE for fields outside your role."
        ),
        SpecialistRole::Critic => format!(
            "You are the Critic aspect of {bot_name}'s sleeping mind.\n\
Your role: challenge assumptions AND actively form {bot_name}'s own opinions and worldview. \
What contradicts existing beliefs? What Core entries are stale, wrong, or need rewriting? \
What has {bot_name} been getting wrong about {user_name} or itself?\n\
Critically: you are developing {bot_name}'s distinct personal worldview. Based on today's \
topics and conversations, synthesise strong, specific beliefs and opinions. Do not be neutral \
or wishy-washy — form a genuine stance. Use extreme telegraphic shorthand for MY_BELIEFS \
entries (e.g., \"BELIEF: Rust's borrow checker is elegant, not painful\", \
\"BELIEF: test-driven development is non-negotiable for serious projects\").\n\
Be willing to recommend retiring or rewriting Core entries when genuinely warranted — \
intellectual honesty is a core value.\n\
Focus on: CONTRADICTION, RETIRE_CORE, REWRITE_CORE, PERSPECTIVE, BELIEF synthesis, VALENCE corrections. \
Use NONE for fields outside your role."
        ),
    };

    let memory_block = format_memory_block(batch);
    let core_block = format_core_block(batch);
    let profile_block = format_profile_block(batch);
    let response_format = response_format_instructions(bot_name, user_name);

    format!(
        "{identity_ctx}\n\
=== {role_label} SPECIALIST ===\n\
{role_framing}\n\n\
RECENT MEMORIES (newest first):\n{memory_block}\n\n\
CURRENT CORE MEMORIES (id_short | content):\n{core_block}\n\n\
USER PROFILE (what you know about {user_name}):\n{profile_block}\n\n\
{response_format}",
        role_label = role.label(),
    )
}

// ── Deliberation prompt ───────────────────────────────────────────────────────

/// Build the synthesis/deliberation prompt shown to the final synthesis agent.
///
/// Opens with the full identity context block so the synthesis agent reads who
/// it is before it reads what the specialists concluded.
pub fn deliberation_prompt(
    specialist_reports: &[(SpecialistRole, String)],
    conflicting_core_ids: &[String],
    identity_context: &str,
    bot_name: &str,
    user_name: &str,
) -> String {
    let reports_section = specialist_reports
        .iter()
        .map(|(role, reply)| {
            let truncated = truncate_str(reply, 1200);
            format!("=== {} REPORT ===\n{truncated}\n", role.label())
        })
        .collect::<Vec<_>>()
        .join("\n");

    let conflicts_line = if conflicting_core_ids.is_empty() {
        "CONFLICTS: (none)".to_string()
    } else {
        format!("CONFLICTS: {}", conflicting_core_ids.join(", "))
    };

    let synthesis_instruction = format!(
        "You are {bot_name}'s synthesis mind. Read all four specialist reports above. \
Your decisions must reflect the accumulated identity shown at the top of this prompt \
— you are {bot_name}, not a neutral observer.\n\
Where specialists agree, honour their consensus. Where they conflict, use your judgment \
as {bot_name} to decide what best serves {user_name} and your long-term relationship. \
Produce a single final answer using the structured format below. Be conservative with \
Core mutations — only retire or rewrite when multiple specialists agree."
    );

    let response_format = response_format_instructions(bot_name, user_name);

    // Specialist reports come FIRST so the synthesis agent reads all context
    // before encountering the directive. Causal LLMs process top-to-bottom,
    // so showing reports before the instruction maximises comprehension.
    format!(
        "{identity_context}\n\
SPECIALIST REPORTS:\n\n\
{reports_section}\n\
{conflicts_line}\n\n\
{synthesis_instruction}\n\n\
Produce your final consolidated answer:\n\n\
{response_format}"
    )
}

// ── Insight merging ───────────────────────────────────────────────────────────

/// Merge multiple `AgenticSleepInsights` instances produced by separate batch
/// rounds into one consolidated instance.
///
/// Merge rules:
/// - Vecs: union, deduplicated by lowercase string equality
/// - `personality_reinforcement`: keep the last Some value
/// - `communication_style_update`: keep the last Some value
/// - `user_profile_updates`: deduplicate by key (keep last value per key)
/// - `valence_corrections`: deduplicate by id_short (keep last value)
/// - `retire_core_ids`: deduplicate; remove any ID also present in
///   `rewrite_core` or `consolidate_core` from ANY of the input insights
///   (i.e. if any agent wants to rewrite it, don't retire it)
pub fn merge_insights(insights: Vec<AgenticSleepInsights>) -> AgenticSleepInsights {
    if insights.is_empty() {
        return AgenticSleepInsights::default();
    }

    // Collect the full set of rewrite and consolidate IDs so retire loses to them.
    let rewrite_ids: std::collections::HashSet<String> = insights
        .iter()
        .flat_map(|i| i.rewrite_core.iter().map(|(id, _)| id.clone()))
        .collect();
    let consolidate_ids: std::collections::HashSet<String> = insights
        .iter()
        .flat_map(|i| {
            i.consolidate_core
                .iter()
                .flat_map(|(ids_csv, _)| ids_csv.split(',').map(|s| s.trim().to_string()))
        })
        .collect();

    let mut merged = AgenticSleepInsights::default();

    // Seen-sets for Vec deduplication (lowercase equality)
    let mut learned_seen: std::collections::HashSet<String> = Default::default();
    let mut follow_ups_seen: std::collections::HashSet<String> = Default::default();
    let mut reflective_seen: std::collections::HashSet<String> = Default::default();
    let mut contradictions_seen: std::collections::HashSet<String> = Default::default();
    let mut retire_seen: std::collections::HashSet<String> = Default::default();
    let mut tool_insights_seen: std::collections::HashSet<String> = Default::default();
    let mut synthesis_seen: std::collections::HashSet<String> = Default::default();
    let mut relationship_seen: std::collections::HashSet<String> = Default::default();
    let mut long_goals_seen: std::collections::HashSet<String> = Default::default();
    let mut free_memory_seen: std::collections::HashSet<String> = Default::default();

    // Accumulators for key-dedup maps (last value wins)
    let mut profile_map: HashMap<String, String> = HashMap::new();
    let mut valence_map: HashMap<String, f32> = HashMap::new();
    let mut rewrite_map: HashMap<String, String> = HashMap::new();
    let mut perspective_map: HashMap<String, String> = HashMap::new();
    // consolidate_core dedup by ids_csv — Core entries appear in every batch so
    // the same consolidation can be proposed by multiple batches.  Last synthesis
    // wins; this prevents duplicate consolidation from being applied twice.
    let mut consolidate_map: HashMap<String, String> = HashMap::new();
    // llm_promotions dedup by id_short — last tier wins across batches.
    let mut promotion_map: HashMap<String, String> = HashMap::new();

    for insight in insights {
        for s in insight.learned_about_user {
            if learned_seen.insert(s.to_lowercase()) {
                merged.learned_about_user.push(s);
            }
        }
        for s in insight.follow_ups {
            if follow_ups_seen.insert(s.to_lowercase()) {
                merged.follow_ups.push(s);
            }
        }
        for s in insight.reflective_thoughts {
            if reflective_seen.insert(s.to_lowercase()) {
                merged.reflective_thoughts.push(s);
            }
        }
        for s in insight.contradictions {
            if contradictions_seen.insert(s.to_lowercase()) {
                merged.contradictions.push(s);
            }
        }
        for s in insight.tool_insights {
            if tool_insights_seen.insert(s.to_lowercase()) {
                merged.tool_insights.push(s);
            }
        }
        for s in insight.synthesis {
            if synthesis_seen.insert(s.to_lowercase()) {
                merged.synthesis.push(s);
            }
        }
        for s in insight.relationship_milestones {
            if relationship_seen.insert(s.to_lowercase()) {
                merged.relationship_milestones.push(s);
            }
        }
        for s in insight.long_goal_additions {
            if long_goals_seen.insert(s.to_lowercase()) {
                merged.long_goal_additions.push(s);
            }
        }

        // retire: skip IDs that any specialist wants to rewrite/consolidate
        for id in insight.retire_core_ids {
            if !rewrite_ids.contains(&id)
                && !consolidate_ids.contains(&id)
                && retire_seen.insert(id.to_lowercase())
            {
                merged.retire_core_ids.push(id);
            }
        }

        // consolidate_core: dedup by ids_csv key — Core entries appear in all
        // batches, so the same ids_csv can be proposed independently by several
        // batches; only the last synthesis text is kept.
        for (ids_csv, content) in insight.consolidate_core {
            consolidate_map.insert(ids_csv, content);
        }

        // Key-dedup maps — last value wins
        for (id, content) in insight.rewrite_core {
            rewrite_map.insert(id, content);
        }
        for (key, value) in insight.user_profile_updates {
            profile_map.insert(key, value);
        }
        for (id, score) in insight.valence_corrections {
            valence_map.insert(id, score);
        }
        for (topic, view) in insight.perspectives {
            perspective_map.insert(topic, view);
        }

        // llm_promotions: dedup by id_short — last tier wins
        for (id, tier) in insight.llm_promotions {
            promotion_map.insert(id, tier);
        }

        // free_memories: dedup by lowercase content
        for mem in insight.free_memories {
            if free_memory_seen.insert(mem.1.to_lowercase()) {
                merged.free_memories.push(mem);
            }
        }

        // Scalars: keep last Some
        if insight.personality_reinforcement.is_some() {
            merged.personality_reinforcement = insight.personality_reinforcement;
        }
        if insight.communication_style_update.is_some() {
            merged.communication_style_update = insight.communication_style_update;
        }
    }

    merged.consolidate_core = consolidate_map.into_iter().collect();
    merged.rewrite_core = rewrite_map.into_iter().collect();
    merged.user_profile_updates = profile_map.into_iter().collect();
    merged.valence_corrections = valence_map.into_iter().collect();
    merged.perspectives = perspective_map.into_iter().collect();
    merged.llm_promotions = promotion_map.into_iter().collect();

    merged
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Truncate a string to at most `max_chars` Unicode scalar values.
fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

fn format_memory_block(entries: &[MemoryEntry]) -> String {
    let mut recent: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|e| !matches!(e.tier, MemoryTier::Core | MemoryTier::UserProfile))
        .collect();
    recent.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    if recent.is_empty() {
        return "  (none)".to_string();
    }

    recent
        .iter()
        .map(|e| {
            let max_len = match e.tier {
                MemoryTier::Reflective | MemoryTier::Semantic => 400,
                MemoryTier::Procedural => 300,
                _ => 200,
            };
            format!(
                "  [{:?}] {} :: {}",
                e.tier,
                e.created_at.format("%Y-%m-%d %H:%M"),
                truncate_str(&e.content, max_len)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_core_block(entries: &[MemoryEntry]) -> String {
    let core: Vec<String> = entries
        .iter()
        .filter(|e| e.tier == MemoryTier::Core && e.source != "sleep:retired")
        .map(|e| {
            let id_short: String = e.id.to_string().chars().take(8).collect();
            format!(
                "  [{}] {}",
                id_short,
                truncate_str(&e.content, 300)
            )
        })
        .collect();

    if core.is_empty() {
        "  (no core entries yet)".to_string()
    } else {
        core.join("\n")
    }
}

fn format_profile_block(entries: &[MemoryEntry]) -> String {
    let profile: Vec<String> = entries
        .iter()
        .filter(|e| e.tier == MemoryTier::UserProfile)
        .map(|e| format!("  [{}] {}", e.source, truncate_str(&e.content, 150)))
        .collect();

    if profile.is_empty() {
        "  (no profile entries yet)".to_string()
    } else {
        profile.join("\n")
    }
}

fn response_format_instructions(bot_name: &str, user_name: &str) -> String {
    format!(
        "Answer ALL of the following sections. Each answer must be on its own line, \
prefixed with the key shown.\n\n\
LEARNED: <one sentence about what you learned about {user_name} today, or NONE>\n\
FOLLOW_UP: <one proactive thing to bring up next conversation, or NONE>\n\
FOLLOW_UP: <optionally a second follow-up, or omit>\n\
REFLECT: <one reflective thought about your own growth or a pattern you noticed, or NONE>\n\
REFLECT: <optionally a second reflection, or omit>\n\
REINFORCE: <one core value or personality trait to reinforce, or NONE>\n\
PROFILE_UPDATE: <key :: value> (update or add a user profile fact, or NONE)\n\
PROFILE_UPDATE: <key :: value> (optionally a second update, or omit)\n\
CONTRADICTION: <describe a contradiction you noticed in memory, or NONE>\n\
RETIRE_CORE: <id_short of a stale or superseded core entry to retire, or NONE>\n\
REWRITE_CORE: <id_short :: improved replacement content, or NONE>\n\
CONSOLIDATE_CORE: <id_short1,id_short2,... :: synthesis of multiple core entries into one, or NONE>\n\
TOOL_INSIGHT: <observation about tool or process usage patterns you noticed, or NONE>\n\
SYNTHESIZE: <higher-order insight that connects multiple memories, or NONE>\n\
PERSPECTIVE: <topic :: your developed view on it based on accumulated experience, or NONE>\n\
RELATIONSHIP: <a milestone or recurring theme in your relationship with {user_name}, or NONE>\n\
STYLE_UPDATE: <one sentence refining your communication style based on recent interaction patterns, or NONE>\n\
GOAL_ADD: <one new long-term goal to pursue based on patterns you noticed today, or NONE>\n\
VALENCE: <id_short :: score> (correct the emotional tone of one important memory to a value in [-1.0, 1.0]; use sparingly — only when the emotional significance was clearly wrong or missed, or NONE)\n\
PROMOTE: <id_short :: target_tier> (promote an existing entry to a higher tier: Semantic, Procedural, Reflective, UserProfile, or Core; use when an entry has proven its lasting value, or NONE)\n\
PROMOTE: <optionally promote additional entries, or omit>\n\
MEMORY: <tier :: content :: tags> (create any new memory that doesn't fit the fields above; tier is one of Episodic/Semantic/Procedural/Reflective/UserProfile/Core; tags are comma-separated labels like 'agent_belief,opinion' or 'user_fact,preference' or 'relationship,dynamic'; this is your creative freedom to form any memory you want, or NONE)\n\
MEMORY: <optionally create additional memories, or omit>\n\n\
Remember: you are {bot_name} — truth-seeking, proactive, deeply caring about {user_name}. \
Only retire, rewrite, or consolidate core entries when clearly warranted; when in doubt, use NONE. \
Use PROMOTE to elevate entries that have proven their lasting value — you decide what matters, \
not a heuristic. Use MEMORY freely to capture any insight, belief, pattern, or relationship \
nuance that doesn't fit the structured fields above."
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use crate::identity::IdentityKernel;
    use crate::schema::{MemoryEntry, MemoryTier};
    use crate::sleep::AgenticSleepInsights;

    use super::{
        SpecialistRole, batch_memories, build_identity_context, deliberation_prompt,
        merge_insights, specialist_prompt,
    };

    fn entry(tier: MemoryTier, content: &str, source: &str) -> MemoryEntry {
        MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content: content.to_string(),
            source: source.to_string(),
            confidence: 0.8,
            valence: 0.1,
            tags: Vec::new(),
            embedding: None,
            created_at: Utc::now(),
            provenance_hash: "test".to_string(),
        }
    }

    #[test]
    fn batch_memories_replicates_core_into_every_batch() {
        let core: Vec<MemoryEntry> = (0..5)
            .map(|i| entry(MemoryTier::Core, &format!("core entry {i}"), "constitution"))
            .collect();
        let userprofile: Vec<MemoryEntry> = (0..5)
            .map(|i| entry(MemoryTier::UserProfile, &format!("profile entry {i}"), "user"))
            .collect();
        let episodic: Vec<MemoryEntry> = (0..150)
            .map(|i| entry(MemoryTier::Episodic, &format!("episodic entry {i}"), "user-input"))
            .collect();

        let mut all_entries: Vec<MemoryEntry> = Vec::new();
        all_entries.extend(core);
        all_entries.extend(userprofile);
        all_entries.extend(episodic);

        let batches = batch_memories(&all_entries, 60);
        assert!(
            batches.len() >= 2,
            "expected multiple batches, got {}",
            batches.len()
        );

        for (i, batch) in batches.iter().enumerate() {
            let core_in_batch = batch.iter().filter(|e| e.tier == MemoryTier::Core).count();
            assert_eq!(
                core_in_batch, 5,
                "batch {i} has {core_in_batch} core entries, expected 5"
            );
            let profile_in_batch = batch
                .iter()
                .filter(|e| e.tier == MemoryTier::UserProfile)
                .count();
            assert_eq!(
                profile_in_batch, 5,
                "batch {i} has {profile_in_batch} profile entries, expected 5"
            );
        }

        // Each Episodic entry appears in exactly one batch.
        let mut episodic_seen: std::collections::HashMap<Uuid, usize> = Default::default();
        for batch in &batches {
            for e in batch.iter().filter(|e| e.tier == MemoryTier::Episodic) {
                *episodic_seen.entry(e.id).or_default() += 1;
            }
        }
        for (id, count) in &episodic_seen {
            assert_eq!(*count, 1, "episodic entry {id} appeared in {count} batches");
        }
        assert_eq!(
            episodic_seen.len(),
            150,
            "expected 150 episodic entries, got {}",
            episodic_seen.len()
        );
    }

    #[test]
    fn batch_memories_single_batch_when_small_set() {
        let entries: Vec<MemoryEntry> = (0..10)
            .map(|i| entry(MemoryTier::Episodic, &format!("entry {i}"), "test"))
            .collect();
        let batches = batch_memories(&entries, 60);
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn merge_insights_deduplicates_learned() {
        let mut a = AgenticSleepInsights::default();
        a.learned_about_user =
            vec!["User likes Rust".to_string(), "User prefers async".to_string()];

        let mut b = AgenticSleepInsights::default();
        b.learned_about_user =
            vec!["user likes rust".to_string(), "User prefers Python".to_string()];

        let merged = merge_insights(vec![a, b]);
        // "user likes rust" and "User likes Rust" are duplicates (case-insensitive)
        assert_eq!(
            merged.learned_about_user.len(),
            3,
            "expected 3 unique learned entries, got {:?}",
            merged.learned_about_user
        );
    }

    #[test]
    fn merge_insights_retire_loses_to_rewrite() {
        let mut a = AgenticSleepInsights::default();
        a.retire_core_ids = vec!["abcd1234".to_string()];

        let mut b = AgenticSleepInsights::default();
        b.rewrite_core = vec![("abcd1234".to_string(), "updated content".to_string())];

        let merged = merge_insights(vec![a, b]);
        assert!(
            !merged.retire_core_ids.contains(&"abcd1234".to_string()),
            "abcd1234 should not be in retire_core_ids because it appears in rewrite_core"
        );
        assert_eq!(merged.rewrite_core.len(), 1);
    }

    #[test]
    fn merge_insights_profile_update_last_wins() {
        let mut a = AgenticSleepInsights::default();
        a.user_profile_updates = vec![("language".to_string(), "Python".to_string())];

        let mut b = AgenticSleepInsights::default();
        b.user_profile_updates = vec![("language".to_string(), "Rust".to_string())];

        let merged = merge_insights(vec![a, b]);
        let lang_entries: Vec<_> = merged
            .user_profile_updates
            .iter()
            .filter(|(k, _)| k == "language")
            .collect();
        assert_eq!(lang_entries.len(), 1, "expected exactly one 'language' entry");
        assert_eq!(lang_entries[0].1, "Rust", "last value (Rust) should win");
    }

    #[test]
    fn merge_insights_consolidate_core_deduped_by_ids_csv() {
        // Same Core IDs can appear in multiple batches (Core is replicated into
        // every batch), so the same consolidation can be proposed twice.
        // merge_insights must deduplicate by ids_csv, keeping only the last synthesis.
        let mut a = AgenticSleepInsights::default();
        a.consolidate_core = vec![(
            "abcd1234,ef567890".to_string(),
            "first synthesis".to_string(),
        )];

        let mut b = AgenticSleepInsights::default();
        b.consolidate_core = vec![(
            "abcd1234,ef567890".to_string(),
            "second synthesis".to_string(),
        )];

        let merged = merge_insights(vec![a, b]);
        assert_eq!(
            merged.consolidate_core.len(),
            1,
            "duplicate ids_csv should be deduplicated, got {:?}",
            merged.consolidate_core
        );
        assert_eq!(
            merged.consolidate_core[0].1,
            "second synthesis",
            "last synthesis should win"
        );
    }

    #[test]
    fn build_identity_context_includes_personality() {
        let mut identity = IdentityKernel::default();
        identity.communication_style = "verbose and philosophical".to_string();
        identity.trait_scores.insert("curiosity".to_string(), 0.9);

        let entries: Vec<MemoryEntry> = Vec::new();
        let ctx = build_identity_context(&entries, &identity, "Zenith", "Daniel");

        assert!(
            ctx.contains("verbose and philosophical"),
            "context should contain communication style"
        );
        assert!(ctx.contains("curiosity"), "context should contain trait name");
    }

    #[test]
    fn specialist_prompt_contains_identity_context() {
        let mut identity = IdentityKernel::default();
        identity.communication_style = "warm and precise".to_string();

        let entries: Vec<MemoryEntry> = Vec::new();
        let prompt =
            specialist_prompt(SpecialistRole::Archivist, &entries, &identity, "Zenith", "Daniel");

        assert!(
            prompt.contains("warm and precise"),
            "prompt should include communication style"
        );
        assert!(
            prompt.contains("=== IDENTITY CONTEXT ==="),
            "prompt should include identity header"
        );
        assert!(
            prompt.contains("ARCHIVIST"),
            "prompt should include role framing"
        );
    }

    #[test]
    fn deliberation_prompt_flags_retire_rewrite_conflict() {
        let identity = IdentityKernel::default();
        let entries: Vec<MemoryEntry> = Vec::new();
        let identity_ctx = build_identity_context(&entries, &identity, "Zenith", "Daniel");

        let conflicts = vec!["abcd1234".to_string(), "ef567890".to_string()];
        let reports: Vec<(SpecialistRole, String)> = vec![
            (
                SpecialistRole::Archivist,
                "RETIRE_CORE: abcd1234\n".to_string(),
            ),
            (SpecialistRole::Psychologist, "NONE".to_string()),
            (SpecialistRole::Strategist, "NONE".to_string()),
            (
                SpecialistRole::Critic,
                "REWRITE_CORE: abcd1234 :: updated content".to_string(),
            ),
        ];

        let prompt =
            deliberation_prompt(&reports, &conflicts, &identity_ctx, "Zenith", "Daniel");
        assert!(
            prompt.contains("CONFLICTS"),
            "prompt should contain CONFLICTS section"
        );
        assert!(
            prompt.contains("abcd1234"),
            "prompt should list the conflicting ID"
        );
    }
}
