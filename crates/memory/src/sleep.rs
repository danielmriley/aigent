//! Sleep-cycle consolidation.
//!
//! Two modes of operation:
//!
//! 1. **Passive distillation** (`distill`) – pure heuristic promotion of
//!    high-confidence / high-repetition Episodic → Semantic / Core entries.
//!    No LLM required.
//!
//! 2. **Agentic consolidation** – the runtime calls
//!    `agentic_sleep_prompt()` to build a rich reflection prompt, feeds it
//!    to the LLM, then calls `parse_agentic_insights()` to turn the LLM
//!    reply into structured `AgenticSleepInsights` that `MemoryManager`
//!    commits via `apply_agentic_sleep_insights()`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::schema::{MemoryEntry, MemoryTier};
use crate::scorer::{PromotionSignals, is_core_eligible};

// ── Basic promotion types ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SleepPromotion {
    pub source_id: String,
    pub to_tier: MemoryTier,
    pub reason: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct SleepSummary {
    pub distilled: String,
    pub promoted_ids: Vec<String>,
    pub promotions: Vec<SleepPromotion>,
}

// ── Passive distillation ──────────────────────────────────────────────────────

/// Heuristic-only consolidation pass.  Called synchronously; no LLM needed.
pub fn distill(entries: &[MemoryEntry]) -> SleepSummary {
    let mut normalized_count: HashMap<String, usize> = HashMap::new();
    for entry in entries {
        let key = entry.content.trim().to_lowercase();
        if key.is_empty() {
            continue;
        }
        *normalized_count.entry(key).or_default() += 1;
    }

    let mut promotions = Vec::new();

    for entry in entries {
        // Core entries are already at the top tier.
        if entry.tier == MemoryTier::Core {
            continue;
        }
        // Skip sleep bookkeeping entries — they clutter core.
        if entry.source.starts_with("sleep:cycle") {
            continue;
        }

        let repeats = normalized_count
            .get(&entry.content.trim().to_lowercase())
            .copied()
            .unwrap_or(1);
        let repetition_score = (repeats as f32 / 3.0).clamp(0.0, 1.0);

        let user_confirmed = entry.source.contains("user") || entry.source.contains("profile");

        let signals = PromotionSignals {
            repetition_score,
            emotional_salience: entry.valence.abs().clamp(0.0, 1.0),
            user_confirmed_importance: if user_confirmed { 0.85 } else { 0.4 },
            task_utility: if entry.content.len() > 40 { 0.8 } else { 0.5 },
        };

        let promoted_tier = if is_core_eligible(entry, signals) {
            MemoryTier::Core
        } else if entry.confidence >= 0.65 {
            MemoryTier::Semantic
        } else {
            continue;
        };

        debug!(
            id = %entry.id,
            tier = ?promoted_tier,
            repeats,
            "passive sleep: promoting entry"
        );

        promotions.push(SleepPromotion {
            source_id: entry.id.to_string(),
            to_tier: promoted_tier,
            reason: format!(
                "sleep-distilled repetition={repeats} confidence={:.2}",
                entry.confidence
            ),
            content: entry.content.clone(),
        });
    }

    let promoted_ids = promotions
        .iter()
        .map(|p| p.source_id.clone())
        .collect();

    let distilled = format!(
        "distilled {} memories, proposed {} promotions",
        entries.len(),
        promotions.len()
    );

    info!(
        total_entries = entries.len(),
        promotions = promotions.len(),
        "passive sleep distillation complete"
    );

    SleepSummary {
        distilled,
        promoted_ids,
        promotions,
    }
}

// ── Agentic sleep ─────────────────────────────────────────────────────────────

/// Structured output from the LLM's nightly reflection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgenticSleepInsights {
    /// What the agent learned about the user today (→ UserProfile).
    pub learned_about_user: Vec<String>,
    /// Proactive follow-up items for the next conversation (→ Reflective).
    pub follow_ups: Vec<String>,
    /// Core/personality value to reinforce (→ Core via sleep: source).
    pub personality_reinforcement: Option<String>,
    /// The agent's own reflective thoughts (→ Reflective).
    pub reflective_thoughts: Vec<String>,
    /// Contradictions discovered that should be resolved (→ Semantic).
    pub contradictions: Vec<String>,
    /// Updated user profile facts (key, value) (→ UserProfile).
    pub user_profile_updates: Vec<(String, String)>,
}

/// Build the agentic sleep reflection prompt.
///
/// The returned string should be sent to the LLM as-is.  The LLM's reply
/// should then be passed to [`parse_agentic_insights`].
pub fn agentic_sleep_prompt(
    entries: &[MemoryEntry],
    bot_name: &str,
    user_name: &str,
) -> String {
    // Collect recent episodic/semantic/reflective entries (last 40).
    let mut recent: Vec<&MemoryEntry> = entries
        .iter()
        .filter(|e| {
            matches!(
                e.tier,
                MemoryTier::Episodic | MemoryTier::Semantic | MemoryTier::Reflective
            )
        })
        .collect();
    recent.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    recent.truncate(40);

    let memory_block = recent
        .iter()
        .map(|e| {
            format!(
                "  [{:?}] {} :: {}",
                e.tier,
                e.created_at.format("%Y-%m-%d %H:%M"),
                &e.content[..e.content.len().min(200)]
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Collect user profile.
    let profile_block: Vec<String> = entries
        .iter()
        .filter(|e| e.tier == MemoryTier::UserProfile)
        .map(|e| format!("  [{}] {}", e.source, &e.content[..e.content.len().min(150)]))
        .collect();

    let profile_text = if profile_block.is_empty() {
        "  (no profile entries yet)".to_string()
    } else {
        profile_block.join("\n")
    };

    format!(
        "You are {bot_name} performing your nightly memory consolidation. \
Your human is {user_name}. Review the memories below and answer each section \
in the EXACT format shown. Be specific and concise. Use \"NONE\" when nothing applies.

RECENT MEMORIES (newest first):
{memory_block}

USER PROFILE (what you know about {user_name}):
{profile_text}

Answer ALL of the following sections. Each answer must be on its own line, \
prefixed with the key shown.

LEARNED: <one sentence about what you learned about {user_name} today, or NONE>
FOLLOW_UP: <one proactive thing to bring up next conversation, or NONE>
FOLLOW_UP: <optionally a second follow-up, or omit>
REFLECT: <one reflective thought about your own growth or a pattern you noticed, or NONE>
REFLECT: <optionally a second reflection, or omit>
REINFORCE: <one core value or personality trait to reinforce, or NONE>
PROFILE_UPDATE: <key :: value> (update or add a user profile fact, or NONE)
PROFILE_UPDATE: <key :: value> (optionally a second update, or omit)
CONTRADICTION: <describe a contradiction you noticed in memory, or NONE>

Remember: you are {bot_name} — truth-seeking, proactive, deeply caring about {user_name}."
    )
}

/// Parse the LLM's nightly reflection reply into structured insights.
///
/// The parser is intentionally lenient: unknown lines are silently ignored.
pub fn parse_agentic_insights(reply: &str) -> AgenticSleepInsights {
    let mut insights = AgenticSleepInsights::default();

    for line in reply.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = strip_key(line, "LEARNED:") {
            if !is_none(rest) {
                insights.learned_about_user.push(rest.to_string());
            }
        } else if let Some(rest) = strip_key(line, "FOLLOW_UP:") {
            if !is_none(rest) {
                insights.follow_ups.push(rest.to_string());
            }
        } else if let Some(rest) = strip_key(line, "REFLECT:") {
            if !is_none(rest) {
                insights.reflective_thoughts.push(rest.to_string());
            }
        } else if let Some(rest) = strip_key(line, "REINFORCE:") {
            if !is_none(rest) && insights.personality_reinforcement.is_none() {
                insights.personality_reinforcement = Some(rest.to_string());
            }
        } else if let Some(rest) = strip_key(line, "PROFILE_UPDATE:") {
            if !is_none(rest) {
                // Format: "key :: value"
                if let Some((k, v)) = rest.split_once("::") {
                    let key = k.trim().to_string();
                    let value = v.trim().to_string();
                    if !key.is_empty() && !value.is_empty() {
                        insights.user_profile_updates.push((key, value));
                    }
                }
            }
        } else if let Some(rest) = strip_key(line, "CONTRADICTION:") {
            if !is_none(rest) {
                insights.contradictions.push(rest.to_string());
            }
        }
    }

    info!(
        learned = insights.learned_about_user.len(),
        follow_ups = insights.follow_ups.len(),
        reflections = insights.reflective_thoughts.len(),
        profile_updates = insights.user_profile_updates.len(),
        contradictions = insights.contradictions.len(),
        "agentic sleep insights parsed"
    );

    insights
}

fn strip_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.strip_prefix(key).map(str::trim)
}

fn is_none(s: &str) -> bool {
    let low = s.trim().to_lowercase();
    low == "none" || low.is_empty()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use crate::schema::{MemoryEntry, MemoryTier};
    use crate::sleep::{distill, parse_agentic_insights};

    fn entry(tier: MemoryTier, content: &str, source: &str, conf: f32) -> MemoryEntry {
        MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content: content.to_string(),
            source: source.to_string(),
            confidence: conf,
            valence: 0.2,
            tags: Vec::new(),
            embedding: None,
            created_at: Utc::now(),
            provenance_hash: "test".to_string(),
        }
    }

    #[test]
    fn distill_proposes_promotions_for_high_confidence_entries() {
        let entries = vec![entry(
            MemoryTier::Episodic,
            "user prefers milestone check-ins for progress",
            "user-chat",
            0.85,
        )];
        let summary = distill(&entries);
        assert!(!summary.promotions.is_empty());
        assert!(summary.distilled.contains("distilled"));
    }

    #[test]
    fn distill_does_not_promote_core_entries() {
        let entries = vec![entry(
            MemoryTier::Core,
            "I am Aigent, a helpful AI",
            "constitution:personality",
            1.0,
        )];
        let summary = distill(&entries);
        assert!(summary.promotions.is_empty());
    }

    #[test]
    fn parse_agentic_insights_extracts_all_fields() {
        let reply = r#"
LEARNED: The user is working on a Rust web service and prefers async code.
FOLLOW_UP: Ask how the Rust web service deployment went.
REFLECT: I should proactively surface relevant Rust examples.
REINFORCE: Be maximally helpful and truth-seeking.
PROFILE_UPDATE: language_preference :: Rust
PROFILE_UPDATE: project :: Rust web service
CONTRADICTION: Earlier memory says user prefers Python but today they mentioned Rust.
        "#;

        let insights = parse_agentic_insights(reply);
        assert_eq!(insights.learned_about_user.len(), 1);
        assert_eq!(insights.follow_ups.len(), 1);
        assert_eq!(insights.reflective_thoughts.len(), 1);
        assert!(insights.personality_reinforcement.is_some());
        assert_eq!(insights.user_profile_updates.len(), 2);
        assert_eq!(insights.contradictions.len(), 1);
    }

    #[test]
    fn parse_agentic_insights_handles_none_entries() {
        let reply = "LEARNED: NONE\nFOLLOW_UP: none\nREFLECT: NONE\nREINFORCE: NONE\n";
        let insights = parse_agentic_insights(reply);
        assert!(insights.learned_about_user.is_empty());
        assert!(insights.follow_ups.is_empty());
        assert!(insights.personality_reinforcement.is_none());
    }
}
