use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Memory tiers from most volatile to most protected.
///
/// | Tier          | Purpose                                                     |
/// |---------------|-------------------------------------------------------------|
/// | `Episodic`    | Raw conversation turns, temporary observations              |
/// | `Semantic`    | Distilled facts, condensed knowledge                        |
/// | `Procedural`  | Learned skills, workflows, how-to knowledge                 |
/// | `Reflective`  | Agent's own thoughts, plans, self-critiques, goals          |
/// | `UserProfile` | Persistent user facts: preferences, goals, life context     |
/// | `Core`        | Immutable identity & constitution (consistency-firewalled)  |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    Episodic,
    Semantic,
    Procedural,
    Reflective,
    UserProfile,
    Core,
}

impl MemoryTier {
    /// Canonical display label used in prompts and vault exports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Episodic => "Episodic",
            Self::Semantic => "Semantic",
            Self::Procedural => "Procedural",
            Self::Reflective => "Reflective",
            Self::UserProfile => "UserProfile",
            Self::Core => "Core",
        }
    }

    /// Kebab-case slug used for file names, index keys, and log lines.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
            Self::Reflective => "reflective",
            Self::UserProfile => "user-profile",
            Self::Core => "core",
        }
    }

    /// Parse a tier from its label (case-insensitive).
    ///
    /// Accepts the canonical label names as well as common abbreviations
    /// the LLM might produce (e.g. "user_profile", "user-profile").
    pub fn from_label(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "episodic" => Some(Self::Episodic),
            "semantic" => Some(Self::Semantic),
            "procedural" => Some(Self::Procedural),
            "reflective" => Some(Self::Reflective),
            "userprofile" | "user_profile" | "user-profile" | "profile" => {
                Some(Self::UserProfile)
            }
            "core" => Some(Self::Core),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub tier: MemoryTier,
    pub content: String,
    pub source: String,
    pub confidence: f32,
    pub valence: f32,
    pub created_at: DateTime<Utc>,
    pub provenance_hash: String,
    /// Searchable tags extracted or assigned at record time.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Pre-computed embedding vector — **in-memory only, never written to JSONL**.
    /// Populated when an embedding backend is configured on [`crate::MemoryManager`].
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    /// Cached tokenized term set — **in-memory only, never written to JSONL**.
    ///
    /// Computed once at record/replay time from `content` so lexical scoring
    /// needs O(1) `contains` per entry instead of re-scanning the full string
    /// on every turn.  Empty for entries created before this field existed;
    /// those score 0.0 on lexical (correct — no false positives).
    #[serde(skip)]
    pub tokens: HashSet<String>,
}

impl MemoryEntry {
    /// First 8 characters of the UUID, used as a compact display identifier.
    pub fn id_short(&self) -> String {
        self.id.to_string()[..8].to_string()
    }

    /// Parse the raw source string into a typed [`SourceKind`].
    ///
    /// This is a zero-allocation helper for pattern-matching on source values
    /// without scattering raw string comparisons across the codebase.
    pub fn source_kind(&self) -> SourceKind {
        SourceKind::from_source(&self.source)
    }
}

// ── SourceKind ───────────────────────────────────────────────────────────────

/// Typed representation of a [`MemoryEntry::source`] string.
///
/// The on-disk event log format is unchanged — the raw `source` field is
/// still stored and serialised as a plain string.  This enum exists only for
/// in-code pattern matching; use [`MemoryEntry::source_kind()`] to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    Belief,
    BeliefRetracted(Uuid),
    OnboardingIdentity,
    SleepDistillation,
    SleepRetraction(Uuid),
    SleepPromotion,
    SleepMultiAgentCycle,
    SleepConsolidation,
    FollowUp,
    /// `"assistant-reply"` or `"assistant-reply:<suffix>"`.
    AssistantReply,
    AgentExplicit,
    AgentReasoning,
    VaultEdit,
    /// Any source starting with `"sleep:"` that is not a specific variant above.
    SleepOther(String),
    /// Any source not matched by the variants above.
    Other(String),
}

impl SourceKind {
    /// Parse a raw source string.  Never panics — unknown strings map to
    /// [`SourceKind::Other`] or [`SourceKind::SleepOther`].
    ///
    /// Named `from_source` (not `from_str`) to avoid shadowing the
    /// `std::str::FromStr` trait method, which would require a `Result` return
    /// type — unnecessary here since parsing is always successful.
    pub fn from_source(s: &str) -> Self {
        match s {
            "belief"                  => Self::Belief,
            "follow-up"               => Self::FollowUp,
            "onboarding:identity"     => Self::OnboardingIdentity,
            "agent-explicit"          => Self::AgentExplicit,
            "agent-reasoning"         => Self::AgentReasoning,
            "vault-edit"              => Self::VaultEdit,
            "sleep:distillation"      => Self::SleepDistillation,
            "sleep:promotion"         => Self::SleepPromotion,
            "sleep:multi-agent-cycle" => Self::SleepMultiAgentCycle,
            "sleep:consolidation"     => Self::SleepConsolidation,
            _ if s.starts_with("belief:retracted:") => {
                let id = s.strip_prefix("belief:retracted:")
                    .and_then(|p| p.parse().ok());
                id.map(Self::BeliefRetracted)
                    .unwrap_or_else(|| Self::Other(s.to_string()))
            }
            _ if s.starts_with("sleep:retraction:") => {
                let id = s.strip_prefix("sleep:retraction:")
                    .and_then(|p| p.parse().ok());
                id.map(Self::SleepRetraction)
                    .unwrap_or_else(|| Self::Other(s.to_string()))
            }
            _ if s.starts_with("assistant-reply") => Self::AssistantReply,
            _ if s.starts_with("sleep:")          => Self::SleepOther(s.to_string()),
            _                                      => Self::Other(s.to_string()),
        }
    }

    /// Returns `true` for any sleep-cycle source (distillation, retraction,
    /// promotion, multi-agent-cycle, consolidation, or unknown `sleep:*`).
    pub fn is_sleep(&self) -> bool {
        matches!(
            self,
            Self::SleepDistillation
                | Self::SleepRetraction(_)
                | Self::SleepPromotion
                | Self::SleepMultiAgentCycle
                | Self::SleepConsolidation
                | Self::SleepOther(_)
        )
    }
}

/// Truncate `s` to at most `max_chars` Unicode scalar values, returning a
/// sub-slice.  Shared helper used by sleep and multi-sleep prompt builders.
pub fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}
