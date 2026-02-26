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
}

impl MemoryEntry {
    /// First 8 characters of the UUID, used as a compact display identifier.
    pub fn id_short(&self) -> String {
        self.id.to_string()[..8].to_string()
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
