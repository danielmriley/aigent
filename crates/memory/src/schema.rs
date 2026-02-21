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
