use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTier {
    Episodic,
    Semantic,
    Procedural,
    Core,
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
}
