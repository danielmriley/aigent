pub mod consistency;
pub mod constitution;
pub mod event_log;
pub mod identity;
pub mod index;
pub mod manager;
pub mod multi_sleep;
pub mod profile;
pub mod retrieval;
pub mod schema;
pub mod scorer;
pub mod sentiment;
pub mod sleep;
pub mod store;
pub mod vector_store;
pub mod vault;

pub use identity::IdentityKernel;
pub use event_log::{MemoryLogEvent, MemoryRecordEvent};
pub use index::{
    ConfidenceCheckpoint, IndexCacheStats, IndexedEntry, MemoryIndex, NodeRegistryEntry,
    NodeState,
};
pub use manager::{ConsolidationFn, EmbedFn, MemoryManager, MemoryStats, BeliefNode};
pub use multi_sleep::{SpecialistRole, batch_memories, merge_insights};
pub use schema::{
    BeliefConsolidatedEvent, BeliefKind, BeliefRelationshipEvent, ConfidenceReason,
    ConfidenceSource, ConfidenceUpdateEvent, EdgeKind, FailureClass, MemoryEntry, MemoryTier,
    SourceKind, truncate_str,
};
pub use sleep::{AgenticSleepInsights, SleepSummary, parse_agentic_insights};
pub use vector_store::{FlatVectorStore, VectorBackend, VectorMatch, cosine_similarity};
pub use vault::{
    KV_CORE, KV_REFLECTIVE, KV_TIER_LIMIT, KV_USER_PROFILE, NARRATIVE_MD, WATCHED_SUMMARIES,
    VaultEditEvent, VaultExportSummary, VaultFileStatus,
    check_vault_checksums, read_kv_for_injection, spawn_vault_watcher, sync_kv_summaries,
};
