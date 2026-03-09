use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── BeliefKind ────────────────────────────────────────────────────────────────

/// Semantic category of a belief, orthogonal to [`MemoryTier`].
///
/// Tier describes *how distilled* a belief is (Episodic → Core).
/// `BeliefKind` describes *what kind of thing* the belief is about —
/// two independent axes over the same entry.
///
/// Each category has a distinct learning-rate profile, decay rate, and
/// rendering policy in the system prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BeliefKind {
    /// Facts about the current state of the world (API availability, code
    /// state, external information).  Highest decay rate; contradicted quickly
    /// by fresh tool results.
    #[default]
    Empirical,
    /// How to do things: tool invocation patterns, coding conventions,
    /// interaction styles that produce good results.  Slow to form and decay;
    /// resistant to single-event contradiction.
    Procedural,
    /// What the agent can and cannot do.  Extremely low starting confidence
    /// for *negative* self-assessments.  Positive assessments start higher.
    /// Must never reach high confidence from a single failure.
    SelfModel,
    /// Preferences, aesthetic stances, values, relational tendencies.  Never
    /// written from a single experience — always distilled from many episodic
    /// observations by the sleep pipeline.  Starting confidence 0.25.
    Opinion,
}

// ── FailureClass ──────────────────────────────────────────────────────────────

/// How permanent a tool failure is — determines which belief tier and
/// confidence delta it warrants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureClass {
    /// Timeout, malformed JSON, network error.  Very low-confidence Episodic
    /// fact that decays within days.  Never justifies a negative self-model.
    Transient,
    /// Missing API key, tool not loaded, wrong config.  Written to Semantic;
    /// revalidated on next success.
    Configuration,
    /// Genuinely impossible with the current setup.  The *only* class that
    /// may contribute to a stable negative [`BeliefKind::SelfModel`] belief,
    /// and only after multi-session confirmation.
    Architectural,
}

// ── ConfidenceReason ──────────────────────────────────────────────────────────

/// Structured reason for a [`ConfidenceUpdateEvent`] delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceReason {
    /// The same tool ran again and produced a consistent result.
    ToolConfirmation,
    /// A different source type confirmed the same claim.
    CrossSourceConfirmation,
    /// The user explicitly agreed with or affirmed the belief.
    UserConfirmation,
    /// The sleep-synthesis specialist assessed this belief as consistent with
    /// the day's observations.
    SleepAgreement,
    /// A tool ran and its result contradicts this belief.
    ToolContradiction,
    /// The user explicitly corrected or contradicted this belief.
    UserCorrection,
    /// A tool that previously failed now succeeded, contradicting a negative
    /// self-model or failure-belief.
    ToolSuccessContradiction,
    /// Nightly decay delta for a belief that received no confirmation signals
    /// since the last sleep cycle.
    StaleDecay,
    /// Dampened propagation from a connected belief's confidence change.
    /// `depth` is 1 (50% of delta) or 2 (25% of delta).
    GraphPropagation { depth: u8 },
}

// ── ConfidenceSource ──────────────────────────────────────────────────────────

/// Who or what triggered a confidence update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceSource {
    /// A tool execution result.
    Tool { name: String },
    /// One of the five nightly sleep pipeline passes (1–5).
    SleepPipeline { pass: u8 },
    /// Explicit user statement in a conversation turn.
    UserMessage,
    /// Cascade from a neighbouring belief's confidence change.
    GraphPropagation,
}

// ── EdgeKind ──────────────────────────────────────────────────────────────────

/// Directed edge type in the belief graph stored in the redb adjacency tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Source belief increases the evidential weight of the target.
    Supports,
    /// Source belief is in direct epistemic conflict with the target.
    Contradicts,
    /// Target was logically derived or synthesised from the source.
    DerivedFrom,
    /// Source is a general principle; target is a specific instance.
    Generalizes,
    /// Target replaces and obsoletes the source (post-consolidation).
    Supersedes,
    /// General semantic relationship without strict logical entailment.
    RelatesTo,
}

impl EdgeKind {
    /// Kebab-case slug used as part of the redb adjacency table key.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Supports     => "supports",
            Self::Contradicts  => "contradicts",
            Self::DerivedFrom  => "derived-from",
            Self::Generalizes  => "generalizes",
            Self::Supersedes   => "supersedes",
            Self::RelatesTo    => "relates-to",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s {
            "supports"     => Some(Self::Supports),
            "contradicts"  => Some(Self::Contradicts),
            "derived-from" => Some(Self::DerivedFrom),
            "generalizes"  => Some(Self::Generalizes),
            "supersedes"   => Some(Self::Supersedes),
            "relates-to"   => Some(Self::RelatesTo),
            _              => None,
        }
    }
}

// ── New event types ───────────────────────────────────────────────────────────

/// Appended to the event log whenever a belief's confidence changes.
///
/// Confidence is **never stored as a mutable field** on `MemoryEntry`.
/// The current confidence of any UUID is `initial_confidence` + Σ(all deltas
/// for that UUID), clamped to [0.0, 1.0].  Full learning history is auditable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceUpdateEvent {
    /// Log-internal sequence ID for this event.
    pub event_id: Uuid,
    pub occurred_at: DateTime<Utc>,
    /// UUID of the [`MemoryEntry`] whose confidence is being updated.
    pub target_id: Uuid,
    /// Signed delta to apply (positive = more confident, negative = less).
    /// Clamped at application time so the belief never exceeds [0.0, 1.0].
    pub delta: f32,
    pub reason: ConfidenceReason,
    pub source: ConfidenceSource,
}

/// Appended by the sleep pipeline when N episodic beliefs are distilled into
/// one consolidated higher-tier belief.
///
/// Source beliefs transition to `Consolidated` node state; a forwarding pointer
/// is written in the redb node registry for each `source_ids` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefConsolidatedEvent {
    pub event_id: Uuid,
    pub occurred_at: DateTime<Utc>,
    /// UUIDs of the source (absorbed) beliefs.
    pub source_ids: Vec<Uuid>,
    /// UUID of the newly created consolidated belief.
    pub canonical_id: Uuid,
    /// Which sleep pass triggered the consolidation.
    pub sleep_pass: u8,
}

/// Appended when the sleep pipeline identifies a directed relationship between
/// two beliefs.  The redb adjacency tables materialise these for O(log n)
/// traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefRelationshipEvent {
    pub event_id: Uuid,
    pub occurred_at: DateTime<Utc>,
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub edge_kind: EdgeKind,
    /// How confident the system is in this relationship itself.
    pub relationship_confidence: f32,
}

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

    /// Numeric discriminant (0–5) used in redb compact storage.
    /// Mirrors the `tier_to_u8` mapping in [`crate::index`].
    pub fn discriminant(self) -> u8 {
        match self {
            Self::Episodic    => 0,
            Self::Semantic    => 1,
            Self::Procedural  => 2,
            Self::Reflective  => 3,
            Self::UserProfile => 4,
            Self::Core        => 5,
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
    /// Stored initial confidence.  **Do not read this directly for the
    /// agent's current belief strength** — use `MemoryManager::current_confidence`
    /// which adds all `ConfidenceUpdateEvent` deltas on top of this value.
    /// This field is kept as the anchor for checkpoint + replay.
    pub confidence: f32,
    pub valence: f32,
    pub created_at: DateTime<Utc>,
    pub provenance_hash: String,
    /// Semantic category of the belief, orthogonal to tier.
    /// Defaults to `Empirical` so all pre-existing entries continue to work.
    #[serde(default)]
    pub belief_kind: BeliefKind,
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
    /// `"tool-success:<tool_name>"` — tool ran and returned data.
    ToolSuccess { tool_name: String },
    /// `"tool-failure:{transient,config,arch}:<tool_name>"` — tool ran and failed.
    ToolFailure { tool_name: String, failure_class: FailureClass },
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
            _ if s.starts_with("tool-success:") => {
                let tool_name = s.strip_prefix("tool-success:").unwrap_or("").to_string();
                Self::ToolSuccess { tool_name }
            }
            _ if s.starts_with("tool-failure:transient:") => {
                let tool_name = s.strip_prefix("tool-failure:transient:").unwrap_or("").to_string();
                Self::ToolFailure { tool_name, failure_class: FailureClass::Transient }
            }
            _ if s.starts_with("tool-failure:config:") => {
                let tool_name = s.strip_prefix("tool-failure:config:").unwrap_or("").to_string();
                Self::ToolFailure { tool_name, failure_class: FailureClass::Configuration }
            }
            _ if s.starts_with("tool-failure:arch:") => {
                let tool_name = s.strip_prefix("tool-failure:arch:").unwrap_or("").to_string();
                Self::ToolFailure { tool_name, failure_class: FailureClass::Architectural }
            }
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

    /// Returns `true` for a successful tool invocation source.
    pub fn is_tool_success(&self) -> bool {
        matches!(self, Self::ToolSuccess { .. })
    }

    /// Returns `true` for any failed tool invocation source.
    pub fn is_tool_failure(&self) -> bool {
        matches!(self, Self::ToolFailure { .. })
    }

    /// Returns the tool name if this is a `ToolSuccess` or `ToolFailure` source.
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Self::ToolSuccess { tool_name } | Self::ToolFailure { tool_name, .. } => {
                Some(tool_name.as_str())
            }
            _ => None,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── BeliefKind serde ─────────────────────────────────────────────────────

    #[test]
    fn belief_kind_serde_round_trip() {
        for kind in [BeliefKind::Empirical, BeliefKind::Procedural, BeliefKind::SelfModel, BeliefKind::Opinion] {
            let json = serde_json::to_string(&kind).unwrap();
            let decoded: BeliefKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, decoded, "round-trip failed for {kind:?}");
        }
    }

    #[test]
    fn belief_kind_default_is_empirical() {
        let kind = BeliefKind::default();
        assert_eq!(kind, BeliefKind::Empirical);
    }

    #[test]
    fn memory_entry_missing_belief_kind_defaults_to_empirical() {
        // Simulate a pre-Phase-1 JSON record that has no `belief_kind` field.
        let json = r#"{"id":"00000000-0000-0000-0000-000000000001","tier":"Episodic","content":"test","source":"test","confidence":0.5,"valence":0.0,"created_at":"2025-01-01T00:00:00Z","provenance_hash":"abc","tags":[]}"]"#;
        // Strip the trailing quote/bracket we accidentally added above — parse the raw object.
        let json = r#"{"id":"00000000-0000-0000-0000-000000000001","tier":"Episodic","content":"test","source":"test","confidence":0.5,"valence":0.0,"created_at":"2025-01-01T00:00:00Z","provenance_hash":"abc","tags":[]}
"#;
        let entry: MemoryEntry = serde_json::from_str(json.trim()).unwrap();
        assert_eq!(entry.belief_kind, BeliefKind::Empirical,
            "missing belief_kind field should default to Empirical for backward compat");
    }

    // ── ConfidenceUpdateEvent serde ──────────────────────────────────────────

    #[test]
    fn confidence_update_event_serde_round_trip() {
        use chrono::Utc;
        use uuid::Uuid;
        let event = ConfidenceUpdateEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            target_id: Uuid::new_v4(),
            delta: -0.15,
            reason: ConfidenceReason::StaleDecay,
            source: ConfidenceSource::SleepPipeline { pass: 1 },
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: ConfidenceUpdateEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event.event_id, decoded.event_id);
        assert_eq!(event.target_id, decoded.target_id);
        assert!((event.delta - decoded.delta).abs() < f32::EPSILON);
    }

    #[test]
    fn confidence_reason_graph_propagation_round_trip() {
        let reason = ConfidenceReason::GraphPropagation { depth: 2 };
        let json = serde_json::to_string(&reason).unwrap();
        let decoded: ConfidenceReason = serde_json::from_str(&json).unwrap();
        assert_eq!(reason, decoded);
    }

    // ── EdgeKind slug / composite key ────────────────────────────────────────

    #[test]
    fn edge_kind_slug_round_trip() {
        let kinds = [
            EdgeKind::Supports,
            EdgeKind::Contradicts,
            EdgeKind::DerivedFrom,
            EdgeKind::Generalizes,
            EdgeKind::Supersedes,
            EdgeKind::RelatesTo,
        ];
        for kind in kinds {
            let slug = kind.slug();
            let decoded = EdgeKind::from_slug(slug).expect("from_slug must round-trip");
            assert_eq!(kind, decoded, "slug round-trip failed for {kind:?} (slug={slug})");
        }
    }

    #[test]
    fn edge_kind_composite_key_format() {
        use uuid::Uuid;
        // The redb adjacency key format is "{uuid}:{edge_kind_slug}".
        let id = Uuid::new_v4();
        let kind = EdgeKind::Supports;
        let key = format!("{}:{}", id, kind.slug());
        assert!(key.starts_with(&id.to_string()));
        assert!(key.ends_with(":supports"));
        // Parse back: split on first ':', remainder is the slug.
        let (uuid_part, slug_part) = key.split_once(':').unwrap();
        assert_eq!(uuid_part.parse::<Uuid>().unwrap(), id);
        assert_eq!(EdgeKind::from_slug(slug_part).unwrap(), kind);
    }

    // ── SourceKind parsing for new variants ──────────────────────────────────

    #[test]
    fn source_kind_tool_success_parsing() {
        let s = SourceKind::from_source("tool-success:web_search");
        match s {
            SourceKind::ToolSuccess { tool_name } => assert_eq!(tool_name, "web_search"),
            other => panic!("expected ToolSuccess, got {other:?}"),
        }
    }

    #[test]
    fn source_kind_tool_failure_transient() {
        let s = SourceKind::from_source("tool-failure:transient:read_file");
        match s {
            SourceKind::ToolFailure { tool_name, failure_class } => {
                assert_eq!(tool_name, "read_file");
                assert_eq!(failure_class, FailureClass::Transient);
            }
            other => panic!("expected ToolFailure, got {other:?}"),
        }
    }

    #[test]
    fn source_kind_tool_failure_config() {
        let s = SourceKind::from_source("tool-failure:config:some_tool");
        match s {
            SourceKind::ToolFailure { failure_class, .. } => {
                assert_eq!(failure_class, FailureClass::Configuration);
            }
            other => panic!("expected ToolFailure, got {other:?}"),
        }
    }

    #[test]
    fn source_kind_tool_failure_architectural() {
        let s = SourceKind::from_source("tool-failure:arch:web_search");
        match s {
            SourceKind::ToolFailure { tool_name, failure_class } => {
                assert_eq!(tool_name, "web_search");
                assert_eq!(failure_class, FailureClass::Architectural);
            }
            other => panic!("expected ToolFailure, got {other:?}"),
        }
    }

    #[test]
    fn source_kind_tool_success_is_tool_success() {
        let s = SourceKind::from_source("tool-success:calc");
        assert!(s.is_tool_success());
        assert!(!s.is_tool_failure());
        assert_eq!(s.tool_name(), Some("calc"));
    }

    #[test]
    fn source_kind_tool_failure_is_tool_failure() {
        let s = SourceKind::from_source("tool-failure:transient:calc");
        assert!(s.is_tool_failure());
        assert!(!s.is_tool_success());
    }

    #[test]
    fn source_kind_existing_variants_still_parse() {
        // Existing source kinds must continue to work unchanged.
        assert_eq!(SourceKind::from_source("belief"), SourceKind::Belief);
        assert_eq!(SourceKind::from_source("sleep:distillation"), SourceKind::SleepDistillation);
        assert_eq!(SourceKind::from_source("vault-edit"), SourceKind::VaultEdit);
        let unknown = SourceKind::from_source("completely-unknown-source");
        assert!(matches!(unknown, SourceKind::Other(_)));
    }

    // ── Backward compat: old bare MemoryEntry lines ──────────────────────────

    #[test]
    fn belief_consolidated_event_serde_round_trip() {
        use chrono::Utc;
        use uuid::Uuid;
        let event = BeliefConsolidatedEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            source_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
            canonical_id: Uuid::new_v4(),
            sleep_pass: 2,
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: BeliefConsolidatedEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event.event_id, decoded.event_id);
        assert_eq!(event.source_ids.len(), decoded.source_ids.len());
        assert_eq!(event.canonical_id, decoded.canonical_id);
    }

    #[test]
    fn belief_relationship_event_serde_round_trip() {
        use chrono::Utc;
        use uuid::Uuid;
        let event = BeliefRelationshipEvent {
            event_id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            source_id: Uuid::new_v4(),
            target_id: Uuid::new_v4(),
            edge_kind: EdgeKind::Contradicts,
            relationship_confidence: 0.85,
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: BeliefRelationshipEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event.event_id, decoded.event_id);
        assert_eq!(event.edge_kind, decoded.edge_kind);
        assert!((event.relationship_confidence - decoded.relationship_confidence).abs() < f32::EPSILON);
    }
}
