//! Phase-2 node state machine and confidence-signal write path.
//!
//! This module adds four capabilities to [`MemoryManager`]:
//!
//! 1. **Forwarding pointer resolution** — `resolve_forwarding` follows the
//!    redb node-registry chain from a possibly-consolidated UUID to the
//!    current canonical Active belief.
//!
//! 2. **Orphan prevention** — `has_active_supporters` checks the reverse
//!    adjacency table before any decay or archival so beliefs that other
//!    Active beliefs depend on are never silently removed.
//!
//! 3. **Node state transitions** — `decay_node` / `archive_node` remove an
//!    entry from the working store and redb retrieval index while preserving
//!    it in the append-only JSONL log.
//!
//! 4. **Confidence write path** — `record_confidence_signal` is the single
//!    correct mechanism for changing a belief's confidence.  It appends a
//!    `ConfidenceUpdateEvent` to the log (preserving the audit trail) and
//!    updates the in-memory `confidence_overrides` cache for O(1) reads.
//!    `current_confidence` reads from that cache.
//!
//! ## Complexity
//!
//! | Operation                 | Complexity |
//! |---------------------------|------------|
//! | `resolve_forwarding`      | O(log n) per hop, ≤ 6 hops (bounded by tier count) |
//! | `has_active_supporters`   | O(log n) — one redb read per `Supports` source     |
//! | `decay_node` / `archive_node` | O(n) store retain + O(log n) index remove     |
//! | `record_confidence_signal`| O(1) cache write + O(1) log append               |
//! | `current_confidence`      | O(1) cache read                                   |

use anyhow::{Result, bail};
use chrono::Utc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::event_log::MemoryLogEvent;
use crate::index::{NodeState};
use crate::schema::{ConfidenceReason, ConfidenceSource, ConfidenceUpdateEvent, EdgeKind};

use super::MemoryManager;

impl MemoryManager {
    // ── Forwarding pointer resolution ─────────────────────────────────────────

    /// Resolve `id` through any chain of forwarding pointers in the redb node
    /// registry to the current canonical Active belief.
    ///
    /// Returns `id` unchanged when:
    /// - No index is attached (no forwarding information available), or
    /// - The UUID has no node registry entry, or
    /// - The node is already Active, Decayed, or Archived (no forwarding pointer).
    ///
    /// Traversal is bounded by the number of memory tiers (≤ 6 hops, capped at 8
    /// for safety), so this is O(k · log n) where k ≤ 8 and n is the registry size.
    pub fn resolve_forwarding(&self, id: Uuid) -> Uuid {
        if let Some(idx) = &self.index {
            idx.resolve_forwarding(&id).unwrap_or(id)
        } else {
            id
        }
    }

    // ── Orphan prevention ─────────────────────────────────────────────────────

    /// Returns `true` when at least one Active belief has a `Supports` edge
    /// pointing *to* `id` in the redb reverse adjacency table.
    ///
    /// This is the orphan-prevention gate: a belief that other Active beliefs
    /// depend on must not be decayed or archived until those dependents are
    /// handled first.  O(k · log n) where k is the number of `Supports` sources.
    ///
    /// Returns `false` (safe to proceed) when no index is attached.
    fn has_active_supporters(&self, id: &Uuid) -> bool {
        let Some(idx) = &self.index else { return false };
        let supporters = match idx.reverse_edges(id, EdgeKind::Supports) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = ?e, %id, "reverse_edges lookup failed during orphan check");
                return false;
            }
        };
        supporters.iter().any(|src_id| {
            matches!(
                idx.get_node_registry(src_id),
                Ok(Some(entry)) if entry.state == NodeState::Active
            )
        })
    }

    // ── Node state transitions ────────────────────────────────────────────────

    /// Transition a belief node to `Decayed` state.
    ///
    /// Removes the entry from the working in-memory store and the redb
    /// entries/tier index so it is invisible to retrieval, then sets its state
    /// to `Decayed` in the node registry.
    ///
    /// **Orphan prevention**: errors without making any changes if any Active
    /// belief has a `Supports` edge pointing to `id`.  Handle those dependents
    /// before decaying the supported node.
    ///
    /// The entry is **never removed** from the JSONL event log — the
    /// append-only invariant is preserved in all cases.
    pub fn decay_node(&mut self, id: Uuid) -> Result<()> {
        let canonical = self.resolve_forwarding(id);

        if self.has_active_supporters(&canonical) {
            bail!(
                "decay_node: cannot decay {canonical} — it has Active Supports-edge dependents \
                 (orphan prevention check failed)"
            );
        }

        // Remove from working store.
        self.store.retain(|e| e.id != canonical);

        // Remove from redb entries/tier index and transition node state.
        if let Some(idx) = &mut self.index {
            if let Err(e) = idx.remove(&canonical) {
                warn!(error = ?e, %canonical, "index remove failed during decay_node — index may be stale");
            }
            if let Err(e) = idx.set_node_state(&canonical, NodeState::Decayed, None) {
                warn!(error = ?e, %canonical, "set_node_state(Decayed) failed");
            }
        }

        // Drop any cached confidence override — the node no longer exists.
        self.confidence_overrides.remove(&canonical);

        info!(%canonical, "belief node decayed (confidence reached zero)");
        Ok(())
    }

    /// Transition a belief node to `Archived` state.
    ///
    /// Semantically identical to [`decay_node`] except the state in the node
    /// registry is set to `Archived` rather than `Decayed`.  Archived beliefs
    /// reached the end of the episodic hot window without consolidating or
    /// losing confidence to zero — they are historical records, not errors.
    ///
    /// The same orphan-prevention check applies.
    pub fn archive_node(&mut self, id: Uuid) -> Result<()> {
        let canonical = self.resolve_forwarding(id);

        if self.has_active_supporters(&canonical) {
            bail!(
                "archive_node: cannot archive {canonical} — it has Active Supports-edge dependents \
                 (orphan prevention check failed)"
            );
        }

        self.store.retain(|e| e.id != canonical);

        if let Some(idx) = &mut self.index {
            if let Err(e) = idx.remove(&canonical) {
                warn!(error = ?e, %canonical, "index remove failed during archive_node — index may be stale");
            }
            if let Err(e) = idx.set_node_state(&canonical, NodeState::Archived, None) {
                warn!(error = ?e, %canonical, "set_node_state(Archived) failed");
            }
        }

        self.confidence_overrides.remove(&canonical);

        info!(%canonical, "belief node archived (episodic hot window expired)");
        Ok(())
    }

    // ── Confidence write path ─────────────────────────────────────────────────

    /// Append a confidence signal to the event log and update the in-memory
    /// confidence cache.
    ///
    /// This is the **only** correct mechanism for changing a belief's confidence.
    /// No `MemoryEntry.confidence` field is ever mutated — all changes are
    /// expressed as new `ConfidenceUpdateEvent` records in the append-only log.
    ///
    /// Steps:
    /// 1. Resolve `target_id` through forwarding pointers to the canonical UUID.
    /// 2. Read current confidence from the in-memory cache (O(1)).
    /// 3. Clamp `delta` so the resulting confidence stays within [0.0, 1.0].
    /// 4. Update the in-memory cache immediately (visible within the same turn).
    /// 5. Update `last_accessed_at` in the node registry.
    /// 6. Append a `ConfidenceUpdateEvent` to the event log for durability.
    ///
    /// Returns `Ok(())` when no event log is configured (no-op in tests).
    pub async fn record_confidence_signal(
        &mut self,
        target_id: Uuid,
        delta: f32,
        reason: ConfidenceReason,
        source: ConfidenceSource,
    ) -> Result<()> {
        let canonical = self.resolve_forwarding(target_id);

        // Clamp: ensure the new confidence stays within [0.0, 1.0].
        let current = self.current_confidence(canonical);
        let effective_delta = if delta > 0.0 {
            delta.min(1.0 - current)
        } else {
            delta.max(-current)
        };

        // Update in-memory cache immediately for O(1) reads within the same turn.
        let new_conf = (current + effective_delta).clamp(0.0, 1.0);
        self.confidence_overrides.insert(canonical, new_conf);

        // Write a redb confidence checkpoint so startup replay can use it as
        // a base instead of replaying all events from the beginning.
        // LSN 0 is used in Phase 3 since full event-sequence tracking is
        // deferred to Phase 4+ — this checkpoint still provides O(1) reads
        // for the current live confidence value via get_confidence_checkpoint.
        if let Some(idx) = &mut self.index {
            if let Err(e) = idx.write_confidence_checkpoint(&canonical, new_conf, 0) {
                warn!(error = ?e, %canonical, "write_confidence_checkpoint failed (non-fatal)");
            }
        }

        // Update last_accessed_at in the node registry so stale-decay skips
        // beliefs that received a signal this cycle.
        if let Some(idx) = &mut self.index {
            match idx.get_node_registry(&canonical) {
                Ok(Some(mut reg)) => {
                    reg.last_accessed_at = Utc::now();
                    if let Err(e) = idx.upsert_node_registry(&reg) {
                        warn!(error = ?e, %canonical, "failed to update last_accessed_at in node registry");
                    }
                }
                Ok(None) => {
                    // Node not in registry yet (pre-Phase-1 entry or no index rebuild) — ok.
                }
                Err(e) => {
                    warn!(error = ?e, %canonical, "node registry lookup failed during confidence signal");
                }
            }
        }

        // Append to the event log for durability and audit trail.
        if let Some(event_log) = &self.event_log {
            let cu = ConfidenceUpdateEvent {
                event_id: Uuid::new_v4(),
                occurred_at: Utc::now(),
                target_id: canonical,
                delta: effective_delta,
                reason,
                source,
            };
            event_log
                .append_log_event(&MemoryLogEvent::ConfidenceUpdate(cu))
                .await?;
        }

        Ok(())
    }

    // ── Confidence read path ──────────────────────────────────────────────────

    /// Return the current computed confidence for a belief UUID.
    ///
    /// Reads from the in-memory `confidence_overrides` cache — O(1).  If no
    /// override has been recorded (i.e. no confidence signals have arrived for
    /// this UUID since startup), falls back to the `confidence` field on the
    /// in-memory `MemoryEntry` (the initial anchor value).
    ///
    /// Returns 0.0 if the UUID is unknown in both the cache and the store.
    ///
    /// **Thread-safety note**: this method does not acquire any lock; callers
    /// must ensure exclusive access to `MemoryManager` when also calling
    /// `record_confidence_signal` (the `&mut self` requirement enforces this at
    /// compile time for single-owner callers).
    pub fn current_confidence(&self, id: Uuid) -> f32 {
        let canonical = self.resolve_forwarding(id);
        if let Some(&cached) = self.confidence_overrides.get(&canonical) {
            return cached;
        }
        self.store.get(canonical).map(|e| e.confidence).unwrap_or(0.0)
    }
}
