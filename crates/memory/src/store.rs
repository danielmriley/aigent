use std::collections::HashMap;
use std::collections::HashSet;

use uuid::Uuid;

use crate::schema::MemoryEntry;

#[derive(Debug, Default)]
pub struct MemoryStore {
    entries: Vec<MemoryEntry>,
    seen_ids: HashSet<String>,
    /// Maps entry UUID → index in `entries` for O(1) lookup.
    by_id: HashMap<Uuid, usize>,
}

impl MemoryStore {
    pub fn insert(&mut self, entry: MemoryEntry) -> bool {
        let entry_id = entry.id.to_string();
        if self.seen_ids.contains(&entry_id) {
            return false;
        }

        let idx = self.entries.len();
        self.by_id.insert(entry.id, idx);
        self.seen_ids.insert(entry_id);
        self.entries.push(entry);
        true
    }

    pub fn all(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// O(1) lookup of a single entry by UUID.
    pub fn get(&self, id: Uuid) -> Option<&MemoryEntry> {
        self.by_id.get(&id).and_then(|&i| self.entries.get(i))
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.seen_ids.clear();
        self.by_id.clear();
    }

    pub fn retain<F>(&mut self, mut keep: F) -> usize
    where
        F: FnMut(&MemoryEntry) -> bool,
    {
        let before = self.entries.len();
        self.entries.retain(|entry| keep(entry));
        // Rebuild both lookup structures after retain.
        self.seen_ids = self.entries.iter().map(|e| e.id.to_string()).collect();
        self.by_id = self.entries.iter().enumerate().map(|(i, e)| (e.id, i)).collect();
        before.saturating_sub(self.entries.len())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a single entry by its full UUID.
    ///
    /// Returns `true` if the entry was found and removed, `false` otherwise.
    pub fn remove(&mut self, id: Uuid) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        if self.entries.len() < before {
            self.seen_ids.remove(&id.to_string());
            self.by_id.remove(&id);
            // Remap positions for entries that shifted.
            self.by_id = self.entries.iter().enumerate().map(|(i, e)| (e.id, i)).collect();
            true
        } else {
            false
        }
    }

    /// Update the `valence` field of the first entry whose UUID string starts
    /// with `id_short` (the first N chars used as a short identifier).
    ///
    /// The value is clamped to `[-1.0, 1.0]`.  Returns `true` if an entry
    /// was found and updated.
    pub fn update_valence_by_id_short(&mut self, id_short: &str, valence: f32) -> bool {
        for entry in &mut self.entries {
            if entry.id.to_string().starts_with(id_short) {
                entry.valence = valence.clamp(-1.0, 1.0);
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use super::MemoryStore;
    use crate::schema::{MemoryEntry, MemoryTier};

    fn make_entry(tier: MemoryTier, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: Uuid::new_v4(),
            tier,
            content: content.to_string(),
            source: "test".to_string(),
            confidence: 0.8,
            valence: 0.0,
            created_at: Utc::now(),
            provenance_hash: "test-hash".to_string(),
            tags: vec![],
            embedding: None,
        }
    }

    #[test]
    fn insert_and_retrieve() {
        let mut store = MemoryStore::default();
        let entry = make_entry(MemoryTier::Episodic, "hello");
        let id = entry.id;
        assert!(store.insert(entry));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(id).unwrap().content, "hello");
    }

    #[test]
    fn insert_deduplicates_by_id() {
        let mut store = MemoryStore::default();
        let entry = make_entry(MemoryTier::Episodic, "hello");
        let dup = entry.clone();
        assert!(store.insert(entry));
        assert!(!store.insert(dup));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn remove_deletes_and_updates_index() {
        let mut store = MemoryStore::default();
        let e1 = make_entry(MemoryTier::Episodic, "first");
        let e2 = make_entry(MemoryTier::Core, "second");
        let id1 = e1.id;
        let id2 = e2.id;
        store.insert(e1);
        store.insert(e2);
        assert_eq!(store.len(), 2);
        assert!(store.remove(id1));
        assert_eq!(store.len(), 1);
        assert!(store.get(id1).is_none());
        assert_eq!(store.get(id2).unwrap().content, "second");
    }

    #[test]
    fn remove_nonexistent_returns_false() {
        let mut store = MemoryStore::default();
        assert!(!store.remove(Uuid::new_v4()));
    }

    #[test]
    fn retain_filters_and_rebuilds_index() {
        let mut store = MemoryStore::default();
        let e1 = make_entry(MemoryTier::Episodic, "ephemeral");
        let e2 = make_entry(MemoryTier::Core, "keep");
        let id2 = e2.id;
        store.insert(e1);
        store.insert(e2);
        let removed = store.retain(|e| e.tier == MemoryTier::Core);
        assert_eq!(removed, 1);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(id2).unwrap().content, "keep");
    }

    #[test]
    fn clear_empties_everything() {
        let mut store = MemoryStore::default();
        store.insert(make_entry(MemoryTier::Episodic, "a"));
        store.insert(make_entry(MemoryTier::Core, "b"));
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn update_valence_by_id_short() {
        let mut store = MemoryStore::default();
        let entry = make_entry(MemoryTier::Episodic, "test");
        let id_str = entry.id.to_string();
        let short = &id_str[..8];
        store.insert(entry);
        assert!(store.update_valence_by_id_short(short, 0.75));
        assert_eq!(store.all()[0].valence, 0.75);
    }

    #[test]
    fn update_valence_clamps_to_range() {
        let mut store = MemoryStore::default();
        let entry = make_entry(MemoryTier::Episodic, "test");
        let id_str = entry.id.to_string();
        let short = &id_str[..8];
        store.insert(entry);
        store.update_valence_by_id_short(short, 5.0);
        assert_eq!(store.all()[0].valence, 1.0);
        store.update_valence_by_id_short(short, -3.0);
        assert_eq!(store.all()[0].valence, -1.0);
    }

    #[test]
    fn update_valence_returns_false_for_missing() {
        let mut store = MemoryStore::default();
        assert!(!store.update_valence_by_id_short("nonexistent", 0.5));
    }

    #[test]
    fn get_after_multiple_removes_has_correct_indices() {
        let mut store = MemoryStore::default();
        let mut ids = Vec::new();
        for i in 0..5 {
            let e = make_entry(MemoryTier::Episodic, &format!("entry-{i}"));
            ids.push(e.id);
            store.insert(e);
        }
        store.remove(ids[1]);
        store.remove(ids[3]);
        assert_eq!(store.len(), 3);
        assert_eq!(store.get(ids[0]).unwrap().content, "entry-0");
        assert_eq!(store.get(ids[2]).unwrap().content, "entry-2");
        assert_eq!(store.get(ids[4]).unwrap().content, "entry-4");
    }
}