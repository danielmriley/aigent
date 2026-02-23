use std::collections::HashSet;

use uuid::Uuid;

use crate::schema::MemoryEntry;

#[derive(Debug, Default)]
pub struct MemoryStore {
    entries: Vec<MemoryEntry>,
    seen_ids: HashSet<String>,
}

impl MemoryStore {
    pub fn insert(&mut self, entry: MemoryEntry) -> bool {
        let entry_id = entry.id.to_string();
        if self.seen_ids.contains(&entry_id) {
            return false;
        }

        self.seen_ids.insert(entry_id);
        self.entries.push(entry);
        true
    }

    pub fn all(&self) -> &[MemoryEntry] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.seen_ids.clear();
    }

    pub fn retain<F>(&mut self, mut keep: F) -> usize
    where
        F: FnMut(&MemoryEntry) -> bool,
    {
        let before = self.entries.len();
        self.entries.retain(|entry| keep(entry));
        self.seen_ids = self
            .entries
            .iter()
            .map(|entry| entry.id.to_string())
            .collect();
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
