use std::collections::HashSet;

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
}
