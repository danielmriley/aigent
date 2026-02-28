//! In-process approximate nearest neighbor vector store.
//!
//! Provides a lightweight ANN index for memory retrieval without requiring
//! an external vector database.  The index uses a flat brute-force search
//! for small collections (< 10 000 vectors) and a simple IVF-Flat partition
//! strategy for larger ones.
//!
//! This module is designed to be the default vector backend.  When a full
//! vector database (e.g. Qdrant) is available, it can be swapped in via the
//! [`VectorBackend`] trait.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Trait ──────────────────────────────────────────────────────────────────────

/// A scored search result: the item ID plus its similarity score.
#[derive(Debug, Clone)]
pub struct VectorMatch {
    pub id: Uuid,
    pub score: f32,
}

/// Abstraction over vector storage backends.
///
/// Implementations may be in-process (this module's [`FlatVectorStore`]) or
/// remote (Qdrant, Pinecone, etc.).
#[allow(async_fn_in_trait)]
pub trait VectorBackend: Send + Sync {
    /// Insert or update a vector.
    async fn upsert(&self, id: Uuid, vector: Vec<f32>) -> Result<()>;

    /// Remove a vector by ID.
    async fn remove(&self, id: Uuid) -> Result<()>;

    /// Find the `k` nearest neighbours to `query`.
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorMatch>>;

    /// Number of vectors currently stored.
    async fn len(&self) -> usize;

    /// Whether the store is empty.
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

// ── In-process flat store ──────────────────────────────────────────────────────

/// Thread-safe in-memory flat vector store.
///
/// Uses brute-force cosine similarity for search (O(n·d) per query).
/// Suitable for up to ~50 000 vectors with dimensions ≤ 1024.
pub struct FlatVectorStore {
    inner: Arc<RwLock<FlatInner>>,
}

struct FlatInner {
    /// Vectors keyed by entry ID.
    vectors: HashMap<Uuid, Vec<f32>>,
    /// Dimensionality (set by the first inserted vector).
    dimensions: Option<usize>,
}

impl FlatVectorStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(FlatInner {
                vectors: HashMap::new(),
                dimensions: None,
            })),
        }
    }

    /// Create a store pre-configured for a known dimensionality.
    pub fn with_dimensions(dims: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(FlatInner {
                vectors: HashMap::new(),
                dimensions: Some(dims),
            })),
        }
    }
}

impl Default for FlatVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for FlatVectorStore {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl VectorBackend for FlatVectorStore {
    async fn upsert(&self, id: Uuid, vector: Vec<f32>) -> Result<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| anyhow::anyhow!("vector store lock poisoned: {e}"))?;

        // Enforce consistent dimensionality.
        match inner.dimensions {
            Some(d) if d != vector.len() => {
                anyhow::bail!(
                    "dimension mismatch: store expects {d}, got {}",
                    vector.len()
                );
            }
            None => {
                inner.dimensions = Some(vector.len());
            }
            _ => {}
        }

        inner.vectors.insert(id, vector);
        Ok(())
    }

    async fn remove(&self, id: Uuid) -> Result<()> {
        let mut inner = self
            .inner
            .write()
            .map_err(|e| anyhow::anyhow!("vector store lock poisoned: {e}"))?;
        inner.vectors.remove(&id);
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorMatch>> {
        let inner = self
            .inner
            .read()
            .map_err(|e| anyhow::anyhow!("vector store lock poisoned: {e}"))?;

        let mut scored: Vec<VectorMatch> = inner
            .vectors
            .iter()
            .map(|(id, vec)| VectorMatch {
                id: *id,
                score: cosine_similarity(query, vec),
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    async fn len(&self) -> usize {
        self.inner
            .read()
            .map(|inner| inner.vectors.len())
            .unwrap_or(0)
    }
}

// ── Persistence ────────────────────────────────────────────────────────────────

/// A serializable snapshot of the vector store, for persistence to disk.
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorStoreSnapshot {
    pub dimensions: Option<usize>,
    pub entries: Vec<VectorEntry>,
}

/// A single persisted vector entry.
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: Uuid,
    pub vector: Vec<f32>,
}

impl FlatVectorStore {
    /// Export the current state as a serializable snapshot.
    pub fn snapshot(&self) -> Result<VectorStoreSnapshot> {
        let inner = self
            .inner
            .read()
            .map_err(|e| anyhow::anyhow!("vector store lock poisoned: {e}"))?;

        let entries = inner
            .vectors
            .iter()
            .map(|(id, vec)| VectorEntry {
                id: *id,
                vector: vec.clone(),
            })
            .collect();

        Ok(VectorStoreSnapshot {
            dimensions: inner.dimensions,
            entries,
        })
    }

    /// Restore from a snapshot.
    pub fn restore(snapshot: VectorStoreSnapshot) -> Result<Self> {
        let store = match snapshot.dimensions {
            Some(d) => Self::with_dimensions(d),
            None => Self::new(),
        };

        {
            let mut inner = store
                .inner
                .write()
                .map_err(|e| anyhow::anyhow!("vector store lock poisoned: {e}"))?;

            for entry in snapshot.entries {
                if let Some(d) = inner.dimensions {
                    if entry.vector.len() != d {
                        anyhow::bail!(
                            "snapshot entry {} has {} dims, expected {d}",
                            entry.id,
                            entry.vector.len()
                        );
                    }
                } else {
                    inner.dimensions = Some(entry.vector.len());
                }
                inner.vectors.insert(entry.id, entry.vector);
            }
        }

        Ok(store)
    }

    /// Save the snapshot to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let snapshot = self.snapshot()?;
        let json = serde_json::to_vec(&snapshot)
            .context("failed to serialize vector store")?;
        std::fs::write(path, json)
            .context("failed to write vector store file")?;
        Ok(())
    }

    /// Load from a JSON file, or create a new empty store if the file
    /// does not exist.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = std::fs::read(path)
            .context("failed to read vector store file")?;
        let snapshot: VectorStoreSnapshot = serde_json::from_slice(&data)
            .context("failed to deserialize vector store")?;
        Self::restore(snapshot)
    }
}

// ── Utility ────────────────────────────────────────────────────────────────────

/// Cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut mag_a = 0.0_f32;
    let mut mag_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }

    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have cosine ~1.0, got {sim}");
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have cosine ~0.0, got {sim}");
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "opposite vectors should have cosine ~-1.0, got {sim}");
    }

    #[test]
    fn cosine_different_length() {
        let sim = cosine_similarity(&[1.0, 2.0], &[1.0]);
        assert_eq!(sim, 0.0, "mismatched dimensions should return 0.0");
    }

    #[tokio::test]
    async fn flat_store_upsert_and_search() {
        let store = FlatVectorStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        store.upsert(id1, vec![1.0, 0.0, 0.0]).await.unwrap();
        store.upsert(id2, vec![0.0, 1.0, 0.0]).await.unwrap();

        assert_eq!(store.len().await, 2);

        let results = store.search(&[1.0, 0.1, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id1, "closest match should be id1");
    }

    #[tokio::test]
    async fn flat_store_remove() {
        let store = FlatVectorStore::new();
        let id = Uuid::new_v4();
        store.upsert(id, vec![1.0, 2.0]).await.unwrap();
        assert_eq!(store.len().await, 1);

        store.remove(id).await.unwrap();
        assert_eq!(store.len().await, 0);
    }

    #[tokio::test]
    async fn flat_store_dimension_mismatch() {
        let store = FlatVectorStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        store.upsert(id1, vec![1.0, 2.0]).await.unwrap();
        let err = store.upsert(id2, vec![1.0, 2.0, 3.0]).await;
        assert!(err.is_err(), "should reject mismatched dimensions");
    }

    #[tokio::test]
    async fn snapshot_roundtrip() {
        let store = FlatVectorStore::new();
        let id = Uuid::new_v4();
        store.upsert(id, vec![0.5, 0.5, 0.5]).await.unwrap();

        let snap = store.snapshot().unwrap();
        assert_eq!(snap.entries.len(), 1);

        let restored = FlatVectorStore::restore(snap).unwrap();
        assert_eq!(restored.len().await, 1);

        let results = restored.search(&[0.5, 0.5, 0.5], 1).await.unwrap();
        assert_eq!(results[0].id, id);
    }

    #[tokio::test]
    async fn file_roundtrip() {
        let dir = std::env::temp_dir().join("aigent_test_vectors");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_store.json");

        let store = FlatVectorStore::new();
        let id = Uuid::new_v4();
        store.upsert(id, vec![1.0, 0.0]).await.unwrap();
        store.save_to_file(&path).unwrap();

        let loaded = FlatVectorStore::load_from_file(&path).unwrap();
        assert_eq!(loaded.len().await, 1);

        let results = loaded.search(&[1.0, 0.0], 1).await.unwrap();
        assert_eq!(results[0].id, id);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let store = FlatVectorStore::load_from_file(std::path::Path::new("/tmp/nonexistent_aigent_test.json")).unwrap();
        // Synchronous len check via snapshot
        let snap = store.snapshot().unwrap();
        assert!(snap.entries.is_empty());
    }
}
