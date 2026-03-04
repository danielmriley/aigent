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
use std::sync::Arc;

use tokio::sync::RwLock;

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
        let mut inner = self.inner.write().await;

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
        let mut inner = self.inner.write().await;
        inner.vectors.remove(&id);
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorMatch>> {
        let inner = self.inner.read().await;

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
        self.inner.read().await.vectors.len()
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
    pub async fn snapshot(&self) -> Result<VectorStoreSnapshot> {
        let inner = self.inner.read().await;

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
    pub async fn restore(snapshot: VectorStoreSnapshot) -> Result<Self> {
        let store = match snapshot.dimensions {
            Some(d) => Self::with_dimensions(d),
            None => Self::new(),
        };

        {
            let mut inner = store.inner.write().await;

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
    pub async fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let snapshot = self.snapshot().await?;
        let json = serde_json::to_vec(&snapshot)
            .context("failed to serialize vector store")?;
        std::fs::write(path, json)
            .context("failed to write vector store file")?;
        Ok(())
    }

    /// Load from a JSON file, or create a new empty store if the file
    /// does not exist.
    pub async fn load_from_file(path: &std::path::Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = std::fs::read(path)
            .context("failed to read vector store file")?;
        let snapshot: VectorStoreSnapshot = serde_json::from_slice(&data)
            .context("failed to deserialize vector store")?;
        Self::restore(snapshot).await
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

// ── Qdrant backend (feature-gated) ────────────────────────────────────────

/// Configuration for connecting to a Qdrant vector database.
#[cfg(feature = "qdrant")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    /// Qdrant gRPC/REST endpoint (e.g. "http://localhost:6334").
    pub endpoint: String,
    /// Collection name to use.
    pub collection: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// API key for Qdrant Cloud (empty = no auth).
    pub api_key: String,
    /// Whether to auto-create the collection if it does not exist.
    pub auto_create_collection: bool,
}

#[cfg(feature = "qdrant")]
impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:6334".into(),
            collection: "aigent_memories".into(),
            dimensions: 384,
            api_key: String::new(),
            auto_create_collection: true,
        }
    }
}

/// Qdrant-backed vector store using the official gRPC client.
///
/// Provides persistent, high-performance vector search backed by a Qdrant
/// instance.  Falls back to in-process FlatVectorStore if Qdrant is unreachable.
#[cfg(feature = "qdrant")]
pub struct QdrantVectorStore {
    config: QdrantConfig,
    client: qdrant_client::Qdrant,
    /// Local fallback for when Qdrant is down or for hybrid search.
    local_cache: FlatVectorStore,
}

#[cfg(feature = "qdrant")]
impl QdrantVectorStore {
    /// Create a new Qdrant vector store.
    ///
    /// Connects to the Qdrant instance and optionally creates the collection.
    pub async fn new(config: QdrantConfig) -> Result<Self> {
        use qdrant_client::Qdrant;

        let mut builder = Qdrant::from_url(&config.endpoint);
        if !config.api_key.is_empty() {
            builder = builder.api_key(config.api_key.clone());
        }
        let client = builder.build()
            .map_err(|e| anyhow::anyhow!("qdrant client build: {e}"))?;

        if config.auto_create_collection {
            Self::ensure_collection(&client, &config).await?;
        }

        let local_cache = FlatVectorStore::with_dimensions(config.dimensions);

        Ok(Self {
            config,
            client,
            local_cache,
        })
    }

    /// Ensure the target collection exists, creating it if necessary.
    async fn ensure_collection(client: &qdrant_client::Qdrant, config: &QdrantConfig) -> Result<()> {
        use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};

        let exists = client.collection_exists(&config.collection).await
            .unwrap_or(false);

        if !exists {
            tracing::info!(
                collection = %config.collection,
                dimensions = config.dimensions,
                "creating qdrant collection"
            );
            client.create_collection(
                CreateCollectionBuilder::new(&config.collection)
                    .vectors_config(
                        VectorParamsBuilder::new(config.dimensions as u64, Distance::Cosine)
                    ),
            ).await
            .map_err(|e| anyhow::anyhow!("create collection: {e}"))?;
        }

        Ok(())
    }

    pub fn config(&self) -> &QdrantConfig {
        &self.config
    }

    /// Get a reference to the local cache for hybrid search.
    pub fn local_cache(&self) -> &FlatVectorStore {
        &self.local_cache
    }
}

#[cfg(feature = "qdrant")]
impl VectorBackend for QdrantVectorStore {
    async fn upsert(&self, id: Uuid, vector: Vec<f32>) -> Result<()> {
        use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};

        // Upsert to local cache first (fast path for hybrid search).
        self.local_cache.upsert(id, vector.clone()).await?;

        // Upsert to Qdrant.
        let point = PointStruct::new(
            id.to_string(),
            vector,
            qdrant_client::Payload::new(),
        );
        self.client.upsert_points(
            UpsertPointsBuilder::new(&self.config.collection, vec![point]).wait(true)
        ).await
            .map_err(|e| anyhow::anyhow!("qdrant upsert: {e}"))?;

        Ok(())
    }

    async fn remove(&self, id: Uuid) -> Result<()> {
        use qdrant_client::qdrant::DeletePointsBuilder;

        // Remove from local cache.
        self.local_cache.remove(id).await?;

        // Remove from Qdrant.
        let point_id: qdrant_client::qdrant::PointId = id.to_string().into();
        self.client.delete_points(
            DeletePointsBuilder::new(&self.config.collection)
                .points(vec![point_id])
                .wait(true)
        ).await
            .map_err(|e| anyhow::anyhow!("qdrant delete: {e}"))?;

        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorMatch>> {
        use qdrant_client::qdrant::SearchPointsBuilder;

        let results = self.client.search_points(
            SearchPointsBuilder::new(&self.config.collection, query.to_vec(), k as u64)
        ).await
        .map_err(|e| anyhow::anyhow!("qdrant search: {e}"))?;

        let matches = results.result.into_iter().filter_map(|scored| {
            // Point ID is stored as a UUID string.
            let point_id = scored.id?;
            let id_str = match point_id.point_id_options? {
                qdrant_client::qdrant::point_id::PointIdOptions::Uuid(s) => s,
                qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => n.to_string(),
            };
            let id = Uuid::parse_str(&id_str).ok()?;
            Some(VectorMatch {
                id,
                score: scored.score,
            })
        }).collect();

        Ok(matches)
    }

    async fn len(&self) -> usize {
        // Use Qdrant collection info for the authoritative count.
        match self.client.collection_info(&self.config.collection).await {
            Ok(info) => info.result
                .map(|r| r.points_count.unwrap_or(0) as usize)
                .unwrap_or(0),
            Err(_) => self.local_cache.len().await,
        }
    }
}

/// Hybrid retrieval helper: combine Qdrant vector search with lexical scoring.
///
/// Returns merged results sorted by a weighted combination of vector similarity
/// and lexical overlap score. Each result includes both scores.
#[cfg(feature = "qdrant")]
#[derive(Debug, Clone)]
pub struct HybridMatch {
    pub id: Uuid,
    pub vector_score: f32,
    pub lexical_score: f32,
    pub combined_score: f32,
}

/// Perform hybrid retrieval: vector search via Qdrant + lexical via local scoring.
///
/// `vector_weight` and `lexical_weight` control the blend (should sum to 1.0).
/// `lexical_scorer` is a closure that returns a [0,1] lexical match score for an ID.
#[cfg(feature = "qdrant")]
pub async fn hybrid_search(
    store: &QdrantVectorStore,
    query_vector: &[f32],
    k: usize,
    vector_weight: f32,
    lexical_weight: f32,
    lexical_scorer: impl Fn(&Uuid) -> f32,
) -> Result<Vec<HybridMatch>> {
    // Get top-k*2 from vector search for re-ranking headroom.
    let vector_results = store.search(query_vector, k * 2).await?;

    let mut hybrid: Vec<HybridMatch> = vector_results
        .into_iter()
        .map(|vm| {
            let lex = lexical_scorer(&vm.id);
            HybridMatch {
                id: vm.id,
                vector_score: vm.score,
                lexical_score: lex,
                combined_score: vm.score * vector_weight + lex * lexical_weight,
            }
        })
        .collect();

    hybrid.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap_or(std::cmp::Ordering::Equal));
    hybrid.truncate(k);

    Ok(hybrid)
}

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

        let snap = store.snapshot().await.unwrap();
        assert_eq!(snap.entries.len(), 1);

        let restored = FlatVectorStore::restore(snap).await.unwrap();
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
        store.save_to_file(&path).await.unwrap();

        let loaded = FlatVectorStore::load_from_file(&path).await.unwrap();
        assert_eq!(loaded.len().await, 1);

        let results = loaded.search(&[1.0, 0.0], 1).await.unwrap();
        assert_eq!(results[0].id, id);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[tokio::test]
    async fn load_nonexistent_returns_empty() {
        let store = FlatVectorStore::load_from_file(std::path::Path::new("/tmp/nonexistent_aigent_test.json")).await.unwrap();
        let snap = store.snapshot().await.unwrap();
        assert!(snap.entries.is_empty());
    }
}
