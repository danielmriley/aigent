//! Generic LRU cache with TTL for tool results.

use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lru::LruCache;

use crate::ToolOutput;

/// A thread-safe LRU cache with per-entry TTL expiration.
pub struct TtlCache {
    inner: Mutex<LruCache<String, CacheEntry>>,
    ttl: Duration,
}

struct CacheEntry {
    output: ToolOutput,
    inserted: Instant,
}

impl TtlCache {
    /// Create a new cache with the given capacity and TTL.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap()),
            )),
            ttl,
        }
    }

    /// Look up a cached entry. Returns `None` if missing or expired.
    pub fn get(&self, key: &str) -> Option<ToolOutput> {
        let mut cache = self.inner.lock().ok()?;
        // Peek first to check expiry without promoting
        let expired = cache
            .peek(key)
            .map(|e| e.inserted.elapsed() >= self.ttl)
            .unwrap_or(false);
        if expired {
            cache.pop(key);
            return None;
        }
        cache.get(key).map(|e| e.output.clone())
    }

    /// Insert or update a cache entry.
    pub fn insert(&self, key: String, output: ToolOutput) {
        if let Ok(mut cache) = self.inner.lock() {
            cache.put(
                key,
                CacheEntry {
                    output,
                    inserted: Instant::now(),
                },
            );
        }
    }
}
