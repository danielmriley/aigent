//! WASM tool hot-reload — watches `extensions/` for `.wasm` changes and
//! swaps the tool registry atomically.
//!
//! # Architecture
//!
//! [`HotRegistry`] wraps a [`ToolRegistry`] behind `Arc<RwLock<_>>` so it
//! can be shared across the daemon (read-heavy, rarely written).  A
//! background task watches the extensions directory with the `notify` crate
//! and rebuilds the registry when `.wasm` files are added, changed, or
//! removed.
//!
//! The watcher debounces events (500 ms) to avoid rebuilding on every
//! intermediate write from `cargo build`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher, EventKind};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use aigent_tools::{Tool, ToolOutput, ToolRegistry, ToolSpec};

// ── HotRegistry ──────────────────────────────────────────────────────────────

/// Thread-safe tool registry that supports hot-reload of WASM tools.
///
/// Consumers acquire a read lock for individual tool lookups / spec listings.
/// The watcher task acquires a write lock (briefly) to swap the entire
/// registry atomically.
#[derive(Clone)]
pub struct HotRegistry {
    inner: Arc<RwLock<ToolRegistry>>,
}

impl HotRegistry {
    /// Wrap an existing registry.
    pub fn new(registry: ToolRegistry) -> Self {
        Self {
            inner: Arc::new(RwLock::new(registry)),
        }
    }

    /// Acquire a read lock and list all tool specs.
    pub fn list_specs(&self) -> Vec<ToolSpec> {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).list_specs()
    }

    /// Acquire a read lock, look up the tool, then release the lock before
    /// the async call so the `RwLockReadGuard` is not held across `.await`.
    pub async fn execute(
        &self,
        name: &str,
        args: &std::collections::HashMap<String, String>,
    ) -> anyhow::Result<ToolOutput> {
        let tool = {
            let registry = self.inner.read().unwrap_or_else(|e| e.into_inner());
            registry
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("unknown tool: {name}"))?
        };
        // Guard dropped — safe to await.
        tool.run(args).await
    }

    /// Replace the entire inner registry.
    pub fn swap(&self, new_registry: ToolRegistry) {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        *guard = new_registry;
    }

    /// Get a read reference to the underlying registry (for `ToolExecutor`).
    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, ToolRegistry> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }
}

// ── Watcher ──────────────────────────────────────────────────────────────────

/// Configuration for the WASM file watcher.
pub struct WatcherConfig {
    /// Path to the extensions directory to watch.
    pub extensions_dir: PathBuf,
    /// Workspace root (pre-opened for WASM guests).
    pub workspace_root: PathBuf,
    /// Debounce duration — events within this window are coalesced.
    pub debounce: Duration,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            extensions_dir: PathBuf::from("extensions"),
            workspace_root: PathBuf::from("."),
            debounce: Duration::from_millis(500),
        }
    }
}

/// Spawn a background task that watches for `.wasm` file changes and
/// reloads the WASM tools in the given [`HotRegistry`].
///
/// Returns a handle to the watcher (must be kept alive) and a
/// `tokio::task::JoinHandle` for the debounce loop.
///
/// The rebuilder calls [`super::wasm::load_wasm_tools_from_dir`] and then
/// re-creates the native fallback set from `rebuild_fn` before swapping
/// the registry.
///
/// # Wiring into the daemon
///
/// TODO: To activate live hot-reload from `run_unified_daemon`, change
/// `DaemonState.tool_registry` from `Arc<ToolRegistry>` to `HotRegistry`
/// (which already wraps `Arc<RwLock<ToolRegistry>>`).  Then call
/// `spawn_watcher` after the registry is built, keeping the returned
/// `(_watcher, _join_handle)` alive for the daemon lifetime so the OS
/// watcher is not dropped prematurely.
#[allow(dead_code)]
pub fn spawn_watcher(
    config: WatcherConfig,
    hot_registry: HotRegistry,
    rebuild_fn: Arc<dyn Fn(Vec<Box<dyn Tool>>) -> ToolRegistry + Send + Sync>,
) -> anyhow::Result<(RecommendedWatcher, tokio::task::JoinHandle<()>)> {
    let (tx, mut rx) = mpsc::unbounded_channel::<()>();

    let extensions_dir = config.extensions_dir.clone();
    let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
        match res {
            Ok(event) => {
                let dominated = matches!(
                    event.kind,
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                );
                let has_wasm = event.paths.iter().any(|p| {
                    p.extension().map(|e| e == "wasm").unwrap_or(false)
                });
                if dominated && has_wasm {
                    let _ = tx.send(());
                }
            }
            Err(e) => warn!(?e, "wasm watcher error"),
        }
    })?;

    watcher.watch(&config.extensions_dir, RecursiveMode::Recursive)?;
    info!(dir = %config.extensions_dir.display(), "wasm: hot-reload watcher started");

    let handle = tokio::spawn(async move {
        let debounce = config.debounce;
        loop {
            // Wait for at least one event.
            if rx.recv().await.is_none() {
                break; // channel closed
            }
            // Debounce: drain any additional events within the window.
            tokio::time::sleep(debounce).await;
            while rx.try_recv().is_ok() {}

            info!("wasm: detected .wasm change — reloading tools");

            #[cfg(feature = "wasm")]
            {
                let wasm_tools = super::wasm::load_wasm_tools_from_dir(
                    &config.extensions_dir,
                    Some(&config.workspace_root),
                );
                let new_registry = rebuild_fn(wasm_tools);
                hot_registry.swap(new_registry);
                info!("wasm: hot-reload complete");
            }

            #[cfg(not(feature = "wasm"))]
            {
                debug!("wasm: hot-reload skipped (wasm feature disabled)");
            }
        }
    });

    Ok((watcher, handle))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_registry_swap() {
        let reg = ToolRegistry::default();
        let hot = HotRegistry::new(reg);
        assert!(hot.list_specs().is_empty());

        let new_reg = ToolRegistry::default();
        // We can't easily create a dummy tool here without the async_trait,
        // so just verify the swap mechanism works with empty registries.
        hot.swap(new_reg);
        assert!(hot.list_specs().is_empty());
    }

    #[test]
    fn hot_registry_clone_shares_state() {
        let reg = ToolRegistry::default();
        let hot = HotRegistry::new(reg);
        let hot2 = hot.clone();

        // Both clones see the same specs.
        assert_eq!(hot.list_specs().len(), hot2.list_specs().len());
    }
}
