//! WASM tool runtime — loads `.wasm` guest tools via Wasmtime + WASIP1.
//!
//! # Architecture
//! Each WASM tool communicates with the host via **stdio JSON**:
//!
//! * **stdin**  → JSON object of `{ "param_name": "value", … }` (tool args).
//! * **stdout** → JSON object of `{ "success": bool, "output": "…" }`.
//!
//! The host instantiates the module fresh for every tool call (stateless /
//! one-shot execution), feeds it stdin, captures stdout, and deserialises the
//! result.  No persistent WASM state is retained between calls.
//!
//! # Discovery (`load_wasm_tools_from_dir`)
//! Scans two layouts:
//!
//! 1. **Direct**: `<extensions_dir>/<name>.wasm`
//! 2. **Sub-workspace** (produced by `extensions/tools-src/`):
//!    `<extensions_dir>/tools-src/<crate>/target/wasm32-wasip1/release/<name>.wasm`
//!
//! Both layouts are tried on every daemon start; missing dirs are silently
//! skipped so the daemon works without any WASM tools present.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use tracing::{debug, warn};
use wasmtime::{Config, Engine, Linker, Module, Store};
use wasmtime_wasi::preview1::{WasiP1Ctx, add_to_linker_sync};
use wasmtime_wasi::pipe::{MemoryInputPipe, MemoryOutputPipe};
use wasmtime_wasi::WasiCtxBuilder;

use aigent_tools::{Tool, ToolOutput, ToolParam, ToolSpec};

// ── Internal store state ────────────────────────────────────────────────────

struct State {
    wasi: WasiP1Ctx,
}

// ── WasmTool ────────────────────────────────────────────────────────────────

/// A [`Tool`] implementation backed by a compiled `.wasm` binary.
///
/// The WASM guest is compiled once at load time and re-instantiated per call.
/// This trades a small per-call overhead for simplicity and full isolation:
/// each invocation gets a fresh linear memory and its own I/O pipes.
pub struct WasmTool {
    spec: ToolSpec,
    engine: Engine,
    module: Module,
}

impl WasmTool {
    /// Load and AOT-compile a `.wasm` file.
    ///
    /// Returns `None` (with a warning log) rather than propagating errors so
    /// that a single malformed guest binary does not prevent the daemon from
    /// starting.
    pub fn load(path: &Path) -> Option<Self> {
        let wasm_bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                warn!(?e, ?path, "wasm: failed to read file");
                return None;
            }
        };

        let mut config = Config::new();
        // Synchronous execution — we will block_in_place on the async side.
        config.async_support(false);

        let engine = match Engine::new(&config) {
            Ok(e) => e,
            Err(e) => {
                warn!(?e, "wasm: failed to create Wasmtime Engine");
                return None;
            }
        };

        let module = match Module::new(&engine, &wasm_bytes) {
            Ok(m) => m,
            Err(e) => {
                warn!(?e, ?path, "wasm: failed to compile module");
                return None;
            }
        };

        // Derive tool name from file stem: `read_file.wasm` → `read_file`.
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let spec = builtin_spec_for(&name);
        Some(Self { spec, engine, module })
    }

    /// Run the WASM guest synchronously on a blocking thread.
    fn run_sync(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let stdin_bytes = bytes::Bytes::from(serde_json::to_vec(args)?);

        // Bounded in-memory output buffers.
        let stdout_pipe = MemoryOutputPipe::new(64 * 1024);
        let stderr_pipe = MemoryOutputPipe::new(4 * 1024);

        let wasi = WasiCtxBuilder::new()
            .stdin(MemoryInputPipe::new(stdin_bytes))
            .stdout(stdout_pipe.clone())
            .stderr(stderr_pipe.clone())
            .build_p1();

        let mut store = Store::new(&self.engine, State { wasi });

        let mut linker: Linker<State> = Linker::new(&self.engine);
        add_to_linker_sync(&mut linker, |s: &mut State| &mut s.wasi)?;

        let instance = linker.instantiate(&mut store, &self.module)?;

        // `_start` is the WASIP1 entry point (equivalent to `main`).
        let start = instance.get_typed_func::<(), ()>(&mut store, "_start")?;

        // proc_exit(0) shows up as a `Trap`; we treat it as clean exit.
        let _ = start.call(&mut store, ());

        drop(store); // release the WasiP1Ctx before consuming the pipes

        let stdout_contents = stdout_pipe.contents();

        #[derive(Deserialize)]
        struct GuestOutput {
            success: bool,
            output: String,
        }

        match serde_json::from_slice::<GuestOutput>(&stdout_contents) {
            Ok(out) => Ok(ToolOutput {
                success: out.success,
                output: out.output,
            }),
            Err(parse_err) => {
                let raw = String::from_utf8_lossy(&stdout_contents);
                // Also surface any stderr for diagnostics.
                let stderr_contents = stderr_pipe.contents();
                let stderr_raw = String::from_utf8_lossy(&stderr_contents);
                warn!(
                    tool = %self.spec.name,
                    ?parse_err,
                    raw = %raw,
                    stderr = %stderr_raw,
                    "wasm: guest produced non-JSON stdout"
                );
                Ok(ToolOutput {
                    success: false,
                    output: format!(
                        "WASM tool '{}' returned unexpected output: {}",
                        self.spec.name,
                        if raw.is_empty() { "(empty)" } else { &raw }
                    ),
                })
            }
        }
    }
}

#[async_trait]
impl Tool for WasmTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        // Wasmtime's synchronous API must not run on a Tokio async thread —
        // use `spawn_blocking` to avoid starving the executor.
        let this_engine = self.engine.clone();
        let this_module = self.module.clone();
        let this_name = self.spec.name.clone();
        let this_params = self.spec.params.clone();
        let this_desc = self.spec.description.clone();
        let args_owned = args.clone();

        tokio::task::spawn_blocking(move || {
            let tool = WasmTool {
                spec: ToolSpec {
                    name: this_name,
                    description: this_desc,
                    params: this_params,
                },
                engine: this_engine,
                module: this_module,
            };
            tool.run_sync(&args_owned)
        })
        .await
        .map_err(|e| anyhow::anyhow!("wasm blocking task panicked: {e}"))?
    }
}

// ── Tool discovery ──────────────────────────────────────────────────────────

/// Discover and load WASM tool binaries from the given `extensions_dir`.
///
/// The returned `Vec` may be empty if no `.wasm` files are found or if no
/// guest tools have been built yet.  The caller should register these tools on
/// top of the native baseline registry so that discovered WASM tools shadow
/// their native counterparts by name.
///
/// # Layouts searched
///
/// **Direct** — `<extensions_dir>/<name>.wasm`
///
/// **Sub-workspace** — produced by running `cargo build --release` in
/// `extensions/tools-src/`:
/// ```text
/// <extensions_dir>/tools-src/<crate-dir>/target/wasm32-wasip1/release/<name>.wasm
/// ```
pub fn load_wasm_tools_from_dir(extensions_dir: &Path) -> Vec<Box<dyn Tool>> {
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();

    if !extensions_dir.is_dir() {
        debug!(
            ?extensions_dir,
            "wasm: extensions dir absent — skipping WASM tool discovery"
        );
        return tools;
    }

    // ── Layout 1: direct .wasm files in extensions/ ──────────────────────
    if let Ok(entries) = fs::read_dir(extensions_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "wasm").unwrap_or(false) {
                debug!(?path, "wasm: loading tool (direct layout)");
                if let Some(tool) = WasmTool::load(&path) {
                    tools.push(Box::new(tool));
                }
            }
        }
    }

    // ── Layout 2: tools-src sub-workspace release builds ─────────────────
    let tools_src = extensions_dir.join("tools-src");
    if tools_src.is_dir() {
        let Ok(crate_dirs) = fs::read_dir(&tools_src) else {
            return tools;
        };

        for crate_entry in crate_dirs.flatten() {
            let crate_dir = crate_entry.path();
            if !crate_dir.is_dir() {
                continue;
            }

            let release_dir = crate_dir
                .join("target")
                .join("wasm32-wasip1")
                .join("release");

            if !release_dir.is_dir() {
                continue;
            }

            let Ok(wasm_entries) = fs::read_dir(&release_dir) else {
                continue;
            };

            for wasm_entry in wasm_entries.flatten() {
                let path = wasm_entry.path();

                // Skip non-.wasm and rlib/rmeta artifacts.
                let is_wasm = path.extension().map(|e| e == "wasm").unwrap_or(false);
                let is_artifact = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with('.'))
                    .unwrap_or(false);

                if is_wasm && !is_artifact {
                    debug!(?path, "wasm: loading tool (tools-src layout)");
                    if let Some(tool) = WasmTool::load(&path) {
                        tools.push(Box::new(tool));
                    }
                }
            }
        }
    }

    if tools.is_empty() {
        debug!(
            ?extensions_dir,
            "wasm: no .wasm tool binaries found — using native tool baseline"
        );
    } else {
        tracing::info!(count = tools.len(), "wasm: loaded WASM tools (shadow native baseline)");
    }

    tools
}

// ── Built-in spec catalogue ─────────────────────────────────────────────────
// When a WASM binary doesn't embed its own spec, we derive it from the tool
// name.  This covers the canonical guest tools in extensions/tools-src/.

fn builtin_spec_for(tool_name: &str) -> ToolSpec {
    match tool_name {
        "read_file" => ToolSpec {
            name: "read_file".to_string(),
            description: "Read the contents of a file within the workspace (WASM guest).".to_string(),
            params: vec![
                ToolParam {
                    name: "path".to_string(),
                    description: "Relative path from workspace root".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "max_bytes".to_string(),
                    description: "Maximum bytes to read (default 65536)".to_string(),
                    required: false,
                },
            ],
        },
        "write_file" => ToolSpec {
            name: "write_file".to_string(),
            description: "Write content to a file within the workspace (WASM guest).".to_string(),
            params: vec![
                ToolParam {
                    name: "path".to_string(),
                    description: "Relative path from workspace root".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "content".to_string(),
                    description: "File content to write".to_string(),
                    required: true,
                },
            ],
        },
        "run_shell" => ToolSpec {
            name: "run_shell".to_string(),
            description: "Execute a shell command within the workspace (WASM guest).".to_string(),
            params: vec![
                ToolParam {
                    name: "command".to_string(),
                    description: "Shell command to execute".to_string(),
                    required: true,
                },
                ToolParam {
                    name: "timeout_secs".to_string(),
                    description: "Max execution time in seconds (default 30)".to_string(),
                    required: false,
                },
            ],
        },
        other => ToolSpec {
            name: other.to_string(),
            description: format!("WASM guest tool: {other}"),
            params: vec![],
        },
    }
}
