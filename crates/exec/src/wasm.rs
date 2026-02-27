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
//! # Tool Metadata Discovery
//!
//! WASM tools can declare their spec (name, description, params) via a
//! **sidecar manifest** — a `<name>.tool.json` file next to the `.wasm`:
//!
//! ```json
//! {
//!   "name": "my_tool",
//!   "description": "Does something useful",
//!   "params": [
//!     { "name": "input", "description": "The input value", "required": true },
//!     { "name": "verbose", "description": "Enable verbose output", "required": false }
//!   ]
//! }
//! ```
//!
//! When no manifest is found, the host falls back to a built-in catalogue
//! for known tools, or a generic one-param spec for unknown tools.
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

use aigent_tools::{Tool, ToolOutput, ToolParam, ToolSpec, ToolMetadata};

// ── Sidecar manifest ────────────────────────────────────────────────────────

/// Sidecar manifest structure for `<name>.tool.json`.
///
/// Allows WASM tools to self-describe their spec without hardcoding in the host.
#[derive(Deserialize)]
struct ToolManifest {
    name: String,
    description: String,
    #[serde(default)]
    params: Vec<ManifestParam>,
    #[serde(default)]
    metadata: Option<ManifestMetadata>,
}

#[derive(Deserialize)]
struct ManifestParam {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    param_type: Option<String>,
    #[serde(default)]
    enum_values: Vec<String>,
    #[serde(default)]
    default: Option<String>,
}

#[derive(Deserialize)]
struct ManifestMetadata {
    #[serde(default)]
    security_level: Option<String>,
    #[serde(default)]
    read_only: Option<bool>,
    #[serde(default)]
    group: Option<String>,
}

/// Try to load a `<stem>.tool.json` manifest next to the `.wasm` file.
fn load_manifest(wasm_path: &Path) -> Option<ToolSpec> {
    let stem = wasm_path.file_stem()?.to_str()?;
    let manifest_path = wasm_path.with_file_name(format!("{stem}.tool.json"));

    // Also check in the crate root (for tools-src layout).
    let paths_to_try = [
        manifest_path.clone(),
        // tools-src/<crate>/tool.json
        wasm_path
            .ancestors()
            .nth(4) // from target/wasm32-wasip1/release/<name>.wasm, go up 4
            .map(|p| p.join("tool.json"))
            .unwrap_or_default(),
    ];

    for path in &paths_to_try {
        if path.is_file() {
            match fs::read_to_string(path) {
                Ok(json_str) => {
                    match serde_json::from_str::<ToolManifest>(&json_str) {
                        Ok(manifest) => {
                            debug!(?path, "wasm: loaded tool manifest");
                            let metadata = if let Some(m) = manifest.metadata {
                                use aigent_tools::SecurityLevel;
                                ToolMetadata {
                                    security_level: match m.security_level.as_deref() {
                                        Some("medium") => SecurityLevel::Medium,
                                        Some("high") => SecurityLevel::High,
                                        _ => SecurityLevel::Low,
                                    },
                                    read_only: m.read_only.unwrap_or(false),
                                    group: m.group.unwrap_or_default(),
                                    ..Default::default()
                                }
                            } else {
                                ToolMetadata::default()
                            };
                            return Some(ToolSpec {
                                name: manifest.name,
                                description: manifest.description,
                                params: manifest.params.into_iter().map(|p| {
                                    use aigent_tools::ParamType;
                                    let param_type = match p.param_type.as_deref() {
                                        Some("number") => ParamType::Number,
                                        Some("integer") => ParamType::Integer,
                                        Some("boolean") => ParamType::Boolean,
                                        Some("array") => ParamType::Array,
                                        Some("object") => ParamType::Object,
                                        _ => ParamType::String,
                                    };
                                    ToolParam {
                                        name: p.name,
                                        description: p.description,
                                        required: p.required,
                                        param_type,
                                        enum_values: p.enum_values,
                                        default: p.default,
                                    }
                                }).collect(),
                                metadata,
                            });
                        }
                        Err(e) => {
                            warn!(?e, ?path, "wasm: failed to parse tool manifest");
                        }
                    }
                }
                Err(e) => {
                    debug!(?e, ?path, "wasm: could not read manifest");
                }
            }
        }
    }

    None
}

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
    /// Workspace root passed to guest tools as a pre-opened directory.
    workspace_root: Option<std::path::PathBuf>,
}

impl WasmTool {
    /// Load and AOT-compile a `.wasm` file.
    ///
    /// Returns `None` (with a warning log) rather than propagating errors so
    /// that a single malformed guest binary does not prevent the daemon from
    /// starting.
    pub fn load(path: &Path) -> Option<Self> {
        Self::load_with_workspace(path, None)
    }

    /// Load a `.wasm` file and optionally bind a workspace root directory
    /// for filesystem access.
    pub fn load_with_workspace(path: &Path, workspace_root: Option<&Path>) -> Option<Self> {
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

        // Try sidecar manifest first, then fall back to built-in catalogue.
        let spec = load_manifest(path).unwrap_or_else(|| builtin_spec_for(&name));

        Some(Self {
            spec,
            engine,
            module,
            workspace_root: workspace_root.map(|p| p.to_path_buf()),
        })
    }

    /// Run the WASM guest synchronously on a blocking thread.
    fn run_sync(&self, args: &HashMap<String, String>) -> Result<ToolOutput> {
        let stdin_bytes = bytes::Bytes::from(serde_json::to_vec(args)?);

        // Bounded in-memory output buffers.
        let stdout_pipe = MemoryOutputPipe::new(256 * 1024);
        let stderr_pipe = MemoryOutputPipe::new(8 * 1024);

        let mut builder = WasiCtxBuilder::new();
        builder
            .stdin(MemoryInputPipe::new(stdin_bytes))
            .stdout(stdout_pipe.clone())
            .stderr(stderr_pipe.clone());

        // Pre-open the workspace directory so guest tools can access files
        // via WASI path_open.  The guest sees it as ".".
        if let Some(ref ws) = self.workspace_root {
            if ws.is_dir() {
                if let Err(e) = builder.preopened_dir(ws, ".", wasmtime_wasi::DirPerms::all(), wasmtime_wasi::FilePerms::all()) {
                    warn!(?e, ?ws, "wasm: failed to preopen workspace dir");
                }
            }
        }

        let wasi = builder.build_p1();

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
        let this_spec = self.spec.clone();
        let this_workspace = self.workspace_root.clone();
        let args_owned = args.clone();

        tokio::task::spawn_blocking(move || {
            let tool = WasmTool {
                spec: this_spec,
                engine: this_engine,
                module: this_module,
                workspace_root: this_workspace,
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
/// `workspace_root` is the agent's workspace directory, pre-opened so guest
/// tools can access files via WASI.  Pass `None` to disable filesystem access.
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
pub fn load_wasm_tools_from_dir(extensions_dir: &Path, workspace_root: Option<&Path>) -> Vec<Box<dyn Tool>> {
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
                if let Some(tool) = WasmTool::load_with_workspace(&path, workspace_root) {
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
                    if let Some(tool) = WasmTool::load_with_workspace(&path, workspace_root) {
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
                    ..Default::default()
                },
                ToolParam {
                    name: "max_bytes".to_string(),
                    description: "Maximum bytes to read (default 65536)".to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata::default(),
        },
        "write_file" => ToolSpec {
            name: "write_file".to_string(),
            description: "Write content to a file within the workspace (WASM guest).".to_string(),
            params: vec![
                ToolParam {
                    name: "path".to_string(),
                    description: "Relative path from workspace root".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "content".to_string(),
                    description: "File content to write".to_string(),
                    required: true,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata::default(),
        },
        "run_shell" => ToolSpec {
            name: "run_shell".to_string(),
            description: "Execute a shell command within the workspace (WASM guest).".to_string(),
            params: vec![
                ToolParam {
                    name: "command".to_string(),
                    description: "Shell command to execute".to_string(),
                    required: true,
                    ..Default::default()
                },
                ToolParam {
                    name: "timeout_secs".to_string(),
                    description: "Max execution time in seconds (default 30)".to_string(),
                    required: false,
                    ..Default::default()
                },
            ],
            metadata: ToolMetadata::default(),
        },
        other => ToolSpec {
            name: other.to_string(),
            description: format!("WASM guest tool: {other}"),
            params: vec![],
            metadata: ToolMetadata::default(),
        },
    }
}
