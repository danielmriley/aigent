//! WASM guest implementation of the `read_file` tool.
//!
//! This module is compiled to a `.wasm` binary and loaded by the aigent-runtime
//! host via wasmtime.  The host calls the exported `run` function with a JSON
//! object of tool parameters; the guest returns a JSON `{ success, output }`.
//!
//! # Building
//!
//! ```sh
//! cd extensions/tools-src
//! cargo build --release
//! # Output: target/wasm32-wasip1/release/read_file.wasm
//! cp target/wasm32-wasip1/release/read_file.wasm ../../extensions/
//! ```
//!
//! # Host contract
//!
//! The host uses the WIT interface defined in `extensions/wit/host.wit`.
//! The host provides `read-file`, `log`, `get-time-unix-ms` as imported
//! host functions; this guest exports `spec` and `run`.
//!
//! # Current implementation strategy
//!
//! Until the wasmtime host-side integration is complete this binary uses
//! a simple stdio JSON protocol so it can be tested standalone:
//!
//! ```sh
//! echo '{"path":"README.md","max_bytes":"1024"}' | \
//!     wasmtime extensions/read_file.wasm
//! ```

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

fn main() {
    // ── Read JSON params from stdin ──────────────────────────────────────────
    let stdin = io::stdin();
    let mut params_json = String::new();
    for line in stdin.lock().lines() {
        params_json.push_str(&line.expect("failed to read stdin"));
        params_json.push('\n');
    }

    let args: HashMap<String, String> =
        serde_json::from_str(params_json.trim()).unwrap_or_default();

    // ── Execute ───────────────────────────────────────────────────────────────
    let result = execute(&args);

    // ── Write JSON result to stdout ───────────────────────────────────────────
    let out = serde_json::json!({
        "success": result.0,
        "output": result.1
    });
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{}", serde_json::to_string(&out).unwrap_or_default());
}

/// Core logic — reads a workspace-relative file path and returns its contents.
///
/// Security note: the host enforces the workspace boundary BEFORE passing
/// the path to this function via the `read-file` host import.  The guest
/// itself performs an additional path sanity check for defence-in-depth.
fn execute(args: &HashMap<String, String>) -> (bool, String) {
    let path = match args.get("path") {
        Some(p) => p,
        None => return (false, "missing required param: path".to_string()),
    };

    // Reject obviously malicious paths inside the WASM sandbox.
    if path.contains("..") || path.starts_with('/') {
        return (
            false,
            format!("path rejected by guest sandbox: {path}"),
        );
    }

    let max_bytes: usize = args
        .get("max_bytes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536);

    // On wasm32-wasip1 std::fs::read_to_string calls the WASI `path_open`
    // host function, which the aigent host restricts to the workspace
    // directory via a pre-opened directory capability.
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let truncated = if content.len() > max_bytes {
                format!(
                    "{}…[truncated at {} bytes]",
                    &content[..max_bytes],
                    max_bytes
                )
            } else {
                content
            };
            (true, truncated)
        }
        Err(e) => (false, format!("read_file error: {e}")),
    }
}
