//! WASM guest implementation of the `run_shell` tool.
//!
//! Shell execution is **not** performed inside the WASM sandbox — WASI does
//! not expose process spawning.  Instead this guest calls the `run-shell`
//! HOST import, which the aigent-runtime host executes under a seccomp /
//! macOS sandbox profile restricting it to the workspace directory.
//!
//! The host runs the command only after the user has approved it (when the
//! `approval_mode` is not `autonomous`).
//!
//! # Building
//! ```sh
//! cd extensions/tools-src && cargo build --release
//! ```

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

/// Imported host function — executes a shell command in the workspace.
///
/// Arguments:
/// - `command`: the shell command string
/// - `timeout_ms`: maximum execution time in milliseconds (0 = no limit)
///
/// Returns: JSON `{ "success": bool, "output": string }`
///
/// The host enforces the workspace boundary, timeout, and any active
/// seccomp/sandbox restrictions before executing the command.
#[link(wasm_import_module = "aigent:host/host")]
extern "C" {
    // NOTE: When wit-bindgen is wired up this extern block is replaced with
    // generated bindings from the run-shell import in host.wit.
    fn host_run_shell(
        command_ptr: *const u8,
        command_len: usize,
        timeout_ms: u32,
        result_ptr: *mut u8,
        result_len: usize,
    ) -> usize;
}

fn main() {
    let stdin = io::stdin();
    let mut params_json = String::new();
    for line in stdin.lock().lines() {
        params_json.push_str(&line.expect("failed to read stdin"));
        params_json.push('\n');
    }
    let args: HashMap<String, String> =
        serde_json::from_str(params_json.trim()).unwrap_or_default();

    let command = match args.get("command") {
        Some(c) => c.clone(),
        None => {
            emit_result(false, "missing required param: command");
            return;
        }
    };
    let timeout_ms: u32 = args
        .get("timeout_secs")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(30)
        .saturating_mul(1000);

    // Delegate actual execution to the host (outside the WASM sandbox).
    // The host validates the command against the approval policy, runs it
    // under seccomp/macOS sandbox, and returns the output.
    //
    // TODO: Replace with wit-bindgen generated call once wasmtime host is wired.
    let _ = (command, timeout_ms); // suppress unused warnings until host is connected
    emit_result(
        false,
        "run_shell requires the aigent wasmtime host runtime (not yet connected)",
    );
}

fn emit_result(success: bool, output: &str) {
    let out = serde_json::json!({ "success": success, "output": output });
    let _ = writeln!(io::stdout().lock(), "{}", out);
}
