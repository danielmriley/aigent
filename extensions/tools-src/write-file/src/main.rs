//! WASM guest implementation of the `write_file` tool.
//!
//! Uses the aigent WIT host interface (`write-file` host import) which
//! enforces the workspace boundary before touching the filesystem.
//!
//! # Building
//! ```sh
//! cd extensions/tools-src && cargo build --release
//! ```

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut params_json = String::new();
    for line in stdin.lock().lines() {
        params_json.push_str(&line.expect("failed to read stdin"));
        params_json.push('\n');
    }
    let args: HashMap<String, String> =
        serde_json::from_str(params_json.trim()).unwrap_or_default();

    let (success, output) = execute(&args);
    let out = serde_json::json!({ "success": success, "output": output });
    let _ = writeln!(io::stdout().lock(), "{}", serde_json::to_string(&out).unwrap_or_default());
}

fn execute(args: &HashMap<String, String>) -> (bool, String) {
    let path = match args.get("path") {
        Some(p) => p,
        None => return (false, "missing required param: path".to_string()),
    };
    let content = match args.get("content") {
        Some(c) => c,
        None => return (false, "missing required param: content".to_string()),
    };

    // Reject path traversal inside the WASM sandbox.
    if path.contains("..") || path.starts_with('/') {
        return (false, format!("path rejected by guest sandbox: {path}"));
    }

    // Create parent directories as needed.
    if let Some(parent) = std::path::Path::new(path).parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return (false, format!("failed to create directories: {e}"));
        }
    }

    match std::fs::write(path, content) {
        Ok(()) => (
            true,
            format!("wrote {} bytes to {path}", content.len()),
        ),
        Err(e) => (false, format!("write_file error: {e}")),
    }
}
