# Creating a New WASM Tool

This guide shows how to create a new WASM guest tool for the aigent runtime.

## Quick Start

1. **Create the crate:**

```bash
cd extensions/tools-src
cargo init --name my-tool my-tool
```

2. **Add it to the workspace:**

Edit `extensions/tools-src/Cargo.toml`:
```toml
members = [
    "read-file",
    "write-file",
    "run-shell",
    "my-tool",  # ← add this
]
```

3. **Create a manifest** — `my-tool/tool.json`:

```json
{
  "name": "my_tool",
  "description": "What this tool does in one sentence",
  "params": [
    { "name": "input", "description": "The input value", "required": true },
    { "name": "verbose", "description": "Enable verbose output", "required": false }
  ]
}
```

4. **Write the tool** — `my-tool/src/main.rs`:

```rust
//! WASM guest tool: my_tool
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

fn main() {
    // ── Read arguments from stdin (JSON object) ──
    let mut input = String::new();
    for line in io::stdin().lock().lines() {
        match line {
            Ok(l) => input.push_str(&l),
            Err(_) => break,
        }
    }

    let args: HashMap<String, String> =
        serde_json::from_str(&input).unwrap_or_default();

    // ── Execute tool logic ──
    let (success, output) = execute(&args);

    // ── Write result to stdout (JSON object) ──
    let result = serde_json::json!({
        "success": success,
        "output": output,
    });
    let _ = io::stdout().write_all(result.to_string().as_bytes());
}

fn execute(args: &HashMap<String, String>) -> (bool, String) {
    let input = match args.get("input") {
        Some(v) => v,
        None => return (false, "missing required param: input".to_string()),
    };

    // Your tool logic here:
    let result = format!("Processed: {input}");

    (true, result)
}
```

5. **Set up Cargo.toml** — `my-tool/Cargo.toml`:

```toml
[package]
name = "my-tool"
version.workspace = true
edition.workspace = true

[[bin]]
name = "my_tool"
path = "src/main.rs"

[dependencies]
serde_json = { workspace = true }
```

6. **Build:**

```bash
# Build just your tool:
./build.sh my-tool

# Or build everything:
./build.sh
```

7. **Test:** Restart the aigent daemon — it auto-discovers WASM tools on startup.

## Protocol

WASM tools communicate with the host via **stdio JSON**:

- **stdin**: `{ "param_name": "value", ... }` — tool arguments
- **stdout**: `{ "success": true, "output": "result text" }` — tool result

## Filesystem Access

Guest tools can read/write files in the workspace directory via standard `std::fs`
operations. The host pre-opens the workspace as the current directory (`.`).

**Important:** Always use relative paths. Absolute paths and `..` traversal
should be rejected for security.

## Available Host Features (WASIP1)

- Standard I/O (stdin, stdout, stderr)
- Filesystem (workspace directory only)
- Clock (monotonic, wall)
- Random number generation

## Notes

- Tools are stateless — each invocation gets a fresh WASM instance
- stdout is buffered (256 KB max) — keep output concise
- The `tool.json` manifest is optional but strongly recommended
- Without a manifest, the host falls back to a generic spec
