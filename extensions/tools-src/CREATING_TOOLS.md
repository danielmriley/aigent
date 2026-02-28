# Creating a New WASM Tool

This guide shows how to create a new WASM guest tool for the aigent runtime.

## Quick Start

Use the scaffolding script:

```bash
cd extensions/tools-src
./new-tool.sh my-tool "What this tool does in one sentence"
```

This creates the directory, Cargo.toml, tool.json, and src/main.rs with all
boilerplate pre-filled. It also adds the tool to the workspace members list.

## Manual Setup

If you prefer to set things up by hand:

### 1. Create the crate

```bash
cd extensions/tools-src
cargo init --name my-tool my-tool
```

### 2. Add to workspace

Edit `extensions/tools-src/Cargo.toml`:
```toml
members = [
    "read-file",
    "write-file",
    "run-shell",
    "my-tool",
]
```

### 3. Create a manifest — `my-tool/tool.json`

```json
{
  "name": "my_tool",
  "description": "What this tool does in one sentence",
  "params": [
    {
      "name": "input",
      "description": "The input value",
      "required": true,
      "param_type": "string"
    },
    {
      "name": "format",
      "description": "Output format",
      "required": false,
      "param_type": "string",
      "enum_values": ["text", "json", "markdown"],
      "default": "text"
    },
    {
      "name": "verbose",
      "description": "Enable verbose output",
      "required": false,
      "param_type": "boolean",
      "default": "false"
    }
  ],
  "metadata": {
    "security_level": "low",
    "read_only": true,
    "group": "custom"
  }
}
```

### 4. Write the tool — `my-tool/src/main.rs`

```rust
//! WASM guest tool: my_tool
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

fn main() {
    let mut input = String::new();
    for line in io::stdin().lock().lines() {
        match line {
            Ok(l) => input.push_str(&l),
            Err(_) => break,
        }
    }
    let args: HashMap<String, String> =
        serde_json::from_str(&input).unwrap_or_default();
    let (success, output) = execute(&args);
    let result = serde_json::json!({ "success": success, "output": output });
    let _ = io::stdout().write_all(result.to_string().as_bytes());
}

fn execute(args: &HashMap<String, String>) -> (bool, String) {
    let input = match args.get("input") {
        Some(v) => v,
        None => return (false, "missing required param: input".to_string()),
    };
    let result = format!("Processed: {input}");
    (true, result)
}
```

### 5. Set up Cargo.toml

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

### 6. Build

```bash
./build.sh my-tool          # Build one tool
./build.sh                  # Build all tools
./build.sh --watch          # Watch mode (auto-rebuild on changes)
./build.sh --watch my-tool  # Watch a specific tool
```

### 7. Test

Restart the aigent daemon — it auto-discovers WASM tools on startup.

## Protocol

WASM tools communicate with the host via **stdio JSON**:

- **stdin**: `{ "param_name": "value", ... }` — tool arguments
- **stdout**: `{ "success": true, "output": "result text" }` — tool result

## tool.json Reference

The manifest is validated at load time. Invalid fields emit warnings but the
tool still loads (lenient validation).

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Tool name (no spaces, use underscores) |
| `description` | string | **yes** | One-sentence description shown to the LLM |
| `params` | array | no | Parameter definitions |
| `metadata` | object | no | Security and grouping metadata |

### Param fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **yes** | Parameter name |
| `description` | string | no | Description shown to the LLM |
| `required` | bool | no | Whether the parameter is required (default: false) |
| `param_type` | string | no | `"string"` (default), `"number"`, `"integer"`, `"boolean"`, `"array"`, `"object"` |
| `enum_values` | string[] | no | Restrict values to this set |
| `default` | string | no | Default value (as string, coerced by host) |

### Metadata fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `security_level` | string | `"low"` | `"low"`, `"medium"`, or `"high"` — affects approval flow |
| `read_only` | bool | `false` | True if the tool never mutates external state |
| `group` | string | `""` | Logical group for `@group` allow/deny lists |

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
- Use `./new-tool.sh` for the fastest way to get started
- Use `./build.sh --watch` during development for instant feedback
