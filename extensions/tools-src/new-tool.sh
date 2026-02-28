#!/usr/bin/env bash
# Scaffold a new WASM guest tool for the aigent runtime.
#
# Usage:
#   ./new-tool.sh my-tool                     # create with default settings
#   ./new-tool.sh my-tool "Tool description"  # create with custom description
#
# Creates:
#   <name>/
#     Cargo.toml
#     tool.json
#     src/main.rs
#
# Then adds the tool to the workspace Cargo.toml members list.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <tool-name> [description]"
    echo ""
    echo "Example:"
    echo "  $0 summarize-text \"Summarize a block of text into key points\""
    exit 1
fi

TOOL_DIR="$1"
TOOL_NAME="${TOOL_DIR//-/_}"  # Rust binary name: hyphens → underscores
DESCRIPTION="${2:-A custom WASM guest tool for aigent}"

if [[ -d "$TOOL_DIR" ]]; then
    echo "Error: directory '$TOOL_DIR' already exists"
    exit 1
fi

echo "▸ Creating tool '$TOOL_NAME' in $TOOL_DIR/..."

# ── Create directory structure ────────────────────────────────────────────────
mkdir -p "$TOOL_DIR/src"

# ── Cargo.toml ────────────────────────────────────────────────────────────────
cat > "$TOOL_DIR/Cargo.toml" << EOF
[package]
name = "$TOOL_DIR"
version.workspace = true
edition.workspace = true

[[bin]]
name = "$TOOL_NAME"
path = "src/main.rs"

[dependencies]
serde_json = { workspace = true }
EOF

# ── tool.json manifest ───────────────────────────────────────────────────────
cat > "$TOOL_DIR/tool.json" << EOF
{
  "name": "$TOOL_NAME",
  "description": "$DESCRIPTION",
  "params": [
    {
      "name": "input",
      "description": "The primary input value",
      "required": true,
      "param_type": "string"
    }
  ],
  "metadata": {
    "security_level": "low",
    "read_only": true,
    "group": "custom"
  }
}
EOF

# ── src/main.rs ───────────────────────────────────────────────────────────────
cat > "$TOOL_DIR/src/main.rs" << 'RUSTEOF'
//! WASM guest tool — communicates with the aigent host via stdio JSON.
//!
//! Protocol:
//!   stdin  → { "param_name": "value", ... }
//!   stdout → { "success": true, "output": "result text" }

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

fn main() {
    // ── Read arguments from stdin ──
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

    // ── Write JSON result to stdout ──
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

    // TODO: Replace with your tool logic.
    let result = format!("Processed: {input}");

    (true, result)
}
RUSTEOF

# ── Add to workspace members ─────────────────────────────────────────────────
if grep -q "members" Cargo.toml 2>/dev/null; then
    # Insert the new tool into the members array (before the closing bracket).
    if ! grep -q "\"$TOOL_DIR\"" Cargo.toml; then
        sed -i "/^members/,/\]/ s|\]|    \"$TOOL_DIR\",\n]|" Cargo.toml
        echo "  ✓ Added '$TOOL_DIR' to workspace members"
    fi
fi

echo ""
echo "✓ Tool '$TOOL_NAME' scaffolded at $TOOL_DIR/"
echo ""
echo "Next steps:"
echo "  1. Edit $TOOL_DIR/tool.json to define your parameters"
echo "  2. Implement your logic in $TOOL_DIR/src/main.rs"
echo "  3. Build: ./build.sh $TOOL_DIR"
echo "  4. Restart the aigent daemon to auto-discover the new tool"
