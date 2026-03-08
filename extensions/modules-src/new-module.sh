#!/usr/bin/env bash
# Scaffold a new WASM module for the aigent runtime.
#
# Usage:
#   ./new-module.sh my-module                      # create with default settings
#   ./new-module.sh my-module "Module description"  # create with custom description
#
# Creates:
#   <name>/
#     Cargo.toml
#     tool.json
#     src/main.rs
#
# The module is automatically added to the workspace Cargo.toml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <module-name> [description]"
    echo ""
    echo "Example:"
    echo "  $0 summarize-text \"Summarize a block of text into key points\""
    exit 1
fi

MODULE_DIR="$1"
MODULE_NAME="${MODULE_DIR//-/_}"  # Rust binary name: hyphens → underscores
DESCRIPTION="${2:-A custom WASM module for aigent}"

if [[ -d "$MODULE_DIR" ]]; then
    echo "Error: directory '$MODULE_DIR' already exists"
    exit 1
fi

echo "▸ Creating module '$MODULE_NAME' in $MODULE_DIR/..."

# ── Create directory structure ────────────────────────────────────────────────
mkdir -p "$MODULE_DIR/src"

# ── Cargo.toml ────────────────────────────────────────────────────────────────
cat > "$MODULE_DIR/Cargo.toml" << EOF
[package]
name = "$MODULE_DIR"
version.workspace = true
edition.workspace = true

[[bin]]
name = "$MODULE_NAME"
path = "src/main.rs"

[dependencies]
serde_json = { workspace = true }
EOF

# ── tool.json manifest ───────────────────────────────────────────────────────
cat > "$MODULE_DIR/tool.json" << EOF
{
  "name": "$MODULE_NAME",
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
cat > "$MODULE_DIR/src/main.rs" << 'RUSTEOF'
//! WASM module — communicates with the aigent host via stdio JSON.
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

    // ── Execute module logic ──
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

    // TODO: Replace with your module logic.
    let result = format!("Processed: {input}");

    (true, result)
}
RUSTEOF

# ── Add to workspace members ─────────────────────────────────────────────────
CARGO_TOML="Cargo.toml"
if [[ -f "$CARGO_TOML" ]]; then
    if ! grep -q "\"$MODULE_DIR\"" "$CARGO_TOML"; then
        # Use Python for reliable TOML member insertion (avoids fragile sed)
        python3 -c "
import re, sys
with open('$CARGO_TOML') as f:
    text = f.read()
# Match the members array and append the new member
def add_member(m):
    bracket = m.group(0)
    # Find the closing bracket
    lines = bracket.rstrip().rstrip(']')
    # Clean up: add new member before the closing ]
    return lines.rstrip().rstrip(',') + (',\n' if lines.strip().endswith('\"') else '') + '    \"$MODULE_DIR\",\n]'
text = re.sub(r'members\s*=\s*\[.*?\]', add_member, text, count=1, flags=re.DOTALL)
with open('$CARGO_TOML', 'w') as f:
    f.write(text)
" 2>/dev/null || {
            # Fallback: simple approach for empty members = []
            sed -i "s|members = \[\]|members = [\n    \"$MODULE_DIR\",\n]|" "$CARGO_TOML"
        }
        echo "  ✓ Added '$MODULE_DIR' to workspace members"
    fi
fi

echo ""
echo "✓ Module '$MODULE_NAME' scaffolded at $MODULE_DIR/"
echo ""
echo "Next steps:"
echo "  1. Edit $MODULE_DIR/tool.json to define your parameters"
echo "  2. Implement your logic in $MODULE_DIR/src/main.rs"
echo "  3. Build & deploy: ./build.sh $MODULE_DIR"
echo "  4. The daemon will auto-discover the module on next reload or restart"
