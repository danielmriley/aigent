#!/usr/bin/env bash
# Scaffold a new WASM skill for the aigent runtime.
#
# Usage:
#   ./new-skill.sh my-skill                     # create with default settings
#   ./new-skill.sh my-skill "Skill description"  # create with custom description
#
# Creates:
#   <name>/
#     Cargo.toml
#     tool.json
#     src/main.rs
#
# The skill is automatically added to the workspace Cargo.toml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <skill-name> [description]"
    echo ""
    echo "Example:"
    echo "  $0 summarize-text \"Summarize a block of text into key points\""
    exit 1
fi

SKILL_DIR="$1"
SKILL_NAME="${SKILL_DIR//-/_}"  # Rust binary name: hyphens → underscores
DESCRIPTION="${2:-A custom WASM skill for aigent}"

if [[ -d "$SKILL_DIR" ]]; then
    echo "Error: directory '$SKILL_DIR' already exists"
    exit 1
fi

echo "▸ Creating skill '$SKILL_NAME' in $SKILL_DIR/..."

# ── Create directory structure ────────────────────────────────────────────────
mkdir -p "$SKILL_DIR/src"

# ── Cargo.toml ────────────────────────────────────────────────────────────────
cat > "$SKILL_DIR/Cargo.toml" << EOF
[package]
name = "$SKILL_DIR"
version.workspace = true
edition.workspace = true

[[bin]]
name = "$SKILL_NAME"
path = "src/main.rs"

[dependencies]
serde_json = { workspace = true }
EOF

# ── tool.json manifest ───────────────────────────────────────────────────────
cat > "$SKILL_DIR/tool.json" << EOF
{
  "name": "$SKILL_NAME",
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
cat > "$SKILL_DIR/src/main.rs" << 'RUSTEOF'
//! WASM skill — communicates with the aigent host via stdio JSON.
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

    // ── Execute skill logic ──
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

    // TODO: Replace with your skill logic.
    let result = format!("Processed: {input}");

    (true, result)
}
RUSTEOF

# ── Add to workspace members ─────────────────────────────────────────────────
CARGO_TOML="Cargo.toml"
if [[ -f "$CARGO_TOML" ]]; then
    if ! grep -q "\"$SKILL_DIR\"" "$CARGO_TOML"; then
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
    return lines.rstrip().rstrip(',') + (',\n' if lines.strip().endswith('\"') else '') + '    \"$SKILL_DIR\",\n]'
text = re.sub(r'members\s*=\s*\[.*?\]', add_member, text, count=1, flags=re.DOTALL)
with open('$CARGO_TOML', 'w') as f:
    f.write(text)
" 2>/dev/null || {
            # Fallback: simple approach for empty members = []
            sed -i "s|members = \[\]|members = [\n    \"$SKILL_DIR\",\n]|" "$CARGO_TOML"
        }
        echo "  ✓ Added '$SKILL_DIR' to workspace members"
    fi
fi

echo ""
echo "✓ Skill '$SKILL_NAME' scaffolded at $SKILL_DIR/"
echo ""
echo "Next steps:"
echo "  1. Edit $SKILL_DIR/tool.json to define your parameters"
echo "  2. Implement your logic in $SKILL_DIR/src/main.rs"
echo "  3. Build & deploy: ./build.sh $SKILL_DIR"
echo "  4. The daemon will auto-discover the skill on next reload or restart"
