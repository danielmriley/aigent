#!/usr/bin/env bash
# Build all WASM guest tools for the aigent runtime.
#
# Usage:
#   ./build.sh              # build all tools in release mode
#   ./build.sh read-file    # build a single tool
#
# Prerequisites:
#   rustup target add wasm32-wasip1
#
# Output: each tool produces a .wasm binary at
#   <crate>/target/wasm32-wasip1/release/<name>.wasm
# These are auto-discovered by the aigent daemon at startup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the WASI target is installed.
if ! rustup target list --installed | grep -q wasm32-wasip1; then
    echo "Installing wasm32-wasip1 target..."
    rustup target add wasm32-wasip1
fi

TOOLS=("$@")
if [[ ${#TOOLS[@]} -eq 0 ]]; then
    # Build all workspace members.
    TOOLS=($(ls -d */ | sed 's/\///'))
fi

BUILT=0
FAILED=0
for tool in "${TOOLS[@]}"; do
    if [[ ! -d "$tool" ]]; then
        echo "⚠ Skipping '$tool' — directory not found"
        continue
    fi

    echo "▸ Building $tool..."
    if (cd "$tool" && cargo build --target wasm32-wasip1 --release 2>&1); then
        BUILT=$((BUILT + 1))
        # Show the output path.
        wasm_file=$(find "$tool/target/wasm32-wasip1/release" -name '*.wasm' -not -name '.*' 2>/dev/null | head -1)
        if [[ -n "$wasm_file" ]]; then
            size=$(du -h "$wasm_file" | cut -f1)
            echo "  ✓ $wasm_file ($size)"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed to build $tool"
    fi
done

echo ""
echo "Built: $BUILT  Failed: $FAILED"
[[ $FAILED -eq 0 ]] || exit 1
