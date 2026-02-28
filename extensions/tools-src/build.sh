#!/usr/bin/env bash
# Build all WASM guest tools for the aigent runtime.
#
# Usage:
#   ./build.sh              # build all tools in release mode
#   ./build.sh read-file    # build a single tool
#   ./build.sh --watch      # rebuild on changes (requires inotifywait or fswatch)
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

# ── Watch mode ────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--watch" ]]; then
    shift
    echo "▸ Watching for changes (Ctrl+C to stop)..."
    if command -v inotifywait &>/dev/null; then
        while true; do
            "$0" "$@"
            echo ""
            echo "Watching for changes..."
            inotifywait -r -e modify,create,delete \
                --include '.*\.(rs|toml|json)$' \
                --exclude 'target/' \
                . 2>/dev/null
            echo ""
        done
    elif command -v fswatch &>/dev/null; then
        fswatch -r --include '.*\.(rs|toml|json)$' --exclude 'target/' . | while read -r _; do
            "$0" "$@"
        done
    else
        echo "Error: --watch requires 'inotifywait' (inotify-tools) or 'fswatch'"
        echo "Install: sudo apt install inotify-tools  (Linux)"
        echo "         brew install fswatch             (macOS)"
        exit 1
    fi
    exit 0
fi

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
