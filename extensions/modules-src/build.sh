#!/usr/bin/env bash
# Build WASM modules and deploy them to the modules directory.
#
# Usage:
#   ./build.sh               # build all modules
#   ./build.sh my-module     # build a single module
#
# Output:
#   Each module produces:
#     ../modules/<name>.wasm       — the compiled WASM binary
#     ../modules/<name>.tool.json  — the sidecar manifest
#
# These are auto-discovered by the aigent daemon at startup or on reload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEPLOY_DIR="$SCRIPT_DIR/../modules"
TARGET_DIR="$SCRIPT_DIR/target/wasm32-wasip1/release"
mkdir -p "$DEPLOY_DIR"

# Ensure the WASI target is installed.
if ! rustup target list --installed | grep -q wasm32-wasip1; then
    echo "Installing wasm32-wasip1 target..."
    rustup target add wasm32-wasip1
fi

MODULES=("$@")
if [[ ${#MODULES[@]} -eq 0 ]]; then
    # Build all workspace members (directories with Cargo.toml).
    MODULES=($(ls -d */ 2>/dev/null | sed 's/\///' | grep -v target || true))
fi

if [[ ${#MODULES[@]} -eq 0 ]]; then
    echo "No modules found to build."
    exit 0
fi

BUILT=0
FAILED=0
for module in "${MODULES[@]}"; do
    if [[ ! -d "$module" ]]; then
        echo "⚠ Skipping '$module' — directory not found"
        continue
    fi

    MODULE_NAME="${module//-/_}"
    echo "▸ Building $module..."

    if cargo build -p "$module" --target wasm32-wasip1 --release 2>&1; then
        # Workspace builds output to the workspace root target/.
        wasm_file="$TARGET_DIR/${MODULE_NAME}.wasm"

        if [[ -f "$wasm_file" ]]; then
            size=$(du -h "$wasm_file" | cut -f1)
            echo "  ✓ Built: $wasm_file ($size)"

            # Deploy .wasm to modules directory.
            cp "$wasm_file" "$DEPLOY_DIR/${MODULE_NAME}.wasm"
            echo "  ✓ Deployed: $DEPLOY_DIR/${MODULE_NAME}.wasm"

            # Deploy sidecar manifest (tool.json → <name>.tool.json).
            if [[ -f "$module/tool.json" ]]; then
                cp "$module/tool.json" "$DEPLOY_DIR/${MODULE_NAME}.tool.json"
                echo "  ✓ Manifest: $DEPLOY_DIR/${MODULE_NAME}.tool.json"
            else
                echo "  ⚠ No tool.json found — module will use fallback spec"
            fi

            BUILT=$((BUILT + 1))
        else
            echo "  ✗ Build succeeded but .wasm not found at $wasm_file"
            FAILED=$((FAILED + 1))
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed to build $module"
    fi
done

echo ""
echo "Built: $BUILT  Failed: $FAILED  Deployed to: $DEPLOY_DIR/"
[[ $FAILED -eq 0 ]] || exit 1
