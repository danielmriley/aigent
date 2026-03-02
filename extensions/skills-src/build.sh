#!/usr/bin/env bash
# Build WASM skills and deploy them to the skills directory.
#
# Usage:
#   ./build.sh              # build all skills
#   ./build.sh my-skill     # build a single skill
#
# Output:
#   Each skill produces:
#     ../skills/<name>.wasm       — the compiled WASM binary
#     ../skills/<name>.tool.json  — the sidecar manifest
#
# These are auto-discovered by the aigent daemon at startup or on reload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEPLOY_DIR="$SCRIPT_DIR/../skills"
TARGET_DIR="$SCRIPT_DIR/target/wasm32-wasip1/release"
mkdir -p "$DEPLOY_DIR"

# Ensure the WASI target is installed.
if ! rustup target list --installed | grep -q wasm32-wasip1; then
    echo "Installing wasm32-wasip1 target..."
    rustup target add wasm32-wasip1
fi

SKILLS=("$@")
if [[ ${#SKILLS[@]} -eq 0 ]]; then
    # Build all workspace members (directories with Cargo.toml).
    SKILLS=($(ls -d */ 2>/dev/null | sed 's/\///' | grep -v target || true))
fi

if [[ ${#SKILLS[@]} -eq 0 ]]; then
    echo "No skills found to build."
    exit 0
fi

BUILT=0
FAILED=0
for skill in "${SKILLS[@]}"; do
    if [[ ! -d "$skill" ]]; then
        echo "⚠ Skipping '$skill' — directory not found"
        continue
    fi

    SKILL_NAME="${skill//-/_}"
    echo "▸ Building $skill..."

    if cargo build -p "$skill" --target wasm32-wasip1 --release 2>&1; then
        # Workspace builds output to the workspace root target/.
        wasm_file="$TARGET_DIR/${SKILL_NAME}.wasm"

        if [[ -f "$wasm_file" ]]; then
            size=$(du -h "$wasm_file" | cut -f1)
            echo "  ✓ Built: $wasm_file ($size)"

            # Deploy .wasm to skills directory.
            cp "$wasm_file" "$DEPLOY_DIR/${SKILL_NAME}.wasm"
            echo "  ✓ Deployed: $DEPLOY_DIR/${SKILL_NAME}.wasm"

            # Deploy sidecar manifest (tool.json → <name>.tool.json).
            if [[ -f "$skill/tool.json" ]]; then
                cp "$skill/tool.json" "$DEPLOY_DIR/${SKILL_NAME}.tool.json"
                echo "  ✓ Manifest: $DEPLOY_DIR/${SKILL_NAME}.tool.json"
            else
                echo "  ⚠ No tool.json found — skill will use fallback spec"
            fi

            BUILT=$((BUILT + 1))
        else
            echo "  ✗ Build succeeded but .wasm not found at $wasm_file"
            FAILED=$((FAILED + 1))
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed to build $skill"
    fi
done

echo ""
echo "Built: $BUILT  Failed: $FAILED  Deployed to: $DEPLOY_DIR/"
[[ $FAILED -eq 0 ]] || exit 1
