#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust toolchain not found. Install from https://rustup.rs first." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cargo build --release --locked -p aigent-app

TARGET_BIN="${SCRIPT_DIR}/target/release/aigent-app"
if [[ ! -f "$TARGET_BIN" ]]; then
  echo "Build completed but binary was not found at ${TARGET_BIN}" >&2
  exit 1
fi

INSTALL_DIR="${AIGENT_INSTALL_DIR:-${HOME}/.local/bin}"
mkdir -p "$INSTALL_DIR"
INSTALL_PATH="${INSTALL_DIR}/aigent"
TEMP_PATH="$(mktemp "${INSTALL_DIR}/.aigent.tmp.XXXXXX")"

cleanup() {
  rm -f "$TEMP_PATH"
}
trap cleanup EXIT

install -m 0755 "$TARGET_BIN" "$TEMP_PATH"
mv -f "$TEMP_PATH" "$INSTALL_PATH"
trap - EXIT

echo "Installed aigent to ${INSTALL_PATH}"
if command -v sha256sum >/dev/null 2>&1; then
  echo "sha256: $(sha256sum "$INSTALL_PATH" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  echo "sha256: $(shasum -a 256 "$INSTALL_PATH" | awk '{print $1}')"
fi

if [[ ":${PATH}:" != *":${INSTALL_DIR}:"* ]]; then
  echo "Note: ${INSTALL_DIR} is not in PATH for this shell."
fi

# Restart the daemon if it is currently running, so it picks up the new binary.
SOCKET_PATH=""
if command -v grep >/dev/null 2>&1 && [[ -f "config/default.toml" ]]; then
  SOCKET_PATH=$(grep -E '^\s*socket_path\s*=' config/default.toml | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
fi
SOCKET_PATH="${SOCKET_PATH:-/tmp/aigent.sock}"
PID_FILE=".aigent/runtime/daemon.pid"
if [[ -S "$SOCKET_PATH" ]] || ([[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null); then
  echo "Restarting daemon to pick up new binary..."
  "$INSTALL_PATH" daemon restart --force && echo "Daemon restarted." || echo "Warning: daemon restart failed — run 'aigent daemon restart' manually."
else
  echo "Run: aigent --version"
  echo "Then: aigent onboard"
fi
