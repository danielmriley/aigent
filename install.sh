#!/usr/bin/env bash
set -euo pipefail

# ── Colours (disabled when piped) ─────────────────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m' DIM='\033[2m' GREEN='\033[0;32m' YELLOW='\033[0;33m'
  RED='\033[0;31m' CYAN='\033[0;36m' RESET='\033[0m'
else
  BOLD='' DIM='' GREEN='' YELLOW='' RED='' CYAN='' RESET=''
fi

info()  { printf "${GREEN}[+]${RESET} %s\n" "$*"; }
warn()  { printf "${YELLOW}[!]${RESET} %s\n" "$*"; }
err()   { printf "${RED}[x]${RESET} %s\n" "$*" >&2; }
ask()   { printf "${BOLD}[?]${RESET} %s " "$*"; }
note()  { printf "${CYAN}    %s${RESET}\n" "$*"; }

usage() {
  cat <<'USAGE'
Aigent installer — build, install, and configure the AI agent.

Usage:
  ./install.sh [OPTIONS]

Options:
  --candle          Enable the Candle local-inference backend (GGUF models).
  --cuda            Enable Candle with CUDA GPU acceleration.
  --metal           Enable Candle with Metal GPU acceleration (macOS).
  --all-features    Build with all optional features.
  --no-candle       Explicitly disable the Candle backend (default).
  --prefix DIR      Install binary to DIR instead of ~/.local/bin.
  --config PATH     Set the config file path (default: config/default.toml).
                    Also sets AIGENT_CONFIG in the shell hint.
  --uninstall       Remove aigent binary, data, and daemon.
  -h, --help        Show this help message.

Environment variables:
  AIGENT_INSTALL_DIR   Override install directory (same as --prefix).
  AIGENT_CONFIG        Override config file path at runtime.
  AIGENT_AUTO_DEPS=1   Auto-install missing system deps without prompting.
  AIGENT_FEATURES      Space-separated extra cargo features (e.g. "qdrant").

Examples:
  # Standard install (Ollama + OpenRouter only):
  ./install.sh

  # With local Candle inference on CPU:
  ./install.sh --candle

  # With Candle + CUDA GPU:
  ./install.sh --cuda

  # With Candle + Metal GPU (macOS):
  ./install.sh --metal

  # Uninstall:
  ./install.sh --uninstall
USAGE
}

# ── OS / distro detection ────────────────────────────────────────────────────
detect_os() {
  case "$(uname -s)" in
    Linux*)  OS="linux" ;;
    Darwin*) OS="macos" ;;
    *)       OS="unknown" ;;
  esac

  DISTRO="unknown"
  PKG_MANAGER="unknown"
  if [[ "$OS" == "linux" ]]; then
    if [[ -f /etc/os-release ]]; then
      # shellcheck disable=SC1091
      . /etc/os-release
      DISTRO="${ID:-unknown}"
    fi
    if command -v apt-get >/dev/null 2>&1; then
      PKG_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then
      PKG_MANAGER="dnf"
    elif command -v yum >/dev/null 2>&1; then
      PKG_MANAGER="yum"
    elif command -v pacman >/dev/null 2>&1; then
      PKG_MANAGER="pacman"
    elif command -v zypper >/dev/null 2>&1; then
      PKG_MANAGER="zypper"
    elif command -v apk >/dev/null 2>&1; then
      PKG_MANAGER="apk"
    fi
  elif [[ "$OS" == "macos" ]]; then
    DISTRO="macos"
    if command -v brew >/dev/null 2>&1; then
      PKG_MANAGER="brew"
    fi
  fi
}

# ── Privilege helper ──────────────────────────────────────────────────────────
sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    echo ""
  elif command -v sudo >/dev/null 2>&1; then
    echo "sudo"
  elif command -v doas >/dev/null 2>&1; then
    echo "doas"
  else
    echo ""
  fi
}

# ── Dependency check & install ────────────────────────────────────────────────

# Required system libraries / headers for the build.
#   openssl (libssl-dev)     — git2 crate + reqwest TLS
#   pkg-config               — locates .pc files for openssl, libssh2, etc.
#   cmake                    — libgit2-sys build (fallback when no system libgit2)
#   zlib (zlib1g-dev)        — libgit2-sys, libssh2-sys compression
#   libssh2 (libssh2-1-dev) — git2 SSH transport

deps_for_manager() {
  case "$1" in
    apt)     echo "pkg-config libssl-dev cmake zlib1g-dev libssh2-1-dev" ;;
    dnf|yum) echo "pkgconfig openssl-devel cmake zlib-devel libssh2-devel" ;;
    pacman)  echo "pkg-config openssl cmake zlib libssh2" ;;
    zypper)  echo "pkg-config libopenssl-devel cmake zlib-devel libssh2-devel" ;;
    apk)     echo "pkgconf openssl-dev cmake zlib-dev libssh2-dev" ;;
    brew)    echo "pkg-config openssl cmake libssh2" ;;
    *)       echo "" ;;
  esac
}

# Quick presence check — returns 0 if the dependency is already satisfied.
check_dep() {
  local dep="$1"
  case "$dep" in
    pkg-config|pkgconfig|pkgconf)
      command -v pkg-config >/dev/null 2>&1 || command -v pkgconf >/dev/null 2>&1
      ;;
    cmake)
      command -v cmake >/dev/null 2>&1
      ;;
    libssl-dev|openssl-devel|libopenssl-devel|openssl-dev|openssl)
      pkg-config --exists openssl 2>/dev/null || \
        [[ -f /usr/include/openssl/ssl.h ]] || \
        [[ -f /usr/local/include/openssl/ssl.h ]] || \
        [[ -d "$(brew --prefix openssl 2>/dev/null)/include" ]] 2>/dev/null
      ;;
    zlib1g-dev|zlib-devel|zlib-dev|zlib)
      pkg-config --exists zlib 2>/dev/null || \
        [[ -f /usr/include/zlib.h ]] || \
        [[ -f /usr/local/include/zlib.h ]]
      ;;
    libssh2-1-dev|libssh2-devel|libssh2-dev|libssh2)
      pkg-config --exists libssh2 2>/dev/null || \
        [[ -f /usr/include/libssh2.h ]] || \
        [[ -f /usr/local/include/libssh2.h ]] || \
        [[ -d "$(brew --prefix libssh2 2>/dev/null)/include" ]] 2>/dev/null
      ;;
    *)
      return 1
      ;;
  esac
}

check_and_install_deps() {
  local manager="$1"
  local all_pkgs
  all_pkgs="$(deps_for_manager "$manager")"

  if [[ -z "$all_pkgs" ]]; then
    warn "Cannot determine packages for package manager '${manager}'."
    warn "Please install these manually: pkg-config, OpenSSL headers, cmake, zlib headers, libssh2 headers."
    return 0
  fi

  local missing=()
  for pkg in $all_pkgs; do
    if ! check_dep "$pkg"; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    info "All build dependencies are satisfied."
    return 0
  fi

  warn "Missing build dependencies: ${missing[*]}"

  local auto_install=false
  if [[ ! -t 0 ]] || [[ "${AIGENT_AUTO_DEPS:-}" == "1" ]]; then
    auto_install=true
  fi

  if [[ "$auto_install" == false ]]; then
    local install_cmd
    install_cmd="$(format_install_cmd "$manager" "${missing[@]}")"
    echo ""
    ask "Install missing dependencies with: ${DIM}${install_cmd}${RESET}  [Y/n] "
    read -r answer
    case "${answer:-Y}" in
      [Yy]*|"") ;;
      *)
        warn "Skipping dependency installation. The build may fail."
        return 0
        ;;
    esac
  fi

  info "Installing: ${missing[*]}"
  run_install "$manager" "${missing[@]}"
  info "Dependencies installed."
}

format_install_cmd() {
  local manager="$1"; shift
  local pkgs=("$@")
  local prefix
  prefix="$(sudo_cmd)"
  case "$manager" in
    apt)          echo "${prefix:+$prefix }apt-get install -y ${pkgs[*]}" ;;
    dnf)          echo "${prefix:+$prefix }dnf install -y ${pkgs[*]}" ;;
    yum)          echo "${prefix:+$prefix }yum install -y ${pkgs[*]}" ;;
    pacman)       echo "${prefix:+$prefix }pacman -S --noconfirm ${pkgs[*]}" ;;
    zypper)       echo "${prefix:+$prefix }zypper install -y ${pkgs[*]}" ;;
    apk)          echo "${prefix:+$prefix }apk add ${pkgs[*]}" ;;
    brew)         echo "brew install ${pkgs[*]}" ;;
  esac
}

run_install() {
  local manager="$1"; shift
  local pkgs=("$@")
  local prefix
  prefix="$(sudo_cmd)"
  case "$manager" in
    apt)
      ${prefix:+$prefix} apt-get update -qq
      ${prefix:+$prefix} apt-get install -y -qq "${pkgs[@]}"
      ;;
    dnf)     ${prefix:+$prefix} dnf install -y "${pkgs[@]}" ;;
    yum)     ${prefix:+$prefix} yum install -y "${pkgs[@]}" ;;
    pacman)  ${prefix:+$prefix} pacman -S --noconfirm "${pkgs[@]}" ;;
    zypper)  ${prefix:+$prefix} zypper install -y "${pkgs[@]}" ;;
    apk)     ${prefix:+$prefix} apk add "${pkgs[@]}" ;;
    brew)    brew install "${pkgs[@]}" ;;
    *)
      err "Unsupported package manager: ${manager}"
      return 1
      ;;
  esac
}

# ── Rust toolchain check ─────────────────────────────────────────────────────

check_rust() {
  if command -v cargo >/dev/null 2>&1; then
    info "Rust toolchain found: $(rustc --version 2>/dev/null || echo 'unknown version')"
    return 0
  fi

  warn "Rust toolchain not found."
  if [[ ! -t 0 ]] || [[ "${AIGENT_AUTO_DEPS:-}" == "1" ]]; then
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
    return 0
  fi

  ask "Install Rust via rustup? [Y/n] "
  read -r answer
  case "${answer:-Y}" in
    [Yy]*|"")
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
      # shellcheck disable=SC1091
      source "${HOME}/.cargo/env"
      ;;
    *)
      err "Rust is required to build Aigent. Install from https://rustup.rs"
      exit 1
      ;;
  esac
}

# ── CUDA / Metal check ───────────────────────────────────────────────────────

check_cuda() {
  if command -v nvcc >/dev/null 2>&1; then
    local cuda_ver
    cuda_ver="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo 'unknown')"
    info "CUDA toolkit found: ${cuda_ver}"
    return 0
  fi

  if [[ -d /usr/local/cuda ]]; then
    info "CUDA found at /usr/local/cuda (nvcc not in PATH)"
    note "You may need: export PATH=/usr/local/cuda/bin:\$PATH"
    return 0
  fi

  warn "CUDA toolkit not found."
  warn "For GPU acceleration, install CUDA from: https://developer.nvidia.com/cuda-downloads"
  warn "Building anyway — Candle will fall back to CPU at runtime."
  return 0
}

check_metal() {
  if [[ "$OS" != "macos" ]]; then
    warn "--metal only works on macOS. Ignoring."
    METAL=false
    return 0
  fi
  # Metal is always available on macOS with a supported GPU.
  info "Metal GPU acceleration enabled (macOS built-in)."
}

# ── Config seeding ────────────────────────────────────────────────────────────

seed_config() {
  local config_path="$1"

  if [[ -f "$config_path" ]]; then
    info "Config file exists at ${config_path}"
    return 0
  fi

  if [[ -f "config/default.toml.example" ]]; then
    local config_dir
    config_dir="$(dirname "$config_path")"
    mkdir -p "$config_dir"
    cp config/default.toml.example "$config_path"
    info "Created ${config_path} from example template"
  else
    warn "No config template found. Run 'aigent onboard' to configure."
  fi
}

# ── Uninstall ─────────────────────────────────────────────────────────────────

do_uninstall() {
  INSTALL_DIR="${AIGENT_INSTALL_DIR:-${HOME}/.local/bin}"
  INSTALL_PATH="${INSTALL_DIR}/aigent"
  DATA_DIR="${HOME}/.aigent"
  MODELS_DIR="${HOME}/.cache/aigent"
  SOCKET_PATH="/tmp/aigent.sock"

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local cfg_path="${AIGENT_CONFIG:-${SCRIPT_DIR}/config/default.toml}"
  if [[ -f "$cfg_path" ]]; then
    local cfg_socket
    cfg_socket=$(grep -E '^\s*socket_path\s*=' "$cfg_path" 2>/dev/null | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
    SOCKET_PATH="${cfg_socket:-$SOCKET_PATH}"
  fi

  info "Aigent uninstaller"
  echo ""

  # 1. Stop the daemon if running.
  PID_FILE="${DATA_DIR}/runtime/daemon.pid"
  if [[ -S "$SOCKET_PATH" ]] || { [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; }; then
    info "Stopping running daemon..."
    if [[ -f "$INSTALL_PATH" ]]; then
      "$INSTALL_PATH" daemon stop 2>/dev/null || true
    fi
    if [[ -f "$PID_FILE" ]]; then
      local pid
      pid="$(cat "$PID_FILE" 2>/dev/null)"
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 1
      fi
    fi
    rm -f "$SOCKET_PATH"
    info "Daemon stopped."
  fi

  # 2. Summarise what will be removed.
  echo ""
  echo "The following will be removed:"
  [[ -f "$INSTALL_PATH" ]] && echo "  Binary:     ${INSTALL_PATH}"
  [[ -d "$DATA_DIR" ]]     && echo "  Data dir:   ${DATA_DIR}  (memory, logs, secrets)"
  [[ -d "$MODELS_DIR" ]]   && echo "  Models dir: ${MODELS_DIR}  (cached GGUF downloads)"
  [[ -S "$SOCKET_PATH" ]]  && echo "  Socket:     ${SOCKET_PATH}"

  if [[ ! -f "$INSTALL_PATH" ]] && [[ ! -d "$DATA_DIR" ]]; then
    warn "Nothing to uninstall — aigent does not appear to be installed."
    return 0
  fi

  echo ""
  if [[ -t 0 ]]; then
    ask "Proceed with uninstall? [y/N] "
    read -r answer
    case "${answer:-N}" in
      [Yy]*) ;;
      *)
        info "Uninstall cancelled."
        return 0
        ;;
    esac
  fi

  # 3. Remove binary.
  if [[ -f "$INSTALL_PATH" ]]; then
    rm -f "$INSTALL_PATH"
    info "Removed ${INSTALL_PATH}"
  fi

  # 4. Remove data directory.
  if [[ -d "$DATA_DIR" ]]; then
    local remove_data=true
    if [[ -t 0 ]]; then
      echo ""
      ask "Also remove ${DATA_DIR}? This deletes all memory, logs, and secrets. [y/N] "
      read -r answer
      case "${answer:-N}" in
        [Yy]*) ;;
        *) remove_data=false ;;
      esac
    fi

    if [[ "$remove_data" == true ]]; then
      rm -rf "$DATA_DIR"
      info "Removed ${DATA_DIR}"
    else
      info "Kept ${DATA_DIR}"
    fi
  fi

  # 5. Remove cached models.
  if [[ -d "$MODELS_DIR" ]]; then
    local remove_models=true
    if [[ -t 0 ]]; then
      echo ""
      ask "Also remove ${MODELS_DIR}? This deletes cached GGUF model files. [y/N] "
      read -r answer
      case "${answer:-N}" in
        [Yy]*) ;;
        *) remove_models=false ;;
      esac
    fi

    if [[ "$remove_models" == true ]]; then
      rm -rf "$MODELS_DIR"
      info "Removed ${MODELS_DIR}"
    else
      info "Kept ${MODELS_DIR}"
    fi
  fi

  # 6. Clean up socket.
  rm -f "$SOCKET_PATH" 2>/dev/null || true

  echo ""
  info "Aigent has been uninstalled."
  info "Source code in ${SCRIPT_DIR} was not removed."
  info "Build artifacts can be cleaned with: cargo clean"
}

# ── Build feature assembly ────────────────────────────────────────────────────

assemble_features() {
  local features=()

  if [[ "$ALL_FEATURES" == true ]]; then
    echo "--all-features"
    return
  fi

  if [[ "$ENABLE_CANDLE" == true ]]; then
    features+=("candle")
  fi

  # Allow extra features from env.
  if [[ -n "${AIGENT_FEATURES:-}" ]]; then
    for f in $AIGENT_FEATURES; do
      features+=("$f")
    done
  fi

  if [[ ${#features[@]} -gt 0 ]]; then
    local joined
    joined=$(IFS=,; echo "${features[*]}")
    echo "--features ${joined}"
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

# Handle --uninstall before anything else (detect_os hasn't run yet).
for arg in "$@"; do
  case "$arg" in
    --uninstall|uninstall)
      do_uninstall
      exit 0
      ;;
  esac
done

# Parse arguments.
ENABLE_CANDLE=false
CUDA=false
METAL=false
ALL_FEATURES=false
CUSTOM_PREFIX=""
CUSTOM_CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candle)       ENABLE_CANDLE=true; shift ;;
    --cuda)         ENABLE_CANDLE=true; CUDA=true; shift ;;
    --metal)        ENABLE_CANDLE=true; METAL=true; shift ;;
    --all-features) ALL_FEATURES=true; ENABLE_CANDLE=true; shift ;;
    --no-candle)    ENABLE_CANDLE=false; shift ;;
    --prefix)       CUSTOM_PREFIX="$2"; shift 2 ;;
    --config)       CUSTOM_CONFIG="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *)
      err "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

detect_os
info "Detected: OS=${OS}, distro=${DISTRO}, package manager=${PKG_MANAGER}"

# Header
echo ""
printf "${BOLD}Aigent Installer${RESET}\n"
if [[ "$ENABLE_CANDLE" == true ]]; then
  if [[ "$CUDA" == true ]]; then
    note "Mode: Candle + CUDA (GPU-accelerated local inference)"
  elif [[ "$METAL" == true ]]; then
    note "Mode: Candle + Metal (GPU-accelerated local inference)"
  else
    note "Mode: Candle (CPU local inference)"
  fi
else
  note "Mode: Standard (Ollama + OpenRouter)"
  note "Tip: use --candle to enable local GGUF model inference"
fi
echo ""

check_rust

if [[ "$PKG_MANAGER" != "unknown" ]]; then
  check_and_install_deps "$PKG_MANAGER"
else
  warn "Could not detect a supported package manager."
  warn "Please ensure these are installed: pkg-config, OpenSSL headers (libssl-dev),"
  warn "  cmake, zlib headers, libssh2 headers."
fi

# Check GPU toolkits if requested.
if [[ "$CUDA" == true ]]; then
  check_cuda
fi
if [[ "$METAL" == true ]]; then
  check_metal
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Seed config if it doesn't exist.
CONFIG_PATH="${CUSTOM_CONFIG:-config/default.toml}"
seed_config "$CONFIG_PATH"

# Assemble cargo build command.
CARGO_FEATURES="$(assemble_features)"
BUILD_CMD="cargo build --release -p aigent-app"
if [[ -n "$CARGO_FEATURES" ]]; then
  BUILD_CMD="${BUILD_CMD} ${CARGO_FEATURES}"
fi

# Add --locked only if Cargo.lock exists.
if [[ -f Cargo.lock ]]; then
  BUILD_CMD="${BUILD_CMD} --locked"
fi

info "Building aigent (release)..."
note "${BUILD_CMD}"
echo ""
eval "$BUILD_CMD"

TARGET_BIN="${SCRIPT_DIR}/target/release/aigent-app"
if [[ ! -f "$TARGET_BIN" ]]; then
  err "Build completed but binary was not found at ${TARGET_BIN}"
  exit 1
fi

INSTALL_DIR="${CUSTOM_PREFIX:-${AIGENT_INSTALL_DIR:-${HOME}/.local/bin}}"
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

info "Installed aigent to ${INSTALL_PATH}"
if command -v sha256sum >/dev/null 2>&1; then
  echo "  sha256: $(sha256sum "$INSTALL_PATH" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  echo "  sha256: $(shasum -a 256 "$INSTALL_PATH" | awk '{print $1}')"
fi

if [[ ":${PATH}:" != *":${INSTALL_DIR}:"* ]]; then
  warn "${INSTALL_DIR} is not in PATH for this shell."
  note "Add to your shell profile:  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi

# If a custom config path was given, suggest the env var.
if [[ -n "$CUSTOM_CONFIG" ]]; then
  echo ""
  note "Custom config: ${CUSTOM_CONFIG}"
  note "Set this in your shell profile:  export AIGENT_CONFIG=\"${CUSTOM_CONFIG}\""
fi

# ── Restart daemon if running ─────────────────────────────────────────────────
SOCKET_PATH=""
if command -v grep >/dev/null 2>&1 && [[ -f "$CONFIG_PATH" ]]; then
  SOCKET_PATH=$(grep -E '^\s*socket_path\s*=' "$CONFIG_PATH" 2>/dev/null | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
fi
SOCKET_PATH="${SOCKET_PATH:-/tmp/aigent.sock}"
PID_FILE=".aigent/runtime/daemon.pid"
if [[ -S "$SOCKET_PATH" ]] || { [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; }; then
  info "Restarting daemon to pick up new binary..."
  "$INSTALL_PATH" daemon restart --force && info "Daemon restarted." || warn "Daemon restart failed — run 'aigent daemon restart' manually."
else
  echo ""
  info "Installation complete!"
  echo ""
  echo "  Ollama Parallel Subagents Setup:"
  echo "    To enable multi-agent concurrency, Ollama needs parallel config."
  echo "    If running manually in a terminal:"
  echo "      OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 ollama serve"
  echo ""
  echo "    If running Ollama as a systemd service (Linux):"
  echo "      sudo systemctl edit ollama.service"
  echo "      [Service]"
  echo "      Environment=\"OLLAMA_NUM_PARALLEL=4\""
  echo "      Environment=\"OLLAMA_MAX_LOADED_MODELS=2\""
  echo ""
  echo "  Quick start:"
  echo "    aigent                    # start interactive mode"
  echo "    aigent onboard            # run the setup wizard"
  echo "    aigent doctor             # check system status"
  echo ""
  if [[ "$ENABLE_CANDLE" == true ]]; then
    echo "  Candle local inference (compiled in):"
    echo "    aigent onboard            # select 'candle' as provider in wizard"
    echo "    /model provider candle    # switch to Candle at runtime"
    echo "    /model set <repo> <gguf>  # set HF model repo + GGUF filename"
    echo "    /model list candle        # show configured Candle model"
    echo ""
    echo "  Config: [inference] section in ${CONFIG_PATH}"
    echo "    candle_enabled = true"
    echo "    candle_model_repo = \"Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF\""
    echo "    candle_model_file = \"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf\""
    echo "    candle_device = \"cpu\"    # or \"cuda\" / \"metal\""
    echo ""
  else
    echo "  Local inference:"
    echo "    Rebuild with: ./install.sh --candle"
    echo "    Or with GPU:  ./install.sh --cuda  (NVIDIA)"
    echo "                  ./install.sh --metal (Apple Silicon)"
    echo ""
  fi
  echo "  Thinking pipeline:"
  echo "    /think low|balanced|deep  # set reasoning depth"
  echo "    /thinking external        # enable external step-by-step reasoning"
  echo "    /thinking model           # use model's internal reasoning (default)"
  echo ""
fi
