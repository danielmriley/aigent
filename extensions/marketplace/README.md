# Aigent Marketplace

This directory contains tool and skill extensions that can be discovered,
installed, and loaded by the Aigent runtime.

## Structure

```
marketplace/
├── README.md           # This file
├── registry.json       # Local registry of installed extensions
└── <extension-name>/
    ├── manifest.toml   # Extension metadata & configuration
    ├── <name>.wasm     # Compiled WASM binary (for WASM tools)
    └── src/            # Optional: Rust source for building
```

## Manifest Format (`manifest.toml`)

```toml
[extension]
name        = "my-tool"
version     = "0.1.0"
description = "A short description of what this tool does"
authors     = ["Your Name <you@example.com>"]
license     = "MIT OR Apache-2.0"

[tool]
# The tool name registered in the runtime
name        = "my_tool"
group       = "custom"
security    = "low"        # low | medium | high
read_only   = false
# WIT world this tool targets (must match extensions/wit/host.wit)
wit_world   = "aigent:host/tools"

[tool.params]
query      = { type = "string",  required = true,  description = "Search query" }
max_results = { type = "integer", required = false, description = "Max results", default = "10" }

[build]
# How to compile the extension (optional — pre-compiled .wasm also accepted)
toolchain = "cargo-component"
target    = "wasm32-wasip1"
```

## CLI Commands

```bash
# List installed extensions
aigent marketplace list

# Install from a local directory
aigent marketplace install ./path/to/extension

# Build an extension from source
aigent marketplace build my-tool

# Remove an extension
aigent marketplace remove my-tool
```

## Feature Flag

The marketplace is gated behind the `marketplace` feature flag:

```bash
cargo build --features marketplace
```
