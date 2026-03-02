# Comprehensive Codebase Audit Report

**Scope**: All 10 workspace crates + config, docs, extensions, and WASM tools  
**Rust version**: rustc 1.93.1 (stable), edition 2024  
**Tests**: 352 passing, 0 failures  
**Clippy**: 14 unique warnings (none fatal)  
**Git state**: HEAD `f25a022` on master, clean working tree  
**Date**: 2025-07-15  

---

## Table of Contents

1. [Critical — Fix Immediately](#1-critical--fix-immediately)
2. [High — Bugs That Corrupt Data or Lose State](#2-high--bugs-that-corrupt-data-or-lose-state)
3. [Medium — Logic Errors, Duplication, and Design Debt](#3-medium--logic-errors-duplication-and-design-debt)
4. [Low — Dead Code, Stale Docs, and Hygiene](#4-low--dead-code-stale-docs-and-hygiene)
5. [Clippy Warnings](#5-clippy-warnings)
6. [Positive Findings](#6-positive-findings)
7. [Summary](#7-summary)

---

## 1. Critical — Fix Immediately

### 1.1 — Hardcoded Brave API Key in Version Control

| Field | Value |
|---|---|
| **Severity** | CRITICAL |
| **Category** | Credential leak |
| **File** | `config/default.toml` line 51 |

**Description**: The default config ships with a real Brave Search API key: `brave_api_key = "BSAfNe6KvDj6l_Dd1rcpPSLJ2871Kca"`. This is committed to version control and included in every checkout.

**Action**: (1) Rotate the key in the Brave dashboard immediately. (2) Replace the value with an empty string or placeholder. (3) Load from environment variable or a `.env` file not tracked in git.

### 1.2 — SSRF in WebBrowseTool (Missing Private-IP Guard)

| Field | Value |
|---|---|
| **Severity** | CRITICAL |
| **Category** | Security |
| **File** | `crates/tools/src/builtins/web_browse.rs` — `fetch_direct()` (~line 317) |

**Description**: `WebBrowseTool::fetch_direct()` makes raw HTTP requests to any user-supplied URL without checking whether the resolved IP falls within a private or reserved range (loopback, RFC 1918, link-local, CGNAT, etc.). The newer `BrowsePageTool` in `browse.rs` has a comprehensive `is_private_ip()` guard, but `WebBrowseTool` lacks it.

Because both tools are registered simultaneously in `default_registry()`, a prompt-injection attack could call `web_browse` with `target=http://169.254.169.254/latest/meta-data/` to reach cloud metadata services, or hit internal services on `127.0.0.1`/`10.x.x.x`.

**Suggested fix**: Remove `WebBrowseTool` and `FetchPageTool` entirely — `BrowsePageTool` already supersedes both (its own doc-comment says so). Or extract `is_private_ip()` into a shared utility and gate `fetch_direct()`.

---

## 2. High — Bugs That Corrupt Data or Lose State

### 2.1 — `std::mem::take` Amnesia Window in SubmitTurn

| Field | Value |
|---|---|
| **Severity** | HIGH |
| **Category** | Data loss |
| **File** | `crates/runtime/src/server/connection.rs` — lines 114, 353, 896 |

**Description**: `SubmitTurn` extracts the live `MemoryManager` from `AgentConnection` via `std::mem::take(&mut self.memory)`, leaving an empty default in place. During the potentially long-running LLM call, any concurrent writes (proactive check, scheduled task, second connection) go to the empty default and are silently lost when the real memory is restored.

The nightly consolidation path was already fixed in an earlier commit (using a snapshot pattern), but the three `SubmitTurn` instances were not.

**Suggested fix**: Clone or `Arc`-wrap the memory manager so the canonical instance remains in place, or take a snapshot and merge back after the turn.

### 2.2 — `record_belief` Confidence Not Persisted

| Field | Value |
|---|---|
| **Severity** | HIGH |
| **Category** | Logic bug |
| **File** | `crates/memory/src/manager/mod.rs` — lines 314–325 |

**Description**: `record_belief()` creates an entry via `record_inner_tagged()`, then sets `entry.confidence = confidence.clamp(0.0, 1.0)`. But `record_inner_tagged()` already wrote the entry to the store and event log with the default confidence (0.7). The mutation only affects the clone returned to the caller — the persisted entry still has 0.7.

**Suggested fix**: Pass the confidence value into `record_inner_tagged()` so it's set *before* the entry is written to the store and logged.

### 2.3 — `run_forgetting_pass` Doesn't Persist to Event Log

| Field | Value |
|---|---|
| **Severity** | HIGH |
| **Category** | Data loss |
| **File** | `crates/memory/src/manager/maintenance.rs` — line 57 |

**Description**: `run_forgetting_pass()` synchronously removes entries from the in-memory store via `store.retain()`, but never records the deletions in the event log. On next restart the event log is replayed, and "forgotten" entries reappear.

**Suggested fix**: After `store.retain()`, append `MemoryEvent::Deleted { id }` entries to the event log for every removed entry.

### 2.4 — Candle Feature Won't Compile

| Field | Value |
|---|---|
| **Severity** | HIGH |
| **Category** | Build failure |
| **File** | `crates/llm/src/lib.rs` — ~line 525 |

**Description**: The `candle` feature-gated code uses `ChatRole` and `Option<String>` in a `format!()` macro, but neither type implements `Display`. Building with `--features candle` will fail.

**Suggested fix**: Add `Display` impls, or use `{:?}` formatting, or restructure the output.

### 2.5 — Candle → OpenRouter Wrong Provider Mapping

| Field | Value |
|---|---|
| **Severity** | HIGH |
| **Category** | Logic bug |
| **File** | `crates/llm/src/lib.rs` — `From<ModelProvider>` impl |

**Description**: The `From<ModelProvider> for Provider` implementation maps `ModelProvider::Candle` to `Provider::OpenRouter`. Candle is a local inference engine; OpenRouter is a cloud API. A user selecting `candle` as their backend would unknowingly route traffic to a remote service.

**Suggested fix**: Map `Candle` to a dedicated `Provider::Candle` variant, or return an error if candle isn't compiled in.

---

## 3. Medium — Logic Errors, Duplication, and Design Debt

### 3.1 — `passive_interval_hours` Set From Turn Count

| Field | Value |
|---|---|
| **File** | `crates/runtime/src/server/connection.rs` — ~line 1016 |

**Description**: The `SleepStatusPayload` field `passive_interval_hours` is set from `auto_sleep_turn_interval`, which is a turn count, not hours. The UI or caller would interpret this as a time-based interval, when it's actually counting turns.

### 3.2 — Three Overlapping HTTP-Fetch Tools

| Field | Value |
|---|---|
| **Files** | `builtins/browse.rs`, `builtins/web_browse.rs`, `builtins/web.rs` |

**Description**: Three tools all fetch web pages and are registered simultaneously:

| Name | Struct | File |
|---|---|---|
| `browse_page` | `BrowsePageTool` | `browse.rs` |
| `web_browse` | `WebBrowseTool` | `web_browse.rs` |
| `fetch_page` | `FetchPageTool` | `web.rs` |

`BrowsePageTool`'s doc-comment states it *"Combines the best of the old `fetch_page` and `web_browse` tools."* — meaning the other two are superseded. The LLM sees three near-identical tool descriptions and picks unpredictably.

**Suggested fix**: Remove `WebBrowseTool` and `FetchPageTool` registrations from `default_registry()`.

### 3.3 — Massive Duplication in `connection.rs`

| Field | Value |
|---|---|
| **File** | `crates/runtime/src/server/connection.rs` (1,078 lines) |

**Description**: The native and legacy code paths duplicate ~400 lines of nearly identical logic:
- Auto-sleep turn counter and sleep checks (~30 lines × 2)
- Turn finalization and memory recording (~100 lines × 2)
- `RunSleepCycle` / `RunMultiAgentSleepCycle` handlers (~50 lines × 2)
- Proactive agent loop logic overlaps with `sleep.rs` (~130 lines)

**Suggested fix**: Extract shared logic into helper functions. The native and legacy paths should only differ in how they obtain the LLM response, not in how they handle memory, tools, or sleep.

### 3.4 — Index Grows Unboundedly / No Removal on Delete

| Field | Value |
|---|---|
| **File** | `crates/memory/src/index.rs` |

**Description**: (a) `tier_index` inserts entries on `upsert` but never deduplicates by ID — the same entry can appear multiple times. (b) The index has no `remove()` method, so deleting an entry from the store leaves stale references in the index. Over time this causes memory bloat and degraded search quality.

### 3.5 — `compact_episodic` and `wipe_tiers` Consistency Risks

| Field | Value |
|---|---|
| **File** | `crates/memory/src/manager/maintenance.rs`, `manager/mod.rs` |

**Description**: `wipe_tiers` modifies the in-memory store before attempting to overwrite the event log file. If the file write fails, memory is inconsistent. `compact_episodic` has the same in-memory vs. disk divergence risk.

### 3.6 — `provenance_hash` Always Placeholder

| Field | Value |
|---|---|
| **File** | `crates/memory/src/manager/mod.rs` — ~line 266 |

**Description**: Every memory entry gets `provenance_hash = "local-dev-placeholder"`. The provenance chain feature — intended to detect tampering or provide lineage — is entirely stubbed.

### 3.7 — `infer_valence` Never Called; Valence Always 0.0

| Field | Value |
|---|---|
| **File** | `crates/memory/src/sentiment.rs` |

**Description**: The `infer_valence()` function exists and returns meaningful sentiment values, but it's never called from any memory recording path. Every entry has `valence: 0.0`, making sentiment-based retrieval inert.

### 3.8 — Silently Swallowed Memory/Recording Errors

| Field | Value |
|---|---|
| **File** | `crates/runtime/src/server/connection.rs` |

**Description**: In the native path, `memory.record(...)` return values are discarded with `let _: Result<_, _> = ...`. Assistant reply recording errors are similarly swallowed. Failures to persist memories are invisible.

### 3.9 — Provider Selection Duplicated 6+ Times

| Field | Value |
|---|---|
| **File** | `crates/runtime/src/server/connection.rs`, throughout |

**Description**: The pattern `if provider == "openrouter" { ... }` appears 6+ times for determining which API to call. This should be a single method on a provider type.

### 3.10 — Config Env Overrides Documented But Not Implemented

| Field | Value |
|---|---|
| **File** | `config/default.toml`, `crates/config/src/lib.rs` |

**Description**: Config comments say environment variables like `TAVILY_API_KEY`, `SEARXNG_BASE_URL`, `SERPER_API_KEY`, `EXA_API_KEY` can override config values, but the config crate has no code to read these environment variables or merge them.

### 3.11 — Safety Defaults Contradictory

| Field | Value |
|---|---|
| **File** | `config/default.toml` |

**Description**: `approval_required = true` and `approval_mode = "autonomous"` coexist. The code default for `allow_shell` is `false` (safe), but the config file sets `allow_shell = true`. A new user cloning the repo gets a configuration that contradicts itself.

### 3.12 — UTF-8 Boundary Panic in WASM read-file Tool

| Field | Value |
|---|---|
| **File** | `extensions/tools-src/read-file/src/lib.rs` |

**Description**: `&content[..max_bytes]` slices a UTF-8 string at an arbitrary byte position. If `max_bytes` falls on a multi-byte character boundary, this panics at runtime.

**Suggested fix**: Use `content.char_indices()` to find the nearest valid boundary, or `content.floor_char_boundary(max_bytes)` (nightly) / a manual equivalent.

### 3.13 — Telegram Model-Listing Duplication

| Field | Value |
|---|---|
| **Files** | `crates/interfaces/telegram/src/lib.rs` vs `crates/interfaces/cli/src/interactive.rs` |

**Description**: The Telegram crate reimplements `ModelProviderFilter`, `parse_model_provider_from_command()`, and `collect_model_lines()` — all functionally identical to the CLI versions.

---

## 4. Low — Dead Code, Stale Docs, and Hygiene

### 4.1 — Dead Code in Runtime Crate

| Item | Location |
|---|---|
| `TurnSource` enum | `agent_loop.rs` — defined/exported, never used |
| `EvalScore` struct | `agent_loop.rs` — defined/exported, never used |
| `ServerEvent::ToolInfoList` | `commands.rs` — variant never constructed |
| `run_proactive_check_from_summaries` | `proactive.rs` — defined, never called |
| `stream_turn` (deprecated) | `tools.rs` — marked deprecated, has duplicate doc-comments |
| `AgentRuntime::run()` | No-op method, never called in production |
| `respond_and_remember` (non-streaming) | Only used in test code |

### 4.2 — Dead Code in Memory Crate

| Item | Location |
|---|---|
| `SOURCE_TRIGGER` constant | `lib.rs` — defined, never referenced |
| `decay_stale_semantic` | `maintenance.rs` — defined, never called |
| `format_user_profile_block` / `user_profile_entries` | `profile.rs` — defined, never called externally |

### 4.3 — Dead Code in TUI Crate

| Item | Location |
|---|---|
| `Action` enum (16 variants) | `action.rs` — never imported or referenced |
| `AgentPanel` struct | `agent_panel.rs` — never instantiated |
| `AdvancedTui` type alias | `lib.rs` — both cfg branches alias to `App`, never used |

### 4.4 — Cosine Similarity Duplicated

| Field | Value |
|---|---|
| **Files** | `crates/memory/src/scorer.rs` and `crates/memory/src/retrieval.rs` |

**Description**: `cosine_similarity()` is implemented identically in two files. Should be a single utility function.

### 4.5 — Sleep/Multi-Sleep Format Duplication

| Field | Value |
|---|---|
| **Files** | `crates/memory/src/sleep.rs` and `multi_sleep.rs` |

**Description**: Sleep response formatting and memory block construction are duplicated between the single and multi-agent sleep paths.

### 4.6 — Stale Documentation

| Document | Issue |
|---|---|
| `MIGRATION.md` | References `crates/audit/` and `crates/core/` — both are present but may have drifted from the documented structure |
| `docs/phase-review-gate.md` | References a non-existent `lancedb` backend |

### 4.7 — MSRV Mismatch

| Field | Value |
|---|---|
| **File** | `clippy.toml` |

**Description**: Claims MSRV 1.85, but the codebase uses APIs available only from 1.89+ (e.g. `split_once` flagged by clippy). The actual minimum is likely 1.89 or higher.

### 4.8 — Other Hygiene Issues

| Issue | Location |
|---|---|
| `duration_ms: 0` hardcoded | Legacy tool path in `connection.rs` |
| Float precision artifact | `config/default.toml`: `0.30000001192092896` instead of `0.3` |
| `deny.toml` missing `[advisories]` | Root `deny.toml` |
| WASM tools use edition 2021 | `extensions/tools-src/*/Cargo.toml` — workspace uses edition 2024 |
| Dead `extern "C"` block | `extensions/tools-src/run-shell/src/lib.rs` |
| `Provider` / `ModelProvider` duplication | `crates/llm/src/lib.rs` — two nearly identical enums |
| Index doc says "bincode" | `crates/memory/src/index.rs` — actually uses serde_json |
| Redundant `event_id` field | `MemoryRecordEvent` in `event_log.rs` |
| `request_events` missing variants | Missing terminal conditions for `ProactiveStats`/`SleepStatus` |
| `ToolCommands::Status` hardcoded list | `main.rs` — `KNOWN_TOOLS` is a static `&[&str]` of 8 names; new tools not shown |
| `tool_chain.rs` stale TODO | Line 262 — stubbed self-evaluation never implemented |

---

## 5. Clippy Warnings

14 unique warnings across the workspace (none fatal):

| Warning | Count | Location(s) |
|---|---|---|
| Manual `split_once` | 3 | Various string parsing sites |
| Too many function arguments | 4 | `respond_and_remember_stream`, etc. |
| MSRV-gated API usage | 2 | Uses 1.89+ APIs with 1.85 MSRV |
| Redundant closure | 1 | `.map_or()` could be `.unwrap_or()` |
| `map_or` preference | 1 | Could use `is_some_and` |
| Empty line after doc comment | 1 | Minor formatting |
| Borrowed expression | 1 | Unnecessary `&` |
| `contains()` on range | 1 | Could use `matches!` |
| Large `Result::Err` variant | 1 | Consider boxing |

---

## 6. Positive Findings

| Area | Notes |
|---|---|
| **CLI structure** | `main.rs` properly delegates via `mod daemon;`. No duplicate function bodies between main/daemon/interactive/memory_cmds. |
| **Error handling (exec)** | `wasm.rs` has zero production `unwrap()` calls. `sandbox.rs` uses proper `Result` propagation. |
| **Error handling (tools)** | All tool `run()` impls return `Result`. `ToolRegistry` lock usage is idiomatic. |
| **Error handling (CLI)** | All async command handlers propagate errors via `?`. |
| **SSRF in BrowsePageTool** | `browse.rs` has comprehensive `is_private_ip()` covering all private/reserved ranges. |
| **Feature gating** | `sandbox` and `wasm` features properly `#[cfg]`-guarded throughout. |
| **Auto-sleep bug fix** | The fix from commit f25a022 is correctly applied — `SleepCycleRunning`/`Done` are not broadcast in the native path. |
| **Test coverage** | 352 tests passing, covering runtime, memory, tools, and exec crates. |
| **Zero `todo!()` / `unimplemented!()`** | No runtime-panicking stub macros anywhere in the codebase. |
| **Zero unused imports** | Compiler reports no unused import warnings. |

---

## 7. Summary

| Severity | Count | Key Findings |
|---|---|---|
| **CRITICAL** | 2 | API key in VCS (§1.1), SSRF bypass (§1.2) |
| **HIGH** | 5 | `std::mem::take` amnesia (§2.1), `record_belief` (§2.2), forgetting pass (§2.3), candle compile (§2.4), candle→openrouter (§2.5) |
| **MEDIUM** | 13 | connection.rs duplication (§3.3), index growth (§3.4), config contradictions (§3.10-3.11), UTF-8 panic (§3.12), etc. |
| **LOW** | ~20 | Dead code (~15 items across runtime/memory/TUI), stale docs, clippy warnings, hygiene |

### Recommended Priority Order

1. **Rotate the Brave API key** — immediate, independent of code changes
2. **Remove `WebBrowseTool` and `FetchPageTool`** — eliminates SSRF and tool overlap in one step
3. **Fix `record_belief` confidence persistence** — small targeted fix
4. **Fix `run_forgetting_pass` event log persistence** — small targeted fix
5. **Address `std::mem::take` amnesia** — requires architectural decision (Arc, snapshot, or clone)
6. **Fix candle feature** — if candle support is desired; otherwise remove the feature flag
7. **Refactor `connection.rs`** — extract shared native/legacy logic into helpers
8. **Clean dead code** — batch removal of unused types/functions
9. **Fix config defaults and docs** — straightforward but low-risk
