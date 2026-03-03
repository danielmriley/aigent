# Aigent Reform 2026 — Architectural Overhaul Plan
Commit baseline: 67497fc0a6091314291a6154a11a6d5d7fa7e097
Date: March 2026

## Goals (in order)
1. Cut end-to-end latency for simple questions from ~8–12s → <3s
2. Make the codebase understandable by a new contributor in <30 minutes
3. Enable safe addition of marketplace + multi-agent without breaking everything
4. Keep 100% of current functionality (no regression)

## What We Will NOT Do
- Full rewrite from scratch (loses momentum + reintroduces every bug we already fixed)
- Keep `runtime` as a god crate
- Keep verbose prompts or duplicated logic
- Keep Python refactoring scripts in git history

## Current Strengths (Preserve)
- External thinker state machine (now works)
- 6-tier memory
- WASM skills + ToolRegistry
- TUI + event system
- Ollama JSON mode + activity timeout

## Known Anti-Patterns to Eliminate Forever
- Giant `write!` macros for prompts
- Hardcoded date/month strings in challenge logic
- Re-sending full system prompt on every thinker step
- `should_challenge` blocks with 50+ string patterns
- Python one-liners to edit Rust code
- Dead `use_native_calling` paths
- String-contains tests for prompt content

## Phase 0: Immediate Wins (Today–Tomorrow, 3 files, ~2 hours)
1. Fast-path bypass for trivial queries (date, list skills, etc.) in `connection.rs`
2. Delete 80% of the `should_challenge` block in `ext_think.rs` (keep only first-step guard)
3. Finalize `CURRENT_DATETIME` injection + explicit "NEVER call run_shell for date" rule in thinker prompt
4. Add native `get_current_datetime` tool (10 lines)

→ Expected result: "What day is it today?" becomes <3s with zero tools.

## Phase 1: Extract the Thinker (Next 3–5 days — highest ROI)
Create new crate: `crates/thinker`
- `external.rs` ← move entire `run_external_thinking_loop`
- `prompt.rs` ← dedicated thinker prompt builder (separate from main prompt)
- `state_machine.rs` + `parser.rs` + `types.rs`

`runtime` depends on `thinker::ExternalThinker::run(...)`

This alone makes `ext_think.rs` disappear and the loop unit-testable.

## Phase 2: Prompt Engine Service (Week 2)
New crate: `crates/prompt`
- `PromptEngine` struct with template registry + memory compressor
- `build_for_turn(inputs)` → clean, cacheable, testable
- Kill `prompt_builder.rs` entirely

## Phase 3: Runtime Split + Polish (Week 3)
- Split `runtime` → `aigent-agent` + `aigent-server`
- Delete legacy paths
- Add conversation summarizer (every 8 turns)
- Proper error types (`thiserror`)
- Integration test suite for external mode
- Benchmarks (latency + token usage)

## File Cleanup List (Do These in Phase 0)
- Delete: `install.sh.bak`, old verbose prompt tests, dead `prose_tool_specs` code
- Archive: Any remaining Python refactoring scripts
- Move: All `ext_think_tests` → proper integration tests under `tests/`
- Create: `ARCHITECTURE.md` + `PERFORMANCE.md`

## Success Metrics After Reform
- Simple question latency <3s on Qwen 3.5 35B
- `cargo test` passes in <15s
- New contributor can understand core loop from `thinker/lib.rs` + `prompt/lib.rs`
- Can add `browse_page` tool in <30 minutes without touching 5 files

## Timeline Commitment
- Phase 0: complete by end of weekend
- Phase 1: complete by next weekend
- No new features until Phase 1 ships
