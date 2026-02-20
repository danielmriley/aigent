# Aigent Phase 0–2 Memory-First Plan

This plan intentionally prioritizes memory as the core of agent helpfulness, continuity, and personality formation.

## Guiding Principles
- Memory is a first-class runtime component, not a peripheral feature.
- The agent uses two memory horizons: short-term conversation memory and long-term durable memory.
- Core personality evolves autonomously through sleep distillation, bounded by consistency and safety checks.
- Obsidian-like readability is delivered via markdown vault projection from canonical memory state.

## Phase 0 — Foundation + Onboarding (Completion)

### Goals
- Ensure onboarding seeds identity and memory policy correctly.
- Ensure install/run paths validate memory prerequisites.

### Tasks
1. Replace prompt-only onboarding with multi-step TUI onboarding wizard.
2. Add onboarding screens for:
   - bot name/personality seed
   - LLM provider/model/key setup
   - memory policy defaults (night sleep window, promotion strictness)
   - workspace and safety profile checks
3. Add startup health checks for memory path access and config validity.
4. Align config defaults with current architecture (`memory.backend = "eventlog"`).
5. Add onboarding completion summary including memory contract.

### Exit Criteria
- First run produces valid identity/core seed memory and complete config.
- User can see memory status and policy immediately after onboarding.

## Phase 1 — Obsidian-Like Memory Integration (Core Priority)

### Goals
- Keep canonical event-log memory while exposing human-readable, linked markdown memory.
- Improve retrieval quality and autonomous core-memory formation.

### Tasks
1. Keep canonical storage in append-only JSONL event log.
2. Build vault projection under `.aigent/vault/`:
   - tier-specific indexes
   - per-topic notes with frontmatter
   - wiki-links/backlinks
   - daily memory notes
3. Add memory retrieval improvements:
   - weighted blend of recency + semantic relevance + core priority
   - strict inclusion of high-confidence core memories
4. Refine sleep distillation:
   - nightly schedule by local time window
   - explicit promotion reasons and provenance
   - one-cycle-per-night behavior
5. Add memory inspection commands:
   - `memory stats`
   - `memory inspect core`
   - `memory export-vault`
   - `memory promotions`

### Exit Criteria
- Agent demonstrates cross-session recall improvements.
- Core memories evolve with clear provenance.
- Vault stays in sync with canonical memory state.

## Phase 2 — Unified Agent Loop + Channel Parity (Completion)

### Goals
- Ensure TUI and Telegram share the same memory semantics and behavior.
- Ensure daemon loop handles chat + sleep + persistence coherently.

### Tasks
1. Implement full daemon event loop for:
   - TUI turns
   - Telegram turns
   - scheduled sleep cycles
   - graceful shutdown persistence
2. Ensure prompt construction always includes:
   - environment context
   - recent conversation memory
   - long-term memory retrieval
3. Implement Telegram parity for core commands (`status`, `context`, memory actions).
4. Add loopiness controls:
   - de-dup response heuristics
   - repeated-context suppression
5. Add parity smoke tests and consistency checks.

### Exit Criteria
- Behavior and memory continuity are consistent between TUI and Telegram.
- Sleep and memory updates run predictably in long-lived sessions.

## Review Gate Before Phase 3
- All Phase 0–2 exit criteria met.
- Memory retrieval and promotion quality manually verified with multi-day test script.
- Audit summary produced for memory writes/promotions/wipes.
