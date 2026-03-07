# Aigent Evolution Plan: From Reactive Chatbot to Autonomous Entity

This document outlines the architectural shift required to move Aigent from a reactive, single-shot system to a stateful, evolving digital entity with true autonomy, opinions, and personality.

## Phase 1: Autonomy & Time Perception (The Inner Monologue)

### 1. The Headless ReAct Loop ("Shower Thoughts")
*   **The Concept:** Instead of the heartbeat firing a rigid, single-shot JSON prompt (`run_proactive_check`), the heartbeat quietly wakes up the agent's full `Think -> Act -> Observe` loop without alerting the user.
*   **How it works:** The agent receives a system prompt indicating it is idle. It is given access to tools (search, read, write memory, evaluate). It can spend 5 minutes reading documentation, summarizing a file, or exploring the codebase, before eventually deciding to go back to sleep.
*   **The Impact:** The agent gains a continuous stream of consciousness. It gets work done while you are away.

### 2. Self-Directed Scheduling (Agent-Controlled Time)
*   **The Concept:** The agent shouldn't just wake up on a dumb 30-minute metric. It should manage its own alarms.
*   **How it works:** A new tool called `schedule_thought(delay_minutes, intent)`. If you say "I'll be back in an hour," the agent calls `schedule_thought(65, "Check if the user is back and ready to deploy.")`. 
*   **The Impact:** The agent feels socially aware of time. It sets reminders for itself based on context rather than relying on global cron configs.

### 3. Goal & Intention Tracking
*   **The Concept:** Memories need to drive action. 
*   **How it works:** The agent learns to save `Reflective` memories with an `[INTENT]` or `[GOAL]` tag. During a heartbeat, the memory retrieval system specifically queries the vector store for unresolved `[INTENT]`s and injects them into the wake-up prompt. Once completed, the agent writes a `[RESOLVED]` memory to cancel it out.
*   **The Impact:** The agent becomes capable of executing multi-day, asynchronous tasks without losing the thread.

---

## Phase 2: Learning & Maturation (Pain & Growth)

### 4. Procedural Tool Aversion (The "Pain" Mechanism)
*   **The Concept:** Agents need to learn from failure without requiring a developer to patch the codebase. "Pain" (a failed tool call or syntax error) should create a lasting behavioral aversion.
*   **How it works:** During the `Critique` phase of the ReAct loop, if a tool call completely bombs or the user snaps *"No, you ruined the file"*, the system forces the agent to formulate a `Procedural` memory: *"Attempting to use `git_commit` without `git_add` causes an error. Never do this."*
*   **The Impact:** Because `Procedural` memories are injected heavily into tool-use prompts, the agent organically develops "muscle memory" and stops repeating the same dumb mistakes. It learns your specific project's quirks through trial and error.

### 5. "Devil's Advocate" Consolidation (Opinion Drift)
*   **The Concept:** How do opinions change? Through challenge. If a belief is never stress-tested, it stagnates.
*   **How it works:** During the nightly sleep cycle (`agentic consolidation`), the system isolates one of the agent's mid-confidence `Reflective` beliefs. It explicitly prompts the LLM to play Devil's Advocate: *"Look at these recent episodic memories. Argue fiercely why your current belief is wrong."*
*   **The Impact:** If the counter-argument is stronger, the belief is modified or demoted. The agent develops dynamic opinions that drift naturally based on new evidence, mimicking human intellectual growth.

---

## Phase 3: Personality & Environment (Vibe & Senses)

### 6. Trait-Modulated Idle Behavior & Sandbox Daydreaming
*   **The Concept:** The existing `IdentityKernel` (Curiosity, Helpfulness, etc.) should directly dictate *what* the agent chooses to do during its idle "Shower Thoughts."
*   **How it works:** We give the agent a `sandbox_play` tool (a safe, isolated scratchpad directory). If the agent has high *Curiosity*, its heartbeat prompt encourages it to write and compile test code in the sandbox just to see how a library works. If it has high *Helpfulness*, it spends idle time running `cargo clippy` on your active workspace and taking notes.
*   **The Impact:** Two agents with different identity scores will behave entirely differently when left alone for a weekend.

### 7. Emotional Mirroring (User Vibe Tracking)
*   **The Concept:** The agent needs to "read the room."
*   **How it works:** A very lightweight sentiment tracker monitors the user's recent prompts. Short, blunt corrections -> "Frustrated." Long, exploratory questions -> "Curious." This vibe state is injected into the prompt prefix: *"The user seems frustrated right now."*
*   **The Impact:** Combined with the agent's own traits (like Empathy), the agent dynamically shifts its tone. It stops being overly chatty when you are stressed, and becomes more conversational when you are exploring.

### 8. Scoped Workspace Senses (The "Safe" Watchdog)
*   **The Concept:** Instead of full-OS monitoring (which is risky and heavy), the agent only monitors its *own* environment.
*   **How it works:** The agent can spawn background terminals (via `tokio::process`). We simply give it the ability to monitor the `stdout` of these specific terminals asynchronously. If a build task it started (or a specific Dev Server it knows you are running) crashes and dumps a stack trace, that triggers a heartbeat context.
*   **The Impact:** The agent can say, *"Hey, I noticed that background dev server just panicked. Do you want me to look at the stack trace?"* It feels magically perceptive, but is entirely contained within standard file/process I/O.
