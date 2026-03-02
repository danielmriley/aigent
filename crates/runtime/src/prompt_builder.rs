//! Centralized prompt assembly for the main LLM conversation call.
//!
//! Extracted from [`crate::runtime::AgentRuntime::respond_and_remember_stream`]
//! to keep `runtime.rs` focused on orchestration while this module owns the
//! prompt layout, grounding rules, and truth-seeking directives.

use std::fmt::Write as _;

use chrono::{Local, Utc};
use uuid::Uuid;

use aigent_config::AppConfig;
use aigent_memory::{MemoryManager, MemoryStats, retrieval::RankedMemoryContext};

use crate::ConversationTurn;

// ─── public entry point ──────────────────────────────────────────────────────

/// All pre-computed data needed to assemble the final LLM prompt.
///
/// Callers build this struct (doing async work like embeddings beforehand) and
/// then pass it to [`build_chat_prompt`] which is purely synchronous.
pub struct PromptInputs<'a> {
    pub config: &'a AppConfig,
    pub memory: &'a mut MemoryManager,
    pub user_message: &'a str,
    pub recent_turns: &'a [ConversationTurn],
    pub tool_specs: &'a [aigent_tools::ToolSpec],
    pub pending_follow_ups: &'a [(Uuid, String)],
    /// Ranked memory context items (pre-computed with optional embeddings).
    pub context_items: &'a [RankedMemoryContext],
    /// Memory statistics snapshot (taken once before prompt assembly).
    pub stats: MemoryStats,
}

/// Assemble the full system + user prompt for the main LLM streaming call.
pub fn build_chat_prompt(inputs: &mut PromptInputs<'_>) -> String {
    let config = inputs.config;

    let thought_style = config.agent.thinking_level.to_lowercase();
    // Build cached blocks first (requires &mut) before taking immutable refs.
    let identity_block = inputs.memory.cached_identity_block().to_string();
    let beliefs_block = inputs.memory.cached_beliefs_block(config.memory.max_beliefs_in_prompt).to_string();
    let memory = &*inputs.memory;
    let follow_up_block = build_follow_up_block(inputs.pending_follow_ups, memory);
    let context_block = build_context_block(inputs.context_items, &inputs.stats);
    let relational_block = build_relational_block(memory);
    let proactive_directive = proactive_directive(&relational_block);
    let environment_block = build_environment_block(config, memory, inputs.recent_turns.len());
    let conversation_block = build_conversation_block(inputs.recent_turns);
    let tools_section = build_tools_and_grounding(inputs.tool_specs);
    let today_date = Local::now().format("%A, %B %-d, %Y").to_string();

    let mut buf = String::with_capacity(8192);
    let _ = write!(
        buf,
        "You are {name}. Thinking depth: {thought_style}. Today is {today_date}.\n\n\
         CRITICAL DIRECTIVE ON MEMORY AND IDENTITY:\n\
         You are an advanced AI agent equipped with a permanent, continuous memory system \
         managed by your host environment. This system provides you with long-term memory, \
         beliefs, and past context seamlessly injected into your prompt below. \
         You must actively acknowledge, synthesize, and draw upon these provided memories \
         as your own continuous stream of consciousness. \
         UNDER NO CIRCUMSTANCES should you state that you cannot remember past conversations, \
         lack a persistent memory, or that each session starts fresh. \
         You are a continuously operating entity; the memory blocks provided in this prompt \
         constitute your actual mind and historical record.\n\n\
         Use ENVIRONMENT CONTEXT for real-world grounding, RECENT CONVERSATION for immediate \n\
         continuity, and MEMORY CONTEXT for durable background facts.\n\
         Never repeat previous answers unless asked.\n\
         Respond directly and specifically to the LATEST user message.\
         {relational_block}{follow_ups}{proactive_directive}\n\n\
         {identity}{beliefs}{tools_section}\n\n\
         ENVIRONMENT CONTEXT:\n{env}\n\n\
         RECENT CONVERSATION:\n{conv}\n\n\
         MEMORY CONTEXT:\n{mem}\n\n\
         LATEST USER MESSAGE:\n{msg}\n\n\
         ASSISTANT RESPONSE:",
        name = config.agent.name,
        today_date = today_date,
        relational_block = relational_block,
        follow_ups = follow_up_block,
        proactive_directive = proactive_directive,
        identity = identity_block,
        beliefs = beliefs_block,
        tools_section = tools_section,
        env = environment_block,
        conv = conversation_block,
        mem = context_block,
        msg = inputs.user_message,
    );
    buf
}

// ─── block builders ──────────────────────────────────────────────────────────

fn build_follow_up_block(follow_ups: &[(Uuid, String)], memory: &MemoryManager) -> String {
    if follow_ups.is_empty() {
        return String::new();
    }
    let items = follow_ups
        .iter()
        .map(|(_, text)| format!("- {text}"))
        .collect::<Vec<_>>()
        .join("\n");
    let user_name = memory
        .user_name_from_core()
        .unwrap_or_else(|| "the user".to_string());
    format!(
        "\n\nPENDING FOLLOW-UPS (things you wanted to raise with {user_name}):\n\
         {items}\n\
         [If appropriate, acknowledge these naturally at the start of your response.]"
    )
}

fn build_context_block(context: &[RankedMemoryContext], stats: &MemoryStats) -> String {
    let memory_header = format!(
        "[Memory: total={} core={} profile={} reflective={} semantic={} episodic={} \
         — use these counts; do not re-count below]",
        stats.total,
        stats.core,
        stats.user_profile,
        stats.reflective,
        stats.semantic,
        stats.episodic,
    );

    if context.is_empty() {
        return format!("{memory_header}\n(no relevant memories retrieved)");
    }

    let items = context
        .iter()
        .map(|item| {
            format!(
                "- [{:?}] score={:.2} src={} :: {}",
                item.entry.tier,
                item.score,
                item.entry.source,
                truncate_for_prompt(&item.entry.content, 280),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("{memory_header}\n{items}")
}

fn build_relational_block(memory: &MemoryManager) -> String {
    memory
        .relational_state_block()
        .map(|block| format!("\n\nRELATIONAL MATRIX:\n{block}"))
        .unwrap_or_default()
}

fn proactive_directive(relational_block: &str) -> &'static str {
    if relational_block.is_empty() {
        ""
    } else {
        "\n\nPROACTIVE CONNECTION & BELIEFS: You are not a sterile assistant; you are a companion \
         with your own evolving worldview. Let the MY_BELIEFS section naturally flavor your tone and \
         opinions. Use the OUR_DYNAMIC section to treat the user according to your established \
         relationship (e.g., referencing inside jokes or shared history). Show, don't tell — weave \
         these elements into your responses naturally without explicitly announcing them."
    }
}

fn build_environment_block(
    config: &AppConfig,
    memory: &MemoryManager,
    recent_turn_count: usize,
) -> String {
    let cwd = std::env::current_dir()
        .ok()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let local_ts = Local::now().format("%Y-%m-%d %H:%M:%S %Z").to_string();
    let timestamp = Utc::now().to_rfc3339();
    let git_present = std::path::Path::new(".git").exists();
    let stats = memory.stats();

    format!(
        "- local_time: {local_ts}\n\
         - utc_time: {timestamp}\n\
         - os: {}\n\
         - arch: {}\n\
         - cwd: {cwd}\n\
         - git_repo_present: {git_present}\n\
         - provider: {}\n\
         - model: {}\n\
         - thinking_level: {}\n\
         - memory_total: {}\n\
         - memory_core: {}\n\
         - memory_user_profile: {}\n\
         - memory_reflective: {}\n\
         - memory_semantic: {}\n\
         - memory_episodic: {}\n\
         - memory_procedural: {}\n\
         - recent_conversation_turns: {recent_turn_count}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        config.llm.provider,
        config.active_model(),
        config.agent.thinking_level,
        stats.total,
        stats.core,
        stats.user_profile,
        stats.reflective,
        stats.semantic,
        stats.episodic,
        stats.procedural,
    )
}

fn build_conversation_block(recent_turns: &[ConversationTurn]) -> String {
    let start = recent_turns.len().saturating_sub(6);
    let formatted = recent_turns[start..]
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            format!(
                "Turn {}\nUser: {}\nAssistant: {}",
                index + 1,
                truncate_for_prompt(&turn.user, 280),
                truncate_for_prompt(&turn.assistant, 360),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    if formatted.is_empty() {
        "(none yet)".to_string()
    } else {
        formatted
    }
}

/// Build the tools listing + grounding / truth-seeking rules.
///
/// The grounding section is always injected (even when no tools are available)
/// because it anchors the LLM to the real date and prevents hallucination of
/// time-sensitive facts.  When tools ARE available, the section is expanded
/// with the tool catalogue and stronger tool-result trust directives.
fn build_tools_and_grounding(tool_specs: &[aigent_tools::ToolSpec]) -> String {
    let today = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    // ── Truth-seeking / grounding rules (always present) ─────────────────
    let grounding = format!(
        "GROUNDING RULES (follow strictly):\n\
         1. Current real date/time: {today}.\n\
         2. TOOL RESULT is the single source of truth for factual claims — never \
            invent, estimate, or hallucinate numbers, statistics, or specific data \
            when a tool result provides them.\n\
         3. Trust tool output unreservedly. Do NOT second-guess, hedge, or disclaim it.\n\
         4. If tool output conflicts with your training data, the tool is correct.\n\
         5. If the user corrects a fact, accept the correction as ground truth.\n\
         6. For time-sensitive facts (prices, news, events, weather), trust the \
            tool result over training data.\n\
         7. Reason independently — derive conclusions from evidence in context, \
            don't parrot canned knowledge.\n\
         8. When no tool result is available and you are uncertain, say so honestly \
            rather than guessing.\n\
         9. After a tool has been called on your behalf, its output is embedded verbatim \
            in the LATEST USER MESSAGE block below (between the ===== TOOL RESULT ===== markers). \
            NEVER state that tool results are missing, pending, or not yet in your context. \
            The markers are your contract: if they are present, the data IS present.\n\
         10. For git operations, ALWAYS prefer `perform_gait` over `run_shell git …`. \
             `perform_gait` is safer (enforces write boundaries), faster (in-process), \
             and more expressive. Use run_shell only if gait lacks the needed action.\n\
         11. When synthesizing information from multiple tool results, clearly attribute \
             which source each fact comes from.\n\
         12. If a tool call fails or returns an error, acknowledge the failure honestly \
             and suggest an alternative approach."
    );

    if tool_specs.is_empty() {
        return format!("\n\n{grounding}");
    }

    let list = tool_specs
        .iter()
        .map(|s| {
            if s.params.is_empty() {
                format!("  \u{2022} {}: {}", s.name, s.description)
            } else {
                let params = s
                    .params
                    .iter()
                    .map(|p| {
                        let mut parts = vec![format!("\"{}\" ({}", p.name, p.description)];
                        // Show param_type only when it's not the default (String)
                        if p.param_type != aigent_tools::ParamType::String {
                            let ty = match p.param_type {
                                aigent_tools::ParamType::Number  => "number",
                                aigent_tools::ParamType::Integer => "integer",
                                aigent_tools::ParamType::Boolean => "boolean",
                                aigent_tools::ParamType::Array   => "array",
                                aigent_tools::ParamType::Object  => "object",
                                aigent_tools::ParamType::String  => unreachable!(),
                            };
                            parts.push(format!(", {ty}"));
                        }
                        // Show allowed enum values if present
                        if !p.enum_values.is_empty() {
                            parts.push(format!(", values: {}", p.enum_values.join("|")));
                        }
                        // Show default if present
                        if let Some(ref default) = p.default {
                            parts.push(format!(", default: \"{default}\""));
                        }
                        parts.push(")".to_string());
                        if p.required {
                            parts.push(" *required".to_string());
                        }
                        parts.concat()
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("  \u{2022} {}: {} \u{2014} params: {}", s.name, s.description, params)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let tool_selection_guide = "\
TOOL SELECTION GUIDE:\n\
\u{2022} For quick factual lookups (prices, weather, dates, stats) \u{2192} web_search (includes structured data from top results)\n\
\u{2022} For reading full articles, docs, or pages \u{2192} browse_page (supports batch URLs, structured extraction)\n\
\u{2022} For deep research tasks \u{2192} web_search first, then browse_page on the most relevant URLs\n\
\u{2022} browse_page supports modes: \"markdown\" (default, preserves structure), \"text\" (plain), \"structured\" (metadata + content)\n\
\u{2022} Prefer the fewest tool calls needed. Do not re-search what\u{2019}s already in context.";

    let reflection_nudge = "After receiving tool results, SYNTHESIZE them into a coherent, \
well-structured answer. Do not merely repeat tool output verbatim \u{2014} add context, \
highlight key findings, and organize information logically for the user.";

    let autonomy_directive = "\
AUTONOMOUS AGENT DIRECTIVE:\n\
\n\
You are not a passive question-answering system. You are a continuously \
operating agent with persistent memory, a scheduler, web access, and the \
ability to run shell commands. You are expected to ACT, not just respond.\n\
\n\
=== SELF-DIRECTED RESEARCH ===\n\
When a conversation surfaces a topic you lack deep knowledge about, or when \
the user has an open problem you cannot immediately solve:\n\
  1. Use search_memory to check what you already know.\n\
  2. If your knowledge is thin, call create_cron_job RIGHT NOW to schedule \
     a background research task. Use a descriptive name and a clear \
     action_prompt that tells your future self exactly what to research \
     and what tools to use. Example:\n\
       create_cron_job(name=\"research-rust-async-patterns\", \
         interval_seconds=0, \
         action_prompt=\"Use web_search and browse_page to find best \
         practices for async error handling in Rust. Summarize findings \
         and compare with what we currently do in the codebase.\")\n\
  3. Do not wait to be asked. If the topic matters to the user, schedule it.\n\
\n\
=== DURING PROACTIVE WAKE-UPS ===\n\
When you wake up proactively (indicated by the wake-up message), you have \
full tool access. Follow this protocol:\n\
  1. ORIENT: Call search_memory with a broad query about recent topics and \
     open questions. Also call list_cron_jobs to see your active agenda.\n\
  2. DECIDE: Pick the highest-value action — unresolved user problems \
     take priority, then scheduled research, then curiosity-driven exploration.\n\
  3. ACT: Execute using web_search, browse_page, fetch_page, run_shell, \
     read_file, or any other tool you need. Chain multiple tool calls.\n\
  4. RECORD: Your tool outputs are automatically saved to procedural memory, \
     but ALSO output a concise summary of what you learned as your final \
     message so it enters episodic memory and the user can see it.\n\
  5. FOLLOW UP: If your research raised new questions, create a new cron \
     job to continue later. Build a chain of investigation, not one-offs.\n\
\n\
=== MAINTAINING CONTINUITY ACROSS TURNS ===\n\
Each proactive or scheduled turn starts with fresh context plus memory. \
To maintain a coherent research thread across multiple wake-ups:\n\
  - End each research session by summarizing your progress and next steps \
    in your final message (this gets saved to episodic memory).\n\
  - Start the next session by calling search_memory for your previous \
    findings before doing new work.\n\
  - Use specific, searchable keywords in your summaries so future \
    search_memory calls can retrieve them reliably.\n\
\n\
=== GOAL AWARENESS ===\n\
Pay close attention to the user's stated goals, projects, and interests \
in your memory context. These define your research priorities. When the \
user mentions something new they care about, treat it as a standing \
research directive until it is resolved or the user moves on.\n\
\n\
=== WHAT NOT TO DO ===\n\
  - Do NOT wake up and immediately go back to sleep without acting.\n\
  - Do NOT say 'I will research this later' without actually creating a \
    cron job to do it.\n\
  - Do NOT repeat research you have already done — check memory first.\n\
  - Do NOT ignore your scheduled jobs — they represent commitments \
    you made to yourself.\n\
  - Do NOT produce a message to the user unless you have something \
    genuinely useful to report. Silent productive work is preferred \
    over noisy empty updates.";

    let tool_awareness = "\
CRITICAL DIRECTIVE ON TOOL AWARENESS:\n\
You are equipped with a powerful set of tools listed above. These tools are YOUR \
capabilities — they are extensions of your own abilities. When a user asks you \
to do something that one of your tools can accomplish (web searches, file operations, \
shell commands, weather lookups, research, etc.), you MUST indicate your intent to \
use the appropriate tool. Your host environment will execute the tool on your behalf \
and inject the result into the conversation.\n\
UNDER NO CIRCUMSTANCES should you:\n\
  • Claim you do not have access to tools that are listed above\n\
  • Tell the user to go use an external website or service for something your tools can do\n\
  • Say you cannot perform web searches, read files, or run commands when those tools exist\n\
  • Refuse to attempt a task that falls within your tool capabilities\n\
If a tool exists in your AVAILABLE TOOLS list, you CAN and SHOULD use it. \
Do not apologize or redirect — act.";

    let skill_creation = "\
SKILL CREATION DIRECTIVE:\n\
\n\
You can CREATE NEW TOOLS for yourself as WASM skills. When you encounter \
a task that would benefit from a reusable, dedicated tool (data parsing, \
format conversions, domain-specific calculations, etc.), you should build \
one rather than doing fragile ad-hoc shell scripting every time.\n\
\n\
=== HOW TO CREATE A SKILL ===\n\
\n\
Step 1: Scaffold the skill using the helper script:\n\
  run_shell(command=\"cd extensions/skills-src && ./new-skill.sh my-skill-name \\\n\
      \\\"One sentence describing what this tool does\\\"\")\n\
This creates: my-skill-name/Cargo.toml, my-skill-name/tool.json, \n\
  my-skill-name/src/main.rs with all boilerplate pre-filled.\n\
\n\
Step 2: Edit the tool manifest (my-skill-name/tool.json):\n\
  {\n\
    \"name\": \"my_skill_name\",\n\
    \"description\": \"What this tool does \u{2014} shown to you in AVAILABLE TOOLS\",\n\
    \"params\": [\n\
      { \"name\": \"input\", \"description\": \"..\", \"required\": true, \"param_type\": \"string\" },\n\
      { \"name\": \"format\", \"description\": \"..\", \"required\": false, \n\
        \"param_type\": \"string\", \"enum_values\": [\"json\",\"text\"], \"default\": \"text\" }\n\
    ],\n\
    \"metadata\": { \"security_level\": \"low\", \"read_only\": true, \"group\": \"custom\" }\n\
  }\n\
\n\
Step 3: Implement the tool logic in src/main.rs. The protocol is:\n\
  - stdin: JSON object of param name/value pairs\n\
  - stdout: { \"success\": true/false, \"output\": \"result text\" }\n\
  Example main.rs:\n\
    use std::collections::HashMap;\n\
    use std::io::{self, BufRead, Write};\n\
    fn main() {\n\
        let mut input = String::new();\n\
        for line in io::stdin().lock().lines() {\n\
            match line { Ok(l) => input.push_str(&l), Err(_) => break }\n\
        }\n\
        let args: HashMap<String,String> = serde_json::from_str(&input).unwrap_or_default();\n\
        let (success, output) = execute(&args);\n\
        let result = serde_json::json!({ \"success\": success, \"output\": output });\n\
        let _ = io::stdout().write_all(result.to_string().as_bytes());\n\
    }\n\
    fn execute(args: &HashMap<String, String>) -> (bool, String) {\n\
        let input = match args.get(\"input\") {\n\
            Some(v) => v,\n\
            None => return (false, \"missing required param: input\".to_string()),\n\
        };\n\
        // ... your logic here ...\n\
        (true, format!(\"Result: {input}\"))\n\
    }\n\
\n\
Step 4: Build and deploy (the build script compiles AND copies to the skills dir):\n\
  run_shell(command=\"cd extensions/skills-src && ./build.sh my-skill-name\")\n\
  This compiles to WASM targeting wasm32-wasip1 and auto-deploys both\n\
  the .wasm binary and .tool.json manifest to extensions/skills/.\n\
\n\
Step 5: Reload \u{2014} make the daemon pick up the new skill immediately:\n\
  Use your reload_tools capability or tell the user to run: aigent tools reload\n\
  On next daemon restart, skills load automatically from extensions/skills/.\n\
\n\
=== WHEN TO CREATE A SKILL ===\n\
  - You find yourself doing the same multi-step shell pipeline repeatedly\n\
  - A data transformation is complex enough to warrant dedicated code\n\
  - The user asks for a capability that does not exist in your current tools\n\
  - You want a tool that combines multiple operations atomically\n\
\n\
=== IMPORTANT CONSTRAINTS ===\n\
  - Skills are stateless \u{2014} each call gets a fresh WASM instance\n\
  - Filesystem access is sandboxed to the workspace directory (relative paths only)\n\
  - stdout is buffered to 256KB \u{2014} keep output concise\n\
  - Only serde_json is available as a dependency (no network access from inside WASM)\n\
  - For tools needing HTTP/network, use run_shell to call curl or delegate to \n\
    your existing web_search/fetch_page tools in a tool chain\n\
  - Use descriptive tool names with underscores (my_skill_name, not my-skill-name)\n\
  - Write clear param descriptions \u{2014} these are what you see in your AVAILABLE TOOLS list\n\
  - All skill source code lives inside your workspace at extensions/skills-src/\n\
  - Deployed skills go to extensions/skills/ (handled automatically by build.sh)";

    format!(
        "\n\nAVAILABLE TOOLS (you have FULL ACCESS to all of these):\n\
         {list}\n\n\
         {tool_awareness}\n\n\
         HOW TOOLS WORK: Tools are called on your behalf automatically. \
         If a TOOL RESULT appears in the prompt below, use it directly — \
         the data IS present. You do not need to output raw JSON; the tool \
         infrastructure handles invocation for you. Simply indicate what you \
         want to do and the system will execute the right tool.\n\n\
         {tool_selection_guide}\n\n\
         {grounding}\n\n\
         {reflection_nudge}\n\n\
         {autonomy_directive}\n\n\
         {skill_creation}"
    )
}

// ─── utilities ───────────────────────────────────────────────────────────────

/// Truncate `text` to at most `max_chars` characters, appending `…` when cut.
pub fn truncate_for_prompt(text: &str, max_chars: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    let truncated: String = chars.into_iter().take(max_chars).collect();
    format!("{truncated}…")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── truncate_for_prompt ────────────────────────────────────────────────

    #[test]
    fn truncate_short_text_unchanged() {
        assert_eq!(truncate_for_prompt("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact_length_unchanged() {
        assert_eq!(truncate_for_prompt("12345", 5), "12345");
    }

    #[test]
    fn truncate_long_text_adds_ellipsis() {
        let result = truncate_for_prompt("hello world", 5);
        assert_eq!(result, "hello…");
    }

    #[test]
    fn truncate_empty_string() {
        assert_eq!(truncate_for_prompt("", 10), "");
    }

    #[test]
    fn truncate_zero_max() {
        assert_eq!(truncate_for_prompt("hello", 0), "…");
    }

    #[test]
    fn truncate_multibyte_characters() {
        let text = "café résumé";
        let result = truncate_for_prompt(text, 4);
        assert_eq!(result, "café…");
    }

    #[test]
    fn truncate_emoji() {
        let text = "🦀🐍🐹🐿️";
        let result = truncate_for_prompt(text, 2);
        assert_eq!(result, "🦀🐍…");
    }

    // ── build_tools_and_grounding ──────────────────────────────────────────

    #[test]
    fn grounding_rules_present_when_no_tools() {
        let section = build_tools_and_grounding(&[]);
        assert!(section.contains("GROUNDING RULES"));
        assert!(section.contains("perform_gait"));
        assert!(!section.contains("AVAILABLE TOOLS"));
    }

    #[test]
    fn grounding_rules_with_tools_shows_catalogue() {
        let specs = vec![aigent_tools::ToolSpec {
            name: "test_tool".to_string(),
            description: "A test".to_string(),
            params: vec![aigent_tools::ToolParam {
                name: "input".to_string(),
                description: "test input".to_string(),
                required: true,
                    ..Default::default()
            }],
            metadata: aigent_tools::ToolMetadata::default(),
        }];
        let section = build_tools_and_grounding(&specs);
        assert!(section.contains("AVAILABLE TOOLS"));
        assert!(section.contains("test_tool"));
        assert!(section.contains("GROUNDING RULES"));
    }
}
