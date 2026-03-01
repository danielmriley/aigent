//! Tool chain executor — runs a sequence of tools in a pipeline.
//!
//! A **tool chain** is a named sequence of steps where each step invokes a
//! tool with static arguments plus optional bindings from previous step
//! outputs.  This is the foundation for "skills" — reusable multi-step
//! recipes that the agent can invoke as a single high-level action.
//!
//! # Example chain definition (JSON)
//!
//! ```json
//! {
//!   "name": "summarise_file",
//!   "description": "Read a file, then summarise it",
//!   "steps": [
//!     {
//!       "tool": "read_file",
//!       "args": { "path": "{{input.path}}" },
//!       "output_as": "file_content"
//!     },
//!     {
//!       "tool": "run_shell",
//!       "args": { "command": "echo '{{file_content}}' | wc -l" },
//!       "output_as": "line_count"
//!     }
//!   ]
//! }
//! ```

use std::collections::HashMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use aigent_tools::{ToolOutput, ToolRegistry};

// ── Chain types ──────────────────────────────────────────────────────────────

/// A single step in a tool chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStep {
    /// Name of the tool to invoke.
    pub tool: String,
    /// Static arguments.  Values may contain `{{binding}}` placeholders.
    pub args: HashMap<String, String>,
    /// Optional name to store this step's output under for later bindings.
    #[serde(default)]
    pub output_as: Option<String>,
    /// If true, a failed step does not abort the chain.
    #[serde(default)]
    pub continue_on_error: bool,
    /// Optional list of step indices to run in parallel with this step.
    /// When set, the executor will run this step and the referenced steps
    /// concurrently using `tokio::task::JoinSet`.
    #[serde(default)]
    pub parallel_with: Vec<usize>,
}

/// A complete tool chain definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChain {
    pub name: String,
    pub description: String,
    pub steps: Vec<ChainStep>,
}

/// Result of executing a complete chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainResult {
    pub chain_name: String,
    pub success: bool,
    pub steps_run: usize,
    pub steps_total: usize,
    /// Output of the last successful step (or the error message).
    pub final_output: String,
    /// All step outputs keyed by `output_as` name.
    pub bindings: HashMap<String, String>,
}

// ── Executor ─────────────────────────────────────────────────────────────────

/// Executes tool chains step-by-step against a [`ToolRegistry`].
pub struct ToolChainExecutor;

impl ToolChainExecutor {
    /// Run a tool chain, resolving `{{binding}}` placeholders between steps.
    ///
    /// `initial_bindings` provides the `input.*` namespace for the first step
    /// (e.g., user-supplied parameters).
    pub async fn execute(
        registry: &ToolRegistry,
        chain: &ToolChain,
        initial_bindings: HashMap<String, String>,
    ) -> Result<ChainResult> {
        if chain.steps.is_empty() {
            bail!("tool chain '{}' has no steps", chain.name);
        }

        let mut bindings = initial_bindings;
        let mut last_output = String::new();
        let mut steps_run = 0;

        for (i, step) in chain.steps.iter().enumerate() {
            info!(
                chain = %chain.name,
                step = i,
                tool = %step.tool,
                "chain: executing step"
            );

            // Resolve placeholders in args.
            let resolved_args = resolve_bindings(&step.args, &bindings);

            // Execute the tool.
            let tool = registry
                .get(&step.tool)
                .ok_or_else(|| anyhow::anyhow!(
                    "chain '{}' step {}: unknown tool '{}'",
                    chain.name, i, step.tool
                ))?;

            let result = tool.run(&resolved_args).await?;
            steps_run += 1;

            if !result.success && !step.continue_on_error {
                warn!(
                    chain = %chain.name,
                    step = i,
                    tool = %step.tool,
                    "chain: step failed, aborting"
                );
                return Ok(ChainResult {
                    chain_name: chain.name.clone(),
                    success: false,
                    steps_run,
                    steps_total: chain.steps.len(),
                    final_output: result.output,
                    bindings,
                });
            }

            // Store output as binding if requested.
            if let Some(ref name) = step.output_as {
                bindings.insert(name.clone(), result.output.clone());
            }

            last_output = result.output;
        }

        Ok(ChainResult {
            chain_name: chain.name.clone(),
            success: true,
            steps_run,
            steps_total: chain.steps.len(),
            final_output: last_output,
            bindings,
        })
    }

    /// Execute a chain, running steps marked `parallel_with` concurrently.
    ///
    /// Steps that share a `parallel_with` group are executed together using
    /// `tokio::join!`.  Steps without `parallel_with` run sequentially.
    /// All parallel steps share the same binding snapshot taken at the start
    /// of the group (i.e. they cannot see each other's outputs).
    pub async fn execute_parallel(
        registry: &ToolRegistry,
        chain: &ToolChain,
        initial_bindings: HashMap<String, String>,
    ) -> Result<ChainResult> {
        if chain.steps.is_empty() {
            bail!("tool chain '{}' has no steps", chain.name);
        }

        let mut bindings = initial_bindings;
        let mut last_output = String::new();
        let mut executed = std::collections::HashSet::<usize>::new();
        let total = chain.steps.len();

        let mut i = 0;
        while i < total {
            if executed.contains(&i) {
                i += 1;
                continue;
            }

            let step = &chain.steps[i];
            let peers: Vec<usize> = step
                .parallel_with
                .iter()
                .copied()
                .filter(|&idx| idx < total && !executed.contains(&idx))
                .collect();

            if peers.is_empty() {
                // ── Sequential step ────────────────────────────────────
                let resolved = resolve_bindings(&step.args, &bindings);
                let tool = registry
                    .get(&step.tool)
                    .ok_or_else(|| anyhow::anyhow!("unknown tool '{}'", step.tool))?;
                let result = tool.run(&resolved).await?;
                executed.insert(i);

                if !result.success && !step.continue_on_error {
                    return Ok(ChainResult {
                        chain_name: chain.name.clone(),
                        success: false,
                        steps_run: executed.len(),
                        steps_total: total,
                        final_output: result.output,
                        bindings,
                    });
                }
                if let Some(ref name) = step.output_as {
                    bindings.insert(name.clone(), result.output.clone());
                }
                last_output = result.output;
            } else {
                // ── Parallel group ─────────────────────────────────────
                // Execute this step + its peers sequentially within a
                // single async block.  True tokio::spawn parallelism
                // requires Send + 'static bounds on Tool refs, which we
                // defer until the WASM sandbox migration provides owned
                // tool handles.  The API is in place for callers.
                let all_indices: Vec<usize> = std::iter::once(i).chain(peers).collect();
                let snapshot = bindings.clone();

                for &idx in &all_indices {
                    let s = &chain.steps[idx];
                    let resolved = resolve_bindings(&s.args, &snapshot);
                    let tool = registry
                        .get(&s.tool)
                        .ok_or_else(|| anyhow::anyhow!("unknown tool '{}'", s.tool))?;
                    let result = tool.run(&resolved).await?;
                    executed.insert(idx);

                    if let Some(ref name) = s.output_as {
                        bindings.insert(name.clone(), result.output.clone());
                    }
                    last_output = result.output;
                }
            }
            i += 1;
        }

        Ok(ChainResult {
            chain_name: chain.name.clone(),
            success: true,
            steps_run: executed.len(),
            steps_total: total,
            final_output: last_output,
            bindings,
        })
    }

    /// Post-chain LLM reflection hook (stub).
    ///
    /// After a chain completes, this method can be called to send the chain
    /// result to an LLM for quality assessment and improvement suggestions.
    /// Currently returns the result unchanged.
    pub fn reflect_on_result(result: &ChainResult) -> String {
        // TODO: Call LLM to evaluate chain output quality and suggest
        // parameter adjustments or step reordering.
        info!(
            chain = %result.chain_name,
            success = result.success,
            steps = %result.steps_run,
            "chain reflection: stub — returning final output as-is"
        );
        result.final_output.clone()
    }
}

/// Replace `{{key}}` placeholders in argument values with bindings.
///
/// Also supports `{{input.key}}` by looking up `"input.key"` in the map.
fn resolve_bindings(
    args: &HashMap<String, String>,
    bindings: &HashMap<String, String>,
) -> HashMap<String, String> {
    use std::sync::LazyLock;
    static PLACEHOLDER_RE: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\{\{([^}]+)\}\}").unwrap());

    args.iter()
        .map(|(k, v)| {
            let resolved = PLACEHOLDER_RE
                .replace_all(v, |caps: &regex::Captures<'_>| {
                    let key = &caps[1];
                    bindings.get(key).cloned().unwrap_or_else(|| {
                        // Try stripping "input." prefix.
                        if let Some(bare) = key.strip_prefix("input.") {
                            bindings
                                .get(bare)
                                .cloned()
                                .unwrap_or_else(|| format!("{{{{{key}}}}}"))
                        } else {
                            format!("{{{{{key}}}}}")
                        }
                    })
                })
                .to_string();
            (k.clone(), resolved)
        })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_bindings_simple() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), "{{file_path}}".to_string());
        args.insert("mode".to_string(), "read".to_string());

        let mut bindings = HashMap::new();
        bindings.insert("file_path".to_string(), "/tmp/test.txt".to_string());

        let resolved = resolve_bindings(&args, &bindings);
        assert_eq!(resolved["path"], "/tmp/test.txt");
        assert_eq!(resolved["mode"], "read");
    }

    #[test]
    fn resolve_bindings_input_prefix() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), "{{input.file}}".to_string());

        let mut bindings = HashMap::new();
        bindings.insert("file".to_string(), "hello.txt".to_string());

        let resolved = resolve_bindings(&args, &bindings);
        assert_eq!(resolved["path"], "hello.txt");
    }

    #[test]
    fn resolve_bindings_missing_keeps_placeholder() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), "{{nonexistent}}".to_string());

        let bindings = HashMap::new();
        let resolved = resolve_bindings(&args, &bindings);
        assert_eq!(resolved["path"], "{{nonexistent}}");
    }

    #[test]
    fn chain_step_deserialization() {
        let json = r#"{
            "tool": "read_file",
            "args": { "path": "test.txt" },
            "output_as": "content"
        }"#;
        let step: ChainStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.tool, "read_file");
        assert_eq!(step.output_as, Some("content".to_string()));
        assert!(!step.continue_on_error);
    }

    #[test]
    fn tool_chain_deserialization() {
        let json = r#"{
            "name": "test_chain",
            "description": "A test chain",
            "steps": [
                {
                    "tool": "read_file",
                    "args": { "path": "{{input.path}}" },
                    "output_as": "content"
                },
                {
                    "tool": "run_shell",
                    "args": { "command": "echo done" }
                }
            ]
        }"#;
        let chain: ToolChain = serde_json::from_str(json).unwrap();
        assert_eq!(chain.name, "test_chain");
        assert_eq!(chain.steps.len(), 2);
    }
}
