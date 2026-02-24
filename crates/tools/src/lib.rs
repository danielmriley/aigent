use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ── Tool trait and registry ──────────────────────────────────────────────────

/// Describes a single parameter that a tool accepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// Static metadata about a tool, used by the LLM to decide which tool to call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub params: Vec<ToolParam>,
}

/// The result returned after a tool runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub success: bool,
    pub output: String,
}

/// Trait implemented by every tool (built-in or WASM-loaded).
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput>;
}

/// Central registry for all available tools.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn list_specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|t| t.spec()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.spec().name == name)
            .map(|t| t.as_ref())
    }
}

// ── Built-in tools ───────────────────────────────────────────────────────────

pub mod builtins;
pub use builtins::{
    CalendarAddEventTool, DraftEmailTool, ReadFileTool, RemindMeTool, RunShellTool, WebSearchTool,
    WriteFileTool,
};
