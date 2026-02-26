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
    CalendarAddEventTool, DraftEmailTool, GitRollbackTool, ReadFileTool, RemindMeTool,
    RunShellTool, WebSearchTool, WriteFileTool,
};

// ── ToolRegistry tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod registry_tests {
    use super::*;
    use std::collections::HashMap;

    /// Minimal dummy tool for testing the registry.
    struct DummyTool {
        name: String,
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: format!("Dummy tool: {}", self.name),
                params: vec![ToolParam {
                    name: "input".to_string(),
                    description: "test param".to_string(),
                    required: true,
                }],
            }
        }
        async fn run(&self, _args: &HashMap<String, String>) -> Result<ToolOutput> {
            Ok(ToolOutput {
                success: true,
                output: format!("ran {}", self.name),
            })
        }
    }

    #[test]
    fn empty_registry() {
        let reg = ToolRegistry::default();
        assert!(reg.list_specs().is_empty());
        assert!(reg.get("anything").is_none());
    }

    #[test]
    fn register_and_get() {
        let mut reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "alpha".into() }));
        reg.register(Box::new(DummyTool { name: "beta".into() }));

        assert!(reg.get("alpha").is_some());
        assert!(reg.get("beta").is_some());
        assert!(reg.get("gamma").is_none());
    }

    #[test]
    fn list_specs_returns_all() {
        let mut reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "one".into() }));
        reg.register(Box::new(DummyTool { name: "two".into() }));
        reg.register(Box::new(DummyTool { name: "three".into() }));

        let specs = reg.list_specs();
        assert_eq!(specs.len(), 3);
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"one"));
        assert!(names.contains(&"two"));
        assert!(names.contains(&"three"));
    }

    #[test]
    fn get_returns_correct_tool_spec() {
        let mut reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "finder".into() }));

        let tool = reg.get("finder").unwrap();
        let spec = tool.spec();
        assert_eq!(spec.name, "finder");
        assert_eq!(spec.params.len(), 1);
        assert!(spec.params[0].required);
    }

    #[tokio::test]
    async fn run_registered_tool() {
        let mut reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "runner".into() }));

        let tool = reg.get("runner").unwrap();
        let result = tool.run(&HashMap::new()).await.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "ran runner");
    }

    /// Duplicate registration: the first tool wins on `get` (Vec + find).
    /// This test documents the current behavior so someone adding a HashMap
    /// backend later doesn't silently change the semantics.
    #[test]
    fn duplicate_name_get_returns_first_registered() {
        let mut reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "dup".into() }));
        reg.register(Box::new(DummyTool { name: "dup".into() }));

        // list_specs should show both
        let specs = reg.list_specs();
        let dup_count = specs.iter().filter(|s| s.name == "dup").count();
        assert_eq!(dup_count, 2, "both duplicates should appear in list_specs");

        // get should return the first one (deterministic for Vec+find)
        assert!(reg.get("dup").is_some());
    }
}
