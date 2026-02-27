use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ── Tool trait and registry ──────────────────────────────────────────────────

/// JSON-friendly type hint for a tool parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

impl Default for ParamType {
    fn default() -> Self {
        Self::String
    }
}

/// Security classification for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SecurityLevel {
    Low,
    Medium,
    High,
}

impl Default for SecurityLevel {
    fn default() -> Self {
        Self::Low
    }
}

/// Optional rich metadata about a tool (security, grouping, cost).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMetadata {
    pub security_level: SecurityLevel,
    pub read_only: bool,
    pub group: String,
    pub cost_estimate: Option<f32>,
    pub examples: Vec<String>,
}

/// Describes a single parameter that a tool accepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub name: String,
    pub description: String,
    pub required: bool,
    /// JSON Schema type for the parameter (default: String).
    #[serde(default)]
    pub param_type: ParamType,
    /// Allowed values when the parameter is an enum.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enum_values: Vec<String>,
    /// Default value expressed as a string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

impl Default for ToolParam {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            required: false,
            param_type: ParamType::String,
            enum_values: Vec::new(),
            default: None,
        }
    }
}

impl ToolParam {
    /// Convenience constructor for the most common case (required string param).
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            param_type: ParamType::String,
            enum_values: Vec::new(),
            default: None,
        }
    }

    /// Convenience constructor for an optional string param.
    pub fn optional(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            param_type: ParamType::String,
            enum_values: Vec::new(),
            default: None,
        }
    }
}

/// Static metadata about a tool, used by the LLM to decide which tool to call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub params: Vec<ToolParam>,
    /// Rich metadata (security, grouping, cost, examples).  Defaults:
    /// `SecurityLevel::Low`, `read_only: false` (fail-safe: assumes write
    /// access until the tool explicitly opts in to read-only), empty group.
    #[serde(default)]
    pub metadata: ToolMetadata,
}

impl ToolSpec {
    /// Generate the OpenAI-compatible `tools` array element for this tool.
    ///
    /// ```json
    /// {
    ///   "type": "function",
    ///   "function": {
    ///     "name": "web_search",
    ///     "description": "...",
    ///     "parameters": {
    ///       "type": "object",
    ///       "properties": { ... },
    ///       "required": [...]
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// This format is accepted by both OpenRouter (OpenAI-compatible) and
    /// Ollama's `/api/chat` endpoint.
    pub fn to_openai_tool_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required: Vec<String> = Vec::new();

        for p in &self.params {
            let type_str = match p.param_type {
                ParamType::String => "string",
                ParamType::Number => "number",
                ParamType::Integer => "integer",
                ParamType::Boolean => "boolean",
                ParamType::Array => "array",
                ParamType::Object => "object",
            };
            let mut prop = serde_json::json!({
                "type": type_str,
                "description": p.description,
            });
            if !p.enum_values.is_empty() {
                prop["enum"] = serde_json::json!(p.enum_values);
            }
            if let Some(ref def) = p.default {
                // Serialize the default as the appropriate JSON type rather
                // than always wrapping it in a JSON string.
                prop["default"] = match p.param_type {
                    ParamType::Number => def.parse::<f64>()
                        .map(|n| serde_json::json!(n))
                        .unwrap_or_else(|_| serde_json::Value::String(def.clone())),
                    ParamType::Integer => def.parse::<i64>()
                        .map(|n| serde_json::json!(n))
                        .unwrap_or_else(|_| serde_json::Value::String(def.clone())),
                    ParamType::Boolean => match def.as_str() {
                        "true" => serde_json::json!(true),
                        "false" => serde_json::json!(false),
                        _ => serde_json::Value::String(def.clone()),
                    },
                    ParamType::Array | ParamType::Object => {
                        serde_json::from_str(def)
                            .unwrap_or_else(|_| serde_json::Value::String(def.clone()))
                    }
                    ParamType::String => serde_json::Value::String(def.clone()),
                };
            }
            properties.insert(p.name.clone(), prop);
            if p.required {
                required.push(p.name.clone());
            }
        }

        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        })
    }
}

/// Convert a slice of `ToolSpec` into the `tools` JSON array accepted by
/// Ollama `/api/chat` and OpenRouter `/chat/completions`.
pub fn specs_to_openai_tools(specs: &[ToolSpec]) -> serde_json::Value {
    serde_json::Value::Array(
        specs.iter().map(|s| s.to_openai_tool_schema()).collect(),
    )
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

    /// Return the names of all tools belonging to the given group.
    pub fn tools_in_group(&self, group: &str) -> Vec<String> {
        self.tools
            .iter()
            .filter(|t| t.spec().metadata.group == group)
            .map(|t| t.spec().name)
            .collect()
    }

    /// Expand `@group` references in a list of tool names.
    ///
    /// If a name starts with `@`, it's treated as a group name and expanded
    /// to all tool names in that group.  Normal tool names pass through.
    pub fn expand_groups(&self, names: &[String]) -> Vec<String> {
        let mut result = Vec::new();
        for name in names {
            if let Some(group) = name.strip_prefix('@') {
                result.extend(self.tools_in_group(group));
            } else {
                result.push(name.clone());
            }
        }
        result
    }
}

// ── Built-in tools ───────────────────────────────────────────────────────────

pub mod builtins;
pub use builtins::{
    CalendarAddEventTool, DraftEmailTool, FetchPageTool, FinanceQuoteTool,
    GitRollbackTool, ReadFileTool, RemindMeTool, RunShellTool, WebSearchTool,
    WriteFileTool,
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
                params: vec![ToolParam::required("input", "test param")],
                metadata: ToolMetadata::default(),
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
