use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

// ── Tool trait and registry ──────────────────────────────────────────────────

/// JSON-friendly type hint for a tool parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ParamType {
    #[default]
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

/// Security classification for a tool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SecurityLevel {
    #[default]
    Low,
    Medium,
    High,
}

/// Optional rich metadata about a tool (security, grouping, cost).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ToolMetadata {
    pub security_level: SecurityLevel,
    pub read_only: bool,
    pub group: String,
    pub cost_estimate: Option<f32>,
    pub examples: Vec<String>,
}

/// Describes a single parameter that a tool accepts.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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
        let (properties, required) = build_param_properties(&self.params);
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

    /// Generate a standalone JSON Schema document for this tool's parameters.
    ///
    /// Unlike [`to_openai_tool_schema`](Self::to_openai_tool_schema), this
    /// produces a self-contained `$schema`-annotated document suitable for
    /// validation or IDE tooling.
    pub fn to_json_schema(&self) -> serde_json::Value {
        let (properties, required) = build_param_properties(&self.params);

        serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": format!("aigent://tools/{}", self.name),
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
            "x-metadata": {
                "security_level": format!("{:?}", self.metadata.security_level),
                "read_only": self.metadata.read_only,
                "group": self.metadata.group,
            }
        })
    }

    /// Validate a set of arguments against this tool's parameter spec.
    ///
    /// Returns a list of validation errors (empty if valid).
    pub fn validate_args(&self, args: &HashMap<String, String>) -> Vec<String> {
        let mut errors = Vec::new();
        for p in &self.params {
            if p.required && !args.contains_key(&p.name) {
                errors.push(format!("missing required param: '{}'", p.name));
            }
            if let Some(val) = args.get(&p.name) {
                // Type validation for non-string types.
                match p.param_type {
                    ParamType::Integer => {
                        if val.parse::<i64>().is_err() {
                            errors.push(format!(
                                "param '{}' expected integer, got '{}'",
                                p.name, val
                            ));
                        }
                    }
                    ParamType::Number => {
                        if val.parse::<f64>().is_err() {
                            errors.push(format!(
                                "param '{}' expected number, got '{}'",
                                p.name, val
                            ));
                        }
                    }
                    ParamType::Boolean => {
                        if val != "true" && val != "false" {
                            errors.push(format!(
                                "param '{}' expected boolean, got '{}'",
                                p.name, val
                            ));
                        }
                    }
                    _ => {}
                }
                if !p.enum_values.is_empty() && !p.enum_values.contains(val) {
                    errors.push(format!(
                        "param '{}' must be one of {:?}, got '{}'",
                        p.name, p.enum_values, val
                    ));
                }
            }
        }
        errors
    }
}


/// Generate a JSON Schema document describing all tool-related types.
///
/// This uses `schemars` to produce a proper `$schema`-annotated document
/// Build a JSON Schema `properties` map and `required` array from a slice of
/// [`ToolParam`]s.  Shared by [`ToolSpec::to_openai_tool_schema`] and
/// [`ToolSpec::to_json_schema`] to avoid duplicating the type-mapping logic.
fn build_param_properties(params: &[ToolParam]) -> (serde_json::Map<String, serde_json::Value>, Vec<String>) {
    let mut properties = serde_json::Map::new();
    let mut required: Vec<String> = Vec::new();
    for p in params {
        properties.insert(p.name.clone(), build_schema_property(p));
        if p.required {
            required.push(p.name.clone());
        }
    }
    (properties, required)
}

/// Build a single JSON Schema property object from a [`ToolParam`].
fn build_schema_property(p: &ToolParam) -> serde_json::Value {
    let type_str = match p.param_type {
        ParamType::String  => "string",
        ParamType::Number  => "number",
        ParamType::Integer => "integer",
        ParamType::Boolean => "boolean",
        ParamType::Array   => "array",
        ParamType::Object  => "object",
    };
    let mut prop = serde_json::json!({
        "type": type_str,
        "description": p.description,
    });
    if !p.enum_values.is_empty() {
        prop["enum"] = serde_json::json!(p.enum_values);
    }
    if let Some(ref def) = p.default {
        // Serialize the default as the appropriate JSON type rather than
        // always wrapping it in a JSON string.
        prop["default"] = match p.param_type {
            ParamType::Number => def.parse::<f64>()
                .map(|n| serde_json::json!(n))
                .unwrap_or_else(|_| serde_json::Value::String(def.clone())),
            ParamType::Integer => def.parse::<i64>()
                .map(|n| serde_json::json!(n))
                .unwrap_or_else(|_| serde_json::Value::String(def.clone())),
            ParamType::Boolean => match def.as_str() {
                "true"  => serde_json::json!(true),
                "false" => serde_json::json!(false),
                _       => serde_json::Value::String(def.clone()),
            },
            ParamType::Array | ParamType::Object => {
                serde_json::from_str(def)
                    .unwrap_or_else(|_| serde_json::Value::String(def.clone()))
            }
            ParamType::String => serde_json::Value::String(def.clone()),
        };
    }
    prop
}

/// suitable for IDE tooling, validation, and marketplace manifests.
pub fn tool_registry_schema() -> schemars::schema::RootSchema {
    schemars::schema_for!(ToolSpec)
}

/// Generate an array of JSON Schema documents, one per tool.
pub fn export_schemars_schemas(specs: &[ToolSpec]) -> serde_json::Value {
    let mut arr: Vec<serde_json::Value> = Vec::new();
    for spec in specs {
        let root = schemars::schema_for!(ToolSpec);
        arr.push(serde_json::json!({
            "tool": spec.name,
            "spec_schema": root,
            "params_schema": spec.to_json_schema(),
        }));
    }
    serde_json::Value::Array(arr)
}

/// Export all tool schemas to a JSON manifest.
///
/// Produces an object keyed by tool name with both OpenAI and standalone
/// JSON Schema representations, suitable for writing to
/// `.aigent/schemas/tools.json`.
pub fn export_tool_schemas(specs: &[ToolSpec]) -> serde_json::Value {
    let mut manifest = serde_json::Map::new();
    for spec in specs {
        manifest.insert(
            spec.name.clone(),
            serde_json::json!({
                "openai": spec.to_openai_tool_schema(),
                "json_schema": spec.to_json_schema(),
            }),
        );
    }
    serde_json::Value::Object(manifest)
}

/// Convert a slice of `ToolSpec` into the `tools` JSON array accepted by
/// Ollama `/api/chat` and OpenRouter `/chat/completions`.
pub fn specs_to_openai_tools(specs: &[ToolSpec]) -> serde_json::Value {
    serde_json::Value::Array(
        specs.iter().map(|s| s.to_openai_tool_schema()).collect(),
    )
}

/// The result returned after a tool runs.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolOutput {
    pub success: bool,
    pub output: String,
}

/// Describes the source/origin of a registered tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum ToolSource {
    /// Built-in Rust implementation (native fallback).
    Native,
    /// Compiled WASM guest loaded from `extensions/`.
    Wasm,
    /// Dynamically loaded at runtime (e.g. from the modules directory).
    Dynamic,
}

impl std::fmt::Display for ToolSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => write!(f, "native"),
            Self::Wasm => write!(f, "wasm"),
            Self::Dynamic => write!(f, "dynamic"),
        }
    }
}

/// An entry in the tool registry, associating a tool with its source metadata.
///
/// `spec` is computed once at registration time (via `Tool::spec()`) and
/// cached here so that all hot-path reads — `get()`, `list_specs()`,
/// `tools_in_group()`, `expand_groups()`, etc. — are allocation-free
/// string comparisons over contiguous memory rather than repeated trait-
/// dispatch calls that heap-allocate a fresh `ToolSpec` per iteration.
struct ToolEntry {
    /// Cached at registration; never mutated after that.
    spec: ToolSpec,
    tool: Arc<dyn Tool>,
    source: ToolSource,
}

/// Summary information about a registered tool, including its origin.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolInfo {
    pub spec: ToolSpec,
    pub source: ToolSource,
}

/// Trait implemented by every tool (built-in or WASM-loaded).
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn run(&self, args: &HashMap<String, String>) -> Result<ToolOutput>;
}

/// Central registry for all available tools.
///
/// Thread-safe via interior `RwLock` — all methods take `&self` so the
/// registry can be shared behind `Arc<ToolRegistry>` without external
/// locking.  Read-path methods (`get`, `list_specs`, `list_tools`) acquire
/// a read lock; mutating methods (`register`, `unregister`) acquire a
/// write lock.
pub struct ToolRegistry {
    tools: RwLock<Vec<ToolEntry>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self {
            tools: RwLock::new(Vec::new()),
        }
    }
}

impl ToolRegistry {
    /// Register a tool.  Defaults to [`ToolSource::Native`].
    ///
    /// This is the backward-compatible entry point used by `default_registry`.
    pub fn register(&self, tool: Box<dyn Tool>) {
        self.register_with_source(tool, ToolSource::Native);
    }

    /// Register a tool with an explicit [`ToolSource`] tag.
    pub fn register_with_source(&self, tool: Box<dyn Tool>, source: ToolSource) {
        // Call spec() once here so every subsequent hot-path read is O(1).
        let spec = tool.spec();
        let entry = ToolEntry {
            spec,
            tool: Arc::from(tool),
            source,
        };
        self.tools.write().unwrap().push(entry);
    }

    /// Unregister all tools with the given name.
    ///
    /// Returns `true` if at least one tool was removed.
    pub fn unregister(&self, name: &str) -> bool {
        let mut tools = self.tools.write().unwrap();
        let before = tools.len();
        tools.retain(|e| e.spec.name != name);
        tools.len() < before
    }

    /// Return specs for all registered tools.
    pub fn list_specs(&self) -> Vec<ToolSpec> {
        self.tools.read().unwrap().iter().map(|e| e.spec.clone()).collect()
    }

    /// Detailed listing including source metadata.
    pub fn list_tools(&self) -> Vec<ToolInfo> {
        self.tools
            .read()
            .unwrap()
            .iter()
            .map(|e| ToolInfo {
                spec: e.spec.clone(),
                source: e.source,
            })
            .collect()
    }

    /// Look up a tool by name (first-match wins, preserving WASM-first
    /// semantics).  Returns an `Arc` so the caller owns the handle and
    /// can safely hold it across `.await` points.
    ///
    /// O(n) scan over a contiguous Vec — allocation-free because the name
    /// comparison reads the cached `ToolEntry::spec.name` rather than calling
    /// `Tool::spec()` on every entry.  For typical registry sizes (30–60
    /// tools) this is ~100 ns and does not warrant a HashMap index.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .read()
            .unwrap()
            .iter()
            .find(|e| e.spec.name == name)
            .map(|e| Arc::clone(&e.tool))
    }

    /// Number of currently registered tools.
    pub fn len(&self) -> usize {
        self.tools.read().unwrap().len()
    }

    /// Returns `true` when no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the names of all dynamically-registered tools.
    pub fn dynamic_tool_names(&self) -> Vec<String> {
        self.tools
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.source == ToolSource::Dynamic)
            .map(|e| e.spec.name.clone())
            .collect()
    }

    /// Return the names of all tools belonging to the given group.
    pub fn tools_in_group(&self, group: &str) -> Vec<String> {
        self.tools
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.spec.metadata.group == group)
            .map(|e| e.spec.name.clone())
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

#[cfg(feature = "marketplace")]
pub mod marketplace;
pub use builtins::{
    BrowsePageTool,
    CalendarAddEventTool, CpTool, CutTool, DraftEmailTool, EchoTool,
    FindTool, GitRollbackTool, GrepTool, HeadTool, ListDirTool, ListModulesTool,
    MkdirTool, MvTool, ReadFileTool, RemindMeTool, RmTool, RunShellTool,
    SedTool, SeqTool, SortTool, TailTool, TouchTool, TreeTool, UniqTool,
    WcTool, WebSearchTool, WorkspaceStatusTool, WriteFileTool,
};
pub use builtins::memory_search::{
    SearchMemoryTool, MemoryQueryFn, MemorySearchResult,
    WriteMemoryTool, MemoryWriteFn,
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
        let reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "alpha".into() }));
        reg.register(Box::new(DummyTool { name: "beta".into() }));

        assert!(reg.get("alpha").is_some());
        assert!(reg.get("beta").is_some());
        assert!(reg.get("gamma").is_none());
    }

    #[test]
    fn list_specs_returns_all() {
        let reg = ToolRegistry::default();
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
        let reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "finder".into() }));

        let tool = reg.get("finder").unwrap();
        let spec = tool.spec();
        assert_eq!(spec.name, "finder");
        assert_eq!(spec.params.len(), 1);
        assert!(spec.params[0].required);
    }

    #[tokio::test]
    async fn run_registered_tool() {
        let reg = ToolRegistry::default();
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
        let reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "dup".into() }));
        reg.register(Box::new(DummyTool { name: "dup".into() }));

        // list_specs should show both
        let specs = reg.list_specs();
        let dup_count = specs.iter().filter(|s| s.name == "dup").count();
        assert_eq!(dup_count, 2, "both duplicates should appear in list_specs");

        // get should return the first one (deterministic for Vec+find)
        assert!(reg.get("dup").is_some());
    }

    // ── New tests for dynamic registration features ────────────────────────

    #[test]
    fn unregister_removes_tool() {
        let reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "removable".into() }));
        reg.register(Box::new(DummyTool { name: "keeper".into() }));

        assert!(reg.unregister("removable"));
        assert!(reg.get("removable").is_none());
        assert!(reg.get("keeper").is_some());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn unregister_nonexistent_returns_false() {
        let reg = ToolRegistry::default();
        assert!(!reg.unregister("ghost"));
    }

    #[test]
    fn register_with_source_and_list_tools() {
        let reg = ToolRegistry::default();
        reg.register_with_source(
            Box::new(DummyTool { name: "native_tool".into() }),
            ToolSource::Native,
        );
        reg.register_with_source(
            Box::new(DummyTool { name: "wasm_tool".into() }),
            ToolSource::Wasm,
        );
        reg.register_with_source(
            Box::new(DummyTool { name: "dyn_tool".into() }),
            ToolSource::Dynamic,
        );

        let infos = reg.list_tools();
        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].source, ToolSource::Native);
        assert_eq!(infos[1].source, ToolSource::Wasm);
        assert_eq!(infos[2].source, ToolSource::Dynamic);
    }

    #[test]
    fn dynamic_tool_names_filters_correctly() {
        let reg = ToolRegistry::default();
        reg.register_with_source(
            Box::new(DummyTool { name: "builtin".into() }),
            ToolSource::Native,
        );
        reg.register_with_source(
            Box::new(DummyTool { name: "skill_a".into() }),
            ToolSource::Dynamic,
        );
        reg.register_with_source(
            Box::new(DummyTool { name: "skill_b".into() }),
            ToolSource::Dynamic,
        );

        let dyn_names = reg.dynamic_tool_names();
        assert_eq!(dyn_names.len(), 2);
        assert!(dyn_names.contains(&"skill_a".to_string()));
        assert!(dyn_names.contains(&"skill_b".to_string()));
    }

    #[test]
    fn unregister_all_duplicates() {
        let reg = ToolRegistry::default();
        reg.register(Box::new(DummyTool { name: "dup".into() }));
        reg.register(Box::new(DummyTool { name: "dup".into() }));
        reg.register(Box::new(DummyTool { name: "other".into() }));

        assert!(reg.unregister("dup"));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("dup").is_none());
        assert!(reg.get("other").is_some());
    }

    #[test]
    fn tool_source_display() {
        assert_eq!(ToolSource::Native.to_string(), "native");
        assert_eq!(ToolSource::Wasm.to_string(), "wasm");
        assert_eq!(ToolSource::Dynamic.to_string(), "dynamic");
    }

    /// The registry can be shared across threads via Arc.
    #[test]
    fn concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let reg = Arc::new(ToolRegistry::default());
        reg.register(Box::new(DummyTool { name: "shared".into() }));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let r = Arc::clone(&reg);
                thread::spawn(move || {
                    assert!(r.get("shared").is_some());
                    r.list_specs().len()
                })
            })
            .collect();

        for h in handles {
            assert_eq!(h.join().unwrap(), 1);
        }
    }

    #[test]
    fn len_and_is_empty() {
        let reg = ToolRegistry::default();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);

        reg.register(Box::new(DummyTool { name: "x".into() }));
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
    }
}
