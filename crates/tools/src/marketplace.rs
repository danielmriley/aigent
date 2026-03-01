//! Marketplace: discover, install, and manage tool extensions.
//!
//! Gated behind the `marketplace` feature flag. Provides:
//!
//! * `ExtensionManifest` — parsed `manifest.toml` for an extension
//! * `MarketplaceRegistry` — local index of installed extensions
//! * Discovery helpers to scan the `extensions/marketplace/` directory

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

// ── Manifest types ───────────────────────────────────────────────────────────

/// Parsed `manifest.toml` for a marketplace extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionManifest {
    pub extension: ExtensionMeta,
    pub tool: ToolManifest,
    #[serde(default)]
    pub build: Option<BuildManifest>,
}

/// Top-level metadata about the extension package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionMeta {
    pub name: String,
    pub version: String,
    pub description: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub license: Option<String>,
}

/// Tool-specific metadata matching the `ToolSpec` schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolManifest {
    pub name: String,
    #[serde(default)]
    pub group: String,
    #[serde(default = "default_security")]
    pub security: String,
    #[serde(default)]
    pub read_only: bool,
    #[serde(default)]
    pub wit_world: Option<String>,
    #[serde(default)]
    pub params: HashMap<String, ParamManifest>,
}

fn default_security() -> String {
    "low".to_string()
}

/// A parameter definition in the manifest (simplified form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamManifest {
    #[serde(rename = "type", default = "default_param_type")]
    pub param_type: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub enum_values: Vec<String>,
}

fn default_param_type() -> String {
    "string".to_string()
}

/// Build instructions for compiling an extension from source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildManifest {
    #[serde(default = "default_toolchain")]
    pub toolchain: String,
    #[serde(default = "default_target")]
    pub target: String,
}

fn default_toolchain() -> String {
    "cargo-component".to_string()
}

fn default_target() -> String {
    "wasm32-wasip1".to_string()
}

// ── Registry ─────────────────────────────────────────────────────────────────

/// An entry in the local marketplace registry (`registry.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub version: String,
    pub tool_name: String,
    pub wasm_path: Option<String>,
    pub installed_at: String,
}

/// Local registry of installed marketplace extensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceRegistry {
    pub version: u32,
    pub extensions: HashMap<String, RegistryEntry>,
}

impl Default for MarketplaceRegistry {
    fn default() -> Self {
        Self {
            version: 1,
            extensions: HashMap::new(),
        }
    }
}

impl MarketplaceRegistry {
    /// Load the registry from `registry.json` in the given directory.
    pub fn load(marketplace_dir: &Path) -> Result<Self> {
        let path = marketplace_dir.join("registry.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let reg: Self = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(reg)
    }

    /// Persist the registry to `registry.json`.
    pub fn save(&self, marketplace_dir: &Path) -> Result<()> {
        let path = marketplace_dir.join("registry.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    /// Install an extension by reading its manifest and recording it.
    pub fn install(&mut self, ext_dir: &Path) -> Result<String> {
        let manifest = load_manifest(ext_dir)?;
        let name = manifest.extension.name.clone();

        // Look for a `.wasm` binary.
        let wasm_path = find_wasm(ext_dir);

        let entry = RegistryEntry {
            name: name.clone(),
            version: manifest.extension.version,
            tool_name: manifest.tool.name,
            wasm_path: wasm_path.map(|p| p.to_string_lossy().to_string()),
            installed_at: chrono::Utc::now().to_rfc3339(),
        };
        self.extensions.insert(name.clone(), entry);
        Ok(name)
    }

    /// Remove an extension by name.
    pub fn remove(&mut self, name: &str) -> bool {
        self.extensions.remove(name).is_some()
    }

    /// List all installed extensions.
    pub fn list(&self) -> Vec<&RegistryEntry> {
        self.extensions.values().collect()
    }
}

// ── Discovery helpers ────────────────────────────────────────────────────────

/// Load and parse a `manifest.toml` from an extension directory.
pub fn load_manifest(ext_dir: &Path) -> Result<ExtensionManifest> {
    let path = ext_dir.join("manifest.toml");
    if !path.exists() {
        bail!("no manifest.toml in {}", ext_dir.display());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let manifest: ExtensionManifest = toml::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(manifest)
}

/// Scan a marketplace directory for sub-directories containing `manifest.toml`.
pub fn discover_extensions(marketplace_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    if !marketplace_dir.is_dir() {
        return Ok(found);
    }
    for entry in std::fs::read_dir(marketplace_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.join("manifest.toml").exists() {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

/// Find the first `.wasm` file in an extension directory.
fn find_wasm(ext_dir: &Path) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(ext_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "wasm") {
                return Some(p);
            }
        }
    }
    None
}

/// Convert a marketplace `ToolManifest` into a `ToolSpec` for runtime
/// registration.
pub fn manifest_to_tool_spec(manifest: &ExtensionManifest) -> crate::ToolSpec {
    use crate::{ParamType, SecurityLevel, ToolMetadata, ToolParam, ToolSpec};

    let security_level = match manifest.tool.security.as_str() {
        "high" => SecurityLevel::High,
        "medium" => SecurityLevel::Medium,
        _ => SecurityLevel::Low,
    };

    let params: Vec<ToolParam> = manifest
        .tool
        .params
        .iter()
        .map(|(name, pm)| {
            let param_type = match pm.param_type.as_str() {
                "integer" => ParamType::Integer,
                "number" => ParamType::Number,
                "boolean" => ParamType::Boolean,
                "array" => ParamType::Array,
                "object" => ParamType::Object,
                _ => ParamType::String,
            };
            ToolParam {
                name: name.clone(),
                description: pm.description.clone(),
                required: pm.required,
                param_type,
                enum_values: pm.enum_values.clone(),
                default: pm.default.clone(),
            }
        })
        .collect();

    ToolSpec {
        name: manifest.tool.name.clone(),
        description: manifest.extension.description.clone(),
        params,
        metadata: ToolMetadata {
            security_level,
            read_only: manifest.tool.read_only,
            group: if manifest.tool.group.is_empty() {
                "marketplace".to_string()
            } else {
                manifest.tool.group.clone()
            },
            ..Default::default()
        },
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_manifest_roundtrip() {
        let toml_str = r#"
[extension]
name = "example-tool"
version = "0.1.0"
description = "An example marketplace tool"
authors = ["Test Author"]
license = "MIT"

[tool]
name = "example_tool"
group = "demo"
security = "low"
read_only = true

[tool.params]
query = { type = "string", required = true, description = "The query" }
limit = { type = "integer", required = false, description = "Max items", default = "10" }

[build]
toolchain = "cargo-component"
target = "wasm32-wasip1"
"#;
        let manifest: ExtensionManifest = toml::from_str(toml_str).unwrap();
        assert_eq!(manifest.extension.name, "example-tool");
        assert_eq!(manifest.tool.name, "example_tool");
        assert_eq!(manifest.tool.params.len(), 2);
        assert!(manifest.tool.read_only);
        assert!(manifest.tool.params.contains_key("query"));

        // Convert to ToolSpec.
        let spec = manifest_to_tool_spec(&manifest);
        assert_eq!(spec.name, "example_tool");
        assert_eq!(spec.metadata.group, "demo");
        assert!(spec.metadata.read_only);
    }

    #[test]
    fn registry_crud() {
        let tmp = std::env::temp_dir().join("aigent_test_marketplace");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Create an extension dir with manifest.
        let ext_dir = tmp.join("my-ext");
        std::fs::create_dir_all(&ext_dir).unwrap();
        std::fs::write(
            ext_dir.join("manifest.toml"),
            r#"
[extension]
name = "my-ext"
version = "1.0.0"
description = "Test extension"

[tool]
name = "my_tool"
"#,
        )
        .unwrap();

        let mut reg = MarketplaceRegistry::load(&tmp).unwrap();
        assert!(reg.extensions.is_empty());

        // Install.
        let name = reg.install(&ext_dir).unwrap();
        assert_eq!(name, "my-ext");
        assert_eq!(reg.list().len(), 1);

        // Save and reload.
        reg.save(&tmp).unwrap();
        let reg2 = MarketplaceRegistry::load(&tmp).unwrap();
        assert_eq!(reg2.extensions.len(), 1);
        assert_eq!(reg2.extensions["my-ext"].tool_name, "my_tool");

        // Remove.
        assert!(reg.remove("my-ext"));
        assert!(reg.list().is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn discover_extensions_finds_manifests() {
        let tmp = std::env::temp_dir().join("aigent_test_discover");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("ext-a")).unwrap();
        std::fs::create_dir_all(tmp.join("ext-b")).unwrap();
        std::fs::create_dir_all(tmp.join("no-manifest")).unwrap();

        std::fs::write(
            tmp.join("ext-a/manifest.toml"),
            "[extension]\nname = \"a\"\nversion = \"0.1.0\"\ndescription = \"A\"\n[tool]\nname = \"a\"",
        )
        .unwrap();
        std::fs::write(
            tmp.join("ext-b/manifest.toml"),
            "[extension]\nname = \"b\"\nversion = \"0.1.0\"\ndescription = \"B\"\n[tool]\nname = \"b\"",
        )
        .unwrap();

        let found = discover_extensions(&tmp).unwrap();
        assert_eq!(found.len(), 2);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn default_registry_is_empty() {
        let reg = MarketplaceRegistry::default();
        assert_eq!(reg.version, 1);
        assert!(reg.extensions.is_empty());
    }
}
