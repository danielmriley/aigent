use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    async fn run(&self, input: &str) -> Result<String>;
}

#[derive(Default)]
pub struct ToolRegistry {
    names: Vec<String>,
}

impl ToolRegistry {
    pub fn register(&mut self, name: impl Into<String>) {
        self.names.push(name.into());
    }

    pub fn list(&self) -> &[String] {
        &self.names
    }
}
