//! Core agent runtime — configuration, LLM routing, and conversation orchestration.

mod chat;
mod proactive;
mod reflection;
mod sleep;
mod tools;

use anyhow::Result;

use aigent_config::AppConfig;
use aigent_llm::{LlmRouter, Provider};

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug, Clone)]
pub struct AgentRuntime {
    pub config: AppConfig,
    pub llm: LlmRouter,
}

impl AgentRuntime {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            llm: LlmRouter::default(),
        }
    }

    pub async fn run(&self) -> Result<()> {
        Ok(())
    }

    pub async fn test_model_connection(&self) -> Result<String> {
        let primary = if self.config.llm.provider.to_lowercase() == "openrouter" {
            Provider::OpenRouter
        } else {
            Provider::Ollama
        };

        let prompt = format!(
            "[healthcheck][bot-name:{}][thinking:{}] Reply with a short single-line confirmation.",
            self.config.agent.name, self.config.agent.thinking_level
        );

        let (provider_used, reply) = self
            .llm
            .chat_with_fallback(
                primary,
                &self.config.llm.ollama_model,
                &self.config.llm.openrouter_model,
                &prompt,
            )
            .await?;

        Ok(format!(
            "provider={provider_used:?} model={} reply={reply}",
            self.config.active_model()
        ))
    }

}
