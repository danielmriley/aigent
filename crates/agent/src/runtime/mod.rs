//! Core agent runtime — configuration, LLM routing, and conversation orchestration.

mod chat;
mod proactive;
mod reflection;
mod sleep;
mod tools;

pub use sleep::SleepGenerationResult;

use crate::AgentResult;

use aigent_config::AppConfig;
use aigent_llm::{LlmRouter, Provider};

// ConversationTurn is defined in the prompt crate; re-export for backward compat.
pub use aigent_prompt::ConversationTurn;

#[derive(Debug, Clone)]
pub struct AgentRuntime {
    pub config: AppConfig,
    pub llm: LlmRouter,
}

impl AgentRuntime {
    pub fn new(config: AppConfig) -> Self {
        #[allow(unused_mut)]
        let mut llm = LlmRouter::new()
            .with_suppress_thinking(config.agent.external_thinking);

        // Wire Candle local inference backend when the feature is compiled in
        // and the user has enabled it in [inference].
        #[cfg(feature = "candle")]
        if config.inference.candle_enabled {
            let candle_cfg = aigent_llm::candle_backend::CandleConfig {
                model_id: config.inference.candle_model_repo.clone(),
                gguf_file: config.inference.candle_model_file.clone(),
                model_path: if config.inference.candle_model_path.is_empty() {
                    None
                } else {
                    Some(config.inference.candle_model_path.clone())
                },
                max_seq_len: config.inference.candle_max_seq_len,
                temperature: config.inference.candle_temperature,
                top_p: config.inference.candle_top_p,
                repeat_penalty: config.inference.candle_repeat_penalty,
                repeat_penalty_last_n: 64,
                device: config.inference.candle_device.clone(),
            };
            tracing::info!(
                model = %candle_cfg.model_id,
                device = %candle_cfg.device,
                "candle local inference configured"
            );
            llm = llm.with_candle_config(candle_cfg);
        }

        Self { config, llm }
    }

    pub async fn run(&self) -> AgentResult<()> {
        Ok(())
    }

    pub async fn test_model_connection(&self) -> AgentResult<String> {
        let primary = Provider::from(self.config.llm.provider.as_str());

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
