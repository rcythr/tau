use crate::context::{ContextPolicy, OutputCompressor};

pub struct AgentType {
    pub system_prompt: String,
    pub model: String,
    pub max_turns: usize,
    pub max_context: usize,
    pub context_policy: ContextPolicy,
    pub compressor: OutputCompressor,
}

impl Default for AgentType {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful assistant.".to_string(),
            model: "default".to_string(),
            max_turns: 50,
            max_context: 8192,
            context_policy: ContextPolicy::PinPrefix { pinned: 4, recent: 20 },
            compressor: OutputCompressor::default(),
        }
    }
}
