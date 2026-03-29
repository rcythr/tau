pub mod config;
pub mod events;

pub use config::AgentType;
pub use events::AgentEvent;

use crate::llm::types::{CompletionRequest, Message, StopReason};
use crate::llm::LlmClient;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

#[allow(dead_code)]
pub struct Agent {
    pub id: String,
    pub agent_type: AgentType,
    client: Arc<dyn LlmClient>,
    history: Vec<Message>,
    event_tx: broadcast::Sender<AgentEvent>,
}

impl Agent {
    pub fn new(agent_type: AgentType, client: Arc<dyn LlmClient>) -> Self {
        let (event_tx, _) = broadcast::channel(64);
        let system_prompt = agent_type.system_prompt.clone();
        let mut history = Vec::new();
        if !system_prompt.is_empty() {
            history.push(Message::system(&system_prompt));
        }
        Agent {
            id: Uuid::new_v4().to_string(),
            agent_type,
            client,
            history,
            event_tx,
        }
    }

    #[allow(dead_code)]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    pub async fn run(&mut self, initial_prompt: &str) -> anyhow::Result<String> {
        let max_turns = self.agent_type.max_turns;
        self.history.push(Message::user(initial_prompt));

        for _ in 0..max_turns {

            let req = CompletionRequest {
                model: self.agent_type.model.clone(),
                messages: self.history.clone(),
                tools: None,
                stream: false,
                max_tokens: None,
                temperature: None,
            };

            let response = self.client.complete(&req).await?;
            let text = response.text();
            let stop_reason = response.stop_reason();

            match stop_reason {
                StopReason::EndTurn => {
                    self.history.push(Message::assistant_text(&text));
                    return Ok(text);
                }
                StopReason::ToolUse => {
                    // Phase 3: tool handling; for now return text
                    self.history.push(Message::assistant_text(&text));
                    return Ok(text);
                }
                StopReason::MaxTokens => {
                    anyhow::bail!("max_tokens exceeded");
                }
                StopReason::Other(_) => {
                    self.history.push(Message::assistant_text(&text));
                    return Ok(text);
                }
            }
        }

        anyhow::bail!("max_turns ({}) exhausted without a final response", max_turns)
    }
}
