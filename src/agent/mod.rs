pub mod config;
pub mod events;

pub use config::AgentType;
pub use events::AgentEvent;

use crate::context::{
    ApproxTokenizer, ContextBudget, ContextManager, ConversationHistory, Tokenizer,
};
use crate::llm::types::{CompletionRequest, StopReason};
use crate::llm::LlmClient;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

pub struct Agent {
    pub id: String,
    pub agent_type: AgentType,
    client: Arc<dyn LlmClient>,
    history: ConversationHistory,
    context_mgr: ContextManager,
    event_tx: broadcast::Sender<AgentEvent>,
}

impl Agent {
    pub fn new(agent_type: AgentType, client: Arc<dyn LlmClient>) -> Self {
        let (event_tx, _) = broadcast::channel(64);

        let tokenizer = Arc::new(ApproxTokenizer);
        let system_tokens = tokenizer.count(&agent_type.system_prompt);
        let budget = ContextBudget::new(
            agent_type.max_context,
            system_tokens,
            0,   // tool_schema_tokens — Phase 3
            1024, // reserved_for_response
        );
        let context_mgr = ContextManager::new(
            budget,
            agent_type.context_policy.clone(),
            tokenizer,
        );

        let history = ConversationHistory::new(&agent_type.system_prompt);

        Agent {
            id: Uuid::new_v4().to_string(),
            agent_type,
            client,
            history,
            context_mgr,
            event_tx,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    fn emit(&self, event: AgentEvent) {
        let _ = self.event_tx.send(event);
    }

    pub async fn run(&mut self, initial_prompt: &str) -> anyhow::Result<String> {
        let max_turns = self.agent_type.max_turns;
        self.history.push(crate::llm::types::Message::user(initial_prompt));

        for turn in 0..max_turns {
            let conversation = self.history.conversation();
            let trimmed = self.context_mgr.trim(conversation);
            let tokens = self.context_mgr.estimate_tokens(&trimmed);

            self.emit(AgentEvent::TurnStart {
                agent_id: self.id.clone(),
                turn,
                tokens_in_context: tokens,
            });

            if trimmed.len() < conversation.len() {
                let dropped = conversation.len() - trimmed.len();
                self.emit(AgentEvent::ContextTrimmed {
                    agent_id: self.id.clone(),
                    policy: format!("{:?}", self.context_mgr.policy),
                    dropped,
                    tokens_freed: 0,
                });
            }

            let messages_to_send = {
                let mut v = vec![self.history.system().clone()];
                v.extend(trimmed);
                v
            };

            let req = CompletionRequest {
                model: self.agent_type.model.clone(),
                messages: messages_to_send,
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
                    self.history.push(crate::llm::types::Message::assistant_text(&text));
                    return Ok(text);
                }
                StopReason::ToolUse => {
                    // Phase 3: tool handling; for now return text
                    self.history.push(crate::llm::types::Message::assistant_text(&text));
                    return Ok(text);
                }
                StopReason::MaxTokens => {
                    anyhow::bail!("max_tokens exceeded");
                }
                StopReason::Other(_) => {
                    self.history.push(crate::llm::types::Message::assistant_text(&text));
                    return Ok(text);
                }
            }
        }

        anyhow::bail!("max_turns ({}) exhausted without a final response", max_turns)
    }
}
