use crate::llm::types::ToolCall;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    TurnStart {
        agent_id: String,
        turn: usize,
        tokens_in_context: usize,
    },
    ContextTrimmed {
        agent_id: String,
        policy: String,
        dropped: usize,
        tokens_freed: usize,
    },
    ToolCalled {
        agent_id: String,
        call: ToolCall,
    },
    ToolBlocked {
        agent_id: String,
        call: ToolCall,
        reason: String,
    },
    ToolResult {
        agent_id: String,
        call_id: String,
        output: String,
        compressed: bool,
    },
}
