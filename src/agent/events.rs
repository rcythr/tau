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
    Other,
}
