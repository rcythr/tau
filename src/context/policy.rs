#[derive(Debug, Clone)]
pub enum ContextPolicy {
    /// Never trim. Return all messages. Fail (caller's responsibility) if budget exceeded.
    AppendOnly,

    /// Keep first `pinned` messages + last `recent` messages.
    /// If total fits in budget, returns all.
    /// Middle messages are dropped when budget is tight.
    PinPrefix { pinned: usize, recent: usize },

    /// Append-only until `trigger` fraction of budget is used.
    /// When triggered: caller must call `compact()` (Phase 4, needs LLM).
    /// Until compaction is available, falls back to PinPrefix behavior.
    RollingCompact {
        pinned: usize,
        recent: usize,
        trigger: f32,
    },

    /// Always return empty slice (system prompt only).
    /// Perfect cache reuse; for sub-agents with discrete single tasks.
    FreshContext,
}
