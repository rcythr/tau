# Phase 2 — Context Management

**Goal**: Replace the Phase 1 stub `ConversationHistory` / context code with the full
KV-cache-aware implementation. The agent loop uses `ContextManager` to trim history
before every LLM call.

**Prerequisite**: Phase 1 `cargo test` passes.

**Exit criteria**:
```
cargo test context::   # all context module tests pass
```
Running the agent with `--max-context 512` on a long conversation must emit
`ContextTrimmed` events and not exceed the configured budget.

---

## Background: KV Cache & Policy Choice

llama.cpp caches KV pairs by token position. Any mutation to tokens at position N
invalidates the cache for N+∞. This means:

- **Best**: append-only — prefix never changes, 100% cache reuse
- **Good**: pin first K + keep last N (PinPrefix) — the system prompt prefix is always stable
- **Good**: periodic compaction into a stable summary block (RollingCompact)
- **Worst**: drop from the front (naïve sliding window) — invalidates everything after the drop

`SlidingWindow` is intentionally NOT a first-class `ContextPolicy` variant. Users who want
it can set `PinPrefix { pinned: 0, recent: N }` — making the cache trade-off visible.

---

## Step 2.1 — Tokenizer (`src/context/tokenizer.rs`)

Replace stub with full implementation.

```rust
pub trait Tokenizer: Send + Sync {
    fn count(&self, text: &str) -> usize;
}

/// Default: chars / 4. Zero dependencies, ~15% overcount for English.
pub struct ApproxTokenizer;
impl Tokenizer for ApproxTokenizer {
    fn count(&self, text: &str) -> usize { (text.len() + 3) / 4 }
}

/// Worst-case: 1 char = 1 token. Use when you must never exceed budget.
pub struct ConservativeTokenizer;
impl Tokenizer for ConservativeTokenizer {
    fn count(&self, text: &str) -> usize { text.len() }
}
```

**`count_message(msg: &Message) -> usize`** helper — sums `content` + all `tool_calls`
arguments JSON. Add ~4 tokens overhead per message for role + formatting tokens.

**Tests**:
```rust
#[test] fn approx_count_ascii()    // "hello world" -> 2-3 tokens
#[test] fn approx_count_empty()    // "" -> 0
#[test] fn conservative_count()    // always >= approx
```

---

## Step 2.2 — Context Budget (`src/context/mod.rs`)

```rust
pub struct ContextBudget {
    pub model_limit:           usize,  // e.g. 8192
    pub system_tokens:         usize,  // computed once from system prompt
    pub tool_schema_tokens:    usize,  // computed once at registry build time
    pub reserved_for_response: usize,  // default: 1024
    pub available_for_history: usize,  // = model_limit - system - tools - reserved
}

impl ContextBudget {
    pub fn new(
        model_limit: usize,
        system_tokens: usize,
        tool_schema_tokens: usize,
        reserved_for_response: usize,
    ) -> Self {
        let available = model_limit
            .saturating_sub(system_tokens)
            .saturating_sub(tool_schema_tokens)
            .saturating_sub(reserved_for_response);
        Self { model_limit, system_tokens, tool_schema_tokens, reserved_for_response, available_for_history: available }
    }
}
```

**`ContextManager`**:
```rust
pub struct ContextManager {
    pub budget:    ContextBudget,
    pub policy:    ContextPolicy,
    tokenizer:     Arc<dyn Tokenizer>,
}

impl ContextManager {
    /// Given the full history (excluding system prompt), return the slice to send.
    /// The system prompt is managed separately and always prepended by the caller.
    pub fn trim(&self, messages: &[Message]) -> Vec<Message>;

    /// Estimate total tokens for a slice of messages.
    pub fn estimate_tokens(&self, messages: &[Message]) -> usize;

    /// True if the current history already exceeds the budget.
    pub fn is_over_budget(&self, messages: &[Message]) -> bool;
}
```

**Tests**:
```rust
#[test] fn budget_available_calculation()
#[test] fn budget_never_negative_on_underflow()  // saturating_sub
#[test] fn manager_estimates_tokens()
```

---

## Step 2.3 — Context Policies (`src/context/policy.rs`)

```rust
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
    RollingCompact { pinned: usize, recent: usize, trigger: f32 },

    /// Always return empty slice (system prompt only).
    /// Perfect cache reuse; for sub-agents with discrete single tasks.
    FreshContext,
}
```

**`ContextManager::trim()` implementation**:

```
AppendOnly:
  return messages.to_vec()

FreshContext:
  return vec![]

PinPrefix { pinned, recent }:
  if estimate_tokens(messages) <= budget.available_for_history:
      return messages.to_vec()  // fits, no trimming needed
  let prefix = messages[..pinned.min(messages.len())]
  let suffix = messages[messages.len().saturating_sub(recent)..]
  // Avoid double-counting overlap if pinned + recent >= total
  combine prefix + suffix (dedup by index)
  if still over budget: reduce recent until fits
  return combined

RollingCompact:
  same as PinPrefix for now (compaction hook added in Phase 4)
```

**Tests**:
```rust
#[test] fn append_only_returns_all()
#[test] fn fresh_context_returns_empty()
#[test] fn pin_prefix_no_trim_when_fits()
#[test] fn pin_prefix_trims_middle_when_over_budget()
#[test] fn pin_prefix_preserves_pinned_count()
#[test] fn pin_prefix_preserves_recent_count()
#[test] fn pin_prefix_handles_empty_input()
#[test] fn pin_prefix_handles_fewer_messages_than_pinned_plus_recent()
```

---

## Step 2.4 — Conversation History (`src/context/history.rs`)

Replace stub with full implementation.

```rust
pub type SnapshotId = usize;

pub struct ConversationHistory {
    system_prompt: Message,           // always role=system, never trimmed
    messages:      Vec<Message>,      // append-only conversation history
    snapshots:     Vec<(SnapshotId, Vec<Message>)>,
    next_snapshot: SnapshotId,
}

impl ConversationHistory {
    pub fn new(system_prompt: &str) -> Self;

    /// Append a message to the history.
    pub fn push(&mut self, msg: Message);

    /// All messages including system prompt (for display/logging).
    pub fn all(&self) -> Vec<Message>;

    /// Conversation messages only (excluding system prompt).
    /// Pass this to ContextManager::trim().
    pub fn conversation(&self) -> &[Message];

    /// System prompt message.
    pub fn system(&self) -> &Message;

    /// Save current state; returns a snapshot ID.
    pub fn checkpoint(&mut self) -> SnapshotId;

    /// Restore to a snapshot. Returns Err if ID not found.
    pub fn restore(&mut self, id: SnapshotId) -> anyhow::Result<()>;

    /// Clone current state (for parallel exploration / sub-agent forking).
    pub fn fork(&self) -> Self;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

**Tests**:
```rust
#[test] fn new_history_has_system_message()
#[test] fn push_adds_to_conversation_not_system()
#[test] fn all_includes_system_first()
#[test] fn checkpoint_and_restore()
#[test] fn restore_unknown_id_returns_err()
#[test] fn fork_is_independent()  // mutations to fork don't affect original
```

---

## Step 2.5 — Output Compressor (`src/context/compressor.rs`)

Applied to tool output before it enters the history. Prevents a single large
bash output from consuming the entire context budget.

```rust
#[derive(Debug, Clone)]
pub struct OutputCompressor {
    pub max_bytes: usize,
    pub strategy:  CompressionStrategy,
}

#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    /// Keep first `head` bytes + last `tail` bytes; insert "[... N bytes omitted ...]" in middle.
    HeadTail { head: usize, tail: usize },

    /// Keep last `size` bytes only.
    TailOnly { size: usize },

    /// Hard truncate at max_bytes with "[truncated]" suffix.
    HardTruncate,
}

impl Default for OutputCompressor {
    fn default() -> Self {
        Self {
            max_bytes: 8 * 1024,  // 8 KiB
            strategy: CompressionStrategy::HeadTail { head: 2048, tail: 2048 },
        }
    }
}

impl OutputCompressor {
    /// Compress `output` if it exceeds `max_bytes`. Returns (compressed, was_compressed).
    pub fn compress(&self, output: &str) -> (String, bool);
}
```

**`HeadTail` output format**:
```
<first head bytes>
... [12345 bytes omitted] ...
<last tail bytes>
```

**Tests**:
```rust
#[test] fn compress_short_string_unchanged()
#[test] fn compress_head_tail_elides_middle()
#[test] fn compress_head_tail_omit_count_correct()
#[test] fn compress_tail_only()
#[test] fn compress_hard_truncate()
#[test] fn compress_returns_was_compressed_flag()
```

---

## Step 2.6 — Wire into Agent Loop (`src/agent/mod.rs`)

Update `Agent` to use the full context management stack:

```rust
pub struct Agent {
    pub id:          String,
    pub agent_type:  AgentType,
    client:          Arc<dyn LlmClient>,
    history:         ConversationHistory,    // full impl
    context_mgr:     ContextManager,
    event_tx:        broadcast::Sender<AgentEvent>,
    trajectory:      Option<Arc<tokio::sync::Mutex<TrajectoryLogger>>>,
}
```

Update `AgentType` to carry `max_context` and `context_policy`:
```rust
pub struct AgentType {
    pub system_prompt:   String,
    pub model:           String,
    pub max_turns:       usize,
    pub max_context:     usize,          // added
    pub context_policy:  ContextPolicy,  // added
    pub compressor:      OutputCompressor, // added
}
```

In `run()`, before building `CompletionRequest`:
```rust
let conversation = self.history.conversation();
let trimmed = self.context_mgr.trim(conversation);
let tokens  = self.context_mgr.estimate_tokens(&trimmed);
self.emit(AgentEvent::TurnStart { agent_id: self.id.clone(), turn, tokens_in_context: tokens });

// If trim dropped messages, emit ContextTrimmed event
if trimmed.len() < conversation.len() {
    let dropped = conversation.len() - trimmed.len();
    self.emit(AgentEvent::ContextTrimmed {
        agent_id: self.id.clone(),
        policy:   format!("{:?}", self.context_mgr.policy),
        dropped,
        tokens_freed: 0,  // exact value added in Phase 4
    });
}

let messages_to_send = {
    let mut v = vec![self.history.system().clone()];
    v.extend(trimmed);
    v
};
```

---

## Step 2.7 — CLI flags

Add to `src/main.rs`:
- `--max-context <TOKENS>` → `AgentType.max_context` (default: 8192)
- `--context-policy <POLICY>` → parse `"append-only"` | `"pin-prefix"` | `"fresh"` (default: `"pin-prefix"`)

---

## Checklist

- [ ] `src/context/tokenizer.rs` — full impl + tests
- [ ] `src/context/mod.rs` — `ContextBudget` + `ContextManager::trim()` + tests
- [ ] `src/context/policy.rs` — all `ContextPolicy` variants + trim logic + tests
- [ ] `src/context/history.rs` — `ConversationHistory` + checkpoint/fork + tests
- [ ] `src/context/compressor.rs` — `OutputCompressor` all strategies + tests
- [ ] `src/agent/mod.rs` — wired to `ContextManager`, emits `ContextTrimmed`
- [ ] `src/agent/config.rs` — `AgentType` carries context config
- [ ] `src/main.rs` — `--max-context` and `--context-policy` flags
- [ ] `cargo test context::` passes
- [ ] Manual test: small `--max-context` forces trim, agent still completes
