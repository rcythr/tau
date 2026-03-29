# Phase 4 — Streaming & Observability

**Goal**: Token stream from the LLM prints in real time. All agent activity is
observable via `AgentEvent`. Trajectory is written to JSONL. `tracing` subscriber
configured. `RollingCompact` context policy gets its LLM-summarization hook.

**Prerequisite**: Phase 3 `cargo test` passes.

**Exit criteria**:
```
cargo test telemetry::
TAU_BASE_URL=http://localhost:8080/v1 cargo run -- \
    --prompt "count to 10" \
    --trajectory /tmp/tau-test.jsonl
# Tokens stream to stdout in real time
# /tmp/tau-test.jsonl is valid JSONL after the run
```

---

## Step 4.1 — SSE Streaming in OpenAiClient (`src/llm/openai.rs`)

Replace the Phase 1 stub with a real `complete_stream()`.

### OpenAI streaming SSE format

```
data: {"id":"x","choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}
data: {"id":"x","choices":[{"delta":{"content":" world"},"finish_reason":null}]}
data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}
data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":null}]}
data: {"id":"x","choices":[{"delta":{},"finish_reason":"tool_calls"}]}
data: [DONE]
```

### Implementation

Use `tokio_util::codec::LinesCodec` to split the response byte stream into lines:

```rust
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio::io::AsyncReadExt;
use futures::StreamExt;

async fn complete_stream(&self, req: &CompletionRequest)
    -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<StreamDelta>> + Send>>>
{
    let mut body = serde_json::to_value(req)?;
    body["stream"] = json!(true);

    let response = self.http
        .post(format!("{}/chat/completions", self.base_url))
        .json(&body)
        .send()
        .await?
        .error_for_status()?;

    let byte_stream = response.bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));

    let reader = tokio_util::io::StreamReader::new(byte_stream);
    let lines  = FramedRead::new(reader, LinesCodec::new());

    let stream = lines
        .filter_map(|line_result| async move {
            match line_result {
                Err(e) => Some(Err(anyhow::anyhow!(e))),
                Ok(line) => {
                    let line = line.trim().to_string();
                    if line.is_empty() || line == "data: [DONE]" { return None; }
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        match serde_json::from_str::<StreamChunk>(json_str) {
                            Ok(chunk) => Some(Ok(StreamDelta::from_chunk(chunk))),
                            Err(e)    => Some(Err(anyhow::anyhow!("SSE parse: {e}: {json_str}"))),
                        }
                    } else {
                        None  // ignore non-data lines (comments, etc.)
                    }
                }
            }
        });

    Ok(Box::pin(stream))
}
```

**`StreamChunk` type** for deserializing a single SSE data line:
```rust
#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
    usage:   Option<Usage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta:         DeltaContent,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct DeltaContent {
    content:    Option<String>,
    tool_calls: Option<Vec<PartialToolCall>>,
}
```

**`StreamDelta`** (public type, already stubbed in Phase 1):
```rust
pub struct StreamDelta {
    pub content:       Option<String>,
    pub tool_calls:    Option<Vec<PartialToolCall>>,
    pub finish_reason: Option<String>,
    pub usage:         Option<Usage>,
}
```

### Accumulating streaming tool calls

Tool call arguments arrive in multiple chunks with an `index` field:
```
chunk 1: tool_calls: [{ index: 0, id: "c1", type: "function", function: { name: "bash", arguments: "" } }]
chunk 2: tool_calls: [{ index: 0, function: { arguments: "{\"cmd\"" } }]
chunk 3: tool_calls: [{ index: 0, function: { arguments: ": \"ls\"}" } }]
```

Accumulate in a `HashMap<usize, ToolCallAccumulator>`:
```rust
struct ToolCallAccumulator {
    id:        String,
    name:      String,
    arguments: String,
}
```

After `finish_reason` arrives, flush accumulators to `Vec<ToolCall>`.

### Agent loop streaming mode

When `--no-stream` is NOT set, use `complete_stream()`:
1. Collect `StreamDelta`s
2. Print `delta.content` chunks to stdout immediately (no newline flush needed if using `print!` + `std::io::stdout().flush()`)
3. Accumulate full content + tool calls
4. After `[DONE]`: construct a synthetic `CompletionResponse` from accumulated state

---

## Step 4.2 — AgentEvent enum (full) (`src/agent/events.rs`)

Replace stub with full enum:

```rust
use crate::llm::types::{ToolCall, StopReason};

#[derive(Debug, Clone)]
pub enum AgentEvent {
    TurnStart {
        agent_id:          String,
        turn:              usize,
        tokens_in_context: usize,
    },
    TurnEnd {
        agent_id:    String,
        turn:        usize,
        stop_reason: StopReason,
    },
    StreamDelta {
        agent_id: String,
        content:  String,
    },
    ToolCalled {
        agent_id: String,
        call:     ToolCall,
    },
    ToolBlocked {
        agent_id: String,
        call:     ToolCall,
        reason:   String,
    },
    ToolResult {
        agent_id:   String,
        call_id:    String,
        output:     String,
        compressed: bool,
    },
    ContextTrimmed {
        agent_id:     String,
        policy:       String,
        dropped:      usize,
        tokens_freed: usize,
    },
    ContextCompacted {
        agent_id:      String,
        before_tokens: usize,
        after_tokens:  usize,
    },
    AgentSpawned {
        parent_id: Option<String>,
        child_id:  String,
        agent_type: String,
    },
    AgentCompleted {
        agent_id: String,
        output:   String,
    },
    AgentFailed {
        agent_id: String,
        error:    String,
    },
    MessageReceived {
        agent_id: String,
        from:     String,
        content:  String,
    },
}
```

---

## Step 4.3 — Trajectory Logger (`src/telemetry/trajectory.rs`)

```rust
pub struct TrajectoryLogger {
    writer:     tokio::io::BufWriter<tokio::fs::File>,
    format:     TrajectoryFormat,
    session_id: uuid::Uuid,
}

#[derive(Debug, Clone)]
pub enum TrajectoryFormat {
    OpenAiMessages,   // one JSON line per message (fine-tune compatible)
    Extended,         // one JSON line per turn with full metadata
    Dual {            // write both simultaneously
        openai_path:   std::path::PathBuf,
        extended_path: std::path::PathBuf,
    },
}

impl TrajectoryLogger {
    pub async fn new(path: &std::path::Path, format: TrajectoryFormat) -> anyhow::Result<Self>;

    /// Log a completed turn (call after tool results are appended to history).
    pub async fn log_turn(&mut self, record: &TurnRecord) -> anyhow::Result<()>;

    pub async fn flush(&mut self) -> anyhow::Result<()>;
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TurnRecord {
    pub session_id:         uuid::Uuid,
    pub agent_id:           String,
    pub turn:               usize,
    pub timestamp:          chrono::DateTime<chrono::Utc>,
    pub messages_sent:      Vec<crate::llm::types::Message>,
    pub response_text:      String,
    pub tool_calls:         Vec<ToolCallRecord>,
    pub context_budget:     crate::context::mod::ContextBudget,  // adjust path
    pub latency_ms:         u64,
    pub prompt_tokens:      usize,
    pub completion_tokens:  usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolCallRecord {
    pub id:           String,
    pub name:         String,
    pub arguments:    String,
    pub allowed:      bool,
    pub output:       Option<String>,
    pub compressed:   bool,
    pub duration_ms:  u64,
}
```

**OpenAI fine-tune format** (one record per session):
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]}
```

**Extended format** (one record per turn):
```json
{"session_id": "...", "agent_id": "...", "turn": 1, "timestamp": "...", "messages_sent": [...], ...}
```

**Tests**:
```rust
#[tokio::test] async fn trajectory_writes_valid_jsonl()
#[tokio::test] async fn trajectory_extended_format_has_metadata()
#[tokio::test] async fn trajectory_openai_format_messages_only()
```

---

## Step 4.4 — Tracing integration (`src/telemetry/tracing.rs`)

```rust
pub fn init_tracing(verbose: bool) {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            if verbose { EnvFilter::new("tau=debug") }
            else       { EnvFilter::new("tau=info")  }
        });
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();
}
```

In `main.rs`: call `init_tracing(cli.verbose)` before anything else.

Add `tracing::debug!` / `tracing::info!` spans in the agent loop:
- `debug!(turn, tokens_in_context, "turn start")`
- `debug!(tool_name, "tool called")`
- `info!(turn, "turn complete")`

---

## Step 4.5 — RollingCompact compaction hook

Add a compaction method to `Agent` (called by `ContextManager` when policy is `RollingCompact`):

```rust
impl Agent {
    async fn compact_history(&mut self) -> anyhow::Result<()> {
        // 1. Identify messages to compact: conversation[pinned..len-recent]
        // 2. Build a compaction prompt:
        //    "Summarize the following conversation history concisely, preserving key facts and decisions:\n<messages>"
        // 3. Call self.client.complete() with a one-shot request (no tools)
        // 4. Replace the compacted range with a single summary Message::user(summary)
        // 5. Emit AgentEvent::ContextCompacted { before_tokens, after_tokens }
    }
}
```

Wire into the trimming path in `run()`:
```rust
if matches!(policy, ContextPolicy::RollingCompact { trigger, .. })
    && (current_tokens as f32 / budget.available_for_history as f32) > trigger
{
    self.compact_history().await?;
}
```

---

## Step 4.6 — Verbose mode (`-v`)

When `--verbose` / `-v` is set, print per-turn stats to stderr:
```
[turn 1] context: 1234 tokens | latency: 450ms | tools: 2
[turn 2] context: 2100 tokens | latency: 230ms | tools: 0
```

Subscribe to `event_tx` in `main.rs` and format `TurnStart` / `TurnEnd` events.

---

## Checklist

- [ ] `src/llm/openai.rs` — `complete_stream()` fully implemented
- [ ] `src/llm/types.rs` — `StreamDelta`, `PartialToolCall`, `StreamChunk` types
- [ ] Streaming accumulator for tool calls across chunks
- [ ] `src/agent/events.rs` — full `AgentEvent` enum
- [ ] `src/telemetry/trajectory.rs` — `TrajectoryLogger` + tests
- [ ] `src/telemetry/tracing.rs` — `init_tracing()`
- [ ] `src/agent/mod.rs` — streaming mode + trajectory logging + `compact_history()`
- [ ] `src/main.rs` — `--trajectory`, `--log-format`, `-v` flags; stream tokens to stdout
- [ ] `cargo test telemetry::` passes
- [ ] Tokens stream in real time to stdout
- [ ] Trajectory JSONL is valid after run
