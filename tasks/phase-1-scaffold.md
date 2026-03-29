# Phase 1 — Scaffold

**Goal**: A `cargo build --release` binary that sends a prompt to a llama.cpp instance
and prints the response. No tools, no context trimming, append-only history.

**Exit criteria**:
```
cargo check          # zero errors
cargo test           # all tests pass
TAU_BASE_URL=http://localhost:8080/v1 cargo run -- --prompt "say hello" --no-stream
```

---

## Step 1.1 — Cargo project

**Action**: Run `cargo init` in the repo root (or create `Cargo.toml` manually, since
there is already a LICENSE).

**`Cargo.toml`**:
```toml
[package]
name = "tau"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "tau"
path = "src/main.rs"

[dependencies]
tokio          = { version = "1",    features = ["full"] }
reqwest        = { version = "0.12", features = ["json", "stream"] }
serde          = { version = "1",    features = ["derive"] }
serde_json     = "1"
anyhow         = "1"
async-trait    = "0.1"
clap           = { version = "4",    features = ["derive"] }
futures        = "0.3"
tracing        = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
uuid           = { version = "1",    features = ["v4"] }
chrono         = { version = "0.4",  features = ["serde"] }
regex          = "1"
glob           = "0.3"
dashmap        = "6"
tokio-util     = { version = "0.7",  features = ["codec", "io"] }
bytes          = "1"

[target.'cfg(target_os = "linux")'.dependencies]
nix = { version = "0.29", features = ["process", "user", "signal"] }

[features]
default   = []
docker    = ["bollard"]
k8s       = ["kube", "k8s-openapi"]

[dependencies.bollard]
version  = "0.17"
optional = true

[dependencies.kube]
version  = "0.91"
features = ["runtime", "derive"]
optional = true

[dependencies.k8s-openapi]
version  = "0.22"
features = ["v1_29"]
optional = true
```

**`src/lib.rs`** — empty for now, filled in later phases:
```rust
// Public re-exports added per phase.
```

---

## Step 1.2 — LLM wire types (`src/llm/types.rs`)

All OpenAI `/v1/chat/completions` request and response types.

**Key types**:

```
Message          — role (system/user/assistant/tool), content, tool_calls, tool_call_id
Role             — System | User | Assistant | Tool
ToolCall         — id, type ("function"), function: FunctionCall
FunctionCall     — name, arguments (JSON string)
ToolDefinition   — type ("function"), function: FunctionDefinition
FunctionDefinition — name, description, parameters (serde_json::Value / JSON Schema)
CompletionRequest  — model, messages, tools, stream, max_tokens, temperature
CompletionResponse — id, choices: Vec<Choice>, usage: Option<Usage>
Choice           — message: Message, finish_reason: Option<String>
Usage            — prompt_tokens, completion_tokens, total_tokens
StopReason       — EndTurn | ToolUse | MaxTokens | Other(String)
StreamDelta      — content: Option<String>, tool_calls: Option<Vec<PartialToolCall>>
PartialToolCall  — index, id, call_type, function: PartialFunctionCall
PartialFunctionCall — name, arguments (both Option<String>, arrive in chunks)
```

**`Message` constructors** (inherent methods):
- `Message::system(content: &str) -> Self`
- `Message::user(content: &str) -> Self`
- `Message::assistant_text(content: &str) -> Self`
- `Message::assistant_with_tool_calls(content: Option<String>, calls: Vec<ToolCall>) -> Self`
- `Message::tool_result(call_id: &str, content: &str) -> Self`

**`CompletionResponse` helpers**:
- `fn text(&self) -> String` — first choice content, empty if None
- `fn stop_reason(&self) -> StopReason` — derived from `finish_reason`
- `fn tool_calls(&self) -> Vec<ToolCall>` — from first choice

**Serde notes**:
- `Role` serializes as lowercase string (`"system"`, `"user"`, etc.)
- `Message.tool_calls` and `Message.tool_call_id` use `#[serde(skip_serializing_if = "Option::is_none")]`
- `CompletionRequest.tools` is `Option<Vec<ToolDefinition>>`; skip if None
- `CompletionRequest.stream` skips if false

**Tests** (in `src/llm/types.rs`):
```rust
#[test]
fn test_message_serde_round_trip() { ... }

#[test]
fn test_stop_reason_from_finish_reason() {
    // "stop"       -> StopReason::EndTurn
    // "tool_calls" -> StopReason::ToolUse
    // "length"     -> StopReason::MaxTokens
    // "other"      -> StopReason::Other("other")
}
```

---

## Step 1.3 — LlmClient trait (`src/llm/mod.rs`)

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<CompletionResponse>;

    async fn complete_stream(
        &self,
        req: &CompletionRequest,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<StreamDelta>> + Send>>>;

    /// Override for model-specific tokenizers. Default: chars / 4.
    fn count_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }
}
```

Re-export `OpenAiClient` from this module.

---

## Step 1.4 — OpenAiClient (`src/llm/openai.rs`)

**Struct**:
```rust
pub struct OpenAiClient {
    pub base_url: String,
    pub api_key:  Option<String>,
    pub model:    String,
    http:         reqwest::Client,
}

impl OpenAiClient {
    pub fn new(base_url: impl Into<String>, api_key: Option<String>, model: impl Into<String>) -> Self

    /// Construct from environment:
    ///   TAU_BASE_URL  (default: http://localhost:8080/v1)
    ///   TAU_API_KEY   (optional)
    ///   TAU_MODEL     (default: "default")
    pub fn from_env() -> Self
}
```

**`complete()`** implementation notes:
- POST `{base_url}/chat/completions`
- Set `Authorization: Bearer {api_key}` only if `api_key.is_some()`; otherwise omit the header
- Serialize `req` to JSON body
- On HTTP 4xx/5xx: extract error body and return `Err(anyhow!(...))`
- Deserialize body to `CompletionResponse`

**`complete_stream()`** implementation notes (Phase 1 stub — can just error):
```rust
async fn complete_stream(&self, _req: &CompletionRequest)
    -> anyhow::Result<...>
{
    anyhow::bail!("streaming not yet implemented; use --no-stream")
}
```
Streaming is fully implemented in Phase 4.

**Test** (integration, marked `#[ignore]` so CI doesn't require a live server):
```rust
#[tokio::test]
#[ignore = "requires live llama.cpp at TAU_BASE_URL"]
async fn test_complete_live() { ... }
```

---

## Step 1.5 — Stub modules (empty but compilable)

Create the following with minimal stub content so the crate compiles end-to-end.
Fill them in during their respective phases.

| File | Stub content |
|------|-------------|
| `src/context/mod.rs` | `pub struct ContextManager;` `pub struct ContextBudget { pub available_for_history: usize }` |
| `src/context/policy.rs` | `pub enum ContextPolicy { AppendOnly }` |
| `src/context/tokenizer.rs` | `pub struct ApproxTokenizer; impl ApproxTokenizer { pub fn count(&self, t: &str) -> usize { t.len() / 4 } }` |
| `src/context/history.rs` | `pub struct ConversationHistory { messages: Vec<crate::llm::types::Message> }` |
| `src/context/compressor.rs` | `pub struct OutputCompressor;` |
| `src/tools/mod.rs` | `pub struct ToolRegistry { tools: Vec<Box<dyn crate::tools::Tool + Send + Sync>> }` + empty `Tool` trait |
| `src/harness/mod.rs` | `pub struct ToolHarness;` |
| `src/harness/policy.rs` | `pub struct AllowAll;` |
| `src/harness/sandbox.rs` | `pub struct NoSandbox;` |
| `src/agent/events.rs` | `pub enum AgentEvent { Other }` |
| `src/agent/config.rs` | `pub struct AgentType { pub system_prompt: String, pub model: String, pub max_turns: usize }` |
| `src/orchestrator/mod.rs` | `pub struct OrchestratorHandle;` |
| `src/telemetry/mod.rs` | `// TODO` |

---

## Step 1.6 — Agent struct + bare run loop (`src/agent/mod.rs`)

**Struct**:
```rust
pub struct Agent {
    pub id:         String,
    pub agent_type: AgentType,
    client:         Arc<dyn LlmClient>,
    history:        ConversationHistory,    // Phase 2: full implementation
    event_tx:       broadcast::Sender<AgentEvent>,
}
```

For Phase 1, `ConversationHistory` is just a `Vec<Message>` with a system prompt prepended.

**`run(initial_prompt: &str) -> anyhow::Result<String>`**:

```
loop up to max_turns:
  1. push user/continuation message to history (first call: initial_prompt)
  2. build CompletionRequest { model, messages: history.all(), tools: None, stream: false }
  3. call client.complete()
  4. match stop_reason:
     EndTurn  → push assistant message, return text
     ToolUse  → (Phase 3) for now: return text anyway
     MaxTokens → bail!("max_tokens exceeded")
  5. bail if loop exhausted
```

**`Agent::new()`** convenience constructor — takes `AgentType`, `client`, `system_prompt`.

---

## Step 1.7 — CLI (`src/main.rs`)

```rust
#[derive(Parser)]
#[command(name = "tau", about = "Local LLM agent")]
struct Cli {
    /// Single prompt (non-interactive mode)
    prompt: Option<String>,

    #[arg(long, env = "TAU_BASE_URL", default_value = "http://localhost:8080/v1")]
    base_url: String,

    #[arg(long, env = "TAU_API_KEY")]
    api_key: Option<String>,

    #[arg(long, env = "TAU_MODEL", default_value = "default")]
    model: String,

    #[arg(long, default_value_t = 50)]
    max_turns: usize,

    #[arg(long, default_value_t = 8192)]
    max_context: usize,

    #[arg(long)]
    system: Option<String>,

    /// Disable streaming (required until Phase 4)
    #[arg(long)]
    no_stream: bool,

    /// Verbose: print token counts etc.
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. init tracing (RUST_LOG env var)
    // 2. parse CLI
    // 3. build OpenAiClient
    // 4. build AgentType
    // 5. build Agent
    // 6. run prompt
    // 7. print result
}
```

For Phase 1 there is no interactive REPL — that comes in Phase 5.

---

## Checklist

- [ ] `Cargo.toml` created with all deps
- [ ] `src/llm/types.rs` — all types + constructors + tests
- [ ] `src/llm/mod.rs` — `LlmClient` trait
- [ ] `src/llm/openai.rs` — `OpenAiClient::complete()` working
- [ ] All stub modules created and compile
- [ ] `src/agent/mod.rs` — bare run loop (no tools)
- [ ] `src/main.rs` — CLI parses, runs agent, prints result
- [ ] `cargo check` passes with zero warnings (use `#[allow(dead_code)]` on stubs)
- [ ] `cargo test` passes (type serialization tests)
