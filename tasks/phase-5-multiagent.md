# Phase 5 — Multi-Agent, Actor Tools & REPL

**Goal**: Multiple agents run as tokio tasks and coordinate through a typed
message-passing system. An `OrchestratorBus` trait (injected into actor tools)
decouples tools from the orchestrator implementation. `LocalOrchestrator`
manages the registry. Users write static topologies in Rust or let the LLM
drive spawning dynamically via `SpawnAgentTool`. An interactive REPL is added.

**Prerequisite**: Phase 4 `cargo test` passes.

**Exit criteria**:
```
cargo test orchestrator:: actor_tools::
# Static topology example: planner → 2 parallel coders
cargo run --example static_topology
# Dynamic topology example: one agent with SpawnAgentTool
cargo run --example dynamic_topology
# REPL
cargo run -- --interactive
```

---

## Architecture Note: Avoiding Circular Dependencies

`Tool` implementations (`actor_tools/`) need to call the orchestrator.
The orchestrator creates agents that hold tools. Naive design creates a cycle.

**Resolution**: `OrchestratorBus` is a narrow trait defined in `src/orchestrator/mod.rs`.
Actor tools hold `OrchestratorHandle(Arc<dyn OrchestratorBus>)`. They do NOT import
any concrete orchestrator type. The concrete `LocalOrchestrator` implements
`OrchestratorBus`.

Module dependency directions (no cycles):
```
actor_tools  →  orchestrator::OrchestratorBus  (narrow trait only)
orchestrator →  agent  →  tools  →  (no upward deps)
```

---

## Step 5.1 — Orchestrator types (`src/orchestrator/mod.rs`)

Replace stub with full types.

### `ActorMessage`
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActorMessage {
    pub from:           String,   // AgentId
    pub to:             String,   // AgentId
    pub body:           String,
    pub correlation_id: uuid::Uuid,
}
```

### `AgentCompletion` / `ExitReason`
```rust
#[derive(Debug, Clone)]
pub struct AgentCompletion {
    pub output:      String,
    pub exit_reason: ExitReason,
}

#[derive(Debug, Clone)]
pub enum ExitReason {
    Completed,
    Failed(String),
    Cancelled,
}
```

### `AgentHandle`

The `completion` field is a `watch::Receiver<Option<AgentCompletion>>` — allows
multiple callers to await the same agent's result without consuming a oneshot.

```rust
#[derive(Clone)]
pub struct AgentHandle {
    pub id:         String,
    pub agent_type: String,
    pub mailbox_tx: tokio::sync::mpsc::Sender<ActorMessage>,
    pub completion: tokio::sync::watch::Receiver<Option<AgentCompletion>>,
}

impl AgentHandle {
    /// Block until the agent finishes. Returns its output or error.
    pub async fn wait(&mut self) -> anyhow::Result<String> {
        self.completion.wait_for(|v| v.is_some()).await?;
        match self.completion.borrow().clone().unwrap().exit_reason {
            ExitReason::Completed     => Ok(self.completion.borrow().clone().unwrap().output),
            ExitReason::Failed(e)     => anyhow::bail!(e),
            ExitReason::Cancelled     => anyhow::bail!("agent was cancelled"),
        }
    }

    /// Send a message to this agent's inbox.
    pub async fn send(&self, from: &str, body: &str) -> anyhow::Result<()> {
        self.mailbox_tx.send(ActorMessage {
            from:           from.to_string(),
            to:             self.id.clone(),
            body:           body.to_string(),
            correlation_id: uuid::Uuid::new_v4(),
        }).await.map_err(|_| anyhow::anyhow!("agent mailbox closed"))
    }
}
```

### `OrchestratorBus` — narrow trait for tools

```rust
#[async_trait::async_trait]
pub trait OrchestratorBus: Send + Sync {
    /// Route a message to a named agent.
    async fn send_message(&self, msg: ActorMessage) -> anyhow::Result<()>;

    /// Spawn a sub-agent by type name, wait for it to finish, return its output.
    /// `config` is a JSON object; factory interprets it (e.g. `{"initial_task": "..."}`)
    async fn spawn_and_await(
        &self,
        agent_type:   &str,
        config:       serde_json::Value,
        initiated_by: String,
    ) -> anyhow::Result<String>;
}

/// Cheaply clonable handle. Passed into actor tools at agent construction.
#[derive(Clone)]
pub struct OrchestratorHandle(pub std::sync::Arc<dyn OrchestratorBus>);
```

### `Orchestrator` — full control-plane trait

```rust
#[async_trait::async_trait]
pub trait Orchestrator: OrchestratorBus {
    fn register_factory(&self, type_name: &str, factory: std::sync::Arc<dyn AgentFactory>);
    async fn spawn_agent(&self, type_name: &str, config: serde_json::Value, initiated_by: Option<String>) -> anyhow::Result<AgentHandle>;
    async fn await_agent(&self, id: &str) -> anyhow::Result<String>;
    async fn shutdown(&self) -> anyhow::Result<()>;
    fn event_stream(&self) -> tokio::sync::broadcast::Receiver<AgentEvent>;
    fn handle(&self) -> OrchestratorHandle;
}
```

### `AgentFactory` trait

```rust
#[async_trait::async_trait]
pub trait AgentFactory: Send + Sync {
    fn type_name(&self) -> &str;
    async fn build(
        &self,
        id:     String,
        config: serde_json::Value,
        handle: OrchestratorHandle,
        event_tx: tokio::sync::broadcast::Sender<AgentEvent>,
    ) -> anyhow::Result<Agent>;
}
```

---

## Step 5.2 — Message Router (`src/orchestrator/router.rs`)

```rust
pub struct MessageRouter {
    senders: dashmap::DashMap<String, tokio::sync::mpsc::Sender<ActorMessage>>,
}

impl MessageRouter {
    pub fn new() -> Self { Self { senders: dashmap::DashMap::new() } }
    pub fn register(&self, id: String, tx: tokio::sync::mpsc::Sender<ActorMessage>);
    pub fn unregister(&self, id: &str);
    pub async fn route(&self, msg: ActorMessage) -> anyhow::Result<()>;
}
```

**Tests**:
```rust
#[tokio::test] fn route_delivers_to_registered_agent()
#[tokio::test] fn route_error_for_unknown_agent()
#[tokio::test] fn unregister_removes_agent()
```

---

## Step 5.3 — LocalOrchestrator (`src/orchestrator/local.rs`)

The key implementation challenge: `LocalOrchestrator` needs to return
`OrchestratorHandle` pointing to itself, but it's constructed as `Arc<Self>`.

**Solution**: Store the self-reference in a `OnceLock` initialized immediately
after `Arc::new()`:

```rust
pub struct LocalOrchestrator {
    client:    std::sync::Arc<dyn crate::llm::LlmClient>,
    factories: dashmap::DashMap<String, std::sync::Arc<dyn AgentFactory>>,
    agents:    dashmap::DashMap<String, AgentHandle>,
    router:    std::sync::Arc<MessageRouter>,
    events_tx: tokio::sync::broadcast::Sender<AgentEvent>,
    self_bus:  std::sync::OnceLock<OrchestratorHandle>,
}

impl LocalOrchestrator {
    pub fn new(client: std::sync::Arc<dyn crate::llm::LlmClient>) -> std::sync::Arc<Self> {
        let (events_tx, _) = tokio::sync::broadcast::channel(1024);
        let orch = std::sync::Arc::new(Self {
            client,
            factories: dashmap::DashMap::new(),
            agents:    dashmap::DashMap::new(),
            router:    std::sync::Arc::new(MessageRouter::new()),
            events_tx,
            self_bus:  std::sync::OnceLock::new(),
        });
        // Set self-reference immediately after construction
        let bus: std::sync::Arc<dyn OrchestratorBus> =
            std::sync::Arc::clone(&orch) as std::sync::Arc<dyn OrchestratorBus>;
        orch.self_bus.set(OrchestratorHandle(bus)).ok();
        orch
    }
}
```

**`spawn_agent()` implementation**:
```
1. Look up factory by type_name
2. Generate agent_id = "{type_name}-{uuid}"
3. Create (mailbox_tx, mailbox_rx) mpsc channel
4. Create (completion_tx, completion_rx) watch channel
5. Register mailbox_tx in router
6. Call factory.build(id, config, self.handle(), self.events_tx.clone())
7. tokio::spawn(async move { run agent; send completion; router.unregister })
8. Store AgentHandle in self.agents
9. Return AgentHandle
```

**`OrchestratorBus` impl** (delegates to router and spawn_agent):
```rust
#[async_trait]
impl OrchestratorBus for LocalOrchestrator {
    async fn send_message(&self, msg: ActorMessage) -> anyhow::Result<()> {
        self.router.route(msg).await
    }

    async fn spawn_and_await(&self, agent_type: &str, config: serde_json::Value, initiated_by: String) -> anyhow::Result<String> {
        let mut handle = self.spawn_agent(agent_type, config, Some(initiated_by)).await?;
        handle.wait().await
    }
}
```

**Tests**:
```rust
#[tokio::test] fn spawn_agent_returns_handle()
#[tokio::test] fn spawn_agent_completes()
#[tokio::test] fn spawn_unknown_type_returns_err()
#[tokio::test] fn send_message_routes_to_agent()
```

---

## Step 5.4 — Actor Tools (`src/actor_tools/`)

### `src/actor_tools/send_message.rs`

```rust
pub struct SendMessageTool {
    pub sender_id: String,
    pub handle:    OrchestratorHandle,
}

// Parameters: { "to": "<agent_id>", "body": "<message>" }

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self)        -> &'static str { "send_message" }
    fn description(&self) -> &'static str {
        "Send a message to another running agent by its ID. Fire-and-forget."
    }
    fn parameters_schema(&self) -> serde_json::Value { ... }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let to   = input["to"].as_str().ok_or_else(|| anyhow!("missing to"))?;
        let body = input["body"].as_str().ok_or_else(|| anyhow!("missing body"))?;
        self.handle.0.send_message(ActorMessage {
            from:           self.sender_id.clone(),
            to:             to.to_string(),
            body:           body.to_string(),
            correlation_id: uuid::Uuid::new_v4(),
        }).await?;
        Ok(ToolOutput::text(format!("message sent to {to}")))
    }
}
```

### `src/actor_tools/spawn_agent.rs`

```rust
pub struct SpawnAgentTool {
    pub caller_id:     String,
    pub handle:        OrchestratorHandle,
    pub allowed_types: Vec<String>,   // empty = all types allowed
}

// Parameters: { "agent_type": "<type_name>", "task": "<initial prompt>" }
// Returns: the sub-agent's final output

#[async_trait]
impl Tool for SpawnAgentTool {
    fn name(&self)        -> &'static str { "spawn_agent" }
    fn description(&self) -> &'static str {
        "Spawn a sub-agent by type name and wait for it to complete. Returns the agent's output."
    }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let agent_type = input["agent_type"].as_str().ok_or_else(|| anyhow!("missing agent_type"))?;
        let task       = input["task"].as_str().ok_or_else(|| anyhow!("missing task"))?;

        if !self.allowed_types.is_empty() && !self.allowed_types.iter().any(|t| t == agent_type) {
            return Ok(ToolOutput::error(format!("agent type '{agent_type}' not in allowed list")));
        }

        let config = serde_json::json!({ "initial_task": task });
        let output = self.handle.0
            .spawn_and_await(agent_type, config, self.caller_id.clone())
            .await?;
        Ok(ToolOutput::text(output))
    }
}
```

**Tests**:
```rust
#[tokio::test] fn send_message_tool_routes_message()
#[tokio::test] fn spawn_agent_tool_returns_subagent_output()
#[tokio::test] fn spawn_agent_tool_rejects_disallowed_type()
```

---

## Step 5.5 — Agent Mailbox Integration (`src/agent/mod.rs`)

The `Agent` loop must drain its mailbox alongside the LLM loop. Add mailbox support:

```rust
pub struct Agent {
    // ... existing fields ...
    mailbox_rx: Option<tokio::sync::mpsc::Receiver<ActorMessage>>,
}
```

In `run()`, after each tool execution cycle, drain pending mailbox messages and
append them as user messages before the next LLM call:
```rust
// Drain mailbox (non-blocking)
while let Ok(msg) = self.mailbox_rx.as_mut().map(|r| r.try_recv()).unwrap_or(Err(TryRecvError::Empty)) {
    self.emit(AgentEvent::MessageReceived {
        agent_id: self.id.clone(),
        from:     msg.from.clone(),
        content:  msg.body.clone(),
    });
    self.history.push(Message::user(&format!("[Message from {}]: {}", msg.from, msg.body)));
}
```

---

## Step 5.6 — Topology helpers (`src/orchestrator/topology.rs`)

```rust
pub struct Topology<'o> {
    orchestrator: &'o dyn Orchestrator,
}

impl<'o> Topology<'o> {
    pub fn new(orchestrator: &'o dyn Orchestrator) -> Self { Self { orchestrator } }

    /// Spawn one agent and await its result.
    pub async fn run(&self, agent_type: &str, task: &str) -> anyhow::Result<String> {
        let mut h = self.orchestrator.spawn_agent(agent_type, json!({"initial_task": task}), None).await?;
        h.wait().await
    }

    /// Run N agents in parallel, return all results (in spawn order).
    pub async fn parallel(&self, agents: Vec<(&str, &str)>) -> anyhow::Result<Vec<String>> {
        let handles: Vec<_> = futures::future::try_join_all(
            agents.into_iter().map(|(t, task)| {
                self.orchestrator.spawn_agent(t, json!({"initial_task": task}), None)
            })
        ).await?;

        futures::future::try_join_all(
            handles.into_iter().map(|mut h| async move { h.wait().await })
        ).await
    }

    /// Run agents as a pipeline: output of stage N is the input task for stage N+1.
    pub async fn pipeline(&self, stages: &[&str], initial: &str) -> anyhow::Result<String> {
        let mut input = initial.to_string();
        for &stage in stages {
            input = self.run(stage, &input).await?;
        }
        Ok(input)
    }
}
```

---

## Step 5.7 — AgentType presets (`src/agent/config.rs`)

```rust
impl AgentType {
    /// Planner: writes markdown only, FreshContext, no bash.
    pub fn planner(client: Arc<dyn LlmClient>) -> Self { ... }

    /// Coder: full tools, PinPrefix policy, bash deny-list for dangerous commands.
    pub fn coder(client: Arc<dyn LlmClient>) -> Self { ... }

    /// Shell: BashTool only, FreshContext.
    pub fn shell(client: Arc<dyn LlmClient>) -> Self { ... }
}
```

---

## Step 5.8 — Interactive REPL (`src/main.rs`)

Add `--interactive` / `-i` flag. When set:

```rust
use rustyline::DefaultEditor;

let mut rl = DefaultEditor::new()?;
let mut agent = Agent::new(...);

loop {
    match rl.readline("tau> ") {
        Ok(line) if line.trim().is_empty() => continue,
        Ok(line) => {
            rl.add_history_entry(&line)?;
            match agent.continue_with(&line).await {
                Ok(response) => println!("{response}"),
                Err(e)       => eprintln!("error: {e}"),
            }
        }
        Err(rustyline::error::ReadlineError::Interrupted) => break,
        Err(rustyline::error::ReadlineError::Eof)         => break,
        Err(e) => { eprintln!("readline error: {e}"); break; }
    }
}
```

Add `Agent::continue_with(prompt: &str)` — like `run()` but appends to existing
history instead of resetting (persistent multi-turn conversation).

Add `rustyline = "14"` to `Cargo.toml`.

---

## Step 5.9 — Examples

Create `examples/` directory with two examples:

### `examples/static_topology.rs`
```rust
// Demonstrates: user writes Rust topology, planner → 2 parallel coders
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Arc::new(OpenAiClient::from_env());
    let orch   = LocalOrchestrator::new(Arc::clone(&client));
    // register factories...
    let topo = Topology::new(&*orch);
    let plan = topo.run("planner", "Design a simple key-value store API").await?;
    let results = topo.parallel(vec![
        ("coder", &format!("Implement the GET endpoint. Plan:\n{plan}")),
        ("coder", &format!("Implement the PUT endpoint. Plan:\n{plan}")),
    ]).await?;
    for r in results { println!("{r}"); }
    Ok(())
}
```

### `examples/dynamic_topology.rs`
```rust
// Demonstrates: LLM drives spawning via SpawnAgentTool
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Arc::new(OpenAiClient::from_env());
    let orch   = LocalOrchestrator::new(Arc::clone(&client));
    // register planner_with_spawn (has SpawnAgentTool + allowed_types=["coder"])
    // spawn planner, let it coordinate coders at runtime
    let mut h = orch.spawn_agent("planner_with_spawn",
        json!({"initial_task": "Build a calculator. Spawn coders for each operation."}),
        None
    ).await?;
    println!("{}", h.wait().await?);
    Ok(())
}
```

---

## Step 5.10 — Docker/K8s stubs (feature-gated)

Add stub files that compile (but do nothing useful) when the features are enabled:

**`src/orchestrator/docker.rs`** (feature = "docker"):
```rust
#[cfg(feature = "docker")]
pub struct DockerOrchestrator { /* TODO */ }
// Implements Orchestrator via bollard; placeholder for future work.
```

**`src/orchestrator/k8s.rs`** (feature = "k8s"):
```rust
#[cfg(feature = "k8s")]
pub struct K8sOrchestrator { /* TODO */ }
// Implements Orchestrator via kube-rs; placeholder for future work.
```

These stubs allow the feature flags to exist in `Cargo.toml` without breaking compilation.

---

## Checklist

- [ ] `src/orchestrator/mod.rs` — `ActorMessage`, `AgentCompletion`, `AgentHandle`, `OrchestratorBus`, `OrchestratorHandle`, `Orchestrator`, `AgentFactory`
- [ ] `src/orchestrator/router.rs` — `MessageRouter` + tests
- [ ] `src/orchestrator/local.rs` — `LocalOrchestrator` with `OnceLock` self-ref pattern + tests
- [ ] `src/orchestrator/topology.rs` — `Topology` combinators (`run`, `parallel`, `pipeline`)
- [ ] `src/actor_tools/send_message.rs` — `SendMessageTool` + tests
- [ ] `src/actor_tools/spawn_agent.rs` — `SpawnAgentTool` + tests
- [ ] `src/agent/mod.rs` — mailbox drain, `continue_with()`
- [ ] `src/agent/config.rs` — `AgentType::planner()`, `::coder()`, `::shell()` presets
- [ ] `src/main.rs` — `--interactive` REPL mode, add `rustyline` dep
- [ ] `examples/static_topology.rs` — compiles and runs
- [ ] `examples/dynamic_topology.rs` — compiles and runs
- [ ] `src/orchestrator/docker.rs` + `k8s.rs` — feature-gated stubs
- [ ] `cargo test orchestrator:: actor_tools::` passes
- [ ] `cargo build --release` produces single binary
- [ ] `ldd target/release/tau` shows only libc/libm
