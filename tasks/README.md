# tau — Implementation Task Files

Each file in this directory is a self-contained task for one phase of the implementation.
Work them in order — each phase compiles and has passing tests before the next begins.

## Phases

| File | Phase | Goal |
|------|-------|------|
| [phase-1-scaffold.md](phase-1-scaffold.md) | 1 | Cargo project, OpenAI client, bare agent loop, CLI | ✅ Complete |
| [phase-2-context.md](phase-2-context.md) | 2 | Token counting, context budget, trimming policies, history fork/checkpoint, output compression |
| [phase-3-tools.md](phase-3-tools.md) | 3 | Tool trait, registry, built-in tools, harness, policies, sandbox |
| [phase-4-streaming.md](phase-4-streaming.md) | 4 | SSE streaming, AgentEvent broadcast, trajectory logger, tracing |
| [phase-5-multiagent.md](phase-5-multiagent.md) | 5 | OrchestratorBus, LocalOrchestrator, actor tools, REPL, Docker/K8s stubs |

## Architecture Reference

- **LLM client**: OpenAI-compatible (`POST /v1/chat/completions`), targets llama.cpp. `LlmClient` trait allows future backends.
- **Context management**: KV-cache-aware policies. Prefer `PinPrefix` and `RollingCompact` over sliding window. Tool output runs through `OutputCompressor` before entering history.
- **Tools at build time**: `Tool` trait, compiled-in `ToolRegistry`. Users extend by adding files + re-registering.
- **Harness**: `ToolPolicy` (pre-execution check) + `Sandbox` (execution isolation) wrap every tool call.
- **Orchestrator / Executor split**: `OrchestratorBus` (narrow, lives in core — no circular dep) vs full `Orchestrator` trait. `LocalOrchestrator` runs executors as tokio tasks. Docker/K8s are feature-gated.
- **Actor tools**: `SendMessageTool` + `SpawnAgentTool` hold an `OrchestratorHandle(Arc<dyn OrchestratorBus>)`.

## Dependency Graph (modules within single crate)

```
main.rs
  └─ orchestrator/  ← Orchestrator, OrchestratorBus, OrchestratorHandle
       ├─ local.rs
       └─ router.rs
  └─ agent/         ← Agent run loop, AgentType, AgentEvent
  └─ actor_tools/   ← SendMessageTool, SpawnAgentTool  (depends on orchestrator::OrchestratorBus)
  └─ tools/         ← Tool trait, ToolRegistry, built-ins
  └─ harness/       ← ToolHarness, ToolPolicy, Sandbox
  └─ context/       ← ContextManager, history, tokenizer, compressor, policy
  └─ llm/           ← LlmClient trait, OpenAiClient, wire types
  └─ telemetry/     ← TrajectoryLogger
```

## Invariants

1. `tools/` does NOT import `orchestrator/` — it only imports `orchestrator::OrchestratorBus` via `actor_tools/`
2. `llm/` has no knowledge of tools or agents
3. `context/` has no knowledge of the LLM client or tools
4. Every `cargo check` and `cargo test` must pass before moving to the next phase
