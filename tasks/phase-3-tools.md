# Phase 3 — Tools & Harness

**Goal**: Agent calls real tools. The harness enforces policy before execution
and can wrap bash in a sandbox. Built-in tools: `bash`, `read_file`, `write_file`,
`glob`, `grep`.

**Prerequisite**: Phase 2 `cargo test` passes.

**Exit criteria**:
```
cargo test tools:: harness::
TAU_BASE_URL=http://localhost:8080/v1 cargo run -- \
    --prompt "list the files in /tmp" --no-stream
# Agent calls bash tool, prints ls output, finishes
```

---

## Step 3.1 — Tool trait and registry (`src/tools/mod.rs`)

Replace stub with full implementation.

```rust
use async_trait::async_trait;

/// Output of a tool call.
pub struct ToolOutput {
    pub content:  String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(s: impl Into<String>) -> Self  { Self { content: s.into(), is_error: false } }
    pub fn error(s: impl Into<String>) -> Self { Self { content: format!("Error: {}", s.into()), is_error: true } }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self)        -> &'static str;
    fn description(&self) -> &'static str;

    /// JSON Schema for the tool's input parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given input (parsed from ToolCall.function.arguments).
    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput>;

    /// Per-tool output compressor override. None = use agent's global compressor.
    fn compressor(&self) -> Option<OutputCompressor> { None }
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self { Self { tools: vec![] } }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.push(Box::new(tool));
    }

    /// Find tool by name.
    pub fn find(&self, name: &str) -> Option<&dyn Tool>;

    /// Generate `Vec<ToolDefinition>` for the CompletionRequest.tools field.
    pub fn to_definitions(&self) -> Vec<ToolDefinition>;

    /// Estimate tokens consumed by all tool schemas (for ContextBudget).
    pub fn schema_tokens(&self, tokenizer: &dyn Tokenizer) -> usize;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn names(&self) -> Vec<&str>;
}

/// Build the default registry with all built-in tools.
impl Default for ToolRegistry {
    fn default() -> Self {
        let mut r = Self::new();
        r.register(BashTool);
        r.register(ReadFileTool);
        r.register(WriteFileTool);
        r.register(GlobTool);
        r.register(GrepTool);
        r
    }
}
```

**Tests**:
```rust
#[test] fn registry_find_by_name()
#[test] fn registry_to_definitions_schema_valid()   // each definition has name + parameters
#[test] fn registry_schema_tokens_nonzero()
```

---

## Step 3.2 — BashTool (`src/tools/bash.rs`)

```rust
pub struct BashTool;

// Input schema:
// { "cmd": "<shell command string>", "timeout_secs": <optional u64, default 30> }

#[async_trait]
impl Tool for BashTool {
    fn name(&self)        -> &'static str { "bash" }
    fn description(&self) -> &'static str {
        "Execute a bash shell command. Returns combined stdout+stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "cmd":          { "type": "string",  "description": "Shell command to execute" },
                "timeout_secs": { "type": "integer", "description": "Timeout in seconds (default 30)" }
            },
            "required": ["cmd"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let cmd     = input["cmd"].as_str().ok_or_else(|| anyhow!("missing cmd"))?;
        let timeout = input["timeout_secs"].as_u64().unwrap_or(30);

        let output = tokio::time::timeout(
            Duration::from_secs(timeout),
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(cmd)
                .output(),
        )
        .await
        .map_err(|_| anyhow!("command timed out after {}s", timeout))??;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = if stderr.is_empty() {
            stdout.into_owned()
        } else {
            format!("{}{}", stdout, stderr)
        };

        // Per-tool compressor: HeadTail 4096/4096
        Ok(ToolOutput::text(combined))
    }

    fn compressor(&self) -> Option<OutputCompressor> {
        Some(OutputCompressor {
            max_bytes: 16 * 1024,
            strategy:  CompressionStrategy::HeadTail { head: 4096, tail: 4096 },
        })
    }
}
```

**Tests**:
```rust
#[tokio::test] fn bash_echo()          // cmd="echo hello" -> "hello\n"
#[tokio::test] fn bash_stderr_merged() // cmd="echo err >&2" -> "err\n"
#[tokio::test] fn bash_exit_nonzero()  // cmd="exit 1" -> ToolOutput with error content
#[tokio::test] fn bash_timeout()       // cmd="sleep 100", timeout_secs=1 -> err
```

---

## Step 3.3 — ReadFileTool (`src/tools/read_file.rs`)

```rust
// Input schema: { "path": "<file path>", "offset": <optional line number>, "limit": <optional line count> }
// Output: file contents (or lines offset..offset+limit), prefixed with line numbers (cat -n style)
```

**Implementation notes**:
- Use `tokio::fs::read_to_string`
- `offset` / `limit` are optional; when provided, slice lines accordingly
- If file doesn't exist: `ToolOutput::error("file not found: {path}")`
- Max output: 100 KiB (per-tool compressor TailOnly 65536)

**Tests**:
```rust
#[tokio::test] fn read_existing_file()
#[tokio::test] fn read_with_offset_limit()
#[tokio::test] fn read_missing_file_error()
```

---

## Step 3.4 — WriteFileTool (`src/tools/write_file.rs`)

```rust
// Input schema: { "path": "<file path>", "content": "<file content>" }
// Output: "wrote N bytes to {path}"
```

**Implementation notes**:
- Create parent directories if needed (`tokio::fs::create_dir_all`)
- Overwrite existing file
- Return byte count

**Tests**:
```rust
#[tokio::test] fn write_creates_file()
#[tokio::test] fn write_creates_parent_dirs()
#[tokio::test] fn write_overwrites_existing()
```

---

## Step 3.5 — GlobTool (`src/tools/glob.rs`)

```rust
// Input schema: { "pattern": "<glob pattern>", "path": "<optional base dir, default .>" }
// Output: newline-separated matching file paths (relative to base dir)
```

**Implementation notes**:
- Use `glob::glob()` from the `glob` crate
- Sort results
- Max 1000 matches; truncate with "... (N more)" if exceeded
- If no matches: `ToolOutput::text("(no matches)")`

**Tests**:
```rust
#[tokio::test] fn glob_finds_files()
#[tokio::test] fn glob_no_matches()
#[tokio::test] fn glob_invalid_pattern_error()
```

---

## Step 3.6 — GrepTool (`src/tools/grep.rs`)

```rust
// Input schema:
// {
//   "pattern": "<regex>",
//   "path": "<file or directory>",
//   "glob": "<optional file glob filter, e.g. '*.rs'>",
//   "context_lines": <optional N, show N lines before+after each match>
// }
// Output: matching lines in "path:line_number:content" format
```

**Implementation notes**:
- Use `regex::Regex`
- Walk directory with `walkdir` or `std::fs::read_dir` recursively
- Optional glob filter on filename
- Max 500 matches; truncate with count
- Binary file detection: skip if > 10% non-UTF-8 bytes

**Add `walkdir` to `Cargo.toml`**: `walkdir = "2"`

**Tests**:
```rust
#[tokio::test] fn grep_finds_match()
#[tokio::test] fn grep_no_match()
#[tokio::test] fn grep_invalid_regex_error()
#[tokio::test] fn grep_with_context_lines()
```

---

## Step 3.7 — Tool Harness (`src/harness/`)

### `src/harness/policy.rs`

```rust
use crate::llm::types::ToolCall;

pub trait ToolPolicy: Send + Sync {
    /// Return Ok(()) to allow, Err with reason to block.
    fn check(&self, call: &ToolCall) -> anyhow::Result<()>;
}

pub struct AllowAll;
impl ToolPolicy for AllowAll { fn check(&self, _: &ToolCall) -> anyhow::Result<()> { Ok(()) } }

pub struct DenyAll;
impl ToolPolicy for DenyAll {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        anyhow::bail!("all tools are denied (DenyAll policy)")
    }
}

pub struct AllowList(pub Vec<&'static str>);
impl ToolPolicy for AllowList {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if self.0.contains(&call.function.name.as_str()) { Ok(()) }
        else { anyhow::bail!("tool '{}' not in allow list", call.function.name) }
    }
}

pub struct DenyList(pub Vec<&'static str>);
impl ToolPolicy for DenyList {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if self.0.contains(&call.function.name.as_str()) {
            anyhow::bail!("tool '{}' is in deny list", call.function.name)
        } else { Ok(()) }
    }
}

/// WriteFileTool: only allow writes to files with a `.md` extension.
pub struct MarkdownWritesOnly;
impl ToolPolicy for MarkdownWritesOnly {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if call.function.name != "write_file" { return Ok(()); }
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
        let path = args["path"].as_str().unwrap_or("");
        if path.ends_with(".md") { Ok(()) }
        else { anyhow::bail!("write_file: only .md files allowed (MarkdownWritesOnly policy)") }
    }
}

/// BashTool: reject commands matching any deny regex.
pub struct BashDenyList { pub patterns: Vec<regex::Regex> }
impl ToolPolicy for BashDenyList {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if call.function.name != "bash" { return Ok(()); }
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
        let cmd = args["cmd"].as_str().unwrap_or("");
        for pat in &self.patterns {
            if pat.is_match(cmd) {
                anyhow::bail!("bash: command matches deny pattern '{}'", pat);
            }
        }
        Ok(())
    }
}

/// All sub-policies must pass (logical AND).
pub struct CompositePolicy(pub Vec<Box<dyn ToolPolicy>>);
impl ToolPolicy for CompositePolicy {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        for p in &self.0 { p.check(call)?; }
        Ok(())
    }
}
```

**Tests**:
```rust
#[test] fn allow_all_passes_any()
#[test] fn deny_all_blocks_any()
#[test] fn allow_list_passes_listed()
#[test] fn allow_list_blocks_unlisted()
#[test] fn markdown_writes_only_allows_md()
#[test] fn markdown_writes_only_blocks_py()
#[test] fn bash_deny_list_blocks_rm_rf()
#[test] fn bash_deny_list_allows_safe_cmd()
#[test] fn composite_policy_all_must_pass()
```

### `src/harness/sandbox.rs`

```rust
#[async_trait]
pub trait Sandbox: Send + Sync {
    async fn execute(&self, tool: &dyn Tool, input: serde_json::Value) -> anyhow::Result<ToolOutput>;
}

/// Direct execution (no isolation). Default.
pub struct NoSandbox;

#[async_trait]
impl Sandbox for NoSandbox {
    async fn execute(&self, tool: &dyn Tool, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        tool.execute(input).await
    }
}
```

Linux-only `LandlockSandbox` and `BwrapSandbox` are stubs in Phase 3:
```rust
#[cfg(target_os = "linux")]
pub struct LandlockSandbox {
    pub allowed_rw_paths: Vec<std::path::PathBuf>,
    pub allowed_ro_paths: Vec<std::path::PathBuf>,
}

#[cfg(target_os = "linux")]
#[async_trait]
impl Sandbox for LandlockSandbox {
    async fn execute(&self, tool: &dyn Tool, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        // TODO Phase 3 bonus: apply landlock rules before delegating
        // For now: warn and fall through to NoSandbox
        tracing::warn!("LandlockSandbox: not yet fully implemented, using NoSandbox");
        tool.execute(input).await
    }
}
```

### `src/harness/mod.rs`

```rust
pub struct ToolHarness {
    policy:  Box<dyn ToolPolicy>,
    sandbox: Box<dyn Sandbox>,
}

impl ToolHarness {
    pub fn new(policy: impl ToolPolicy + 'static, sandbox: impl Sandbox + 'static) -> Self {
        Self { policy: Box::new(policy), sandbox: Box::new(sandbox) }
    }

    /// Permissive default: AllowAll + NoSandbox.
    pub fn permissive() -> Self {
        Self::new(AllowAll, NoSandbox)
    }

    /// Check policy then execute via sandbox.
    pub async fn call(
        &self,
        registry: &ToolRegistry,
        tool_call: &ToolCall,
    ) -> anyhow::Result<ToolOutput> {
        // 1. Check policy
        self.policy.check(tool_call)?;

        // 2. Find tool
        let tool = registry.find(&tool_call.function.name)
            .ok_or_else(|| anyhow!("unknown tool: {}", tool_call.function.name))?;

        // 3. Parse arguments
        let input: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
            .unwrap_or(serde_json::Value::Object(Default::default()));

        // 4. Execute via sandbox
        self.sandbox.execute(tool, input).await
    }
}
```

**Tests**:
```rust
#[tokio::test] fn harness_blocks_denied_tool()
#[tokio::test] fn harness_executes_allowed_tool()
#[tokio::test] fn harness_returns_error_for_unknown_tool()
```

---

## Step 3.8 — Wire Tools into Agent Loop

Update `AgentType`:
```rust
pub struct AgentType {
    // ... existing fields ...
    pub tools:   Arc<ToolRegistry>,
    pub harness: Arc<ToolHarness>,
}
```

In `Agent::run()`, handle `StopReason::ToolUse`:
```rust
StopReason::ToolUse => {
    let tool_calls = response.tool_calls();
    // Record assistant turn with tool calls
    self.history.push(Message::assistant_with_tool_calls(
        Some(response.text()),
        tool_calls.clone(),
    ));
    // Execute each call sequentially (parallel execution: Phase 5)
    for call in &tool_calls {
        self.emit(AgentEvent::ToolCalled { agent_id: self.id.clone(), call: call.clone() });

        let result = self.agent_type.harness.call(&self.agent_type.tools, call).await;
        let output = match result {
            Ok(out) => out,
            Err(e)  => {
                self.emit(AgentEvent::ToolBlocked {
                    agent_id: self.id.clone(),
                    call:     call.clone(),
                    reason:   e.to_string(),
                });
                ToolOutput::error(e.to_string())
            }
        };

        // Apply compressor
        let tool_compressor = self.agent_type.tools
            .find(&call.function.name)
            .and_then(|t| t.compressor())
            .unwrap_or_else(|| self.agent_type.compressor.clone());

        let (compressed_content, was_compressed) = tool_compressor.compress(&output.content);

        self.emit(AgentEvent::ToolResult {
            agent_id:   self.id.clone(),
            call_id:    call.id.clone(),
            output:     compressed_content.clone(),
            compressed: was_compressed,
        });

        self.history.push(Message::tool_result(&call.id, &compressed_content));
    }
}
```

---

## Step 3.9 — CLI flags

Add to `src/main.rs`:
- `--tool <NAME>` (repeatable) → `AllowList` policy; omit flag = `AllowAll`
- `--no-tool <NAME>` (repeatable) → `DenyList` policy
- `--sandbox <none|landlock|bwrap>` → configure sandbox (default: none)

---

## Checklist

- [ ] `src/tools/mod.rs` — `Tool` trait, `ToolOutput`, `ToolRegistry` + tests
- [ ] `src/tools/bash.rs` — `BashTool` + tests
- [ ] `src/tools/read_file.rs` — `ReadFileTool` + tests
- [ ] `src/tools/write_file.rs` — `WriteFileTool` + tests
- [ ] `src/tools/glob.rs` — `GlobTool` + tests
- [ ] `src/tools/grep.rs` — `GrepTool` + tests
- [ ] `src/harness/policy.rs` — all policies + tests
- [ ] `src/harness/sandbox.rs` — `NoSandbox` + `LandlockSandbox` stub
- [ ] `src/harness/mod.rs` — `ToolHarness::call()` + tests
- [ ] `src/agent/mod.rs` — tool execution loop wired
- [ ] `src/agent/config.rs` — `AgentType` carries `tools` + `harness`
- [ ] `cargo test tools:: harness::` passes
- [ ] End-to-end: agent calls bash and returns output
