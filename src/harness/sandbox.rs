use crate::tools::{Tool, ToolOutput};
use async_trait::async_trait;

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
