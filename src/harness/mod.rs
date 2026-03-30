pub mod policy;
pub mod sandbox;

pub use policy::{
    AllowAll, AllowList, BashDenyList, CompositePolicy, DenyAll, DenyList, MarkdownWritesOnly,
    ToolPolicy,
};
pub use sandbox::{NoSandbox, Sandbox};

use crate::llm::types::ToolCall;
use crate::tools::{ToolOutput, ToolRegistry};
use anyhow::anyhow;

pub struct ToolHarness {
    policy: Box<dyn ToolPolicy>,
    sandbox: Box<dyn Sandbox>,
}

impl ToolHarness {
    pub fn new(policy: impl ToolPolicy + 'static, sandbox: impl Sandbox + 'static) -> Self {
        Self {
            policy: Box::new(policy),
            sandbox: Box::new(sandbox),
        }
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
        let tool = registry
            .find(&tool_call.function.name)
            .ok_or_else(|| anyhow!("unknown tool: {}", tool_call.function.name))?;

        // 3. Parse arguments
        let input: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
            .unwrap_or(serde_json::Value::Object(Default::default()));

        // 4. Execute via sandbox
        self.sandbox.execute(tool, input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{FunctionCall, ToolCall};

    fn make_call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: "test-id".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    #[tokio::test]
    async fn harness_blocks_denied_tool() {
        let harness = ToolHarness::new(DenyAll, NoSandbox);
        let registry = ToolRegistry::default();
        let call = make_call("bash", r#"{"cmd":"echo hello"}"#);
        let result = harness.call(&registry, &call).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("DenyAll"));
    }

    #[tokio::test]
    async fn harness_executes_allowed_tool() {
        let harness = ToolHarness::permissive();
        let registry = ToolRegistry::default();
        let call = make_call("bash", r#"{"cmd":"echo hello"}"#);
        let result = harness.call(&registry, &call).await;
        assert!(result.is_ok());
        let out = result.unwrap();
        assert!(out.content.contains("hello"));
    }

    #[tokio::test]
    async fn harness_returns_error_for_unknown_tool() {
        let harness = ToolHarness::permissive();
        let registry = ToolRegistry::default();
        let call = make_call("nonexistent_tool", r#"{}"#);
        let result = harness.call(&registry, &call).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown tool"));
    }
}
