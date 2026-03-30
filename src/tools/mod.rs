pub mod bash;
pub mod glob;
pub mod grep;
pub mod read_file;
pub mod write_file;

pub use bash::BashTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

use crate::context::{OutputCompressor, Tokenizer};
use crate::llm::types::{FunctionDefinition, ToolDefinition};
use async_trait::async_trait;

/// Output of a tool call.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(s: impl Into<String>) -> Self {
        Self { content: s.into(), is_error: false }
    }
    pub fn error(s: impl Into<String>) -> Self {
        Self { content: format!("Error: {}", s.into()), is_error: true }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// JSON Schema for the tool's input parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given input (parsed from ToolCall.function.arguments).
    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput>;

    /// Per-tool output compressor override. None = use agent's global compressor.
    fn compressor(&self) -> Option<OutputCompressor> {
        None
    }
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: vec![] }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.push(Box::new(tool));
    }

    /// Find tool by name.
    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    /// Generate `Vec<ToolDefinition>` for the CompletionRequest.tools field.
    pub fn to_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    parameters: t.parameters_schema(),
                },
            })
            .collect()
    }

    /// Estimate tokens consumed by all tool schemas (for ContextBudget).
    pub fn schema_tokens(&self, tokenizer: &dyn Tokenizer) -> usize {
        self.tools
            .iter()
            .map(|t| {
                let schema = t.parameters_schema().to_string();
                let desc = t.description();
                let name = t.name();
                tokenizer.count(&format!("{}{}{}", name, desc, schema))
            })
            .sum()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ApproxTokenizer;

    #[test]
    fn registry_find_by_name() {
        let r = ToolRegistry::default();
        assert!(r.find("bash").is_some());
        assert!(r.find("read_file").is_some());
        assert!(r.find("write_file").is_some());
        assert!(r.find("glob").is_some());
        assert!(r.find("grep").is_some());
        assert!(r.find("nonexistent").is_none());
    }

    #[test]
    fn registry_to_definitions_schema_valid() {
        let r = ToolRegistry::default();
        let defs = r.to_definitions();
        assert_eq!(defs.len(), 5);
        for def in &defs {
            assert!(!def.function.name.is_empty());
            assert!(def.function.parameters.is_object());
        }
    }

    #[test]
    fn registry_schema_tokens_nonzero() {
        let r = ToolRegistry::default();
        let tok = ApproxTokenizer;
        let tokens = r.schema_tokens(&tok);
        assert!(tokens > 0);
    }
}
