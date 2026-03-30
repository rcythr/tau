use crate::context::{CompressionStrategy, OutputCompressor};
use crate::tools::{Tool, ToolOutput};
use anyhow::anyhow;
use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Execute a bash shell command. Returns combined stdout+stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)"
                }
            },
            "required": ["cmd"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let cmd = input["cmd"].as_str().ok_or_else(|| anyhow!("missing cmd"))?;
        let timeout = input["timeout_secs"].as_u64().unwrap_or(30);

        let result = tokio::time::timeout(
            Duration::from_secs(timeout),
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(cmd)
                .output(),
        )
        .await
        .map_err(|_| anyhow!("command timed out after {}s", timeout))??;

        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        let combined = if stderr.is_empty() {
            stdout.into_owned()
        } else {
            format!("{}{}", stdout, stderr)
        };

        if result.status.success() {
            Ok(ToolOutput::text(combined))
        } else {
            let code = result.status.code().unwrap_or(-1);
            Ok(ToolOutput::error(format!(
                "exit code {}: {}",
                code, combined
            )))
        }
    }

    fn compressor(&self) -> Option<OutputCompressor> {
        Some(OutputCompressor {
            max_bytes: 16 * 1024,
            strategy: CompressionStrategy::HeadTail { head: 4096, tail: 4096 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool;
        let input = json!({ "cmd": "echo hello" });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content.trim(), "hello");
    }

    #[tokio::test]
    async fn bash_stderr_merged() {
        let tool = BashTool;
        let input = json!({ "cmd": "echo err >&2" });
        let out = tool.execute(input).await.unwrap();
        assert!(out.content.contains("err"));
    }

    #[tokio::test]
    async fn bash_exit_nonzero() {
        let tool = BashTool;
        let input = json!({ "cmd": "exit 1" });
        let out = tool.execute(input).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("exit code 1"));
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool;
        let input = json!({ "cmd": "sleep 100", "timeout_secs": 1 });
        let result = tool.execute(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timed out"));
    }
}
