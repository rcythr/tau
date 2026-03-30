use crate::tools::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::json;

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Write content to a file. Creates parent directories if needed. Overwrites existing files."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let path = match input["path"].as_str() {
            Some(p) => p.to_string(),
            None => return Ok(ToolOutput::error("missing path")),
        };

        let content = match input["content"].as_str() {
            Some(c) => c.to_string(),
            None => return Ok(ToolOutput::error("missing content")),
        };

        // Create parent directories if needed
        if let Some(parent) = std::path::Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return Ok(ToolOutput::error(format!(
                        "failed to create directories: {}",
                        e
                    )));
                }
            }
        }

        let byte_count = content.len();
        if let Err(e) = tokio::fs::write(&path, content.as_bytes()).await {
            return Ok(ToolOutput::error(format!("failed to write file: {}", e)));
        }

        Ok(ToolOutput::text(format!("wrote {} bytes to {}", byte_count, path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn write_creates_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        let tool = WriteFileTool;
        let input = json!({
            "path": path.to_str().unwrap(),
            "content": "hello world"
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("11"));
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "hello world");
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a").join("b").join("c.txt");
        let tool = WriteFileTool;
        let input = json!({
            "path": path.to_str().unwrap(),
            "content": "nested"
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert!(path.exists());
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "original").unwrap();

        let tool = WriteFileTool;
        let input = json!({
            "path": path.to_str().unwrap(),
            "content": "new content"
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "new content");
    }
}
