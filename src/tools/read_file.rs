use crate::context::{CompressionStrategy, OutputCompressor};
use crate::tools::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::json;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a file from disk. Returns file contents with line numbers. Optionally specify offset (start line) and limit (number of lines)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Start reading from this line number (1-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let path = match input["path"].as_str() {
            Some(p) => p.to_string(),
            None => return Ok(ToolOutput::error("missing path")),
        };

        let contents = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(_) => return Ok(ToolOutput::error(format!("file not found: {}", path))),
        };

        let lines: Vec<&str> = contents.lines().collect();
        let total = lines.len();

        // offset is 1-based line number; default to 1
        let offset = input["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit = input["limit"].as_u64().map(|l| l as usize);

        let start = (offset - 1).min(total);
        let end = match limit {
            Some(lim) => (start + lim).min(total),
            None => total,
        };

        let result: String = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}\t{}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolOutput::text(result))
    }

    fn compressor(&self) -> Option<OutputCompressor> {
        Some(OutputCompressor {
            max_bytes: 100 * 1024,
            strategy: CompressionStrategy::TailOnly { size: 65536 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_temp_file(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[tokio::test]
    async fn read_existing_file() {
        let f = make_temp_file("line one\nline two\nline three\n");
        let tool = ReadFileTool;
        let input = json!({ "path": f.path().to_str().unwrap() });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("line one"));
        assert!(out.content.contains("1\t"));
        assert!(out.content.contains("2\t"));
        assert!(out.content.contains("3\t"));
    }

    #[tokio::test]
    async fn read_with_offset_limit() {
        let f = make_temp_file("a\nb\nc\nd\ne\n");
        let tool = ReadFileTool;
        let input = json!({ "path": f.path().to_str().unwrap(), "offset": 2, "limit": 2 });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        // Should contain lines 2 and 3 (b and c)
        assert!(out.content.contains("b"));
        assert!(out.content.contains("c"));
        assert!(!out.content.contains("a"));
        assert!(!out.content.contains("d"));
    }

    #[tokio::test]
    async fn read_missing_file_error() {
        let tool = ReadFileTool;
        let input = json!({ "path": "/nonexistent/path/file.txt" });
        let out = tool.execute(input).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("file not found"));
    }
}
