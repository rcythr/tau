use crate::tools::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::json;

pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn description(&self) -> &'static str {
        "Find files matching a glob pattern. Returns newline-separated matching file paths."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files against"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current directory)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let pattern = match input["pattern"].as_str() {
            Some(p) => p.to_string(),
            None => return Ok(ToolOutput::error("missing pattern")),
        };

        let base_dir = input["path"].as_str().unwrap_or(".");

        // Build the full pattern: base_dir + "/" + pattern, or just pattern if absolute
        let full_pattern = if std::path::Path::new(&pattern).is_absolute() {
            pattern.clone()
        } else {
            format!("{}/{}", base_dir.trim_end_matches('/'), pattern)
        };

        let matches = match glob::glob(&full_pattern) {
            Ok(paths) => paths,
            Err(e) => return Ok(ToolOutput::error(format!("invalid glob pattern: {}", e))),
        };

        let mut results: Vec<String> = Vec::new();
        let max_matches = 1000;

        for entry in matches {
            match entry {
                Ok(path) => {
                    // Make path relative to base_dir if possible
                    let display = if let Ok(rel) =
                        path.strip_prefix(base_dir.trim_end_matches('/'))
                    {
                        rel.to_string_lossy().into_owned()
                    } else {
                        path.to_string_lossy().into_owned()
                    };
                    results.push(display);
                    if results.len() >= max_matches {
                        break;
                    }
                }
                Err(_) => continue,
            }
        }

        if results.is_empty() {
            return Ok(ToolOutput::text("(no matches)"));
        }

        results.sort();

        // Check if there are more matches beyond our limit
        let total = results.len();
        let output = if total >= max_matches {
            format!("{}\n... ({} more)", results.join("\n"), "truncated at 1000")
        } else {
            results.join("\n")
        };

        Ok(ToolOutput::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    fn setup_dir() -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("foo.txt"), "content").unwrap();
        fs::write(dir.path().join("bar.txt"), "content").unwrap();
        fs::write(dir.path().join("baz.rs"), "content").unwrap();
        dir
    }

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = setup_dir();
        let tool = GlobTool;
        let input = json!({
            "pattern": "*.txt",
            "path": dir.path().to_str().unwrap()
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("foo.txt") || out.content.contains("bar.txt"));
        assert!(!out.content.contains("baz.rs"));
    }

    #[tokio::test]
    async fn glob_no_matches() {
        let dir = TempDir::new().unwrap();
        let tool = GlobTool;
        let input = json!({
            "pattern": "*.xyz",
            "path": dir.path().to_str().unwrap()
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "(no matches)");
    }

    #[tokio::test]
    async fn glob_invalid_pattern_error() {
        let tool = GlobTool;
        // Invalid glob: unclosed bracket
        let input = json!({
            "pattern": "[invalid",
            "path": "."
        });
        let out = tool.execute(input).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("invalid glob pattern"));
    }
}
