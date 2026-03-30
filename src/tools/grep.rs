use crate::tools::{Tool, ToolOutput};
use async_trait::async_trait;
use regex::Regex;
use serde_json::json;
use walkdir::WalkDir;

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn description(&self) -> &'static str {
        "Search files for a regex pattern. Returns matching lines in path:line_number:content format."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in"
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob filter on filename (e.g. '*.rs')"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Show N lines before and after each match"
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> anyhow::Result<ToolOutput> {
        let pattern_str = match input["pattern"].as_str() {
            Some(p) => p,
            None => return Ok(ToolOutput::error("missing pattern")),
        };

        let search_path = match input["path"].as_str() {
            Some(p) => p.to_string(),
            None => return Ok(ToolOutput::error("missing path")),
        };

        let glob_filter = input["glob"].as_str().map(|s| s.to_string());
        let context_lines = input["context_lines"].as_u64().unwrap_or(0) as usize;

        let regex = match Regex::new(pattern_str) {
            Ok(r) => r,
            Err(e) => return Ok(ToolOutput::error(format!("invalid regex: {}", e))),
        };

        // Build glob pattern matcher for filename filtering
        let glob_matcher = match &glob_filter {
            Some(pat) => match glob::Pattern::new(pat) {
                Ok(m) => Some(m),
                Err(e) => return Ok(ToolOutput::error(format!("invalid glob filter: {}", e))),
            },
            None => None,
        };

        let mut results: Vec<String> = Vec::new();
        let max_matches = 500;

        let path_obj = std::path::Path::new(&search_path);

        if path_obj.is_file() {
            search_file(
                path_obj,
                &regex,
                context_lines,
                &mut results,
                max_matches,
            );
        } else {
            for entry in WalkDir::new(&search_path)
                .follow_links(false)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if !entry.file_type().is_file() {
                    continue;
                }

                // Apply glob filter on file name
                if let Some(ref matcher) = glob_matcher {
                    let filename = entry.file_name().to_string_lossy();
                    if !matcher.matches(&filename) {
                        continue;
                    }
                }

                search_file(
                    entry.path(),
                    &regex,
                    context_lines,
                    &mut results,
                    max_matches,
                );

                if results.len() >= max_matches {
                    break;
                }
            }
        }

        if results.is_empty() {
            return Ok(ToolOutput::text("(no matches)"));
        }

        let truncated = results.len() >= max_matches;
        let mut output = results.join("\n");
        if truncated {
            output.push_str(&format!("\n... (truncated at {} matches)", max_matches));
        }

        Ok(ToolOutput::text(output))
    }
}

fn is_likely_binary(content: &[u8]) -> bool {
    if content.is_empty() {
        return false;
    }
    let sample_len = content.len().min(1024);
    let non_utf8 = content[..sample_len]
        .iter()
        .filter(|&&b| b == 0 || (b < 32 && b != b'\n' && b != b'\r' && b != b'\t'))
        .count();
    (non_utf8 * 10) > sample_len
}

fn search_file(
    path: &std::path::Path,
    regex: &Regex,
    context_lines: usize,
    results: &mut Vec<String>,
    max_matches: usize,
) {
    let raw = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return,
    };

    if is_likely_binary(&raw) {
        return;
    }

    let content = String::from_utf8_lossy(&raw);
    let path_str = path.to_string_lossy();
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    let mut i = 0;
    while i < total && results.len() < max_matches {
        if regex.is_match(lines[i]) {
            if context_lines == 0 {
                results.push(format!("{}:{}:{}", path_str, i + 1, lines[i]));
            } else {
                let start = i.saturating_sub(context_lines);
                let end = (i + context_lines + 1).min(total);
                for j in start..end {
                    let marker = if j == i { ":" } else { "-" };
                    results.push(format!("{}{}{}:{}", path_str, marker, j + 1, lines[j]));
                }
                results.push("--".to_string());
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn grep_finds_match() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "hello world\nfoo bar\nhello again\n").unwrap();

        let tool = GrepTool;
        let input = json!({
            "pattern": "hello",
            "path": file.to_str().unwrap()
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello world"));
        assert!(out.content.contains("hello again"));
    }

    #[tokio::test]
    async fn grep_no_match() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "hello world\nfoo bar\n").unwrap();

        let tool = GrepTool;
        let input = json!({
            "pattern": "zzznomatch",
            "path": file.to_str().unwrap()
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "(no matches)");
    }

    #[tokio::test]
    async fn grep_invalid_regex_error() {
        let tool = GrepTool;
        let input = json!({
            "pattern": "[invalid",
            "path": "."
        });
        let out = tool.execute(input).await.unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("invalid regex"));
    }

    #[tokio::test]
    async fn grep_with_context_lines() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "before\nmatch line\nafter\n").unwrap();

        let tool = GrepTool;
        let input = json!({
            "pattern": "match",
            "path": file.to_str().unwrap(),
            "context_lines": 1
        });
        let out = tool.execute(input).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("before"));
        assert!(out.content.contains("match line"));
        assert!(out.content.contains("after"));
    }
}
