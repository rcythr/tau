use crate::llm::types::ToolCall;

pub trait ToolPolicy: Send + Sync {
    /// Return Ok(()) to allow, Err with reason to block.
    fn check(&self, call: &ToolCall) -> anyhow::Result<()>;
}

pub struct AllowAll;
impl ToolPolicy for AllowAll {
    fn check(&self, _: &ToolCall) -> anyhow::Result<()> {
        Ok(())
    }
}

pub struct DenyAll;
impl ToolPolicy for DenyAll {
    fn check(&self, _: &ToolCall) -> anyhow::Result<()> {
        anyhow::bail!("all tools are denied (DenyAll policy)")
    }
}

pub struct AllowList(pub Vec<&'static str>);
impl ToolPolicy for AllowList {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if self.0.contains(&call.function.name.as_str()) {
            Ok(())
        } else {
            anyhow::bail!("tool '{}' not in allow list", call.function.name)
        }
    }
}

pub struct DenyList(pub Vec<&'static str>);
impl ToolPolicy for DenyList {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if self.0.contains(&call.function.name.as_str()) {
            anyhow::bail!("tool '{}' is in deny list", call.function.name)
        } else {
            Ok(())
        }
    }
}

/// WriteFileTool: only allow writes to files with a `.md` extension.
pub struct MarkdownWritesOnly;
impl ToolPolicy for MarkdownWritesOnly {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if call.function.name != "write_file" {
            return Ok(());
        }
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
        let path = args["path"].as_str().unwrap_or("");
        if path.ends_with(".md") {
            Ok(())
        } else {
            anyhow::bail!("write_file: only .md files allowed (MarkdownWritesOnly policy)")
        }
    }
}

/// BashTool: reject commands matching any deny regex.
pub struct BashDenyList {
    pub patterns: Vec<regex::Regex>,
}
impl ToolPolicy for BashDenyList {
    fn check(&self, call: &ToolCall) -> anyhow::Result<()> {
        if call.function.name != "bash" {
            return Ok(());
        }
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
        for p in &self.0 {
            p.check(call)?;
        }
        Ok(())
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

    #[test]
    fn allow_all_passes_any() {
        let call = make_call("bash", r#"{"cmd":"ls"}"#);
        assert!(AllowAll.check(&call).is_ok());
    }

    #[test]
    fn deny_all_blocks_any() {
        let call = make_call("bash", r#"{"cmd":"ls"}"#);
        assert!(DenyAll.check(&call).is_err());
    }

    #[test]
    fn allow_list_passes_listed() {
        let policy = AllowList(vec!["bash", "read_file"]);
        let call = make_call("bash", r#"{"cmd":"ls"}"#);
        assert!(policy.check(&call).is_ok());
    }

    #[test]
    fn allow_list_blocks_unlisted() {
        let policy = AllowList(vec!["read_file"]);
        let call = make_call("bash", r#"{"cmd":"ls"}"#);
        assert!(policy.check(&call).is_err());
    }

    #[test]
    fn markdown_writes_only_allows_md() {
        let policy = MarkdownWritesOnly;
        let call = make_call("write_file", r#"{"path":"readme.md","content":"hi"}"#);
        assert!(policy.check(&call).is_ok());
    }

    #[test]
    fn markdown_writes_only_blocks_py() {
        let policy = MarkdownWritesOnly;
        let call = make_call("write_file", r#"{"path":"script.py","content":"hi"}"#);
        assert!(policy.check(&call).is_err());
    }

    #[test]
    fn bash_deny_list_blocks_rm_rf() {
        let policy = BashDenyList {
            patterns: vec![regex::Regex::new(r"rm\s+-rf").unwrap()],
        };
        let call = make_call("bash", r#"{"cmd":"rm -rf /"}"#);
        assert!(policy.check(&call).is_err());
    }

    #[test]
    fn bash_deny_list_allows_safe_cmd() {
        let policy = BashDenyList {
            patterns: vec![regex::Regex::new(r"rm\s+-rf").unwrap()],
        };
        let call = make_call("bash", r#"{"cmd":"echo hello"}"#);
        assert!(policy.check(&call).is_ok());
    }

    #[test]
    fn composite_policy_all_must_pass() {
        let policy = CompositePolicy(vec![
            Box::new(AllowList(vec!["bash"])),
            Box::new(BashDenyList {
                patterns: vec![regex::Regex::new(r"rm\s+-rf").unwrap()],
            }),
        ]);

        // Allowed tool + safe cmd: pass
        let ok_call = make_call("bash", r#"{"cmd":"echo hello"}"#);
        assert!(policy.check(&ok_call).is_ok());

        // Allowed tool + dangerous cmd: blocked by BashDenyList
        let bad_cmd = make_call("bash", r#"{"cmd":"rm -rf /"}"#);
        assert!(policy.check(&bad_cmd).is_err());

        // Blocked tool: blocked by AllowList
        let bad_tool = make_call("write_file", r#"{"path":"x.py","content":""}"#);
        assert!(policy.check(&bad_tool).is_err());
    }
}
