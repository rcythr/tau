use crate::llm::types::Message;

pub trait Tokenizer: Send + Sync {
    fn count(&self, text: &str) -> usize;
}

/// Default: chars / 4. Zero dependencies, ~15% overcount for English.
pub struct ApproxTokenizer;

impl Tokenizer for ApproxTokenizer {
    fn count(&self, text: &str) -> usize {
        (text.len() + 3) / 4
    }
}

/// Worst-case: 1 char = 1 token. Use when you must never exceed budget.
pub struct ConservativeTokenizer;

impl Tokenizer for ConservativeTokenizer {
    fn count(&self, text: &str) -> usize {
        text.len()
    }
}

/// Estimate tokens for a single message: content + tool_calls args + 4 overhead.
pub fn count_message(tok: &dyn Tokenizer, msg: &Message) -> usize {
    let content_tokens = msg.content.as_deref().map(|s| tok.count(s)).unwrap_or(0);
    let tool_call_tokens: usize = msg
        .tool_calls
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|tc| tok.count(&tc.function.arguments))
        .sum();
    content_tokens + tool_call_tokens + 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_count_ascii() {
        let t = ApproxTokenizer;
        let count = t.count("hello world");
        assert!(count >= 2 && count <= 4);
    }

    #[test]
    fn approx_count_empty() {
        let t = ApproxTokenizer;
        assert_eq!(t.count(""), 0);
    }

    #[test]
    fn conservative_count() {
        let a = ApproxTokenizer;
        let c = ConservativeTokenizer;
        let text = "hello world";
        assert!(c.count(text) >= a.count(text));
    }
}
