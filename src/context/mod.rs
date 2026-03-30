pub mod compressor;
pub mod history;
pub mod policy;
pub mod tokenizer;

pub use compressor::{CompressionStrategy, OutputCompressor};
pub use history::ConversationHistory;
pub use policy::ContextPolicy;
pub use tokenizer::{ApproxTokenizer, Tokenizer};

use crate::llm::types::Message;
use std::sync::Arc;

pub struct ContextBudget {
    pub model_limit: usize,
    pub system_tokens: usize,
    pub tool_schema_tokens: usize,
    pub reserved_for_response: usize,
    pub available_for_history: usize,
}

impl ContextBudget {
    pub fn new(
        model_limit: usize,
        system_tokens: usize,
        tool_schema_tokens: usize,
        reserved_for_response: usize,
    ) -> Self {
        let available = model_limit
            .saturating_sub(system_tokens)
            .saturating_sub(tool_schema_tokens)
            .saturating_sub(reserved_for_response);
        Self {
            model_limit,
            system_tokens,
            tool_schema_tokens,
            reserved_for_response,
            available_for_history: available,
        }
    }
}

pub struct ContextManager {
    pub budget: ContextBudget,
    pub policy: ContextPolicy,
    tokenizer: Arc<dyn Tokenizer>,
}

impl ContextManager {
    pub fn new(budget: ContextBudget, policy: ContextPolicy, tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self { budget, policy, tokenizer }
    }

    /// Estimate total tokens for a slice of messages.
    pub fn estimate_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| tokenizer::count_message(self.tokenizer.as_ref(), m))
            .sum()
    }

    /// True if the current history already exceeds the budget.
    pub fn is_over_budget(&self, messages: &[Message]) -> bool {
        self.estimate_tokens(messages) > self.budget.available_for_history
    }

    /// Given the full history (excluding system prompt), return the slice to send.
    pub fn trim(&self, messages: &[Message]) -> Vec<Message> {
        match &self.policy {
            ContextPolicy::AppendOnly => messages.to_vec(),
            ContextPolicy::FreshContext => vec![],
            ContextPolicy::PinPrefix { pinned, recent } => {
                self.apply_pin_prefix(messages, *pinned, *recent)
            }
            ContextPolicy::RollingCompact { pinned, recent, .. } => {
                self.apply_pin_prefix(messages, *pinned, *recent)
            }
        }
    }

    fn apply_pin_prefix(&self, messages: &[Message], pinned: usize, recent: usize) -> Vec<Message> {
        if !self.is_over_budget(messages) {
            return messages.to_vec();
        }

        let total = messages.len();
        let pinned_count = pinned.min(total);
        let prefix = &messages[..pinned_count];

        // Start with full recent count, reduce until fits
        let mut recent_count = recent.min(total);
        loop {
            let suffix_start = total.saturating_sub(recent_count);
            // Combine prefix and suffix, deduplicating overlapping indices
            let combined: Vec<Message> = if pinned_count >= suffix_start {
                // prefix already covers all of suffix
                messages[..pinned_count.max(total.saturating_sub(recent_count).max(pinned_count))]
                    .to_vec()
            } else {
                let mut v = prefix.to_vec();
                v.extend_from_slice(&messages[suffix_start..]);
                v
            };

            if !self.is_over_budget(&combined) || recent_count == 0 {
                return combined;
            }
            recent_count = recent_count.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::Message;

    fn msgs(n: usize) -> Vec<Message> {
        (0..n).map(|i| Message::user(&"x".repeat(40 * (i + 1)))).collect()
    }

    fn make_mgr_with_policy(available: usize, policy: ContextPolicy) -> ContextManager {
        let budget = ContextBudget {
            model_limit: available + 1024,
            system_tokens: 0,
            tool_schema_tokens: 0,
            reserved_for_response: 0,
            available_for_history: available,
        };
        ContextManager::new(budget, policy, Arc::new(ApproxTokenizer))
    }

    #[test]
    fn budget_available_calculation() {
        let b = ContextBudget::new(8192, 512, 256, 1024);
        assert_eq!(b.available_for_history, 8192 - 512 - 256 - 1024);
    }

    #[test]
    fn budget_never_negative_on_underflow() {
        let b = ContextBudget::new(100, 200, 50, 50);
        assert_eq!(b.available_for_history, 0);
    }

    #[test]
    fn manager_estimates_tokens() {
        let mgr = make_mgr_with_policy(4096, ContextPolicy::AppendOnly);
        let msgs = vec![Message::user("hello world"), Message::assistant_text("hi")];
        let tokens = mgr.estimate_tokens(&msgs);
        assert!(tokens > 0);
    }

    #[test]
    fn append_only_returns_all() {
        let mgr = make_mgr_with_policy(10, ContextPolicy::AppendOnly);
        let input = msgs(5);
        let out = mgr.trim(&input);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn fresh_context_returns_empty() {
        let mgr = make_mgr_with_policy(4096, ContextPolicy::FreshContext);
        let input = msgs(5);
        let out = mgr.trim(&input);
        assert!(out.is_empty());
    }

    #[test]
    fn pin_prefix_no_trim_when_fits() {
        // Large budget — all messages fit
        let mgr = make_mgr_with_policy(100_000, ContextPolicy::PinPrefix { pinned: 2, recent: 2 });
        let input = msgs(6);
        let out = mgr.trim(&input);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn pin_prefix_trims_middle_when_over_budget() {
        // Each message ~10+ tokens; budget is tight
        let mgr = make_mgr_with_policy(30, ContextPolicy::PinPrefix { pinned: 2, recent: 2 });
        let input = msgs(10);
        let out = mgr.trim(&input);
        assert!(out.len() < input.len());
    }

    #[test]
    fn pin_prefix_preserves_pinned_count() {
        let pinned = 2;
        let mgr = make_mgr_with_policy(30, ContextPolicy::PinPrefix { pinned, recent: 2 });
        let input = msgs(10);
        let out = mgr.trim(&input);
        // First `pinned` messages should be the same
        for i in 0..pinned {
            assert_eq!(out[i].content, input[i].content);
        }
    }

    #[test]
    fn pin_prefix_preserves_recent_count() {
        // msgs(10): sizes 40,80,...,400 bytes -> tokens ~14,24,...,104
        // pinned(2)+recent(2) total ~236 tokens; all-10 total ~590 tokens
        // budget=250 forces trim but keeps pinned+recent intact
        let recent = 2;
        let mgr = make_mgr_with_policy(250, ContextPolicy::PinPrefix { pinned: 2, recent });
        let input = msgs(10);
        let out = mgr.trim(&input);
        // Output should be trimmed but last `recent` entries match tail of input
        assert!(out.len() < input.len(), "expected trimming to occur");
        let last_out = &out[out.len().saturating_sub(recent)..];
        let last_in = &input[input.len() - recent..];
        for (a, b) in last_out.iter().zip(last_in.iter()) {
            assert_eq!(a.content, b.content);
        }
    }

    #[test]
    fn pin_prefix_handles_empty_input() {
        let mgr = make_mgr_with_policy(100, ContextPolicy::PinPrefix { pinned: 2, recent: 2 });
        let out = mgr.trim(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn pin_prefix_handles_fewer_messages_than_pinned_plus_recent() {
        let mgr = make_mgr_with_policy(10, ContextPolicy::PinPrefix { pinned: 5, recent: 5 });
        let input = msgs(3);
        let out = mgr.trim(&input);
        // Should return all without panicking
        assert!(out.len() <= input.len());
    }
}
