pub mod openai;
pub mod types;

pub use openai::OpenAiClient;
pub use types::*;

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

#[async_trait]
#[allow(dead_code)]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<CompletionResponse>;

    async fn complete_stream(
        &self,
        req: &CompletionRequest,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<StreamDelta>> + Send>>>;

    /// Override for model-specific tokenizers. Default: chars / 4.
    fn count_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }
}
