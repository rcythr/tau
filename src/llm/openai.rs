use crate::llm::types::{CompletionRequest, CompletionResponse, StreamDelta};
use crate::llm::LlmClient;
use anyhow::anyhow;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

#[allow(dead_code)]
pub struct OpenAiClient {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    http: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
    ) -> Self {
        OpenAiClient {
            base_url: base_url.into(),
            api_key,
            model: model.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Construct from environment:
    ///   TAU_BASE_URL  (default: http://localhost:8080/v1)
    ///   TAU_API_KEY   (optional)
    ///   TAU_MODEL     (default: "default")
    #[allow(dead_code)]
    pub fn from_env() -> Self {
        let base_url = std::env::var("TAU_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:8080/v1".to_string());
        let api_key = std::env::var("TAU_API_KEY").ok();
        let model = std::env::var("TAU_MODEL").unwrap_or_else(|_| "default".to_string());
        Self::new(base_url, api_key, model)
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut builder = self.http.post(&url).json(req);

        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = builder.send().await?;
        let status = response.status();

        if status.is_client_error() || status.is_server_error() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("HTTP {}: {}", status, body));
        }

        let completion = response.json::<CompletionResponse>().await?;
        Ok(completion)
    }

    async fn complete_stream(
        &self,
        _req: &CompletionRequest,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<StreamDelta>> + Send>>> {
        anyhow::bail!("streaming not yet implemented; use --no-stream")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{Message, CompletionRequest};

    #[tokio::test]
    #[ignore = "requires live llama.cpp at TAU_BASE_URL"]
    async fn test_complete_live() {
        let client = OpenAiClient::from_env();
        let req = CompletionRequest {
            model: client.model.clone(),
            messages: vec![Message::user("say hello")],
            tools: None,
            stream: false,
            max_tokens: Some(64),
            temperature: None,
        };
        let resp = client.complete(&req).await.unwrap();
        assert!(!resp.text().is_empty());
    }
}
