use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message {
            role: Role::System,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Message {
            role: Role::User,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_text(content: &str) -> Self {
        Message {
            role: Role::Assistant,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[allow(dead_code)]
    pub fn assistant_with_tool_calls(content: Option<String>, calls: Vec<ToolCall>) -> Self {
        Message {
            role: Role::Assistant,
            content,
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }

    #[allow(dead_code)]
    pub fn tool_result(call_id: &str, content: &str) -> Self {
        Message {
            role: Role::Tool,
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(call_id.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl CompletionResponse {
    pub fn text(&self) -> String {
        self.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string()
    }

    pub fn stop_reason(&self) -> StopReason {
        let reason = self
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .unwrap_or("");
        StopReason::from(reason)
    }

    #[allow(dead_code)]
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

impl From<&str> for StopReason {
    fn from(s: &str) -> Self {
        match s {
            "stop" => StopReason::EndTurn,
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            other => StopReason::Other(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PartialFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PartialToolCall {
    pub index: usize,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<PartialFunctionCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct StreamDelta {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<PartialToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serde_round_trip() {
        let msg = Message::user("hello world");
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, Role::User);
        assert_eq!(decoded.content.as_deref(), Some("hello world"));

        let tool_msg = Message::tool_result("call-123", "result content");
        let json2 = serde_json::to_string(&tool_msg).unwrap();
        let decoded2: Message = serde_json::from_str(&json2).unwrap();
        assert_eq!(decoded2.role, Role::Tool);
        assert_eq!(decoded2.tool_call_id.as_deref(), Some("call-123"));

        // tool_calls should be absent when None
        let assistant = Message::assistant_text("hi");
        let json3 = serde_json::to_string(&assistant).unwrap();
        assert!(!json3.contains("tool_calls"));
    }

    #[test]
    fn test_stop_reason_from_finish_reason() {
        assert_eq!(StopReason::from("stop"), StopReason::EndTurn);
        assert_eq!(StopReason::from("tool_calls"), StopReason::ToolUse);
        assert_eq!(StopReason::from("length"), StopReason::MaxTokens);
        assert_eq!(
            StopReason::from("other"),
            StopReason::Other("other".to_string())
        );
    }
}
