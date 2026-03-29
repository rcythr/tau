use async_trait::async_trait;

#[async_trait]
#[allow(dead_code)]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
}

#[allow(dead_code)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool + Send + Sync>>,
}
