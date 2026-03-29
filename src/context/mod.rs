#[allow(dead_code)]
pub mod compressor;
#[allow(dead_code)]
pub mod history;
#[allow(dead_code)]
pub mod policy;
#[allow(dead_code)]
pub mod tokenizer;

#[allow(dead_code)]
pub struct ContextManager;

#[allow(dead_code)]
pub struct ContextBudget {
    pub available_for_history: usize,
}
