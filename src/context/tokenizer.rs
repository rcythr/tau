#[allow(dead_code)]
pub struct ApproxTokenizer;

impl ApproxTokenizer {
    pub fn count(&self, t: &str) -> usize {
        t.len() / 4
    }
}
