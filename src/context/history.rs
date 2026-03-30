use crate::llm::types::Message;

pub type SnapshotId = usize;

pub struct ConversationHistory {
    system_prompt: Message,
    messages: Vec<Message>,
    snapshots: Vec<(SnapshotId, Vec<Message>)>,
    next_snapshot: SnapshotId,
}

impl ConversationHistory {
    pub fn new(system_prompt: &str) -> Self {
        Self {
            system_prompt: Message::system(system_prompt),
            messages: Vec::new(),
            snapshots: Vec::new(),
            next_snapshot: 0,
        }
    }

    /// Append a message to the history.
    pub fn push(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// All messages including system prompt (for display/logging).
    pub fn all(&self) -> Vec<Message> {
        let mut v = vec![self.system_prompt.clone()];
        v.extend(self.messages.clone());
        v
    }

    /// Conversation messages only (excluding system prompt).
    pub fn conversation(&self) -> &[Message] {
        &self.messages
    }

    /// System prompt message.
    pub fn system(&self) -> &Message {
        &self.system_prompt
    }

    /// Save current state; returns a snapshot ID.
    pub fn checkpoint(&mut self) -> SnapshotId {
        let id = self.next_snapshot;
        self.next_snapshot += 1;
        self.snapshots.push((id, self.messages.clone()));
        id
    }

    /// Restore to a snapshot. Returns Err if ID not found.
    pub fn restore(&mut self, id: SnapshotId) -> anyhow::Result<()> {
        match self.snapshots.iter().find(|(sid, _)| *sid == id) {
            Some((_, msgs)) => {
                self.messages = msgs.clone();
                Ok(())
            }
            None => anyhow::bail!("snapshot id {} not found", id),
        }
    }

    /// Clone current state (for parallel exploration / sub-agent forking).
    pub fn fork(&self) -> Self {
        Self {
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
            snapshots: self.snapshots.clone(),
            next_snapshot: self.next_snapshot,
        }
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::Role;

    #[test]
    fn new_history_has_system_message() {
        let h = ConversationHistory::new("You are helpful.");
        assert_eq!(h.system().role, Role::System);
        assert_eq!(h.system().content.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn push_adds_to_conversation_not_system() {
        let mut h = ConversationHistory::new("sys");
        h.push(Message::user("hello"));
        assert_eq!(h.conversation().len(), 1);
        assert_eq!(h.system().content.as_deref(), Some("sys"));
    }

    #[test]
    fn all_includes_system_first() {
        let mut h = ConversationHistory::new("sys");
        h.push(Message::user("hi"));
        let all = h.all();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].role, Role::System);
        assert_eq!(all[1].role, Role::User);
    }

    #[test]
    fn checkpoint_and_restore() {
        let mut h = ConversationHistory::new("sys");
        h.push(Message::user("first"));
        let id = h.checkpoint();
        h.push(Message::user("second"));
        assert_eq!(h.len(), 2);
        h.restore(id).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h.conversation()[0].content.as_deref(), Some("first"));
    }

    #[test]
    fn restore_unknown_id_returns_err() {
        let mut h = ConversationHistory::new("sys");
        assert!(h.restore(999).is_err());
    }

    #[test]
    fn fork_is_independent() {
        let mut h = ConversationHistory::new("sys");
        h.push(Message::user("original"));
        let mut forked = h.fork();
        forked.push(Message::user("extra"));
        assert_eq!(h.len(), 1);
        assert_eq!(forked.len(), 2);
    }
}
