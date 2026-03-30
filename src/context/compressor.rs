#[derive(Debug, Clone)]
pub struct OutputCompressor {
    pub max_bytes: usize,
    pub strategy: CompressionStrategy,
}

#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    /// Keep first `head` bytes + last `tail` bytes; insert "[... N bytes omitted ...]" in middle.
    HeadTail { head: usize, tail: usize },

    /// Keep last `size` bytes only.
    TailOnly { size: usize },

    /// Hard truncate at max_bytes with "[truncated]" suffix.
    HardTruncate,
}

impl Default for OutputCompressor {
    fn default() -> Self {
        Self {
            max_bytes: 8 * 1024,
            strategy: CompressionStrategy::HeadTail { head: 2048, tail: 2048 },
        }
    }
}

impl OutputCompressor {
    /// Compress `output` if it exceeds `max_bytes`. Returns (compressed, was_compressed).
    pub fn compress(&self, output: &str) -> (String, bool) {
        let bytes = output.as_bytes();
        if bytes.len() <= self.max_bytes {
            return (output.to_string(), false);
        }

        let result = match &self.strategy {
            CompressionStrategy::HeadTail { head, tail } => {
                let head = (*head).min(bytes.len());
                let tail = (*tail).min(bytes.len().saturating_sub(head));
                let tail_start = bytes.len().saturating_sub(tail);
                let omitted = tail_start.saturating_sub(head);

                let head_str = String::from_utf8_lossy(&bytes[..head]);
                let tail_str = String::from_utf8_lossy(&bytes[tail_start..]);
                format!(
                    "{}\n... [{} bytes omitted] ...\n{}",
                    head_str, omitted, tail_str
                )
            }
            CompressionStrategy::TailOnly { size } => {
                let start = bytes.len().saturating_sub(*size);
                String::from_utf8_lossy(&bytes[start..]).into_owned()
            }
            CompressionStrategy::HardTruncate => {
                let truncated = &bytes[..self.max_bytes];
                format!("{}[truncated]", String::from_utf8_lossy(truncated))
            }
        };

        (result, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_short_string_unchanged() {
        let c = OutputCompressor::default();
        let short = "hello world";
        let (out, was_compressed) = c.compress(short);
        assert_eq!(out, short);
        assert!(!was_compressed);
    }

    #[test]
    fn compress_head_tail_elides_middle() {
        let c = OutputCompressor {
            max_bytes: 10,
            strategy: CompressionStrategy::HeadTail { head: 3, tail: 3 },
        };
        let input = "abcdefghijklmno"; // 15 bytes
        let (out, was_compressed) = c.compress(input);
        assert!(was_compressed);
        assert!(out.starts_with("abc"));
        assert!(out.ends_with("mno"));
        assert!(out.contains("omitted"));
    }

    #[test]
    fn compress_head_tail_omit_count_correct() {
        let c = OutputCompressor {
            max_bytes: 10,
            strategy: CompressionStrategy::HeadTail { head: 3, tail: 3 },
        };
        let input = "abcdefghijklmno"; // 15 bytes, head=3, tail=3, omitted = 15-3-3 = 9
        let (out, _) = c.compress(input);
        assert!(out.contains("9 bytes omitted"));
    }

    #[test]
    fn compress_tail_only() {
        let c = OutputCompressor {
            max_bytes: 5,
            strategy: CompressionStrategy::TailOnly { size: 3 },
        };
        let input = "abcdefgh"; // 8 bytes
        let (out, was_compressed) = c.compress(input);
        assert!(was_compressed);
        assert_eq!(out, "fgh");
    }

    #[test]
    fn compress_hard_truncate() {
        let c = OutputCompressor {
            max_bytes: 5,
            strategy: CompressionStrategy::HardTruncate,
        };
        let input = "abcdefgh";
        let (out, was_compressed) = c.compress(input);
        assert!(was_compressed);
        assert!(out.starts_with("abcde"));
        assert!(out.ends_with("[truncated]"));
    }

    #[test]
    fn compress_returns_was_compressed_flag() {
        let c = OutputCompressor {
            max_bytes: 5,
            strategy: CompressionStrategy::HardTruncate,
        };
        let (_, short_compressed) = c.compress("hi");
        let (_, long_compressed) = c.compress("this is longer than 5 bytes");
        assert!(!short_compressed);
        assert!(long_compressed);
    }
}
