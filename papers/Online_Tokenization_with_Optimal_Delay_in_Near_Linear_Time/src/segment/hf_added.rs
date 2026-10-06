//! Hugging Face `AddedToken` semantics that exact literal matching cannot express.

use super::Seg;
use std::sync::Arc;

/// A literal token extracted before ordinary pre-tokenization and BPE.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

impl AddedToken {
    /// Build an exact literal with no boundary or whitespace modifiers.
    pub fn exact(content: impl Into<String>, id: u32) -> Self {
        Self {
            content: content.into(),
            id,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        }
    }

    fn has_matching_modifier(&self) -> bool {
        self.single_word || self.lstrip || self.rstrip
    }
}

pub(super) fn needs_hf_matcher(tokens: &[AddedToken]) -> bool {
    tokens
        .iter()
        .any(|token| token.has_matching_modifier() || !token.content.is_ascii())
        || has_prefix_overlap(tokens)
}

fn has_prefix_overlap(tokens: &[AddedToken]) -> bool {
    let mut literals: Vec<&str> = tokens.iter().map(|token| token.content.as_str()).collect();
    literals.sort_unstable();
    literals.windows(2).any(|pair| pair[1].starts_with(pair[0]))
}

/// Streaming leftmost-longest matcher for Hugging Face added-token modifiers.
///
/// It retains only unsettled lookahead: a partial literal, word-boundary
/// lookahead, or whitespace that an `lstrip`/`rstrip` token may consume.
pub(super) struct HfConfig {
    tokens: Vec<CompiledToken>,
    has_lstrip: bool,
}

/// Matching-only projection of a Hugging Face descriptor. `normalized` and
/// `special` are source metadata; after model validation they play no role in
/// segmentation and need not remain resident.
struct CompiledToken {
    content: Box<str>,
    id: u32,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
}

pub(super) struct HfMatcher {
    config: Arc<HfConfig>,
    pending: String,
    /// Whether the character immediately before `pending` is a Unicode `\w`.
    prev_is_word: bool,
    peak: usize,
    held_sum: u64,
    bytes_seen: u64,
}

impl HfConfig {
    pub(super) fn new(tokens: &[AddedToken]) -> Self {
        let tokens: Vec<CompiledToken> = tokens
            .iter()
            .filter(|token| !token.content.is_empty())
            .map(|token| CompiledToken {
                content: token.content.as_str().into(),
                id: token.id,
                single_word: token.single_word,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            })
            .collect();
        Self {
            has_lstrip: tokens.iter().any(|token| token.lstrip),
            tokens,
        }
    }
}

impl HfMatcher {
    pub(super) fn from_config(config: Arc<HfConfig>) -> Self {
        Self {
            config,
            pending: String::new(),
            prev_is_word: false,
            peak: 0,
            held_sum: 0,
            bytes_seen: 0,
        }
    }

    pub(super) fn peak_held(&self) -> usize {
        self.peak
    }

    pub(super) fn held_sum(&self) -> u64 {
        self.held_sum
    }

    pub(super) fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }

    pub(super) fn reset_peak(&mut self) {
        debug_assert!(
            self.pending.is_empty(),
            "segmenter must be idle between runs"
        );
        self.peak = 0;
    }

    #[inline]
    pub(super) fn feed<const PROFILE: bool>(&mut self, chunk: &str, sink: &mut impl FnMut(Seg)) {
        if PROFILE {
            self.bytes_seen += chunk.len() as u64;
        }
        self.pending.push_str(chunk);
        self.process(false, sink);
        if PROFILE {
            self.peak = self.peak.max(self.pending.len());
            self.held_sum += self.pending.len() as u64 * chunk.len() as u64;
        }
    }

    pub(super) fn finish(&mut self, sink: &mut impl FnMut(Seg)) {
        self.process(true, sink);
        debug_assert!(self.pending.is_empty());
        self.prev_is_word = false;
    }

    fn process(&mut self, eof: bool, sink: &mut impl FnMut(Seg)) {
        loop {
            let partial_start = self.partial_start();
            let Some((start, token_index)) = self.next_complete() else {
                let hold = if eof {
                    self.pending.len()
                } else {
                    self.no_match_hold_start(partial_start)
                };
                self.emit_text_prefix(hold, sink);
                return;
            };

            // An earlier partial literal may still become the leftmost-longest
            // match after another chunk arrives.
            if !eof && partial_start.is_some_and(|partial| partial <= start) {
                let hold = self.adjust_lstrip_start(start.min(partial_start.unwrap()));
                self.emit_text_prefix(hold, sink);
                return;
            }

            let token = &self.config.tokens[token_index];
            let end = start + token.content.len();
            let left_is_word = if start == 0 {
                self.prev_is_word
            } else {
                ends_with_word(&self.pending[..start])
            };

            if token.single_word && end == self.pending.len() && !eof {
                // The next chunk determines the right word boundary.
                let hold = self.adjust_lstrip_start(start);
                self.emit_text_prefix(hold, sink);
                return;
            }
            let right_is_word = end < self.pending.len() && starts_with_word(&self.pending[end..]);
            if token.single_word && (left_is_word || right_is_word) {
                // HF discards a failed leftmost-longest candidate rather than
                // reconsidering an inner overlap.
                self.emit_text_prefix(end, sink);
                continue;
            }

            let match_start = if token.lstrip {
                self.adjust_lstrip_start(start)
            } else {
                start
            };
            let mut match_end = end;
            if token.rstrip {
                match_end += leading_whitespace_len(&self.pending[end..]);
                if match_end == self.pending.len() && !eof {
                    // The whitespace run may continue in the next chunk.
                    self.emit_text_prefix(match_start, sink);
                    return;
                }
            }
            let id = token.id;

            self.emit_text_prefix(match_start, sink);
            let consumed = match_end - match_start;
            self.consume_without_emitting(consumed);
            sink(Seg::Special(id));
        }
    }

    /// Earliest complete literal, breaking same-offset ties by longest match.
    fn next_complete(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (index, token) in self.config.tokens.iter().enumerate() {
            let Some(start) = self.pending.find(token.content.as_ref()) else {
                continue;
            };
            match best {
                None => best = Some((start, index)),
                Some((best_start, best_index))
                    if start < best_start
                        || (start == best_start
                            && token.content.len()
                                > self.config.tokens[best_index].content.len()) =>
                {
                    best = Some((start, index));
                }
                _ => {}
            }
        }
        best
    }

    /// Earliest suffix that is a proper prefix of any configured literal.
    fn partial_start(&self) -> Option<usize> {
        let bytes = self.pending.as_bytes();
        let mut earliest = None;
        for token in &self.config.tokens {
            let literal = token.content.as_bytes();
            let max = bytes.len().min(literal.len().saturating_sub(1));
            for prefix_len in (1..=max).rev() {
                if bytes.ends_with(&literal[..prefix_len]) {
                    let start = bytes.len() - prefix_len;
                    earliest = Some(earliest.map_or(start, |old: usize| old.min(start)));
                    break;
                }
            }
        }
        earliest
    }

    fn no_match_hold_start(&self, partial_start: Option<usize>) -> usize {
        let mut hold = partial_start.unwrap_or(self.pending.len());
        if self.config.has_lstrip {
            hold = self.adjust_lstrip_start(hold);
        }
        hold
    }

    fn adjust_lstrip_start(&self, start: usize) -> usize {
        trailing_whitespace_start(&self.pending[..start])
    }

    fn emit_text_prefix(&mut self, end: usize, sink: &mut impl FnMut(Seg)) {
        if end == 0 {
            return;
        }
        {
            let text = &self.pending[..end];
            sink(Seg::Text(text));
            self.prev_is_word = ends_with_word(text);
        }
        self.pending.drain(..end);
    }

    fn consume_without_emitting(&mut self, end: usize) {
        if end == 0 {
            return;
        }
        self.prev_is_word = ends_with_word(&self.pending[..end]);
        self.pending.drain(..end);
    }
}

fn leading_whitespace_len(text: &str) -> usize {
    text.char_indices()
        .take_while(|(_, character)| character.is_whitespace())
        .map(|(offset, character)| offset + character.len_utf8())
        .last()
        .unwrap_or(0)
}

fn trailing_whitespace_start(text: &str) -> usize {
    text.char_indices()
        .rev()
        .take_while(|(_, character)| character.is_whitespace())
        .map(|(offset, _)| offset)
        .last()
        .unwrap_or(text.len())
}

fn starts_with_word(text: &str) -> bool {
    static STARTS_WITH_WORD: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"^\w").unwrap());
    STARTS_WITH_WORD.is_match(text)
}

fn ends_with_word(text: &str) -> bool {
    static ENDS_WITH_WORD: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"\w$").unwrap());
    ENDS_WITH_WORD.is_match(text)
}
