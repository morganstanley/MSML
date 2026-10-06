//! Streaming segmentation around literal tokens that bypass ordinary BPE.
//!
//! Exact ASCII literals use the small trie in [`exact`]. Hugging Face token
//! sets that need modifiers, Unicode matching, or leftmost-longest overlap use
//! the correctness-oriented matcher in [`hf_added`].

mod exact;
mod hf_added;

#[cfg(test)]
mod tests;

use crate::Encoding;
use exact::{ExactConfig, ExactMatcher};
use hf_added::{HfConfig, HfMatcher, needs_hf_matcher};
use std::sync::Arc;

pub use hf_added::AddedToken;

/// One settled segment of the input stream.
pub enum Seg<'a> {
    /// Text that continues through normal pre-tokenization and BPE.
    Text(&'a str),
    /// A literal token emitted directly as a hard BPE boundary.
    Special(u32),
}

enum Matcher {
    Exact(ExactMatcher),
    HuggingFace(HfMatcher),
}

/// Immutable added-token matching program shared by tokenizer streams.
///
/// Model loading compiles descriptors into this representation once. Each
/// stream then clones only an `Arc` and keeps its own pending-input and
/// profiling state.
#[derive(Clone)]
pub(crate) struct SegmenterConfig {
    matcher: ConfigMatcher,
}

#[derive(Clone)]
enum ConfigMatcher {
    Exact(Arc<ExactConfig>),
    HuggingFace(Arc<HfConfig>),
}

impl SegmenterConfig {
    /// Compile exact ASCII `(literal, token_id)` pairs once.
    pub(crate) fn from_specials(specials: &[(&str, u32)]) -> Self {
        Self {
            matcher: ConfigMatcher::Exact(Arc::new(ExactConfig::new(specials))),
        }
    }

    /// Compile Hugging Face added-token descriptors once.
    pub(crate) fn from_added_tokens(tokens: &[AddedToken]) -> Self {
        let matcher = if needs_hf_matcher(tokens) {
            ConfigMatcher::HuggingFace(Arc::new(HfConfig::new(tokens)))
        } else {
            let pairs: Vec<(&str, u32)> = tokens
                .iter()
                .map(|token| (token.content.as_str(), token.id))
                .collect();
            ConfigMatcher::Exact(Arc::new(ExactConfig::new(&pairs)))
        };
        Self { matcher }
    }

    fn instantiate(&self) -> Matcher {
        match &self.matcher {
            ConfigMatcher::Exact(config) => {
                Matcher::Exact(ExactMatcher::from_config(Arc::clone(config)))
            }
            ConfigMatcher::HuggingFace(config) => {
                Matcher::HuggingFace(HfMatcher::from_config(Arc::clone(config)))
            }
        }
    }
}

/// Streaming matcher for tiktoken specials and Hugging Face added tokens.
///
/// Only unsettled input is retained across calls. Exact matching holds at most
/// the longest literal; Hugging Face whitespace modifiers may additionally
/// retain an unfinished whitespace run.
pub struct SpecialSegmenter {
    matcher: Matcher,
}

impl SpecialSegmenter {
    /// Build from an encoding's built-in exact special tokens.
    pub fn new(encoding: Encoding) -> Self {
        Self::from_specials(encoding.special_tokens())
    }

    /// Build from exact ASCII `(literal, token_id)` pairs.
    pub fn from_specials(specials: &[(&str, u32)]) -> Self {
        Self::from_config(SegmenterConfig::from_specials(specials))
    }

    /// Build from Hugging Face-style added-token descriptors.
    ///
    /// Simple ASCII, prefix-free sets still use the exact trie. Only semantics
    /// that the exact matcher cannot express select the general HF matcher.
    pub fn from_added_tokens(tokens: &[AddedToken]) -> Self {
        Self::from_config(SegmenterConfig::from_added_tokens(tokens))
    }

    /// Start an independent stream from an already compiled immutable config.
    pub(crate) fn from_config(config: SegmenterConfig) -> Self {
        Self {
            matcher: config.instantiate(),
        }
    }

    /// Peak bytes ever retained as unsettled input.
    pub fn peak_held(&self) -> usize {
        match &self.matcher {
            Matcher::Exact(matcher) => matcher.peak_held(),
            Matcher::HuggingFace(matcher) => matcher.peak_held(),
        }
    }

    /// Average retained bytes per input byte.
    pub fn avg_held(&self) -> f64 {
        self.held_sum() as f64 / self.bytes_seen().max(1) as f64
    }

    /// Total bytes fed since construction.
    pub fn bytes_seen(&self) -> u64 {
        match &self.matcher {
            Matcher::Exact(matcher) => matcher.bytes_seen(),
            Matcher::HuggingFace(matcher) => matcher.bytes_seen(),
        }
    }

    pub(crate) fn held_sum(&self) -> u64 {
        match &self.matcher {
            Matcher::Exact(matcher) => matcher.held_sum(),
            Matcher::HuggingFace(matcher) => matcher.held_sum(),
        }
    }

    pub(crate) fn reset_peak(&mut self) {
        match &mut self.matcher {
            Matcher::Exact(matcher) => matcher.reset_peak(),
            Matcher::HuggingFace(matcher) => matcher.reset_peak(),
        }
    }

    /// Feed one chunk and collect held-byte profiling statistics.
    pub fn feed(&mut self, chunk: &str, sink: impl FnMut(Seg)) {
        self.feed_inner::<true>(chunk, sink);
    }

    /// Feed one chunk with profiling bookkeeping compiled out.
    pub(crate) fn feed_unprofiled(&mut self, chunk: &str, sink: impl FnMut(Seg)) {
        self.feed_inner::<false>(chunk, sink);
    }

    #[inline]
    fn feed_inner<const PROFILE: bool>(&mut self, chunk: &str, mut sink: impl FnMut(Seg)) {
        match &mut self.matcher {
            Matcher::Exact(matcher) => matcher.feed::<PROFILE>(chunk, &mut sink),
            Matcher::HuggingFace(matcher) => matcher.feed::<PROFILE>(chunk, &mut sink),
        }
    }

    /// Settle an unfinished literal at end of stream and reset matcher state.
    pub fn finish(&mut self, mut sink: impl FnMut(Seg)) {
        match &mut self.matcher {
            Matcher::Exact(matcher) => matcher.finish(&mut sink),
            Matcher::HuggingFace(matcher) => matcher.finish(&mut sink),
        }
    }
}
