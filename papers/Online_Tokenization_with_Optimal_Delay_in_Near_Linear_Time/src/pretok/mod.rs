//! Pretokenization strategies for the tiktoken split patterns.
//!
//! [`Pretok`] selects between encoding-specific SIMD mask scanners and
//! DFA-based pretokenizers over simplified patterns plus small boundary
//! fixups. They all produce identical pretoken boundaries; they differ only in
//! the stepping strategy. The split pattern itself is selected by [`Encoding`].

mod dfa_stream;
mod encoding;
mod fast_cl100k;
mod fast_cl100k_family;
mod fast_llama3;
mod fast_mask;
mod fast_mistral;
mod fast_o200k;
mod fast_o200k_family;
mod fast_p50k;
mod fast_qwen3;
mod fast_r50k;
mod fast_unicode;
mod lazy;
mod stream_pretok;

pub(crate) use dfa_stream::DfaStream;
pub use encoding::{
    CL100K_DFA_PATTERN, CL100K_SPLIT_PATTERN, Encoding, O200K_DFA_PATTERN, O200K_SPLIT_PATTERN,
    R50K_DFA_PATTERN, R50K_SPLIT_PATTERN,
};
pub(crate) use fast_cl100k::FastCl100kPretokenizer;
pub(crate) use fast_llama3::FastLlama3Pretokenizer;
pub(crate) use fast_mistral::FastMistralPretokenizer;
pub(crate) use fast_o200k::FastO200kPretokenizer;
pub(crate) use fast_qwen3::FastQwen3Pretokenizer;
pub use fast_r50k::FastR50kPretokenizer;
pub use lazy::LazyPretokenizer;
pub use stream_pretok::StreamPretokenizer;
pub(crate) use stream_pretok::{KEYED_BATCH, KeyedBatchConsumer, KeyedBatchStats, KeyedSpan};

/// A pre-split stage applied to a span *before* the main pattern pretokenizer,
/// mirroring HuggingFace `Sequence` pre-tokenizer components that run ahead of
/// `ByteLevel`. It only adds cut points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Presplit {
    /// No pre-split: the whole span goes straight to the pattern pretokenizer.
    #[default]
    None,
    /// HF `Digits { individual_digits: true }`: isolate each numeric char
    /// (`char::is_numeric`, i.e. `\p{N}`) as its own sub-span; maximal
    /// non-numeric runs stay together. Used by StarCoder.
    ///
    /// Example:
    /// "   11" -> "    ", "1", "1" with split and then regex
    /// "   11" -> "   ", " 1", "1" with regex alone
    DigitsIndividual,
}

impl Presplit {
    #[inline]
    pub fn split<'a>(&self, s: &'a str, mut emit: impl FnMut(&'a str)) {
        match self {
            Presplit::None => {
                if !s.is_empty() {
                    emit(s);
                }
            }
            Presplit::DigitsIndividual => {
                let mut last = 0;
                for (idx, c) in s.char_indices() {
                    if c.is_numeric() {
                        if last < idx {
                            emit(&s[last..idx]);
                        }
                        let end = idx + c.len_utf8();
                        emit(&s[idx..end]);
                        last = end;
                    }
                }
                if last < s.len() {
                    emit(&s[last..]);
                }
            }
        }
    }
}

/// Pretokenization strategy. Construct with [`Pretok::build`].
pub enum Pretok {
    /// Hardware mask scanner selected by [`Encoding`].
    Fast(Encoding),
    /// Hybrid lazy DFA stepped one byte at a time (incremental stepping, but
    /// not streaming — requires the full input).
    LazyDfa(LazyPretokenizer),
    /// Streaming dispatcher: encoding-specific SIMD by default, or the DFA
    /// compatibility path when explicitly forced.
    Stream(StreamPretokenizer),
}

impl Pretok {
    /// Build a pretokenizer for `enc` from a mode string: `"simd"`, `"dfa"`, or
    /// `"dfa-stream"`.
    /// `"simd"` selects the encoding's hardware mask scanner.
    pub fn build(mode: &str, enc: Encoding) -> Result<Pretok, Box<dyn std::error::Error>> {
        let pretok = match mode {
            "simd" => Pretok::Fast(enc),
            "dfa" => Pretok::LazyDfa(LazyPretokenizer::new(enc)?),
            "dfa-stream" => Pretok::Stream(StreamPretokenizer::new_with_mode(enc, true)?),
            _ => {
                return Err(format!(
                    "unknown pretokenizer mode '{mode}' (use simd|dfa|dfa-stream)"
                )
                .into());
            }
        };
        Ok(pretok)
    }

    /// Human-readable description of the strategy.
    pub fn description(&self) -> &'static str {
        match self {
            Pretok::Fast(_) => "SIMD mask scanner (NEON/AVX-512/AVX2, scalar Unicode fallback)",
            Pretok::LazyDfa(_) => "lazy DFA, byte-stepping (not streaming)",
            Pretok::Stream(_) => "streaming SIMD scanner (DFA compatibility path available)",
        }
    }

    /// Short name for the results table (combined with the BPE name, e.g.
    /// "mtc+dfa" or "tik+dfa").
    pub fn suffix(&self) -> &'static str {
        match self {
            Pretok::Fast(_) => "simd",
            Pretok::LazyDfa(_) => "dfa",
            Pretok::Stream(_) => "dfastream",
        }
    }

    /// Split `input` into pre-tokens, invoking `emit` once per piece.
    ///
    /// `emit` takes `&str` (any lifetime) rather than `&'a str`: the zero-copy
    /// engines hand back subslices of `input`, but [`Pretok::Stream`] emits
    /// slices of its own internal buffer, so callers must not assume the piece
    /// borrows from `input`. For `Stream`, this feeds `input` as a single chunk
    /// and finishes; true chunked streaming uses [`StreamPretokenizer`] directly.
    pub fn split(&mut self, input: &str, mut emit: impl FnMut(&str)) {
        match self {
            Pretok::Fast(enc) => split_fast(*enc, input, emit),
            Pretok::LazyDfa(p) => p.split(input, emit),
            Pretok::Stream(p) => {
                p.feed(input, &mut emit);
                p.finish(&mut emit);
            }
        }
    }
}

/// Dispatch once per input, keeping the concrete iterator monomorphized inside
/// each arm rather than matching once per pretoken.
#[inline]
fn split_fast(enc: Encoding, input: &str, mut emit: impl FnMut(&str)) {
    macro_rules! emit_all {
        ($scanner:expr) => {
            for piece in $scanner {
                emit(piece);
            }
        };
    }

    match enc {
        Encoding::R50k => emit_all!(FastR50kPretokenizer::new(input)),
        Encoding::P50k => emit_all!(fast_p50k::FastP50kPretokenizer::new(input)),
        Encoding::Cl100k => emit_all!(FastCl100kPretokenizer::new(input)),
        Encoding::O200k => emit_all!(FastO200kPretokenizer::new(input)),
        Encoding::Llama3 => emit_all!(FastLlama3Pretokenizer::new(input)),
        Encoding::Mistral => emit_all!(FastMistralPretokenizer::new(input)),
        Encoding::Qwen3 => emit_all!(FastQwen3Pretokenizer::new(input)),
    }
}

#[cfg(test)]
mod tests {
    use super::{Encoding, Pretok};

    #[test]
    fn rejects_unknown_mode() {
        let error = match Pretok::build("unknown", Encoding::R50k) {
            Ok(_) => panic!("unknown pretokenizer mode was accepted"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "unknown pretokenizer mode 'unknown' (use simd|dfa|dfa-stream)"
        );
    }
}
