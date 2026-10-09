//! Input normalization stage, run between the special-token segmenter and the
//! pre-tokenizer (mirroring a HuggingFace tokenizer's `normalizer`). Most
//! byte-level BPE tokenizers have none — those use the pass-through
//! [`Normalizer::None`], which borrows the input unchanged with no allocation.
//! qwen / neox apply Unicode NFC.

use std::borrow::Cow;

use unicode_normalization::{UnicodeNormalization, char::canonical_combining_class};

/// The normalization a tokenizer applies to text before pre-tokenization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Normalizer {
    /// No normalization: the input passes through unchanged (no allocation).
    #[default]
    None,
    /// Unicode NFC (canonical composition).
    Nfc,
}

impl Normalizer {
    pub fn from_name(name: &str) -> Option<Normalizer> {
        match name {
            "none" => Some(Normalizer::None),
            "NFC" => Some(Normalizer::Nfc),
            _ => None,
        }
    }

    #[inline]
    pub fn normalize<'a>(&self, s: &'a str) -> Cow<'a, str> {
        match self {
            Normalizer::None => Cow::Borrowed(s),
            Normalizer::Nfc => Cow::Owned(s.nfc().collect()),
        }
    }
}

/// A Hangul *conjoining* V (vowel) or T (trailing) jamo. These compose backward
/// with a preceding L / LV under NFC (L+V→LV, LV+T→LVT), so a break must not be
/// placed before one — otherwise the leading jamo would be orphaned.
#[inline]
fn is_hangul_vt(c: char) -> bool {
    matches!(c as u32,
        0x1161..=0x1175 | 0x11A8..=0x11C2   // modern conjoining V / T
        | 0xD7B0..=0xD7C6 | 0xD7CB..=0xD7FB) // extended conjoining V / T
}

/// Byte index of the last *safe NFC boundary* in `s`: the start of the last
/// "stable starter" — a char with canonical combining class 0 that is not a
/// Hangul V/T jamo. NFC composition and canonical ordering are confined within a
/// segment (a starter and its following combining marks), and the only backward
/// composition across a starter is Hangul, so `s[..idx]` normalizes identically
/// regardless of what follows and can be flushed, while `s[idx..]` must be
/// carried (it may still grow / compose with the next input). Returns 0 if there
/// is no such boundary (the whole string must be carried).
///
/// Used by the streaming tokenizer to apply NFC incrementally in O(input) time
/// and O(longest combining sequence) space — the same bound the rest of the
/// streaming pipeline already carries.
pub(crate) fn last_nfc_boundary(s: &str) -> usize {
    let mut last = 0;
    for (i, c) in s.char_indices() {
        if canonical_combining_class(c) == 0 && !is_hangul_vt(c) {
            last = i;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Emulate the streaming NFC: feed `s` in chunks of `step` bytes, flushing up
    /// to the last safe boundary and carrying the rest, then flush at EOF.
    fn incremental_nfc(s: &str, step: usize) -> String {
        let (mut out, mut carry) = (String::new(), String::new());
        let mut i = 0;
        while i < s.len() {
            let mut j = (i + step).min(s.len());
            while !s.is_char_boundary(j) {
                j += 1;
            }
            carry.push_str(&s[i..j]);
            let k = last_nfc_boundary(&carry);
            out.push_str(&Normalizer::Nfc.normalize(&carry[..k]));
            carry.drain(..k);
            i = j;
        }
        out.push_str(&Normalizer::Nfc.normalize(&carry));
        out
    }

    #[test]
    fn incremental_nfc_matches_oneshot() {
        let cases = [
            "cafe\u{0301}",                        // e + combining acute -> é
            "a\u{0301}\u{0323}b\u{0323}\u{0301}c", // reordered combining marks
            "\u{1100}\u{1161}\u{11A8}",            // Hangul L V T jamo -> 각
            "\u{AC00}\u{11A8}",                    // precomposed 가 + T jamo -> 갇
            "A\u{030A}ngstro\u{0308}m saved",      // Å, ö
            "한글 test café déjà 12345",
            "中文字符没有组合标记", // CJK precomposed starters
            "plain ascii no marks at all",
        ];
        for s in cases {
            let oneshot: String = s.nfc().collect();
            for step in [1usize, 2, 3, 4, 7, 1000] {
                assert_eq!(incremental_nfc(s, step), oneshot, "case {s:?} step {step}");
            }
        }
    }
}
