//! SIMD pretokenizer for Qwen 3.
//!
//! This is Gigatoken's MIT-licensed Qwen2 mask scheme: contractions,
//! single-codepoint number pieces, and newline-priority whitespace.

use super::fast_cl100k_family;
use super::fast_mask::{MaskScheme, MaskState};
use super::fast_unicode::{self, ClassTable};

pub(crate) struct Qwen3Scheme;

/// Number of provisional pieces retained at a streaming chunk boundary.
pub(super) const STREAM_RETAIN: usize = 1;

impl MaskScheme for Qwen3Scheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        fast_cl100k_family::advance(bytes, pos, false, false)
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        let classes = ClassTable::get();
        fast_cl100k_family::batch_masks(bytes, scan, false, move |cp| {
            // SAFETY: the family scanner obtains `cp` from valid UTF-8.
            unsafe { classes.class_of(cp) }
        })
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        let classes = ClassTable::get();
        unsafe {
            fast_cl100k_family::batch_masks_x86::<AVX512>(bytes, scan, false, move |cp| {
                // SAFETY: the family scanner obtains `cp` from valid UTF-8.
                classes.class_of(cp)
            })
        }
    }
}

pub(crate) struct FastQwen3Pretokenizer<'a> {
    input: &'a str,
    state: MaskState,
}

impl<'a> FastQwen3Pretokenizer<'a> {
    #[inline]
    pub(crate) fn new(input: &'a str) -> Self {
        fast_unicode::warm();
        Self {
            input,
            state: MaskState::new(input.as_bytes(), 0),
        }
    }

    #[cfg(test)]
    fn new_scalar(input: &'a str) -> Self {
        fast_unicode::warm();
        Self {
            input,
            state: MaskState::new_scalar(0),
        }
    }

    #[inline]
    pub(crate) fn next_span(&mut self) -> Option<(usize, usize)> {
        self.state.next_span::<Qwen3Scheme>(self.input.as_bytes())
    }
}

impl<'a> Iterator for FastQwen3Pretokenizer<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (start, end) = self.next_span()?;
        Some(&self.input[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::FastQwen3Pretokenizer;
    use crate::pretok::Encoding;

    #[test]
    fn matches_reference() {
        super::super::fast_cl100k_family::tests::check(Encoding::Qwen3, |input| {
            let actual = FastQwen3Pretokenizer::new(input)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            assert_eq!(
                FastQwen3Pretokenizer::new_scalar(input)
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
                actual,
                "forced scalar mismatch: {input:?}"
            );
            actual
        });
    }
}
