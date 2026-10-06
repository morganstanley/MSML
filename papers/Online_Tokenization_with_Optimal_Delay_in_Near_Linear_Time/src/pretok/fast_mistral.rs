//! SIMD pretokenizer for Mistral-NeMo's Tekken encoding.
//!
//! Its split expression is Gigatoken's Nemotron member of the o200k family:
//! no contraction suffix and one Unicode number character per token.

use super::fast_mask::{MaskScheme, MaskState};
use super::fast_o200k_family;
use super::fast_unicode::warm_o200k;

pub(crate) struct MistralScheme;

/// Number of provisional pieces retained at a streaming chunk boundary.
pub(super) const STREAM_RETAIN: usize = 2;

impl MaskScheme for MistralScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        fast_o200k_family::advance_pos::<false, false, true, false>(bytes, pos)
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        fast_o200k_family::batch_masks::<false, false, true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: MaskState selects a runtime-detected SIMD tier.
        unsafe {
            fast_o200k_family::batch_masks_x86::<AVX512, false, false, true, false>(bytes, scan)
        }
    }
}

pub(crate) struct FastMistralPretokenizer<'a> {
    input: &'a str,
    state: MaskState,
}

impl<'a> FastMistralPretokenizer<'a> {
    #[inline]
    pub(crate) fn new(input: &'a str) -> Self {
        warm_o200k();
        Self {
            input,
            state: MaskState::new(input.as_bytes(), 0),
        }
    }

    #[cfg(test)]
    fn new_scalar(input: &'a str) -> Self {
        warm_o200k();
        Self {
            input,
            state: MaskState::new_scalar(0),
        }
    }

    #[inline]
    pub(crate) fn next_span(&mut self) -> Option<(usize, usize)> {
        self.state.next_span::<MistralScheme>(self.input.as_bytes())
    }
}

impl<'a> Iterator for FastMistralPretokenizer<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (start, end) = self.next_span()?;
        Some(&self.input[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::FastMistralPretokenizer;
    use crate::pretok::Encoding;

    #[test]
    fn matches_reference() {
        super::super::fast_o200k::tests::check(Encoding::Mistral, |input| {
            let actual = FastMistralPretokenizer::new(input).collect::<Vec<_>>();
            assert_eq!(
                FastMistralPretokenizer::new_scalar(input).collect::<Vec<_>>(),
                actual,
                "forced scalar mismatch: {input:?}"
            );
            actual
        });
    }
}
