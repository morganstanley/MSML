//! SIMD pretokenizer for `o200k_base`.
//!
//! The scheme is a thin specialization of the shared o200k-family mask
//! scanner: contraction suffixes are attached and numbers are grouped in
//! runs of at most three Unicode number characters.

use super::fast_mask::{MaskScheme, MaskState};
use super::fast_o200k_family;
use super::fast_unicode::warm_o200k;

pub(crate) struct O200kScheme;

/// Number of provisional pieces retained at a streaming chunk boundary.
pub(super) const STREAM_RETAIN: usize = 2;

impl MaskScheme for O200kScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        fast_o200k_family::advance_pos::<true, true, true, false>(bytes, pos)
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        fast_o200k_family::batch_masks::<true, true, true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: MaskState selects a runtime-detected SIMD tier.
        unsafe {
            fast_o200k_family::batch_masks_x86::<AVX512, true, true, true, false>(bytes, scan)
        }
    }
}

/// Whole-buffer o200k iterator. Streaming composition retains this scanner's
/// [`MaskState`] and rebases it when the settled prefix is removed.
pub(crate) struct FastO200kPretokenizer<'a> {
    input: &'a str,
    state: MaskState,
}

impl<'a> FastO200kPretokenizer<'a> {
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
        self.state.next_span::<O200kScheme>(self.input.as_bytes())
    }
}

impl<'a> Iterator for FastO200kPretokenizer<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (start, end) = self.next_span()?;
        Some(&self.input[start..end])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::FastO200kPretokenizer;
    use crate::pretok::Encoding;

    pub(crate) const CASES: &[&str] = &[
        "",
        "hello",
        "Hello",
        "HTTPResponse",
        "parseHTMLDocument",
        "don't can'ts x'll'd",
        "'sound 3'ts",
        "1234567 ١٢٣٤٥",
        " hello\tWorld",
        "hello, world!",
        "a/b .\n//x",
        "hello  \n\n  ",
        "café CAFÉ cafÉ",
        "ΑΒΓδε Привет Мир",
        "日本語ABC abc日本語Def",
        "e\u{301}f A\u{301}B",
        "देवनागरी में परीक्षण",
        "עִבְרִית נִקּוּד",
        "الْعَرَبِيَّة",
        "\u{a0}\u{2028}\r\nx",
    ];

    pub(crate) fn check(encoding: Encoding, scan: for<'a> fn(&'a str) -> Vec<&'a str>) {
        fn reference<'a>(regex: &fancy_regex::Regex, input: &'a str) -> Vec<&'a str> {
            regex
                .find_iter(input)
                .map(|m| m.unwrap().as_str())
                .collect()
        }

        let regex = fancy_regex::Regex::new(encoding.split_pattern()).unwrap();

        for input in CASES {
            assert_eq!(
                scan(input),
                reference(&regex, input),
                "{encoding:?}: {input:?}"
            );
            // Move every interesting transition across several 64-byte SIMD
            // batch boundaries, including the 70-byte lookahead cutoff.
            for prefix in [61, 63, 64, 65, 69, 127] {
                let padded = format!("{}{}{}", "x".repeat(prefix), input, " Z9".repeat(30));
                assert_eq!(
                    scan(&padded),
                    reference(&regex, &padded),
                    "{encoding:?}, boundary prefix {prefix}: {input:?}"
                );
            }
        }

        let alphabet = [
            'a', 'Z', 'é', 'ǅ', '日', 'Ж', 'ا', '한', '1', '٢', 'Ⅷ', ' ', '\t', '\n', '\r',
            '\u{a0}', '\u{2028}', '\u{301}', '\u{20dd}', '.', '\'', '/', '€', '\u{200b}',
        ];
        let mut seed = 0x93e3_5eed_12ab_7801u64;
        for round in 0..1_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let len = 1 + (seed as usize % 512);
            let mut input = String::new();
            for _ in 0..len {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                input.push(alphabet[seed as usize % alphabet.len()]);
            }
            assert_eq!(
                scan(&input),
                reference(&regex, &input),
                "{encoding:?}, random case {round}: {input:?}"
            );
        }

        // The generic streaming adapter may split UTF-8 and scanner batches
        // at unrelated boundaries. Verify that retaining only its unsettled
        // suffix is exactly equivalent to the whole-buffer scanner.
        let streaming_input = CASES.join(" | ").repeat(8);
        let expected = reference(&regex, &streaming_input)
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for target in [1, 2, 3, 7, 31, 63, 64, 65, 127, 256] {
            let mut stream = super::super::StreamPretokenizer::new(encoding).unwrap();
            let mut actual = Vec::new();
            let mut start = 0;
            while start < streaming_input.len() {
                let mut end = (start + target).min(streaming_input.len());
                while end > start && !streaming_input.is_char_boundary(end) {
                    end -= 1;
                }
                if end == start {
                    end = streaming_input[start..]
                        .char_indices()
                        .nth(1)
                        .map_or(streaming_input.len(), |(offset, _)| start + offset);
                }
                stream.feed(&streaming_input[start..end], |piece| {
                    actual.push(piece.to_owned())
                });
                start = end;
            }
            stream.finish(|piece| actual.push(piece.to_owned()));
            assert_eq!(actual, expected, "{encoding:?}, streaming target {target}");
        }
    }

    #[test]
    fn matches_reference() {
        check(Encoding::O200k, |input| {
            let actual = FastO200kPretokenizer::new(input).collect::<Vec<_>>();
            assert_eq!(
                FastO200kPretokenizer::new_scalar(input).collect::<Vec<_>>(),
                actual,
                "forced scalar mismatch: {input:?}"
            );
            actual
        });
    }
}
