//! SIMD pretokenizer for the r50k/p50k split expression.
//!
//! This file contains only r50k-specific scalar semantics and boundary-mask
//! algebra. The shared mask engine owns architecture-tier dispatch, two-phase
//! 256-pretoken harvesting, key/hash construction, and streaming composition.
//! The design is adapted from Gigatoken's MIT-licensed r50k scanner.

use super::{
    fast_mask::{self, MaskScheme, MaskState},
    fast_unicode::{self, CharClass, ClassTable, decode_cp, is_ascii_ws, is_digit, is_letter},
};

/// Two provisional pieces cover a contraction split across a chunk boundary
/// (`'` + `l` may become `'ll`) and trailing-whitespace settlement.
pub(super) const STREAM_RETAIN: usize = 2;

#[inline]
pub(crate) fn warm_unicode_classes() {
    fast_unicode::warm();
}

// --------------------------------------------------------------------------
// Scalar ground truth
// --------------------------------------------------------------------------

const HIGH_BITS: u64 = 0x8080_8080_8080_8080;

#[inline(always)]
fn ascii_nonletter_mask(word: u64) -> u64 {
    let lowered = word | 0x2020_2020_2020_2020;
    let ge_a = (lowered | HIGH_BITS).wrapping_sub(0x6161_6161_6161_6161);
    let le_z = 0xfafa_fafa_fafa_fafa_u64.wrapping_sub(lowered);
    !(ge_a & le_z) & HIGH_BITS
}

#[inline(always)]
fn scan_ascii_letters(bytes: &[u8], mut pos: usize) -> usize {
    while pos + 8 <= bytes.len() {
        let word = unsafe { (bytes.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & HIGH_BITS != 0 {
            break;
        }
        let nonletter = ascii_nonletter_mask(word);
        if nonletter != 0 {
            return pos + nonletter.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    while pos < bytes.len() && is_letter(bytes[pos]) {
        pos += 1;
    }
    pos
}

#[inline(always)]
fn class_at(bytes: &[u8], pos: usize) -> (CharClass, usize) {
    let byte = bytes[pos];
    if byte < 0x80 {
        let class = if is_letter(byte) {
            CharClass::Letter
        } else if is_digit(byte) {
            CharClass::Number
        } else if is_ascii_ws(byte) {
            CharClass::Whitespace
        } else {
            CharClass::Other
        };
        return (class, 1);
    }
    // SAFETY: callers pass a valid UTF-8 string and `pos` is a scalar lead.
    let (cp, len) = unsafe { decode_cp(bytes, pos) };
    // SAFETY: valid UTF-8 decoding produces a Unicode scalar value.
    (unsafe { ClassTable::get().class_of(cp) }, len)
}

#[inline(always)]
fn scan_letters(bytes: &[u8], mut pos: usize) -> usize {
    loop {
        pos = scan_ascii_letters(bytes, pos);
        if pos >= bytes.len() {
            return pos;
        }
        let (class, len) = class_at(bytes, pos);
        if class != CharClass::Letter {
            return pos;
        }
        pos += len;
    }
}

#[inline(always)]
fn scan_numbers(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let (class, len) = class_at(bytes, pos);
        if class != CharClass::Number {
            break;
        }
        pos += len;
    }
    pos
}

#[inline(always)]
fn scan_other(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let (class, len) = class_at(bytes, pos);
        if class != CharClass::Other {
            break;
        }
        pos += len;
    }
    pos
}

#[inline(always)]
fn previous_char_start(bytes: &[u8], pos: usize) -> usize {
    let mut start = pos - 1;
    while bytes[start] & 0xc0 == 0x80 {
        start -= 1;
    }
    start
}

#[inline(always)]
fn advance_whitespace(bytes: &[u8], start: usize, resume: usize) -> usize {
    let mut pos = resume.max(start).min(bytes.len());
    let mut last = if pos > start {
        previous_char_start(bytes, pos)
    } else {
        start
    };
    while pos < bytes.len() {
        let (class, len) = class_at(bytes, pos);
        if class != CharClass::Whitespace {
            break;
        }
        last = pos;
        pos += len;
    }
    // `\s+(?!\S)` leaves the final whitespace scalar for the next match
    // when a multi-character run is followed by non-whitespace.
    if pos < bytes.len() && last > start {
        last
    } else {
        pos
    }
}

#[inline(always)]
fn advance(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let first = unsafe { *bytes.get_unchecked(start) };

    if is_letter(first) {
        return scan_letters(bytes, start + 1);
    }
    if first == b' ' {
        if start + 1 == len {
            return len;
        }
        let (class, char_len) = class_at(bytes, start + 1);
        let next = start + 1 + char_len;
        return match class {
            CharClass::Letter => scan_letters(bytes, next),
            CharClass::Number => scan_numbers(bytes, next),
            CharClass::Whitespace => advance_whitespace(bytes, start, next),
            CharClass::Other => scan_other(bytes, next),
        };
    }
    if first >= 0x80 {
        let (class, char_len) = class_at(bytes, start);
        let next = start + char_len;
        return match class {
            CharClass::Letter => scan_letters(bytes, next),
            CharClass::Number => scan_numbers(bytes, next),
            CharClass::Whitespace => advance_whitespace(bytes, start, next),
            CharClass::Other => scan_other(bytes, next),
        };
    }
    if is_digit(first) {
        return scan_numbers(bytes, start + 1);
    }
    if first == b'\'' {
        match bytes.get(start + 1) {
            Some(b's' | b'd' | b'm' | b't') => return start + 2,
            Some(b'l') if bytes.get(start + 2) == Some(&b'l') => return start + 3,
            Some(b'v') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            Some(b'r') if bytes.get(start + 2) == Some(&b'e') => return start + 3,
            _ => return scan_other(bytes, start + 1),
        }
    }
    if is_ascii_ws(first) {
        return advance_whitespace(bytes, start, start + 1);
    }
    scan_other(bytes, start + 1)
}

// --------------------------------------------------------------------------
// SIMD boundary masks
// --------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline(always)]
fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
    use std::arch::aarch64::*;
    if scan + 70 > bytes.len() {
        return (0, u64::MAX);
    }
    unsafe {
        let ptr = bytes.as_ptr().add(scan);
        let zero = vdupq_n_u8(0);
        let mut letters = [zero; 4];
        let mut digits = [zero; 4];
        let mut spaces = [zero; 4];
        let mut whitespace = [zero; 4];
        let mut high = [zero; 4];
        let mut apostrophes = [zero; 4];
        for index in 0..4 {
            let value = vld1q_u8(ptr.add(16 * index));
            let lowered = vorrq_u8(value, vdupq_n_u8(0x20));
            letters[index] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            digits[index] = vcleq_u8(vsubq_u8(value, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            spaces[index] = vceqq_u8(value, vdupq_n_u8(b' '));
            whitespace[index] = vorrq_u8(
                spaces[index],
                vcleq_u8(vsubq_u8(value, vdupq_n_u8(9)), vdupq_n_u8(4)),
            );
            high[index] = vcltzq_s8(vreinterpretq_s8_u8(value));
            apostrophes[index] = vceqq_u8(value, vdupq_n_u8(b'\''));
        }

        let letter_mask = fast_mask::movemask64(letters[0], letters[1], letters[2], letters[3]);
        let digit_mask = fast_mask::movemask64(digits[0], digits[1], digits[2], digits[3]);
        let space_mask = fast_mask::movemask64(spaces[0], spaces[1], spaces[2], spaces[3]);
        let whitespace_mask =
            fast_mask::movemask64(whitespace[0], whitespace[1], whitespace[2], whitespace[3]);
        let apostrophe_any = vorrq_u8(
            vorrq_u8(apostrophes[0], apostrophes[1]),
            vorrq_u8(apostrophes[2], apostrophes[3]),
        );
        let apostrophe_mask = if vmaxvq_u8(apostrophe_any) == 0 {
            0
        } else {
            fast_mask::movemask64(
                apostrophes[0],
                apostrophes[1],
                apostrophes[2],
                apostrophes[3],
            )
        };
        let high_any = vorrq_u8(vorrq_u8(high[0], high[1]), vorrq_u8(high[2], high[3]));
        if vmaxvq_u8(high_any) != 0 {
            let high_mask = fast_mask::movemask64(high[0], high[1], high[2], high[3]);
            return extended_masks(
                bytes,
                scan,
                letter_mask,
                digit_mask,
                space_mask,
                whitespace_mask,
                high_mask,
                apostrophe_mask,
            );
        }
        ascii_masks(
            bytes,
            scan,
            letter_mask,
            digit_mask,
            space_mask,
            whitespace_mask,
            apostrophe_mask,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
    if scan + 70 > bytes.len() {
        return (0, u64::MAX);
    }
    let masks = if AVX512 {
        unsafe { fast_mask::ascii_masks_avx512(bytes, scan) }
    } else {
        unsafe { fast_mask::ascii_masks_avx2(bytes, scan) }
    };
    let whitespace = masks.s | masks.wt | masks.n;
    if masks.hi != 0 {
        // SAFETY: both runtime-selected x86 tiers include the bit features
        // declared by the shared Unicode repair path.
        return unsafe {
            extended_masks(
                bytes, scan, masks.l, masks.d, masks.s, whitespace, masks.hi, masks.ap,
            )
        };
    }
    ascii_masks(bytes, scan, masks.l, masks.d, masks.s, whitespace, masks.ap)
}

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
#[inline(always)]
fn ascii_masks(
    bytes: &[u8],
    scan: usize,
    letters: u64,
    digits: u64,
    spaces: u64,
    whitespace: u64,
    apostrophes: u64,
) -> (u64, u64) {
    let other = !(letters | digits | whitespace);
    let (prev_letter, prev_digit, prev_space, prev_ws, prev_other) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else {
        carries_at(bytes, scan)
    };

    let continuing = (letters & ((letters << 1) | prev_letter))
        | (digits & ((digits << 1) | prev_digit))
        | (other & ((other << 1) | prev_other));
    let after_space = (spaces << 1) | prev_space;
    let non_ws_boundary = !whitespace & !continuing & !after_space;

    let mut split = whitespace & (!whitespace >> 1);
    let lookahead = bytes[scan + 64];
    if lookahead < 0x80 {
        split |= (u64::from(!is_ascii_ws(lookahead)) << 63) & whitespace;
    } else if whitespace >> 63 != 0 && unsafe { fast_mask::nn_at_full(bytes, scan + 64) } {
        split |= 1 << 63;
    }
    let previous_ws = (whitespace << 1) | prev_ws;
    let ws_boundary = whitespace & (!previous_ws | split);
    let mut boundary = non_ws_boundary | ws_boundary;
    let mut bad = 0u64;
    apply_contractions(bytes, scan, apostrophes, &mut boundary, &mut bad);
    (boundary & !bad, bad)
}

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "bmi1,bmi2,lzcnt,popcnt")
)]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn extended_masks(
    bytes: &[u8],
    scan: usize,
    ascii_letters: u64,
    ascii_digits: u64,
    spaces: u64,
    ascii_whitespace: u64,
    high: u64,
    apostrophes: u64,
) -> (u64, u64) {
    let table = ClassTable::get();
    let classify = move |cp| {
        // SAFETY: scanner decodes a valid Unicode scalar value.
        unsafe { table.class_of(cp) }
    };

    let mut claimed = fast_mask::UniClasses::default();
    let (prev_letter, prev_digit, prev_space, prev_ws, prev_other) = if scan == 0 {
        (0, 0, 0, 0, 0)
    } else if bytes[scan - 1] < 0x80 {
        carries_at(bytes, scan)
    } else {
        // SAFETY: batch lookahead guarantees the walk-back decode is in bounds.
        let (class, _, end) = unsafe { fast_mask::char_through(bytes, scan, classify) };
        let char_mask = if end > scan {
            (1u64 << (end - scan)) - 1
        } else {
            0
        };
        claimed.cont = char_mask;
        match class {
            CharClass::Letter => {
                claimed.l = char_mask;
                (1, 0, 0, 0, 0)
            }
            CharClass::Number => {
                claimed.n = char_mask;
                (0, 1, 0, 0, 0)
            }
            CharClass::Whitespace => {
                claimed.ws = char_mask;
                claimed.resid = char_mask;
                (0, 0, u64::from(bytes[scan - 1] == b' '), 1, 0)
            }
            CharClass::Other => {
                claimed.o = char_mask;
                (0, 0, 0, 0, 1)
            }
        }
    };

    let unicode = unsafe {
        fast_mask::classify_uni_chars::<true, false>(bytes, scan, high & !claimed.cont, classify)
    };
    let letters = ascii_letters | claimed.l | unicode.l;
    let digits = ascii_digits | claimed.n | unicode.n;
    let whitespace = ascii_whitespace | claimed.ws | unicode.ws;
    let other = !(ascii_letters | ascii_digits | ascii_whitespace | high) | claimed.o | unicode.o;
    let continuation = claimed.cont | unicode.cont;
    let residual = claimed.resid | unicode.resid;

    let continuing = (letters & ((letters << 1) | prev_letter))
        | (digits & ((digits << 1) | prev_digit))
        | (other & ((other << 1) | prev_other));
    let after_space = (spaces << 1) | prev_space;
    let non_ws_boundary = !whitespace & !continuing & !after_space & !continuation;

    let non_ws = !whitespace;
    let mut split = (ascii_whitespace & (non_ws >> 1))
        | (unicode.w2 & (non_ws >> 2))
        | (unicode.w3 & (non_ws >> 3));
    let whitespace_leads = ascii_whitespace | unicode.w2 | unicode.w3;
    let edge_multibyte = (unicode.w2 & (1 << 62)) | (unicode.w3 & (1 << 61));
    let lookahead = bytes[scan + 64];
    if lookahead < 0x80 && edge_multibyte == 0 {
        split =
            (split & !(1 << 63)) | ((u64::from(!is_ascii_ws(lookahead)) << 63) & ascii_whitespace);
    } else {
        let edge = edge_multibyte | ((1 << 63) & ascii_whitespace);
        if edge != 0 {
            if unsafe { fast_mask::nn_at_full(bytes, scan + 64) } {
                split |= edge;
            } else {
                split &= !edge;
            }
        }
    }
    let previous_ws = (whitespace << 1) | prev_ws;
    let ws_boundary = whitespace_leads & (!previous_ws | split);
    let mut boundary = non_ws_boundary | ws_boundary;
    let mut bad = residual | residual << 1 | residual >> 1;
    apply_contractions(bytes, scan, apostrophes & !bad, &mut boundary, &mut bad);
    (boundary & !bad, bad)
}

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
#[inline(always)]
fn apply_contractions(
    bytes: &[u8],
    scan: usize,
    apostrophes: u64,
    boundary: &mut u64,
    bad: &mut u64,
) {
    let mut candidates = apostrophes & *boundary & !*bad;
    while candidates != 0 {
        let index = candidates.trailing_zeros() as usize;
        candidates &= candidates - 1;
        if index >= 61 {
            *bad |= u64::MAX << index;
            break;
        }
        let length = match bytes[scan + index + 1] {
            b's' | b'd' | b'm' | b't' => 2,
            b'l' if bytes[scan + index + 2] == b'l' => 3,
            b'v' if bytes[scan + index + 2] == b'e' => 3,
            b'r' if bytes[scan + index + 2] == b'e' => 3,
            _ => 0,
        };
        if length != 0 {
            *boundary &= !(1u64 << (index + 1));
            *boundary |= 1u64 << (index + length);
        }
    }
}

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
#[inline(always)]
fn carries_at(bytes: &[u8], scan: usize) -> (u64, u64, u64, u64, u64) {
    let byte = bytes[scan - 1];
    if byte < 0x80 {
        let letter = is_letter(byte);
        let digit = is_digit(byte);
        let whitespace = is_ascii_ws(byte);
        let bit = |value: bool| u64::from(value);
        return (
            bit(letter),
            bit(digit),
            bit(byte == b' '),
            bit(whitespace),
            bit(!letter && !digit && !whitespace),
        );
    }
    match unsafe {
        fast_mask::char_through(bytes, scan, |cp| {
            // SAFETY: the scanner decoded a valid Unicode scalar value.
            fast_unicode::class_of(cp)
        })
    }
    .0
    {
        CharClass::Letter => (1, 0, 0, 0, 0),
        CharClass::Number => (0, 1, 0, 0, 0),
        CharClass::Whitespace => (0, 0, 0, 1, 0),
        CharClass::Other => (0, 0, 0, 0, 1),
    }
}

pub(crate) struct R50kScheme;

impl MaskScheme for R50kScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance(bytes, pos)
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        batch_masks(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        unsafe { batch_masks_x86::<AVX512>(bytes, scan) }
    }
}

/// Whole-buffer iterator. Streaming uses the same [`R50kScheme`] through the
/// shared bounded-memory composer in `stream_pretok`.
pub struct FastR50kPretokenizer<'a> {
    input: &'a str,
    state: MaskState,
}

impl<'a> FastR50kPretokenizer<'a> {
    #[inline]
    pub fn new(input: &'a str) -> Self {
        warm_unicode_classes();
        Self {
            input,
            state: MaskState::new(input.as_bytes(), 0),
        }
    }

    #[cfg(test)]
    fn new_scalar(input: &'a str) -> Self {
        warm_unicode_classes();
        Self {
            input,
            state: MaskState::new_scalar(0),
        }
    }
}

impl<'a> Iterator for FastR50kPretokenizer<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (start, end) = self.state.next_span::<R50kScheme>(self.input.as_bytes())?;
        Some(&self.input[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::FastR50kPretokenizer;
    use crate::pretok::R50K_SPLIT_PATTERN;
    use fancy_regex::Regex;

    fn reference(input: &str) -> Vec<&str> {
        Regex::new(R50K_SPLIT_PATTERN)
            .unwrap()
            .find_iter(input)
            .map(|matched| matched.unwrap().as_str())
            .filter(|piece| !piece.is_empty())
            .collect()
    }

    #[test]
    fn matches_reference_and_scalar_across_edges() {
        let atoms = [
            "a", "B", "9", " ", "  ", "\n", "\t", "'", "'s", "'ll", "!", "é", "日", "🎉", "\u{a0}",
            "word", "12", "’", "€",
        ];
        let cases = [
            "",
            "hello world",
            "  double  spaces  ",
            "don't can't we'll they've you're I'm",
            "a\n\nb\t c\r\nend",
            "café résumé 日本語 🎉 \u{a0} end",
        ];
        for input in cases {
            let expected = reference(input);
            assert_eq!(
                FastR50kPretokenizer::new(input).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                FastR50kPretokenizer::new_scalar(input).collect::<Vec<_>>(),
                expected
            );
        }

        let mut random = 0x9e37_79b9_7f4a_7c15u64;
        for round in 0..1_000 {
            let mut input = String::new();
            while input.len() < 64 + round % 160 {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                input.push_str(atoms[random as usize % atoms.len()]);
            }
            let expected = reference(&input);
            assert_eq!(
                FastR50kPretokenizer::new(&input).collect::<Vec<_>>(),
                expected,
                "SIMD round {round}: {input:?}"
            );
            assert_eq!(
                FastR50kPretokenizer::new_scalar(&input).collect::<Vec<_>>(),
                expected,
                "scalar round {round}: {input:?}"
            );
        }
    }
}
