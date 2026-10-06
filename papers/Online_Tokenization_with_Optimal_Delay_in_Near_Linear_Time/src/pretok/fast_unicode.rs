//! Packed Unicode character classes shared by the hardware pretokenizers.
//!
//! The table layout follows Gigatoken's MIT-licensed fast pretokenizers:
//! <https://github.com/marcelroed/gigatoken/blob/34a1599f0c0ae7d7cd0d1c530e6522320158b360/src/pretokenize/unicode.rs>.
//! We build it from the same `regex` Unicode tables used by Hiriluk's
//! reference patterns, avoiding an additional Unicode-data dependency.

use once_cell::sync::Lazy;
use regex::RegexSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CharClass {
    Letter = 0,
    Number = 1,
    Whitespace = 2,
    Other = 3,
}

/// Four code points share one byte (272 KiB for all Unicode scalar values).
static CLASSES: Lazy<Box<[u8]>> = Lazy::new(|| {
    let regex = RegexSet::new([r"^\p{L}$", r"^\p{N}$", r"^\s$"])
        .expect("static Unicode class expressions must compile");
    let mut packed = vec![0xff; 0x110000 / 4];
    let mut utf8 = [0; 4];
    for cp in 0..=char::MAX as u32 {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let matches = regex.matches(ch.encode_utf8(&mut utf8));
        let class = if matches.matched(0) {
            CharClass::Letter
        } else if matches.matched(1) {
            CharClass::Number
        } else if matches.matched(2) {
            CharClass::Whitespace
        } else {
            CharClass::Other
        };
        let shift = (cp & 3) << 1;
        let slot = &mut packed[(cp >> 2) as usize];
        *slot = (*slot & !(3 << shift)) | ((class as u8) << shift);
    }
    packed.into_boxed_slice()
});

#[derive(Clone, Copy)]
pub(crate) struct ClassTable(&'static [u8]);

impl ClassTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&CLASSES)
    }

    /// Classify one Unicode code point without a bounds check.
    ///
    /// # Safety
    ///
    /// `cp` must be at most `char::MAX as u32`.
    #[inline(always)]
    pub(crate) unsafe fn class_of(self, cp: u32) -> CharClass {
        let byte = unsafe { *self.0.get_unchecked((cp >> 2) as usize) };
        match (byte >> ((cp & 3) << 1)) & 3 {
            0 => CharClass::Letter,
            1 => CharClass::Number,
            2 => CharClass::Whitespace,
            _ => CharClass::Other,
        }
    }
}

/// Classify one Unicode code point without a bounds check.
///
/// # Safety
///
/// `cp` must be at most `char::MAX as u32`.
#[inline(always)]
pub(crate) unsafe fn class_of(cp: u32) -> CharClass {
    unsafe { ClassTable::get().class_of(cp) }
}

#[inline]
pub(crate) fn warm() {
    Lazy::force(&CLASSES);
}

#[inline(always)]
pub(crate) fn is_letter(byte: u8) -> bool {
    (byte | 0x20).wrapping_sub(b'a') < 26
}

#[inline(always)]
pub(crate) fn is_digit(byte: u8) -> bool {
    byte.wrapping_sub(b'0') < 10
}

#[inline(always)]
pub(crate) fn is_ascii_ws(byte: u8) -> bool {
    byte == b' ' || byte.wrapping_sub(9) < 5
}

/// Decode one scalar from valid UTF-8. Pretokenizer inputs are `str`, so the
/// caller only needs to ensure `pos` points to a non-ASCII scalar lead.
#[inline(always)]
pub(crate) unsafe fn decode_cp(bytes: &[u8], pos: usize) -> (u32, usize) {
    unsafe {
        let b0 = *bytes.get_unchecked(pos) as u32;
        let b1 = (*bytes.get_unchecked(pos + 1) & 0x3f) as u32;
        if b0 < 0xe0 {
            return (((b0 & 0x1f) << 6) | b1, 2);
        }
        let b2 = (*bytes.get_unchecked(pos + 2) & 0x3f) as u32;
        if b0 < 0xf0 {
            return (((b0 & 0x0f) << 12) | (b1 << 6) | b2, 3);
        }
        let b3 = (*bytes.get_unchecked(pos + 3) & 0x3f) as u32;
        (((b0 & 7) << 18) | (b1 << 12) | (b2 << 6) | b3, 4)
    }
}

// ---------------------------------------------------------------------------
// o200k-family character classes
// ---------------------------------------------------------------------------

/// Unicode classes needed by o200k's case-structured letter runs.
///
/// `Upper` is `Lu | Lt`, `Lower` is `Ll`, and `Caseless` is `Lm | Lo`.
/// Marks join letter runs but also continue punctuation runs, so they remain
/// distinct rather than being folded into `Caseless`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum O200kCharClass {
    Upper = 0,
    Lower = 1,
    Caseless = 2,
    Mark = 3,
    Number = 4,
    Whitespace = 5,
    Other = 6,
}

/// Two code points share one byte (544 KiB for all Unicode scalar values).
///
/// This is separate from `CLASSES`: cl100k never pays to initialize or retain
/// the larger case-aware table.
static O200K_CLASSES: Lazy<Box<[u8]>> = Lazy::new(|| {
    let regex = RegexSet::new([
        r"^[\p{Lu}\p{Lt}]$",
        r"^\p{Ll}$",
        r"^[\p{Lm}\p{Lo}]$",
        r"^\p{M}$",
        r"^\p{N}$",
        r"^\s$",
    ])
    .expect("static o200k Unicode class expressions must compile");
    let mut packed = vec![0x66; 0x110000 / 2];
    let mut utf8 = [0; 4];
    for cp in 0..=char::MAX as u32 {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let matches = regex.matches(ch.encode_utf8(&mut utf8));
        let class = if matches.matched(0) {
            O200kCharClass::Upper
        } else if matches.matched(1) {
            O200kCharClass::Lower
        } else if matches.matched(2) {
            O200kCharClass::Caseless
        } else if matches.matched(3) {
            O200kCharClass::Mark
        } else if matches.matched(4) {
            O200kCharClass::Number
        } else if matches.matched(5) {
            O200kCharClass::Whitespace
        } else {
            O200kCharClass::Other
        };
        let shift = (cp & 1) << 2;
        let slot = &mut packed[(cp >> 1) as usize];
        *slot = (*slot & !(0xf << shift)) | ((class as u8) << shift);
    }
    packed.into_boxed_slice()
});

#[derive(Clone, Copy)]
pub(crate) struct O200kClassTable(&'static [u8]);

impl O200kClassTable {
    #[inline]
    pub(crate) fn get() -> Self {
        Self(&O200K_CLASSES)
    }

    /// Classify one Unicode code point without a bounds check.
    ///
    /// # Safety
    ///
    /// `cp` must be at most `char::MAX as u32`.
    #[inline(always)]
    pub(crate) unsafe fn class_of(self, cp: u32) -> O200kCharClass {
        let byte = unsafe { *self.0.get_unchecked((cp >> 1) as usize) };
        match (byte >> ((cp & 1) << 2)) & 0xf {
            0 => O200kCharClass::Upper,
            1 => O200kCharClass::Lower,
            2 => O200kCharClass::Caseless,
            3 => O200kCharClass::Mark,
            4 => O200kCharClass::Number,
            5 => O200kCharClass::Whitespace,
            _ => O200kCharClass::Other,
        }
    }
}

/// Classify one Unicode code point without a bounds check.
///
/// # Safety
///
/// `cp` must be at most `char::MAX as u32`.
#[inline(always)]
pub(crate) unsafe fn o200k_class_of(cp: u32) -> O200kCharClass {
    unsafe { O200kClassTable::get().class_of(cp) }
}

#[inline]
pub(crate) fn warm_o200k() {
    Lazy::force(&O200K_CLASSES);
}

#[cfg(test)]
mod o200k_tests {
    use super::{O200kCharClass as C, o200k_class_of};

    #[test]
    fn case_aware_classes_cover_representative_codepoints() {
        for (ch, expected) in [
            ('A', C::Upper),
            ('ǅ', C::Upper),
            ('a', C::Lower),
            ('日', C::Caseless),
            ('ʰ', C::Caseless),
            ('\u{301}', C::Mark),
            ('٣', C::Number),
            ('\u{85}', C::Whitespace),
            ('!', C::Other),
        ] {
            // SAFETY: `ch` is a valid Unicode scalar value.
            assert_eq!(unsafe { o200k_class_of(ch as u32) }, expected, "{ch:?}");
        }
    }
}
