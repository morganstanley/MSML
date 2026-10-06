//! Shared infrastructure for mask-scanner pretokenizers.
//!
//! A mask-scanner pretokenizer processes 64-byte batches: SIMD classifies
//! every byte, bitmask algebra derives "a token starts here" bits, and
//! `next()` pops one bit per token — no per-token dispatch branches, which
//! is what makes it ~2x the serial scalar scanners (see Gigatoken's
//! [pinned optimization log](https://github.com/marcelroed/gigatoken/blob/34a1599f/pretokenizer_optimization_log.md),
//! step 15).
//!
//! A scheme plugs in two functions ([`MaskScheme`]):
//! - `advance`: the scalar ground truth (also the no-SIMD iterator),
//! - `batch_masks`: `(usable, bad)` bitmasks for a 64-byte batch. `usable`
//!   bits are trustworthy token starts; `bad` marks zones (non-ASCII the
//!   scheme doesn't classify in-mask, batch-edge ambiguities) that
//!   [`MaskState`] re-derives through `advance`, never emitting a token
//!   across an unresolved zone.
//!
//! Layering, bottom to top:
//! 1. Platform SIMD primitives (`movemask64`, `ascii_masks` on NEON;
//!    `ascii_masks_avx512` / `ascii_masks_avx2` on x86-64) — the only
//!    per-platform code.
//! 2. Bit-domain helpers shared across schemes — platform-independent
//!    u64 algebra and per-char table classification
//!    (`classify_uni_chars`, `char_through`, `nn_at_full`,
//!    `digit_run_splits3`), parameterized by each scheme's codepoint
//!    classifier.
//! 3. Per-scheme `batch_masks` boundary algebra (in the scheme's module).
//! 4. [`MaskState`] — the scheme-agnostic batch walker: segments, bad-zone
//!    gaps, scalar tail, one-batch-ahead precompute; scalar overruns stay
//!    on the 64-byte grid so the precompute survives them.
//!
//! Adapted from Gigatoken's MIT-licensed mask scanner at commit `34a1599f`:
//! <https://github.com/marcelroed/gigatoken/blob/34a1599f0c0ae7d7cd0d1c530e6522320158b360/src/pretokenize/fast/mask.rs>.
//! Hiriluk adapts its tier-specialized two-phase fill to bounded streaming
//! batches; cache ownership and output buffering remain in the streaming
//! tokenizer.

use super::{
    fast_unicode::{self as unicode, CharClass},
    stream_pretok::{KEYED_BATCH, KeyedBatchConsumer, KeyedSpan},
};
use crate::piece_cache::PieceKeyPacker;

// -----------------------------------------------------------------------
// Platform SIMD primitives: aarch64 NEON (compile-time, always present)
// and x86_64 AVX-512 or AVX2 (runtime-detected; scalar fallback
// otherwise).
// -----------------------------------------------------------------------

/// Does this x86_64 CPU have the full AVX-512 tier (Zen 4/5, Ice
/// Lake+)? Schemes dispatch their batch classifier on this: the AVX-512
/// front-end when true, the AVX2 one otherwise.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx512_scanner_available() -> bool {
    // std's feature cache makes this an atomic load + bit test after the
    // first call.
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("popcnt")
}

/// Whether this x86 CPU can use the AVX-512/VBMI2 fill tier. VBMI2 supplies
/// `vpcompressb`, which flattens one 64-bit boundary mask without the scalar
/// eight-octet table walk. Skylake-X lacks VBMI2 and stays on plain AVX-512.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx512_fill_available() -> bool {
    avx512_scanner_available() && std::arch::is_x86_feature_detected!("avx512vbmi2")
}

/// Does this x86_64 CPU have the AVX2 tier (Haswell+, all Zen)? The bit
/// features (BMI1/2, LZCNT, POPCNT) arrived with or before AVX2 on every
/// AVX2 CPU, but are detected explicitly since the boundary algebra's
/// codegen relies on them.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx2_scanner_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("popcnt")
}

/// Is the SIMD mask scanner usable on this machine? Little-endian aarch64
/// always has NEON; x86_64 requires AVX-512 (Zen 4/5, Ice Lake+) or AVX2
/// (Haswell+, Zen 1-3), detected at runtime. The mask lane-to-bit mapping is
/// little-endian, so big-endian aarch64 deliberately uses the scalar path.
/// When this returns false, [`MaskState`] runs every token through the
/// scheme's scalar `advance`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
    avx512_scanner_available() || avx2_scanner_available()
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn simd_scanner_available() -> bool {
    cfg!(all(target_arch = "aarch64", target_endian = "little"))
}

// The x86-64 batch classifiers are annotated
// `#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]`
// (AVX-512 tier) or `#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]`
// (AVX2 tier). Besides the wide byte ops, the scalar-visible bit features
// (BMI1/2, LZCNT, POPCNT) are enabled so the boundary algebra inlined
// into those functions compiles to tzcnt/lzcnt/blsr instead of
// baseline-x86 bsf sequences. The sets must stay in sync with
// [`avx512_scanner_available`] / [`avx2_scanner_available`].

/// simdjson-style movemask: 4 mask vectors (64 lanes of 0x00/0xFF) -> u64,
/// bit i = lane i.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline(always)]
pub(crate) unsafe fn movemask64(
    v0: std::arch::aarch64::uint8x16_t,
    v1: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
    v3: std::arch::aarch64::uint8x16_t,
) -> u64 {
    use std::arch::aarch64::*;
    unsafe {
        const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(W.as_ptr());
        let mut a0 = vandq_u8(v0, w);
        let a1 = vandq_u8(v1, w);
        let a2 = vandq_u8(v2, w);
        let a3 = vandq_u8(v3, w);
        // The 4-`addp` reduction tree (simdjson's arm64 movemask), pinned
        // as asm. Written with `vpaddq_u8`, LLVM rewrites every pairwise
        // add into a uzp1/uzp2/orr triple — adjacent weighted lanes have
        // disjoint bits, so add == or, and the canonical or-form never
        // re-forms addp — inflating each call from 9 to 17 vector ops
        // (4-7 calls per 64-byte batch across the schemes). The weighted
        // `and`s stay outside so the scheduler still interleaves
        // neighboring calls. `addp(x, x)` lane 0..7 equals the old
        // `addp(x, zero)` lanes 0..7; only lane u64 0 is read.
        core::arch::asm!(
            "addp {a0:v}.16b, {a0:v}.16b, {a1:v}.16b",
            "addp {a2:v}.16b, {a2:v}.16b, {a3:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a2:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a0:v}.16b",
            a0 = inout(vreg) a0,
            a1 = in(vreg) a1,
            a2 = inout(vreg) a2 => _,
            a3 = in(vreg) a3,
            options(pure, nomem, nostack, preserves_flags),
        );
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a0))
    }
}

/// One u64 mask (bit i = byte scan+i) per byte predicate, for 64 bytes.
/// The working currency of scheme boundary algebra: everything after this
/// is platform-independent u64 bit math.
#[derive(Clone, Copy, Default)]
pub(crate) struct AsciiMasks {
    /// ASCII letters.
    pub l: u64,
    /// ASCII digits.
    pub d: u64,
    /// Space (0x20) only.
    pub s: u64,
    /// Non-newline ASCII whitespace: \t, \x0b, \x0c.
    pub wt: u64,
    /// Newlines: \r, \n.
    pub n: u64,
    /// Non-ASCII bytes (>= 0x80).
    pub hi: u64,
    /// ASCII apostrophes.
    pub ap: u64,
}

/// Classify `bytes[scan..scan+64]` with AVX-512 (requires
/// `scan + 64 <= bytes.len()`). One 64-byte load and one k-register
/// compare per predicate: a `__mmask64` IS the u64 the bit algebra wants,
/// so there is no movemask ladder and no lazy any-tests — every field
/// (including `hi` and `ap`) is computed unconditionally.
///
/// Runtime-gated: callers reach this only after
/// [`simd_scanner_available`] reported AVX-512 support (enforced by
/// [`MaskState`], which otherwise never leaves the scalar path).
///
/// # Safety
///
/// The CPU must support the AVX-512 scanner tier and
/// `scan + 64 <= bytes.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
pub(crate) unsafe fn ascii_masks_avx512(bytes: &[u8], scan: usize) -> AsciiMasks {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(bytes.as_ptr().add(scan) as *const _);
        let lowered = _mm512_or_si512(v, _mm512_set1_epi8(0x20));
        let l = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(lowered, _mm512_set1_epi8(b'a' as i8)),
            _mm512_set1_epi8(25),
        );
        let d = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(b'0' as i8)),
            _mm512_set1_epi8(9),
        );
        let s = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b' ' as i8));
        let n = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\r' as i8))
            | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8));
        // \t (9), \x0b (11), \x0c (12): ascii ws minus \r\n and space.
        let wt =
            _mm512_cmple_epu8_mask(_mm512_sub_epi8(v, _mm512_set1_epi8(9)), _mm512_set1_epi8(4))
                & !n;
        let hi = _mm512_movepi8_mask(v) as u64;
        let ap = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\'' as i8));
        AsciiMasks {
            l,
            d,
            s,
            wt,
            n,
            hi,
            ap,
        }
    }
}

/// Classify `bytes[scan..scan+64]` with AVX2 (requires
/// `scan + 64 <= bytes.len()`). Two 32-byte loads; each predicate is one
/// vector compare per half plus a `vpmovmskb` ladder into the u64 the bit
/// algebra wants — more mask-extraction traffic than the AVX-512 version
/// (whose k-register compares ARE the u64s), but the output currency is
/// identical, so everything downstream is shared. AVX2 has no unsigned
/// byte compare; `x <= lim` is `min_epu8(x, lim) == x`.
///
/// Runtime-gated: callers reach this only after
/// [`avx2_scanner_available`] reported AVX2 support (enforced by the
/// schemes' dispatch, behind [`MaskState`]'s `simd_scanner_available`
/// gate).
///
/// `#[inline(never)]` is load-bearing: inlined, LLVM's vector combiner
/// sees the compare vectors behind the returned u64s and pulls the
/// caller's scalar boundary algebra back into the byte-vector domain,
/// expanding every mask<->vector crossing into vpinsrb/vpextrb ladders
/// (~240 byte ops per batch, measured 3.5x slower end to end on Zen 2).
/// The AVX-512 tier has no such domain to return to (k-register compares
/// ARE the u64s), so it stays inline.
///
/// # Safety
///
/// The CPU must support the AVX2 scanner tier and
/// `scan + 64 <= bytes.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline(never)]
pub(crate) unsafe fn ascii_masks_avx2(bytes: &[u8], scan: usize) -> AsciiMasks {
    use std::arch::x86_64::*;
    unsafe {
        // Closures inherit the enclosing fn's target features.
        let le =
            |v: __m256i, lim: __m256i| -> __m256i { _mm256_cmpeq_epi8(_mm256_min_epu8(v, lim), v) };
        let mm = |m0: __m256i, m1: __m256i| -> u64 {
            (_mm256_movemask_epi8(m0) as u32 as u64)
                | ((_mm256_movemask_epi8(m1) as u32 as u64) << 32)
        };

        let p = bytes.as_ptr().add(scan);
        let v0 = _mm256_loadu_si256(p as *const _);
        let v1 = _mm256_loadu_si256(p.add(32) as *const _);

        let x20 = _mm256_set1_epi8(0x20);
        let ca = _mm256_set1_epi8(b'a' as i8);
        let c25 = _mm256_set1_epi8(25);
        let l = mm(
            le(_mm256_sub_epi8(_mm256_or_si256(v0, x20), ca), c25),
            le(_mm256_sub_epi8(_mm256_or_si256(v1, x20), ca), c25),
        );
        let c0 = _mm256_set1_epi8(b'0' as i8);
        let c9 = _mm256_set1_epi8(9);
        let d = mm(
            le(_mm256_sub_epi8(v0, c0), c9),
            le(_mm256_sub_epi8(v1, c0), c9),
        );
        let sp = _mm256_set1_epi8(b' ' as i8);
        let s = mm(_mm256_cmpeq_epi8(v0, sp), _mm256_cmpeq_epi8(v1, sp));
        let cr = _mm256_set1_epi8(b'\r' as i8);
        let lf = _mm256_set1_epi8(b'\n' as i8);
        let n = mm(
            _mm256_or_si256(_mm256_cmpeq_epi8(v0, cr), _mm256_cmpeq_epi8(v0, lf)),
            _mm256_or_si256(_mm256_cmpeq_epi8(v1, cr), _mm256_cmpeq_epi8(v1, lf)),
        );
        // \t (9), \x0b (11), \x0c (12): ascii ws minus \r\n and space.
        let c4 = _mm256_set1_epi8(4);
        let wt = mm(
            le(_mm256_sub_epi8(v0, c9), c4),
            le(_mm256_sub_epi8(v1, c9), c4),
        ) & !n;
        let hi = mm(v0, v1); // vpmovmskb takes the sign bit directly
        let apc = _mm256_set1_epi8(b'\'' as i8);
        let ap = mm(_mm256_cmpeq_epi8(v0, apc), _mm256_cmpeq_epi8(v1, apc));
        AsciiMasks {
            l,
            d,
            s,
            wt,
            n,
            hi,
            ap,
        }
    }
}

// -----------------------------------------------------------------------
// Bit-domain helpers (platform-independent)
// -----------------------------------------------------------------------

/// Is the char starting at `idx` NOT whitespace (`\S` for a `(?!\S)`
/// lookahead)? Full answer via the packed table.
///
/// # Safety
///
/// `idx < bytes.len()`, and when `bytes[idx]` is non-ASCII,
/// `idx + 4 <= bytes.len()` (the guardless [`decode_cp_inbounds`] read).
/// The batch classifiers' `scan + 70 <= len` guard covers every call
/// site's worst case (`idx = scan + 64`).
#[inline(always)]
pub(crate) unsafe fn nn_at_full(bytes: &[u8], idx: usize) -> bool {
    use super::fast_unicode::{decode_cp as decode_cp_inbounds, is_ascii_ws};
    let b = bytes[idx];
    if b < 0x80 {
        return !is_ascii_ws(b);
    }
    // SAFETY: caller guarantees idx + 4 <= len for a non-ASCII byte here
    // (this fn's contract).
    let (cp, _) = unsafe { decode_cp_inbounds(bytes, idx) };
    // SAFETY: valid UTF-8 decoding produces a Unicode scalar value.
    (unsafe { unicode::class_of(cp) }) != CharClass::Whitespace
}

/// The char containing byte `pos - 1` (`pos > 0`, valid UTF-8): its
/// class, lead index, and end (exclusive). `end > pos` iff the char
/// straddles across `pos`. ASCII classifies with the byte predicates;
/// multi-byte chars walk back to their lead (at most 3 bytes) and use
/// the packed table — this is what lets a batch after a unicode char
/// compute true boundary carries instead of deferring to a bad zone.
/// `class`: the scheme's codepoint classifier (`unicode::class_of`, or a
/// mark-folding view like `unicode::class_of_marks_join`).
///
/// # Safety
///
/// `pos > 0`, and when `bytes[pos - 1]` is non-ASCII,
/// `pos + 3 <= bytes.len()`: the walk-back lead `j` satisfies
/// `j <= pos - 1`, so the guardless [`decode_cp_inbounds`] read needs
/// `j + 4 <= pos + 3` in-bounds bytes. The batch classifiers' `scan + 70
/// <= len` guard covers every call site (`pos <= scan + 64`).
#[inline(always)]
pub(crate) unsafe fn char_through(
    bytes: &[u8],
    pos: usize,
    class: impl Fn(u32) -> CharClass,
) -> (CharClass, usize, usize) {
    use super::fast_unicode::{decode_cp as decode_cp_inbounds, is_ascii_ws, is_digit, is_letter};
    let b = bytes[pos - 1];
    if b < 0x80 {
        let cls = if is_letter(b) {
            CharClass::Letter
        } else if is_digit(b) {
            CharClass::Number
        } else if is_ascii_ws(b) {
            CharClass::Whitespace
        } else {
            CharClass::Other
        };
        return (cls, pos - 1, pos);
    }
    let mut j = pos - 1;
    while j > 0 && bytes[j] & 0xC0 == 0x80 {
        j -= 1;
    }
    // SAFETY: j < pos and pos + 3 <= len (this fn's contract), so
    // j + 4 <= len.
    let (cp, l) = unsafe { decode_cp_inbounds(bytes, j) };
    (class(cp), j, j + l)
}

/// Per-byte class masks for a batch's unicode chars, classified with the
/// packed table (`unicode::class_of`) — the same lookup the scalar paths
/// do. Every byte of a classified char carries the char's class, so
/// byte-adjacency == char-adjacency and the schemes' u64 boundary
/// algebra applies unchanged.
#[derive(Clone, Copy, Default)]
pub(crate) struct UniClasses {
    /// Letter / number / other / whitespace bytes.
    pub l: u64,
    pub n: u64,
    pub o: u64,
    pub ws: u64,
    /// Whitespace lead bits by char length, for the char-length-aware
    /// `(?!\S)` shift tests. Deferred ws chars (see `resid`) are not
    /// included.
    pub w2: u64,
    pub w3: u64,
    /// Lead bits of all classified chars by length, for schemes that
    /// shift a test by the previous char's length (the cl100k family's
    /// two-chars-back rule).
    pub lead2: u64,
    pub lead3: u64,
    pub lead4: u64,
    /// Continuation bytes of classified chars.
    pub cont: u64,
    /// Bytes only the scalar path can decide: whitespace chars straddling
    /// the batch end (their run-split bookkeeping crosses the boundary),
    /// number chars when `NUMBERS` is false, and stray continuation
    /// bytes. Class masks stay truthful for these bytes so neighbors'
    /// algebra is exact; callers turn `resid` into bad zones (±1 smear).
    pub resid: u64,
}

/// Classify every unicode char whose lead bit is in `m` (typically
/// `hi & !claimed-straddle-in-bytes`) for `bytes[scan..scan+64]`.
/// A char spilling off the batch end is classified via the lookahead
/// bytes; only its in-batch bytes get class bits, and the next
/// batch's `char_through` walk-back covers the remainder. `NUMBERS`:
/// false for schemes whose digit grouping is char-counted (`\p{N}{1,3}`
/// byte masks can't express multi-byte chars), true otherwise.
/// `LEADS`: whether to fill the per-length lead masks (only schemes with
/// a shift-by-prev-char-length rule need them).
///
/// The loop stays branchy on purpose: a branchless csel-selected
/// decode/classify body measured 0.986x (predicted branches beat data
/// chains, log step 13/17). 2-byte chars (nearly all non-ASCII in western
/// corpora) take a dedicated lane with an inline decode; 3/4-byte chars
/// pay the general ladder.
///
/// # Safety
///
/// `scan + 70 <= bytes.len()` (the batch classifiers' lookahead guard):
/// a lead bit at position 63 puts the guardless [`decode_cp_inbounds`]
/// read at `scan + 63`, which may touch through `scan + 67`.
#[inline(always)]
pub(crate) unsafe fn classify_uni_chars<const NUMBERS: bool, const LEADS: bool>(
    bytes: &[u8],
    scan: usize,
    mut m: u64,
    class: impl Fn(u32) -> CharClass,
) -> UniClasses {
    use super::fast_unicode::decode_cp as decode_cp_inbounds;
    let mut u = UniClasses::default();
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        let b = bytes[scan + i];
        if b < 0xE0 {
            // 2-byte lane (leads 0xC2..0xDF, cp < 0x800): nearly every
            // non-ASCII char in western corpora, so this branch predicts
            // taken and skips the length ladder + general decode.
            if b < 0xC2 {
                u.resid |= 1 << i; // stray continuation byte (invalid UTF-8)
                continue;
            }
            let lead = 1u64 << i;
            let chm = 3u64 << i; // in-batch bytes (excess drops at bit 63)
            // SAFETY: scan + 70 <= len (this fn's # Safety contract),
            // i <= 63, so scan + i + 1 <= scan + 64 < len.
            let b1 = unsafe { *bytes.get_unchecked(scan + i + 1) };
            let cp = ((b as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F);
            match class(cp) {
                CharClass::Letter => u.l |= chm,
                CharClass::Number => {
                    u.n |= chm;
                    if !NUMBERS {
                        u.resid |= chm;
                    }
                }
                CharClass::Other => u.o |= chm,
                CharClass::Whitespace => {
                    u.ws |= chm;
                    if i + 2 > 64 {
                        // Straddling-out ws stays a bad zone; its true
                        // class marks keep neighbors' `(?!\S)` tests
                        // exact.
                        u.resid |= chm;
                    } else {
                        u.w2 |= lead;
                    }
                }
            }
            if LEADS {
                u.lead2 |= lead;
            }
            u.cont |= chm & !lead;
            m &= !chm;
            continue;
        }
        let l = if b < 0xF0 { 3 } else { 4 };
        let chm = ((1u64 << l) - 1) << i; // in-batch bytes (excess drops)
        let lead = 1u64 << i;
        // SAFETY: scan + 70 <= len (this fn's # Safety contract), i <= 63,
        // so scan + i + 4 <= len even for a 4-byte lead at bit 63.
        let (cp, _) = unsafe { decode_cp_inbounds(bytes, scan + i) };
        match class(cp) {
            CharClass::Letter => u.l |= chm,
            CharClass::Number => {
                u.n |= chm;
                if !NUMBERS {
                    u.resid |= chm;
                }
            }
            CharClass::Other => u.o |= chm,
            CharClass::Whitespace => {
                u.ws |= chm;
                if i + l > 64 || l == 4 {
                    // Straddling-out ws (and defensively: no 4-byte cp
                    // is ws in Unicode) stays a bad zone; its true class
                    // marks keep neighbors' `(?!\S)` tests exact.
                    u.resid |= chm;
                } else {
                    u.w3 |= lead;
                }
            }
        }
        if LEADS {
            if l == 3 {
                u.lead3 |= lead;
            } else {
                u.lead4 |= lead;
            }
        }
        u.cont |= chm & !lead;
        m &= !chm;
    }
    u
}

/// Token-start bits inside ASCII digit runs for `\p{N}{1,3}`: each run
/// splits into 3-char tokens, so boundaries sit at run start + 3k. (For a
/// plain `\p{N}` scheme every digit is a start — no helper needed.)
#[inline(always)]
pub(crate) fn digit_run_splits3(d: u64) -> u64 {
    let mut b = d & !(d << 1); // run starts
    // A start at p re-arms at p+3 while the run continues: hop condition
    // c = "p..p+3 all digits". Log-doubling covers 64-bit runs in 5 steps.
    let mut c = d & (d >> 1) & (d >> 2) & (d >> 3);
    let mut sh = 3u32;
    while sh < 64 {
        b |= (b & c) << sh;
        c &= c >> sh;
        sh <<= 1;
    }
    b
}

// -----------------------------------------------------------------------
// The batch walker
// -----------------------------------------------------------------------

/// The two per-scheme hooks of a mask-scanner pretokenizer.
pub(crate) trait MaskScheme {
    /// Scalar ground truth: end of the token starting at `pos`
    /// (`pos < bytes.len()`, `pos` on a token boundary).
    fn advance(bytes: &[u8], pos: usize) -> usize;

    /// `(usable, bad)` for `bytes[scan..scan+64]` (`scan+64 <= len`):
    /// `usable` bit k = trustworthy token start at scan+k; `bad` bit k =
    /// byte scan+k needs the scalar path. `usable & bad` must be 0.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64);

    /// The x86_64 batch classifier, monomorphized on the SIMD tier
    /// (`AVX512` = true → the AVX-512 front-end, false → AVX2); same
    /// `(usable, bad)` contract as the aarch64 `batch_masks`. The fill
    /// wrappers instantiate this inside a matching `#[target_feature]`
    /// region, so the tier function inlines into the fill loop and no
    /// per-batch dispatch survives (the codegen a `-C target-cpu=native`
    /// build gets).
    ///
    /// # Safety
    ///
    /// The selected tier must have been runtime-detected:
    /// [`avx512_scanner_available`] for `AVX512` = true,
    /// [`avx2_scanner_available`] for `AVX512` = false.
    #[cfg(target_arch = "x86_64")]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64);

    /// Runtime-dispatched form of [`Self::batch_masks_x86`] for call
    /// sites outside a tier-monomorphized region (`next_span`): a cached
    /// tier check plus a non-inlined call per batch into a per-tier
    /// `#[target_feature]` wrapper ([`batch_masks_dyn_avx512`] /
    /// [`batch_masks_dyn_avx2`]), so the classifier body still compiles
    /// under the full tier feature set. Must only be called when
    /// [`simd_scanner_available`] is true — [`MaskState`] guarantees this
    /// by never leaving the scalar path otherwise.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64)
    where
        Self: Sized,
    {
        debug_assert!(simd_scanner_available());
        // The tier check is a cached atomic load + bit test and the
        // branch is perfectly predicted, so it is noise next to the
        // batch classification it selects.
        if avx512_scanner_available() {
            // SAFETY: runtime AVX-512 detection right above.
            unsafe { batch_masks_dyn_avx512::<Self>(bytes, scan) }
        } else {
            // SAFETY: MaskState enables the mask-scanner path only after
            // runtime detection (simd_scanner_available); without AVX-512
            // that detection was the AVX2 tier's.
            unsafe { batch_masks_dyn_avx2::<Self>(bytes, scan) }
        }
    }
}

/// AVX-512 feature region for the runtime-dispatched
/// `MaskScheme::batch_masks`: the scheme's `#[inline(always)]`
/// `batch_masks_x86` body fuses into this wrapper, so the per-batch call
/// `next_span` pays runs full-tier codegen (without this region the body
/// would inline into the plain-feature caller, where the inner
/// `#[target_feature]` mask classifiers can't inline and the boundary
/// algebra loses BMI/LZCNT codegen — measured ~25% slower).
///
/// # Safety
///
/// The CPU must support the AVX-512 scanner tier
/// ([`avx512_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn batch_masks_dyn_avx512<S: MaskScheme>(bytes: &[u8], scan: usize) -> (u64, u64) {
    // SAFETY: the caller detected the AVX-512 tier (fn contract).
    unsafe { S::batch_masks_x86::<true>(bytes, scan) }
}

/// AVX2 counterpart of [`batch_masks_dyn_avx512`].
///
/// # Safety
///
/// The CPU must support the AVX2 scanner tier
/// ([`avx2_scanner_available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
unsafe fn batch_masks_dyn_avx2<S: MaskScheme>(bytes: &[u8], scan: usize) -> (u64, u64) {
    // SAFETY: the caller detected the AVX2 tier (fn contract).
    unsafe { S::batch_masks_x86::<false>(bytes, scan) }
}

/// x86 fill tiers. The public fill chooses one tier once per keyed batch;
/// const propagation then removes classifier and boundary-flattener dispatch
/// from the 64-byte harvest loop.
const X86_TIER_DYN: u8 = 0;
#[cfg(target_arch = "x86_64")]
const X86_TIER_AVX2: u8 = 1;
#[cfg(target_arch = "x86_64")]
const X86_TIER_AVX512: u8 = 2;
const X86_TIER_AVX512_VBMI2: u8 = 3;

/// Scheme-agnostic mask-scanner state: pops trusted boundary bits, walks
/// bad zones through the scheme's scalar `advance`, runs the buffer tail
/// scalar, and precomputes one batch ahead so the SIMD chain retires under
/// the previous batch's pops. Without SIMD support (non-aarch64/x86_64
/// targets, or an x86_64 CPU without AVX-512 or AVX2) `scalar_until`
/// starts at `usize::MAX`, so every token takes the scalar path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaskState {
    /// Start of the pending (not yet emitted) token.
    pub pos: usize,
    /// Base of the next batch to scan.
    scan: usize,
    /// Base the `rem`/`batch_*` bits refer to.
    mask_base: usize,
    /// Boundary bits of the current segment (trusted, pop-ready).
    rem: u64,
    /// Full usable mask of the current batch (later segments).
    batch_usable: u64,
    /// Bad zones of the current batch not yet passed.
    batch_bad: u64,
    /// Emit tokens via the scalar advance while `pos < scalar_until`.
    scalar_until: usize,
    /// Eagerly computed masks for the batch at `pre_base` (usize::MAX =
    /// none).
    pre_base: usize,
    pre_usable: u64,
    pre_bad: u64,
}

impl MaskState {
    /// Build a scanner state for one contiguous buffer.
    ///
    /// The mask algebra is substantially faster on ASCII-heavy text, but its
    /// per-codepoint dirty-batch repair is more expensive than the scalar
    /// family scanner on dense UTF-8 (for example Chinese). Sample a few
    /// evenly spaced regions once and select the scalar ground truth for the whole buffer
    /// when at least one quarter of its bytes are non-ASCII. Streaming creates
    /// one state per input chunk, so mixed corpora adapt chunk by chunk without
    /// retaining any additional state or changing token boundaries.
    #[inline]
    pub(crate) fn new(bytes: &[u8], pos: usize) -> Self {
        let scalar_until = if simd_scanner_available() && !dense_non_ascii(bytes, pos) {
            pos
        } else {
            usize::MAX
        };
        Self {
            pos,
            scan: pos,
            mask_base: pos,
            rem: 0,
            batch_usable: 0,
            batch_bad: 0,
            scalar_until,
            pre_base: usize::MAX,
            pre_usable: 0,
            pre_bad: 0,
        }
    }

    /// Force the scalar ground-truth path on SIMD hosts. Differential tests
    /// use this to exercise the production scalar fallback without changing
    /// runtime dispatch or adding a release-build branch.
    #[cfg(test)]
    #[inline]
    pub(crate) fn new_scalar(pos: usize) -> Self {
        Self {
            pos,
            scan: pos,
            mask_base: pos,
            rem: 0,
            batch_usable: 0,
            batch_bad: 0,
            scalar_until: usize::MAX,
            pre_base: usize::MAX,
            pre_usable: 0,
            pre_bad: 0,
        }
    }

    /// Load the segment of `batch_usable` bits in [from_bit, next bad run)
    /// into `rem` and aim `scalar_until` past that bad run at the next
    /// trusted boundary (or the batch end).
    #[cfg(any(
        all(target_arch = "aarch64", target_endian = "little"),
        target_arch = "x86_64"
    ))]
    #[inline(always)]
    fn load_segment(&mut self, from_bit: u32) {
        let live = u64::MAX << from_bit;
        let seg_bad = self.batch_bad & live;
        if seg_bad == 0 {
            self.rem = self.batch_usable & live;
            self.batch_bad = 0;
        } else {
            let nb = seg_bad.trailing_zeros();
            self.rem = self.batch_usable & live & ((1u64 << nb) - 1);
            let rest = self.batch_usable & (u64::MAX << nb);
            self.scalar_until = if rest != 0 {
                self.mask_base + rest.trailing_zeros() as usize
            } else {
                self.mask_base + 64
            };
        }
        // A bit at the pending token's own start is not an end. Branchless:
        // whether the pending token starts exactly at this segment's first
        // bit is a ~20% coin flip on natural text.
        let at_start = self.pos == self.mask_base + from_bit as usize;
        self.rem &= !(u64::from(at_start) << from_bit);
    }

    /// The next token's byte range, or None at end of input.
    #[inline(always)]
    pub(crate) fn next_span<S: MaskScheme>(&mut self, bytes: &[u8]) -> Option<(usize, usize)> {
        let len = bytes.len();
        loop {
            if self.rem != 0 {
                let tz = self.rem.trailing_zeros() as usize;
                let end = self.mask_base + tz;
                self.rem &= self.rem - 1;
                let start = self.pos;
                self.pos = end;
                return Some((start, end));
            }
            if self.pos < self.scalar_until {
                if self.pos >= len {
                    return None;
                }
                let start = self.pos;
                let end = S::advance(bytes, start);
                self.pos = end;
                return Some((start, end));
            }
            #[cfg(any(
                all(target_arch = "aarch64", target_endian = "little"),
                target_arch = "x86_64"
            ))]
            {
                // Continue with the current batch's next trusted segment
                // after a scalar gap (each batch is computed exactly once).
                if self.batch_bad != 0 && self.pos < self.mask_base + 64 {
                    self.load_segment((self.pos - self.mask_base) as u32);
                    continue;
                }
                self.batch_bad = 0;
                // Resume after a scalar overrun WITHOUT leaving the
                // 64-byte grid: the precomputed next batch (and the
                // prefetch chain behind it) stays valid, where rebasing
                // to the token boundary invalidated it on every bad-zone
                // overrun — a large part of a deferral's ~800-cycle
                // cost. Grid bits below `pos` may be stale run-internal
                // bits (a ws or digit run the scalar walked through can
                // cross the grid base); they are masked by the
                // `from_bit` passed to load_segment below, and every
                // path that puts `pos` inside such a run goes through a
                // deferral first, so those bits are never trusted.
                while self.scan + 64 <= self.pos {
                    self.scan += 64;
                }
                if self.scan + 64 > len {
                    // Tail: scalar to the end of the buffer.
                    self.scalar_until = usize::MAX;
                    continue;
                }
                let (usable, bad) = if self.pre_base == self.scan {
                    (self.pre_usable, self.pre_bad)
                } else {
                    S::batch_masks(bytes, self.scan)
                };
                self.mask_base = self.scan;
                self.scan += 64;
                self.batch_usable = usable;
                self.batch_bad = bad;
                // Kick off the next batch now; its SIMD chain overlaps this
                // batch's pops instead of stalling the next refill. Also
                // done for dirty batches: a scalar overrun past the batch
                // end just leaves the precompute unused (`pre_base` misses),
                // while gaps that resolve inside the batch — the common
                // case — keep the pipeline primed. Dirty batches used to
                // skip this, and paying the whole SIMD chain latency at the
                // next refill was a large part of their ~270-cycle cost.
                if self.scan + 64 <= len {
                    let (u2, b2) = S::batch_masks(bytes, self.scan);
                    self.pre_base = self.scan;
                    self.pre_usable = u2;
                    self.pre_bad = b2;
                } else {
                    self.pre_base = usize::MAX;
                }
                // An overrun may have left `pos` inside this grid batch;
                // start from its bit so stale bits below never pop. The
                // no-overrun case keeps the constant argument (and its
                // folded codegen) — schemes with few bad zones take that
                // branch essentially always.
                if self.pos > self.mask_base {
                    self.load_segment((self.pos - self.mask_base) as u32);
                } else {
                    self.load_segment(0);
                }
            }
            #[cfg(not(any(
                all(target_arch = "aarch64", target_endian = "little"),
                target_arch = "x86_64"
            )))]
            {
                // Unreachable: scalar_until is usize::MAX on this arch.
                self.scalar_until = usize::MAX;
            }
        }
    }
}

/// The streaming families retain at most two provisional pretokens. The fused
/// producer may harvest those two lookahead spans in addition to one logical
/// cache batch before the caller decides which prefix is settled.
const FUSED_RETAIN_MAX: usize = 2;

/// One logical keyed batch, one full-mask overshoot, and the dirty-zone scalar
/// tail that may follow it. Only the initialized prefix is ever read.
const BOUNDARY_SCRATCH: usize = KEYED_BATCH + FUSED_RETAIN_MAX + 208;

/// Keep every harvested endpoint representable relative to the current fill.
/// The extra 127 bytes cover the widest classifier/repair lookahead.
const RELATIVE_MASK_LIMIT: isize = u16::MAX as isize - 127;

/// Set-bit positions of one byte, packed into eight u16 lanes. Unused lanes
/// are harmless scratch: the next octet overwrites them at its prefix-popcount
/// offset and callers only read the final population count.
#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
static BIT_POS: [[u16; 8]; 256] = {
    let mut table = [[0u16; 8]; 256];
    let mut byte = 1usize;
    while byte < 256 {
        let mut bit = 0usize;
        let mut lane = 0usize;
        while bit < 8 {
            if byte >> bit & 1 == 1 {
                table[byte][lane] = bit as u16;
                lane += 1;
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// Flatten a 64-bit boundary mask without a branch per boundary. At most
/// seven scratch lanes are written beyond the returned population count.
#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
#[inline(always)]
unsafe fn flatten_boundaries(mask: u64, base: u16, out: *mut u16) -> usize {
    let mut counts = mask;
    counts -= (counts >> 1) & 0x5555_5555_5555_5555;
    counts = (counts & 0x3333_3333_3333_3333) + ((counts >> 2) & 0x3333_3333_3333_3333);
    counts = (counts + (counts >> 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    let inclusive = counts.wrapping_mul(0x0101_0101_0101_0101);
    let exclusive = inclusive << 8;

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    unsafe {
        use std::arch::aarch64::*;
        for octet in 0..8 {
            let byte = (mask >> (8 * octet)) as u8 as usize;
            let write = (exclusive >> (8 * octet)) as u8 as usize;
            let positions = vld1q_u16(BIT_POS[byte].as_ptr());
            let positions = vaddq_u16(
                positions,
                vdupq_n_u16(base.wrapping_add((8 * octet) as u16)),
            );
            vst1q_u16(out.add(write), positions);
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for octet in 0..8 {
            let byte = (mask >> (8 * octet)) as u8 as usize;
            let write = (exclusive >> (8 * octet)) as u8 as usize;
            let octet_base = base.wrapping_add((8 * octet) as u16);
            for lane in 0..8 {
                out.add(write + lane)
                    .write(BIT_POS[byte][lane].wrapping_add(octet_base));
            }
        }
    }
    (inclusive >> 56) as usize
}

/// AVX-512/VBMI2 boundary flattening. `vpcompressb` packs the selected iota
/// lanes, after which both halves are widened to relative u16 endpoints.
/// Two unconditional 64-byte stores scribble `out[0..64]`; callers reserve
/// that slack even though only the returned population count is initialized.
///
/// # Safety
///
/// The CPU must support AVX-512 F/BW/VBMI2 and `out[0..64]` must be writable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
#[inline]
unsafe fn flatten_boundaries_avx512(mask: u64, base: u16, out: *mut u16) -> usize {
    use std::arch::x86_64::*;

    const IOTA: [u8; 64] = {
        let mut values = [0u8; 64];
        let mut index = 0usize;
        while index < 64 {
            values[index] = index as u8;
            index += 1;
        }
        values
    };

    unsafe {
        let iota = _mm512_loadu_si512(IOTA.as_ptr().cast());
        let compressed = _mm512_maskz_compress_epi8(mask, iota);
        let base = _mm512_set1_epi16(base as i16);
        let lo = _mm512_add_epi16(
            _mm512_cvtepu8_epi16(_mm512_castsi512_si256(compressed)),
            base,
        );
        let hi = _mm512_add_epi16(
            _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(compressed)),
            base,
        );
        _mm512_storeu_si512(out.cast(), lo);
        _mm512_storeu_si512(out.add(32).cast(), hi);
    }
    mask.count_ones() as usize
}

/// Flatten one boundary mask using the fill's compile-time-selected x86 tier.
/// The const comparison disappears from every live instantiation.
///
/// # Safety
///
/// As [`flatten_boundaries`]. For the VBMI2 tier, additionally the contract of
/// [`flatten_boundaries_avx512`].
#[inline(always)]
unsafe fn flatten_boundaries_tier<const X86_TIER: u8>(
    mask: u64,
    base: u16,
    out: *mut u16,
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if X86_TIER == X86_TIER_AVX512_VBMI2 {
        return unsafe { flatten_boundaries_avx512(mask, base, out) };
    }
    unsafe { flatten_boundaries(mask, base, out) }
}

/// Phase B of the shared mask pull. Keeping this out of line prevents the
/// key/hash temporaries from polluting the register-heavy boundary harvester.
/// Cache prefetching shares this pass. The final one/two records may only be
/// streaming lookahead, but a prefetch is a non-semantic hint and issuing it
/// here avoids rereading the completed span array before every probe batch.
#[inline(always)]
fn build_keyed_batch_impl<const X86_CRC: bool, C: KeyedBatchConsumer>(
    bytes: &[u8],
    fill_start: usize,
    ends: &[std::mem::MaybeUninit<u16>],
    wide_end: Option<usize>,
    out: &mut [KeyedSpan],
    consumer: &mut C,
) {
    let total = ends.len() + usize::from(wide_end.is_some());
    if total == 0 {
        return;
    }
    debug_assert!(out.len() >= total);
    debug_assert!(wide_end.is_none() || ends.is_empty());

    let last_end = wide_end.unwrap_or_else(|| {
        fill_start + unsafe { ends.last().unwrap_unchecked().assume_init() } as usize
    });
    let last_start = if ends.len() >= 2 {
        fill_start + unsafe { ends.get_unchecked(ends.len() - 2).assume_init() } as usize
    } else {
        fill_start
    };
    u32::try_from(last_end).expect("stream chunk offset exceeds u32");
    u32::try_from(last_end - fill_start).expect("pretoken length exceeds u32");

    let out_ptr = out.as_mut_ptr();
    let packer = PieceKeyPacker::new();
    let all_loads_in_bounds = last_start
        .checked_add(16)
        .is_some_and(|end| end <= bytes.len());
    let mut start = fill_start;

    for index in 0..total {
        let end = if let Some(end) = wide_end {
            end
        } else {
            fill_start + unsafe { ends.get_unchecked(index).assume_init() } as usize
        };
        let len = end - start;
        let key = if all_loads_in_bounds {
            let word = u128::from_le(unsafe {
                (bytes.as_ptr().add(start) as *const u128).read_unaligned()
            });
            packer.key_from_loaded_fill::<X86_CRC>(word, len)
        } else {
            packer.key_from_span_fill::<X86_CRC>(bytes, start, len)
        };
        let span = KeyedSpan {
            key,
            start: start as u32,
            len: len as u32,
        };
        consumer.prefetch(span.key);
        unsafe { out_ptr.add(index).write(span) };
        start = end;
    }
}

/// Baseline/portable phase-B body. Off x86 this preserves the platform's
/// compile-time hash selection (including aarch64 CRC).
#[inline(never)]
fn build_keyed_batch_default<C: KeyedBatchConsumer>(
    bytes: &[u8],
    fill_start: usize,
    ends: &[std::mem::MaybeUninit<u16>],
    wide_end: Option<usize>,
    out: &mut [KeyedSpan],
    consumer: &mut C,
) {
    build_keyed_batch_impl::<false, C>(bytes, fill_start, ends, wide_end, out, consumer);
}

/// SSE4.2 phase-B body. The per-span implementation inlines into this
/// wrapper, allowing both CRC instructions to inline as well; there is one
/// out-of-line call per harvested group, not one per key.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline(never)]
unsafe fn build_keyed_batch_crc<C: KeyedBatchConsumer>(
    bytes: &[u8],
    fill_start: usize,
    ends: &[std::mem::MaybeUninit<u16>],
    wide_end: Option<usize>,
    out: &mut [KeyedSpan],
    consumer: &mut C,
) {
    build_keyed_batch_impl::<true, C>(bytes, fill_start, ends, wide_end, out, consumer);
}

#[inline(always)]
fn build_keyed_batch<const X86_CRC: bool, C: KeyedBatchConsumer>(
    bytes: &[u8],
    fill_start: usize,
    ends: &[std::mem::MaybeUninit<u16>],
    wide_end: Option<usize>,
    out: &mut [KeyedSpan],
    consumer: &mut C,
) {
    #[cfg(target_arch = "x86_64")]
    if X86_CRC {
        debug_assert!(crate::piece_cache::crc_hash_selected());
        // SAFETY: the true fill instantiation is reachable only through an
        // SSE4.2-gated wrapper selected by the same immutable feature bit.
        return unsafe { build_keyed_batch_crc(bytes, fill_start, ends, wide_end, out, consumer) };
    }
    build_keyed_batch_default(bytes, fill_start, ends, wide_end, out, consumer);
}

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
impl MaskState {
    #[inline]
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    /// Fill a keyed cache batch through the same two-phase organization used
    /// by the r50k engine: harvest compact endpoints first, then construct
    /// keys, hashes, and L2 prefetches in one flat counted pass.
    ///
    /// The caller owns streaming settlement (one retained cl100k piece, two
    /// retained o200k pieces). This method only advances exact whole-buffer
    /// boundaries and may write up to `KEYED_BATCH + 2` records.
    #[inline(never)]
    pub(crate) fn fill_keyed_two_phase<S: MaskScheme, C: KeyedBatchConsumer>(
        &mut self,
        bytes: &[u8],
        out: &mut [KeyedSpan],
        capacity: usize,
        consumer: &mut C,
    ) -> usize {
        let capacity = capacity.min(KEYED_BATCH + FUSED_RETAIN_MAX);
        if capacity == 0 || self.pos >= bytes.len() {
            return 0;
        }

        // Choose the classifier, boundary-flattener, and hash arm once per
        // fill. Dense Unicode intentionally remains on the scalar scheme,
        // but still selects CRC once here for its shared phase-B key loop.
        #[cfg(target_arch = "x86_64")]
        if crate::piece_cache::crc_hash_selected() {
            let use_simd = self.scalar_until != usize::MAX && simd_scanner_available();
            if use_simd && avx512_fill_available() {
                // SAFETY: runtime checks establish AVX-512 F/BW/VL, VBMI2,
                // the bit-manipulation tier, and SSE4.2.
                return unsafe {
                    self.fill_keyed_avx512_vbmi2_crc::<S, C>(bytes, out, capacity, consumer)
                };
            }
            if use_simd && avx512_scanner_available() {
                // SAFETY: runtime checks establish the AVX-512 scanner tier
                // and SSE4.2.
                return unsafe {
                    self.fill_keyed_avx512_crc::<S, C>(bytes, out, capacity, consumer)
                };
            }
            if use_simd && avx2_scanner_available() {
                // SAFETY: runtime checks establish the AVX2 scanner tier and
                // SSE4.2.
                return unsafe { self.fill_keyed_avx2_crc::<S, C>(bytes, out, capacity, consumer) };
            }
            // Dense Unicode or an unusual SSE4.2 CPU without either vector
            // tier: keep dynamic/scalar classification but monomorphize CRC.
            return unsafe { self.fill_keyed_crc::<S, C>(bytes, out, capacity, consumer) };
        }

        self.fill_keyed_two_phase_impl::<S, C, false, X86_TIER_DYN>(bytes, out, capacity, consumer)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi1,bmi2,lzcnt,popcnt,sse4.2"
    )]
    unsafe fn fill_keyed_avx512_vbmi2_crc<S: MaskScheme, C: KeyedBatchConsumer>(
        &mut self,
        bytes: &[u8],
        out: &mut [KeyedSpan],
        capacity: usize,
        consumer: &mut C,
    ) -> usize {
        self.fill_keyed_two_phase_impl::<S, C, true, X86_TIER_AVX512_VBMI2>(
            bytes, out, capacity, consumer,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt,sse4.2")]
    unsafe fn fill_keyed_avx512_crc<S: MaskScheme, C: KeyedBatchConsumer>(
        &mut self,
        bytes: &[u8],
        out: &mut [KeyedSpan],
        capacity: usize,
        consumer: &mut C,
    ) -> usize {
        self.fill_keyed_two_phase_impl::<S, C, true, X86_TIER_AVX512>(
            bytes, out, capacity, consumer,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt,sse4.2")]
    unsafe fn fill_keyed_avx2_crc<S: MaskScheme, C: KeyedBatchConsumer>(
        &mut self,
        bytes: &[u8],
        out: &mut [KeyedSpan],
        capacity: usize,
        consumer: &mut C,
    ) -> usize {
        self.fill_keyed_two_phase_impl::<S, C, true, X86_TIER_AVX2>(bytes, out, capacity, consumer)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.2")]
    unsafe fn fill_keyed_crc<S: MaskScheme, C: KeyedBatchConsumer>(
        &mut self,
        bytes: &[u8],
        out: &mut [KeyedSpan],
        capacity: usize,
        consumer: &mut C,
    ) -> usize {
        self.fill_keyed_two_phase_impl::<S, C, true, X86_TIER_DYN>(bytes, out, capacity, consumer)
    }

    #[inline(always)]
    fn fill_keyed_two_phase_impl<
        S: MaskScheme,
        C: KeyedBatchConsumer,
        const X86_CRC: bool,
        const X86_TIER: u8,
    >(
        &mut self,
        bytes: &[u8],
        out: &mut [KeyedSpan],
        capacity: usize,
        consumer: &mut C,
    ) -> usize {
        debug_assert!(capacity <= KEYED_BATCH + FUSED_RETAIN_MAX);
        debug_assert!(out.len() >= capacity);
        let capacity = capacity.min(KEYED_BATCH + FUSED_RETAIN_MAX);
        if capacity == 0 || self.pos >= bytes.len() {
            return 0;
        }

        // Dense Unicode deliberately stays on the scalar ground truth. It
        // still benefits from compact endpoint harvesting and the same flat
        // phase-B key/cache loop.
        let use_simd = self.scalar_until != usize::MAX && simd_scanner_available();
        let len = bytes.len();
        let mut pending = self.pos;
        let mut scan = self.scan;
        if scan > pending {
            scan -= 64 * (scan - pending).div_ceil(64);
        }
        let mut produced = 0usize;

        'refill: while produced < capacity && pending < len {
            if pending >= scan.saturating_add(64) {
                scan += 64 * ((pending - scan) / 64);
            }
            let fill_start = pending;
            let needed = capacity - produced;
            let mut ends = [std::mem::MaybeUninit::<u16>::uninit(); BOUNDARY_SCRATCH];
            let ends_ptr = ends.as_mut_ptr().cast::<u16>();
            let mut count = 0usize;
            let mut resume = pending;
            let mut exhausted = false;
            let mut overflow_end = None;

            'harvest: while count < needed {
                let mask_in_window = use_simd
                    && scan + 64 <= len
                    && scan.wrapping_sub(fill_start) as isize <= RELATIVE_MASK_LIMIT;
                if !mask_in_window {
                    let mut position = if count == 0 {
                        fill_start
                    } else {
                        fill_start + unsafe { *ends_ptr.add(count - 1) } as usize
                    };
                    while position < len && count < needed {
                        let end = S::advance(bytes, position);
                        let relative = end - fill_start;
                        if relative > u16::MAX as usize {
                            overflow_end = Some(end);
                            break 'harvest;
                        }
                        unsafe { ends_ptr.add(count).write(relative as u16) };
                        count += 1;
                        position = end;
                    }
                    exhausted = position >= len;
                    break;
                }

                let base = scan;
                #[cfg(target_arch = "x86_64")]
                let (usable, bad) = match X86_TIER {
                    // SAFETY: these arms are instantiated only through their
                    // matching runtime-gated target-feature wrappers above.
                    X86_TIER_AVX512 | X86_TIER_AVX512_VBMI2 => unsafe {
                        S::batch_masks_x86::<true>(bytes, base)
                    },
                    X86_TIER_AVX2 => unsafe { S::batch_masks_x86::<false>(bytes, base) },
                    _ => S::batch_masks(bytes, base),
                };
                #[cfg(not(target_arch = "x86_64"))]
                let (usable, bad) = S::batch_masks(bytes, base);
                // At the pending token's start, a usable bit is not an end.
                // A bad bit at that same position must remain live so the
                // scalar scheme re-derives the token before any trusted suffix.
                let (mut usable_live, mut bad_live) = if resume >= base {
                    debug_assert!(resume - base < 64);
                    let bit = resume - base;
                    ((u64::MAX << bit) << 1, u64::MAX << bit)
                } else {
                    (u64::MAX, u64::MAX)
                };
                let relative_base = base.wrapping_sub(fill_start) as u16;

                if bad & bad_live == 0 {
                    debug_assert!(
                        count
                            + if X86_TIER == X86_TIER_AVX512_VBMI2 {
                                64
                            } else {
                                72
                            }
                            <= BOUNDARY_SCRATCH
                    );
                    count += unsafe {
                        flatten_boundaries_tier::<X86_TIER>(
                            usable & usable_live,
                            relative_base,
                            ends_ptr.add(count),
                        )
                    };
                    scan = base + 64;
                    continue;
                }

                loop {
                    let segment_bad = bad & bad_live;
                    if segment_bad == 0 {
                        debug_assert!(
                            count
                                + if X86_TIER == X86_TIER_AVX512_VBMI2 {
                                    64
                                } else {
                                    72
                                }
                                <= BOUNDARY_SCRATCH
                        );
                        count += unsafe {
                            flatten_boundaries_tier::<X86_TIER>(
                                usable & usable_live,
                                relative_base,
                                ends_ptr.add(count),
                            )
                        };
                        scan = base + 64;
                        break;
                    }

                    let first_bad = segment_bad.trailing_zeros();
                    let trusted_prefix = usable & usable_live & !(u64::MAX << first_bad);
                    debug_assert!(
                        count
                            + if X86_TIER == X86_TIER_AVX512_VBMI2 {
                                64
                            } else {
                                72
                            }
                            <= BOUNDARY_SCRATCH
                    );
                    count += unsafe {
                        flatten_boundaries_tier::<X86_TIER>(
                            trusted_prefix,
                            relative_base,
                            ends_ptr.add(count),
                        )
                    };
                    let mut position = if count == 0 {
                        fill_start
                    } else {
                        fill_start + unsafe { *ends_ptr.add(count - 1) } as usize
                    };
                    let trusted_suffix = usable & (u64::MAX << first_bad);
                    let scalar_until = if trusted_suffix != 0 {
                        base + trusted_suffix.trailing_zeros() as usize
                    } else {
                        base + 64
                    };
                    while position < scalar_until {
                        let end = S::advance(bytes, position);
                        let relative = end - fill_start;
                        if relative > u16::MAX as usize {
                            overflow_end = Some(end);
                            break 'harvest;
                        }
                        unsafe { ends_ptr.add(count).write(relative as u16) };
                        count += 1;
                        position = end;
                    }
                    if position >= base + 64 {
                        scan = base + 64 * ((position - base) / 64);
                        resume = position;
                        break;
                    }
                    let bit = position - base;
                    debug_assert!(bit < 64);
                    bad_live = u64::MAX << bit;
                    usable_live = bad_live << 1;
                }
            }

            if count == 0 {
                debug_assert!(!exhausted);
                let end = overflow_end.unwrap_or_else(|| S::advance(bytes, fill_start));
                build_keyed_batch::<X86_CRC, C>(
                    bytes,
                    fill_start,
                    &[],
                    Some(end),
                    &mut out[produced..],
                    consumer,
                );
                produced += 1;
                pending = end;
                continue 'refill;
            }

            let emitted = count.min(needed);
            build_keyed_batch::<X86_CRC, C>(
                bytes,
                fill_start,
                &ends[..emitted],
                None,
                &mut out[produced..],
                consumer,
            );
            produced += emitted;
            pending =
                fill_start + unsafe { ends.get_unchecked(emitted - 1).assume_init() } as usize;
            if exhausted {
                debug_assert_eq!(pending, len);
                break;
            }
        }

        // Any mask endpoints harvested past `capacity` are pure functions of
        // the bytes and are cheaply recomputed from the pending token.
        if scan > pending {
            scan -= 64 * (scan - pending).div_ceil(64);
        }
        self.pos = pending;
        self.scan = scan;
        self.mask_base = scan;
        self.rem = 0;
        self.batch_usable = 0;
        self.batch_bad = 0;
        self.scalar_until = if use_simd { pending } else { usize::MAX };
        self.pre_base = usize::MAX;
        produced
    }
}

/// A cheap once-per-buffer density test. At most 4 KiB is inspected; the
/// unaligned word loads and popcounts compile to a short scalar loop and avoid
/// invoking any encoding-specific Unicode classifier.
#[inline]
fn dense_non_ascii(bytes: &[u8], pos: usize) -> bool {
    const WINDOWS: usize = 4;
    const WINDOW: usize = 1024;
    const HIGH_BITS: u64 = 0x8080_8080_8080_8080;

    let bytes = &bytes[pos.min(bytes.len())..];
    if bytes.len() < 64 {
        return false;
    }

    #[inline(always)]
    fn count_high(bytes: &[u8]) -> u32 {
        let mut high = 0u32;
        let mut offset = 0usize;
        while offset + 8 <= bytes.len() {
            let word = unsafe { (bytes.as_ptr().add(offset) as *const u64).read_unaligned() };
            high += (word & HIGH_BITS).count_ones();
            offset += 8;
        }
        while offset < bytes.len() {
            high += u32::from(unsafe { *bytes.get_unchecked(offset) } >= 0x80);
            offset += 1;
        }
        high
    }

    if bytes.len() <= WINDOWS * WINDOW {
        return count_high(bytes) as usize * 4 >= bytes.len();
    }

    let last = bytes.len() - WINDOW;
    let mut high = 0u32;
    for window in 0..WINDOWS {
        let start = last * window / (WINDOWS - 1);
        high += count_high(&bytes[start..start + WINDOW]);
    }
    high as usize * 4 >= WINDOWS * WINDOW
}

#[cfg(test)]
mod density_tests {
    use super::dense_non_ascii;

    #[test]
    fn selects_dense_unicode_without_misclassifying_ascii_or_short_buffers() {
        assert!(!dense_non_ascii(b"plain ASCII text", 0));
        assert!(!dense_non_ascii(&vec![b'a'; 8 * 1024], 0));

        let chinese = "中文语料".repeat(2 * 1024);
        assert!(dense_non_ascii(chinese.as_bytes(), 0));

        let mixed = format!(
            "{}{}{}",
            "a".repeat(2 * 1024),
            "中文".repeat(2 * 1024),
            "z".repeat(2 * 1024)
        );
        assert!(dense_non_ascii(mixed.as_bytes(), 0));
    }
}
