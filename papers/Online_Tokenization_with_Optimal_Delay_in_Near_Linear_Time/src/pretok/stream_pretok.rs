//! Streaming pretokenizer dispatch and encoding-independent chunk composition.
//!
//! The byte-stepped reference engine lives in [`super::dfa_stream`]. Each
//! hardware scanner owns its encoding-specific boundary algebra; this module
//! only selects an engine, retains unsettled suffixes, and forms keyed cache
//! batches.

use super::{
    Encoding,
    dfa_stream::DfaStream,
    fast_cl100k::{self, Cl100kScheme, FastCl100kPretokenizer},
    fast_llama3::{self, FastLlama3Pretokenizer},
    fast_mask::{MaskScheme, MaskState},
    fast_mistral::{self, FastMistralPretokenizer},
    fast_o200k::{self, FastO200kPretokenizer, O200kScheme},
    fast_qwen3::{self, FastQwen3Pretokenizer},
    fast_r50k::{self, FastR50kPretokenizer, R50kScheme},
    fast_unicode,
};
use crate::piece_cache::PieceKey;

pub(crate) const KEYED_BATCH: usize = 256;

/// Statistics accumulated while one keyed span batch is formed.
#[derive(Clone, Copy, Default)]
pub(crate) struct KeyedBatchStats {
    pub(crate) pieces: usize,
    pub(crate) pretok_sum: u64,
    pub(crate) pretok_peak: usize,
}

impl KeyedBatchStats {
    #[inline(always)]
    pub(crate) fn record(&mut self, len: usize) {
        self.pieces += 1;
        self.pretok_peak = self.pretok_peak.max(len);
        let len = len as u64;
        self.pretok_sum += len * (len + 1) / 2;
    }
}

/// Consumer for one chunk-local keyed pretoken batch. `prefetch` is called as
/// records are formed, including provisional streaming lookahead; `consume`
/// follows only for the settled prefix once the complete batch is ready.
pub(crate) trait KeyedBatchConsumer {
    const PROFILE: bool = true;

    fn prefetch(&mut self, key: PieceKey);

    /// Consume `span_count` real records followed by readable lookahead
    /// records used by the fixed-distance cache prefetch loop.
    fn consume(
        &mut self,
        base: &str,
        spans: &mut [KeyedSpan],
        span_count: usize,
        stats: KeyedBatchStats,
    );
}

/// One settled pretoken within the base string supplied to a keyed callback.
/// The 32-byte layout lets one cache line hold two scanner outputs.
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
pub(crate) struct KeyedSpan {
    pub(crate) key: PieceKey,
    pub(crate) start: u32,
    pub(crate) len: u32,
}

impl KeyedSpan {
    pub(crate) const PREFETCH_SLACK: usize = 16;
}

const _: () = assert!(std::mem::size_of::<KeyedSpan>() == 32);
const _: () = assert!(std::mem::align_of::<KeyedSpan>() == 32);

/// Streaming pretokenizer dispatcher.
///
/// Every encoding uses its specialized SIMD stream unless the DFA is forced.
pub struct StreamPretokenizer {
    inner: StreamImpl,
}

enum StreamImpl {
    Fast(FastFamilyStream),
    Dfa(Box<DfaStream>),
}

#[derive(Clone, Copy)]
enum FastFamily {
    R50k,
    P50k,
    Cl100k,
    O200k,
    Llama3,
    Mistral,
    Qwen3,
}

impl FastFamily {
    #[inline]
    fn from_encoding(enc: Encoding) -> Self {
        match enc {
            Encoding::R50k => Self::R50k,
            Encoding::P50k => Self::P50k,
            Encoding::Cl100k => Self::Cl100k,
            Encoding::O200k => Self::O200k,
            Encoding::Llama3 => Self::Llama3,
            Encoding::Mistral => Self::Mistral,
            Encoding::Qwen3 => Self::Qwen3,
        }
    }

    #[inline]
    fn encoding(self) -> Encoding {
        match self {
            Self::R50k => Encoding::R50k,
            Self::P50k => Encoding::P50k,
            Self::Cl100k => Encoding::Cl100k,
            Self::O200k => Encoding::O200k,
            Self::Llama3 => Encoding::Llama3,
            Self::Mistral => Encoding::Mistral,
            Self::Qwen3 => Encoding::Qwen3,
        }
    }

    fn warm(self) {
        match self {
            Self::R50k | Self::P50k | Self::Cl100k | Self::Llama3 | Self::Qwen3 => {
                fast_unicode::warm()
            }
            Self::O200k | Self::Mistral => fast_unicode::warm_o200k(),
        }
    }
}

impl StreamPretokenizer {
    pub fn new(enc: Encoding) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_mode(enc, false)
    }

    pub fn new_with_mode(
        enc: Encoding,
        force_dfa: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = if force_dfa {
            StreamImpl::Dfa(Box::new(DfaStream::new(enc)?))
        } else {
            StreamImpl::Fast(FastFamilyStream::new(FastFamily::from_encoding(enc)))
        };
        Ok(Self { inner })
    }

    pub fn feed<F: FnMut(&str)>(&mut self, chunk: &str, emit: F) {
        match &mut self.inner {
            StreamImpl::Fast(stream) => stream.feed(chunk, emit),
            StreamImpl::Dfa(stream) => stream.feed(chunk, emit),
        }
    }

    pub fn finish<F: FnMut(&str)>(&mut self, emit: F) {
        match &mut self.inner {
            StreamImpl::Fast(stream) => stream.finish(emit),
            StreamImpl::Dfa(stream) => stream.finish(emit),
        }
    }

    pub(crate) fn feed_keyed<C: KeyedBatchConsumer>(&mut self, chunk: &str, consumer: &mut C) {
        match &mut self.inner {
            StreamImpl::Fast(stream) => stream.feed_keyed(chunk, consumer),
            StreamImpl::Dfa(stream) => dfa_feed_keyed(stream, chunk, consumer),
        }
    }

    pub(crate) fn finish_keyed<C: KeyedBatchConsumer>(&mut self, consumer: &mut C) {
        match &mut self.inner {
            StreamImpl::Fast(stream) => stream.finish_keyed(consumer),
            StreamImpl::Dfa(stream) => dfa_finish_keyed(stream, consumer),
        }
    }
}

/// Generic streaming composition for every hardware scanner. Completed pieces
/// are emitted immediately and only the small right-context-dependent suffix
/// is retained. Schemes retain one or two provisional pieces according to
/// their contraction, case, and trailing-whitespace rules.
struct FastFamilyStream {
    family: FastFamily,
    carry: String,
    long: Option<LongCarry>,
    keyed: Box<[KeyedSpan]>,
}

/// A genuinely long unsettled pretoken is rare, but repeatedly rescanning it
/// from byte zero would make chunked input quadratic. Activate the incremental
/// DFA only for that exceptional suffix; ordinary input stays entirely on the
/// SIMD path and pays no DFA construction cost.
struct LongCarry {
    stream: Box<DfaStream>,
    buffered: usize,
}

/// Borrow only this much of a new chunk while finding the first boundary past
/// the carried suffix. Once found, the rest of the chunk is scanned in place.
const FAST_FAMILY_BRIDGE_BYTES: usize = 256;

/// Bound the amount of repeated SIMD work before switching one exceptionally
/// long pending pretoken to the incremental byte-stepped fallback.
const FAST_FAMILY_LINEAR_THRESHOLD: usize = 16 * 1024;

impl FastFamilyStream {
    fn new(family: FastFamily) -> Self {
        family.warm();
        Self {
            family,
            carry: String::new(),
            long: None,
            keyed: vec![KeyedSpan::default(); KEYED_BATCH + KeyedSpan::PREFETCH_SLACK]
                .into_boxed_slice(),
        }
    }

    #[inline]
    fn prefix_end(chunk: &str, start: usize) -> usize {
        let mut end = (start + FAST_FAMILY_BRIDGE_BYTES).min(chunk.len());
        while !chunk.is_char_boundary(end) {
            end -= 1;
        }
        debug_assert!(end > start);
        end
    }

    /// Move an oversized unresolved suffix into the incremental DFA. Returns a
    /// boundary in the current chunk if the DFA's more precise state can
    /// already settle past `prior`.
    fn activate_long<F: FnMut(&str)>(&mut self, prior: &mut usize, mut emit: F) -> Option<usize> {
        debug_assert!(self.long.is_none());
        debug_assert!(!self.carry.is_empty());
        let carry = std::mem::take(&mut self.carry);
        let mut stream =
            DfaStream::new(self.family.encoding()).expect("built-in pretokenizer DFA must compile");
        let mut emitted = 0usize;
        stream.feed(&carry, |piece| {
            emitted += piece.len();
            emit(piece);
        });
        debug_assert!(emitted <= carry.len());

        if emitted >= *prior {
            let direct_from = emitted - *prior;
            // The un-emitted suffix starts at `direct_from` in the caller's
            // chunk and will be rescanned by SIMD. Clear the DFA without
            // forwarding that provisional suffix a second time.
            stream.finish(|_| {});
            return Some(direct_from);
        }

        *prior -= emitted;
        self.long = Some(LongCarry {
            stream: Box::new(stream),
            buffered: carry.len() - emitted,
        });
        None
    }

    /// Settle the carried suffix against bounded prefixes of `chunk`. The
    /// returned offset is a proven pretoken boundary, after which the caller
    /// can scan the remainder directly without copying it.
    fn bridge<F: FnMut(&str)>(&mut self, chunk: &str, emit: &mut F) -> Option<usize> {
        if self.carry.is_empty() && self.long.is_none() {
            return Some(0);
        }
        if chunk.is_empty() {
            return None;
        }

        let mut appended = 0usize;
        let mut prior = self
            .long
            .as_ref()
            .map_or(self.carry.len(), |long| long.buffered);

        loop {
            let end = Self::prefix_end(chunk, appended);
            if self.long.is_some() {
                let old_buffered = self.long.as_ref().unwrap().buffered;
                let mut emitted = 0usize;
                {
                    let long = self.long.as_mut().unwrap();
                    long.stream.feed(&chunk[appended..end], |piece| {
                        emitted += piece.len();
                        emit(piece);
                    });
                    debug_assert!(emitted <= old_buffered + end - appended);
                    long.buffered = old_buffered + end - appended - emitted;
                }
                if emitted >= prior {
                    let direct_from = emitted - prior;
                    let mut long = self.long.take().unwrap();
                    long.stream.finish(|_| {});
                    return Some(direct_from);
                }
                prior -= emitted;
            } else {
                self.carry.push_str(&chunk[appended..end]);
                let keep = scan_fast_settled(self.family, &self.carry, emit);
                if keep >= prior {
                    let direct_from = keep - prior;
                    self.carry.clear();
                    return Some(direct_from);
                }
                if keep != 0 {
                    self.carry.drain(..keep);
                    prior -= keep;
                }
                if self.carry.len() >= FAST_FAMILY_LINEAR_THRESHOLD
                    && let Some(direct_from) = self.activate_long(&mut prior, &mut *emit)
                {
                    return Some(direct_from);
                }
            }

            appended = end;
            if appended == chunk.len() {
                return None;
            }
        }
    }

    fn feed<F: FnMut(&str)>(&mut self, chunk: &str, mut emit: F) {
        let Some(direct_from) = self.bridge(chunk, &mut emit) else {
            return;
        };
        let direct = &chunk[direct_from..];
        if !direct.is_empty() {
            let keep = scan_fast_settled(self.family, direct, &mut emit);
            self.carry.push_str(&direct[keep..]);
        }
    }

    fn finish<F: FnMut(&str)>(&mut self, mut emit: F) {
        if let Some(mut long) = self.long.take() {
            long.stream.finish(&mut emit);
        } else {
            scan_fast_all(self.family, &self.carry, &mut emit);
        }
        self.carry.clear();
    }

    fn bridge_keyed<C: KeyedBatchConsumer>(
        &mut self,
        chunk: &str,
        consumer: &mut C,
    ) -> Option<usize> {
        if self.carry.is_empty() && self.long.is_none() {
            return Some(0);
        }
        if chunk.is_empty() {
            return None;
        }

        let mut appended = 0usize;
        let mut prior = self
            .long
            .as_ref()
            .map_or(self.carry.len(), |long| long.buffered);

        loop {
            let end = Self::prefix_end(chunk, appended);
            if self.long.is_some() {
                let old_buffered = self.long.as_ref().unwrap().buffered;
                let mut emitted = 0usize;
                {
                    let long = self.long.as_mut().unwrap();
                    long.stream.feed(&chunk[appended..end], |piece| {
                        emitted += piece.len();
                        consume_dfa_piece(piece, consumer);
                    });
                    debug_assert!(emitted <= old_buffered + end - appended);
                    long.buffered = old_buffered + end - appended - emitted;
                }
                if emitted >= prior {
                    let direct_from = emitted - prior;
                    let mut long = self.long.take().unwrap();
                    long.stream.finish(|_| {});
                    return Some(direct_from);
                }
                prior -= emitted;
            } else {
                self.carry.push_str(&chunk[appended..end]);
                let keep =
                    scan_fast_keyed_settled(self.family, &self.carry, &mut self.keyed, consumer);
                if keep >= prior {
                    let direct_from = keep - prior;
                    self.carry.clear();
                    return Some(direct_from);
                }
                if keep != 0 {
                    self.carry.drain(..keep);
                    prior -= keep;
                }
                if self.carry.len() >= FAST_FAMILY_LINEAR_THRESHOLD
                    && let Some(direct_from) = self.activate_long(&mut prior, |piece| {
                        consume_dfa_piece(piece, consumer);
                    })
                {
                    return Some(direct_from);
                }
            }

            appended = end;
            if appended == chunk.len() {
                return None;
            }
        }
    }

    fn feed_keyed<C: KeyedBatchConsumer>(&mut self, chunk: &str, consumer: &mut C) {
        let Some(direct_from) = self.bridge_keyed(chunk, consumer) else {
            return;
        };
        let direct = &chunk[direct_from..];
        if !direct.is_empty() {
            let keep = scan_fast_keyed_settled(self.family, direct, &mut self.keyed, consumer);
            self.carry.push_str(&direct[keep..]);
        }
    }

    fn finish_keyed<C: KeyedBatchConsumer>(&mut self, consumer: &mut C) {
        if let Some(mut long) = self.long.take() {
            long.stream
                .finish(|piece| consume_dfa_piece(piece, consumer));
        } else {
            scan_fast_keyed_all(self.family, &self.carry, &mut self.keyed, consumer);
        }
        self.carry.clear();
    }
}

/// Dispatch once per chunk, leaving the concrete scanner loop monomorphized.
fn scan_fast_settled(family: FastFamily, input: &str, emit: &mut impl FnMut(&str)) -> usize {
    let unstable_from = trailing_whitespace_start(input);
    match family {
        FastFamily::R50k | FastFamily::P50k => scan_iter::<_, _, { fast_r50k::STREAM_RETAIN }>(
            input,
            FastR50kPretokenizer::new(input),
            unstable_from,
            emit,
        ),
        FastFamily::Cl100k => scan_iter::<_, _, { fast_cl100k::STREAM_RETAIN }>(
            input,
            FastCl100kPretokenizer::new(input),
            unstable_from,
            emit,
        ),
        FastFamily::O200k => scan_iter::<_, _, { fast_o200k::STREAM_RETAIN }>(
            input,
            FastO200kPretokenizer::new(input),
            unstable_from,
            emit,
        ),
        FastFamily::Llama3 => scan_iter::<_, _, { fast_llama3::STREAM_RETAIN }>(
            input,
            FastLlama3Pretokenizer::new(input),
            unstable_from,
            emit,
        ),
        FastFamily::Mistral => scan_iter::<_, _, { fast_mistral::STREAM_RETAIN }>(
            input,
            FastMistralPretokenizer::new(input),
            unstable_from,
            emit,
        ),
        FastFamily::Qwen3 => scan_iter::<_, _, { fast_qwen3::STREAM_RETAIN }>(
            input,
            FastQwen3Pretokenizer::new(input),
            unstable_from,
            emit,
        ),
    }
}

fn scan_fast_all(family: FastFamily, input: &str, emit: &mut impl FnMut(&str)) {
    match family {
        FastFamily::R50k | FastFamily::P50k => {
            scan_iter::<_, _, 0>(input, FastR50kPretokenizer::new(input), input.len(), emit);
        }
        FastFamily::Cl100k => {
            scan_iter::<_, _, 0>(input, FastCl100kPretokenizer::new(input), input.len(), emit);
        }
        FastFamily::O200k => {
            scan_iter::<_, _, 0>(input, FastO200kPretokenizer::new(input), input.len(), emit);
        }
        FastFamily::Llama3 => {
            scan_iter::<_, _, 0>(input, FastLlama3Pretokenizer::new(input), input.len(), emit);
        }
        FastFamily::Mistral => {
            scan_iter::<_, _, 0>(
                input,
                FastMistralPretokenizer::new(input),
                input.len(),
                emit,
            );
        }
        FastFamily::Qwen3 => {
            scan_iter::<_, _, 0>(input, FastQwen3Pretokenizer::new(input), input.len(), emit);
        }
    }
}

#[inline]
fn scan_iter<'a, I, F, const RETAIN: usize>(
    input: &'a str,
    scanner: I,
    unstable_from: usize,
    emit: &mut F,
) -> usize
where
    I: Iterator<Item = &'a str>,
    F: FnMut(&str),
{
    debug_assert!(RETAIN <= 2);
    let mut pending = [None; 2];
    let mut held = 0usize;
    for piece in scanner {
        if RETAIN == 0 {
            emit(piece);
        } else if held < RETAIN {
            pending[held] = Some(piece);
            held += 1;
        } else {
            let settled = pending[0].expect("retained pretoken initialized");
            let start = settled.as_ptr() as usize - input.as_ptr() as usize;
            if start + settled.len() > unstable_from {
                return start;
            }
            emit(settled);
            if RETAIN == 2 {
                pending[0] = pending[1];
            }
            pending[RETAIN - 1] = Some(piece);
        }
    }
    pending[0].map_or(input.len(), |piece| {
        piece.as_ptr() as usize - input.as_ptr() as usize
    })
}

fn scan_fast_keyed_settled<C: KeyedBatchConsumer>(
    family: FastFamily,
    input: &str,
    keyed: &mut [KeyedSpan],
    consumer: &mut C,
) -> usize {
    let unstable_from = trailing_whitespace_start(input);
    match family {
        FastFamily::R50k | FastFamily::P50k => scan_mask_keyed::<
            R50kScheme,
            _,
            { fast_r50k::STREAM_RETAIN },
        >(input, unstable_from, keyed, consumer),
        FastFamily::Cl100k => scan_mask_keyed::<Cl100kScheme, _, { fast_cl100k::STREAM_RETAIN }>(
            input,
            unstable_from,
            keyed,
            consumer,
        ),
        FastFamily::O200k => scan_mask_keyed::<O200kScheme, _, { fast_o200k::STREAM_RETAIN }>(
            input,
            unstable_from,
            keyed,
            consumer,
        ),
        FastFamily::Llama3 => scan_iter_keyed::<_, _, { fast_llama3::STREAM_RETAIN }>(
            input,
            FastLlama3Pretokenizer::new(input),
            unstable_from,
            keyed,
            consumer,
        ),
        FastFamily::Mistral => scan_iter_keyed::<_, _, { fast_mistral::STREAM_RETAIN }>(
            input,
            FastMistralPretokenizer::new(input),
            unstable_from,
            keyed,
            consumer,
        ),
        FastFamily::Qwen3 => scan_iter_keyed::<_, _, { fast_qwen3::STREAM_RETAIN }>(
            input,
            FastQwen3Pretokenizer::new(input),
            unstable_from,
            keyed,
            consumer,
        ),
    }
}

fn scan_fast_keyed_all<C: KeyedBatchConsumer>(
    family: FastFamily,
    input: &str,
    keyed: &mut [KeyedSpan],
    consumer: &mut C,
) {
    match family {
        FastFamily::R50k | FastFamily::P50k => {
            scan_mask_keyed::<R50kScheme, _, 0>(input, input.len(), keyed, consumer);
        }
        FastFamily::Cl100k => {
            scan_mask_keyed::<Cl100kScheme, _, 0>(input, input.len(), keyed, consumer);
        }
        FastFamily::O200k => {
            scan_mask_keyed::<O200kScheme, _, 0>(input, input.len(), keyed, consumer);
        }
        FastFamily::Llama3 => {
            scan_iter_keyed::<_, _, 0>(
                input,
                FastLlama3Pretokenizer::new(input),
                input.len(),
                keyed,
                consumer,
            );
        }
        FastFamily::Mistral => {
            scan_iter_keyed::<_, _, 0>(
                input,
                FastMistralPretokenizer::new(input),
                input.len(),
                keyed,
                consumer,
            );
        }
        FastFamily::Qwen3 => {
            scan_iter_keyed::<_, _, 0>(
                input,
                FastQwen3Pretokenizer::new(input),
                input.len(),
                keyed,
                consumer,
            );
        }
    }
}

/// Chunk-local fused pull for the shared mask-scanner families. Phase A
/// harvests compact boundaries and phase B derives keys, hashes, and L2
/// prefetches. The final one/two provisional pieces stay in `keyed` until the
/// whole buffer is known, preserving cl100k/o200k streaming semantics.
#[inline(never)]
fn scan_mask_keyed<S, C, const RETAIN: usize>(
    input: &str,
    unstable_from: usize,
    keyed: &mut [KeyedSpan],
    consumer: &mut C,
) -> usize
where
    S: MaskScheme,
    C: KeyedBatchConsumer,
{
    debug_assert!(RETAIN <= 2);
    debug_assert!(keyed.len() >= KEYED_BATCH + KeyedSpan::PREFETCH_SLACK);
    let bytes = input.as_bytes();
    let target = KEYED_BATCH + RETAIN;
    let mut state = MaskState::new(bytes, 0);
    let mut held = 0usize;

    loop {
        let filled =
            state.fill_keyed_two_phase::<S, C>(bytes, &mut keyed[held..], target - held, consumer);
        let total = held + filled;
        if total == 0 {
            return input.len();
        }

        let last = keyed[total - 1];
        let last_end = last.start as usize + last.len as usize;
        let exhausted = state.position() == bytes.len();
        let crosses_unstable = last_end > unstable_from;

        if exhausted || crosses_unstable {
            let stable = if crosses_unstable {
                keyed[..total].partition_point(|span| {
                    span.start as usize + span.len as usize <= unstable_from
                })
            } else {
                total
            };
            let emit = stable.saturating_sub(RETAIN);
            consume_keyed_prefix::<C>(input, keyed, emit, consumer);
            return keyed
                .get(emit)
                .map_or(input.len(), |span| span.start as usize);
        }

        // A full logical batch has RETAIN real lookahead records. Consume the
        // settled prefix and carry only those records into the next fill.
        debug_assert_eq!(total, target);
        consume_keyed_prefix::<C>(input, keyed, KEYED_BATCH, consumer);
        if RETAIN != 0 {
            keyed.copy_within(KEYED_BATCH..target, 0);
        }
        held = RETAIN;
    }
}

#[inline]
fn consume_keyed_prefix<C: KeyedBatchConsumer>(
    input: &str,
    keyed: &mut [KeyedSpan],
    count: usize,
    consumer: &mut C,
) {
    if count == 0 {
        return;
    }
    let mut stats = KeyedBatchStats::default();
    if C::PROFILE {
        for span in &keyed[..count] {
            stats.record(span.len as usize);
        }
    }
    consumer.consume(input, keyed, count, stats);
}

#[inline]
fn scan_iter_keyed<'a, I, C, const RETAIN: usize>(
    input: &'a str,
    scanner: I,
    unstable_from: usize,
    keyed: &mut [KeyedSpan],
    consumer: &mut C,
) -> usize
where
    I: Iterator<Item = &'a str>,
    C: KeyedBatchConsumer,
{
    debug_assert!(RETAIN <= 2);
    let mut pending = [None; 2];
    let mut held = 0usize;
    let mut count = 0usize;
    let mut stats = KeyedBatchStats::default();

    for piece in scanner {
        let current = (
            piece.as_ptr() as usize - input.as_ptr() as usize,
            piece.len(),
        );
        if RETAIN == 0 {
            push_keyed(
                input, keyed, &mut count, &mut stats, consumer, current.0, current.1,
            );
        } else if held < RETAIN {
            pending[held] = Some(current);
            held += 1;
        } else {
            let (start, len) = pending[0].expect("retained pretoken initialized");
            if start + len > unstable_from {
                flush_keyed(input, keyed, &mut count, &mut stats, consumer);
                return start;
            }
            push_keyed(input, keyed, &mut count, &mut stats, consumer, start, len);
            if RETAIN == 2 {
                pending[0] = pending[1];
            }
            pending[RETAIN - 1] = Some(current);
        }
    }

    flush_keyed(input, keyed, &mut count, &mut stats, consumer);
    pending[0].map_or(input.len(), |(start, _)| start)
}

/// Start of the trailing whitespace run whose grouping can still change when
/// a later newline or non-whitespace byte arrives.
#[inline]
fn trailing_whitespace_start(input: &str) -> usize {
    let mut start = input.len();
    for (index, ch) in input.char_indices().rev() {
        // SAFETY: `char` is a Unicode scalar value by construction.
        if unsafe { fast_unicode::class_of(ch as u32) } != fast_unicode::CharClass::Whitespace {
            break;
        }
        start = index;
    }
    start
}

#[inline(always)]
fn push_keyed<C: KeyedBatchConsumer>(
    input: &str,
    keyed: &mut [KeyedSpan],
    count: &mut usize,
    stats: &mut KeyedBatchStats,
    consumer: &mut C,
    start: usize,
    len: usize,
) {
    let key = PieceKey::from_span(input.as_bytes(), start, len);
    consumer.prefetch(key);
    keyed[*count] = KeyedSpan {
        key,
        start: u32::try_from(start).expect("stream chunk offset exceeds u32"),
        len: u32::try_from(len).expect("pretoken length exceeds u32"),
    };
    if C::PROFILE {
        stats.record(len);
    }
    *count += 1;
    if *count == KEYED_BATCH {
        flush_keyed(input, keyed, count, stats, consumer);
    }
}

#[inline]
fn flush_keyed<C: KeyedBatchConsumer>(
    input: &str,
    keyed: &mut [KeyedSpan],
    count: &mut usize,
    stats: &mut KeyedBatchStats,
    consumer: &mut C,
) {
    if *count == 0 {
        return;
    }
    consumer.consume(input, keyed, *count, *stats);
    *count = 0;
    *stats = KeyedBatchStats::default();
}

/// Adapt the callback-oriented DFA to the keyed cache interface. The fast
/// scanners fill large batches; this correctness fallback emits one piece.
fn dfa_feed_keyed<C: KeyedBatchConsumer>(stream: &mut DfaStream, chunk: &str, consumer: &mut C) {
    stream.feed(chunk, |piece| consume_dfa_piece(piece, consumer));
}

fn dfa_finish_keyed<C: KeyedBatchConsumer>(stream: &mut DfaStream, consumer: &mut C) {
    stream.finish(|piece| consume_dfa_piece(piece, consumer));
}

#[inline]
fn consume_dfa_piece<C: KeyedBatchConsumer>(piece: &str, consumer: &mut C) {
    let mut stats = KeyedBatchStats::default();
    if C::PROFILE {
        stats.record(piece.len());
    }
    let span = KeyedSpan {
        key: PieceKey::from_bytes(piece.as_bytes()),
        start: 0,
        len: u32::try_from(piece.len()).expect("pretoken length exceeds u32"),
    };
    let mut spans = [KeyedSpan::default(); 1 + KeyedSpan::PREFETCH_SLACK];
    spans[0] = span;
    consumer.prefetch(span.key);
    consumer.consume(piece, &mut spans, 1, stats);
}

#[cfg(test)]
mod tests {
    use super::{
        Encoding, FAST_FAMILY_LINEAR_THRESHOLD, FastFamily, FastFamilyStream, KEYED_BATCH,
        KeyedBatchConsumer, KeyedBatchStats, KeyedSpan, StreamImpl, StreamPretokenizer,
    };
    use crate::piece_cache::PieceKey;

    #[derive(Default)]
    struct Capture<const PROFILE: bool = true> {
        pieces: Vec<String>,
        prefetches: usize,
        batch_sizes: Vec<usize>,
    }

    impl<const PROFILE: bool> KeyedBatchConsumer for Capture<PROFILE> {
        const PROFILE: bool = PROFILE;

        fn prefetch(&mut self, _key: PieceKey) {
            self.prefetches += 1;
        }

        fn consume(
            &mut self,
            base: &str,
            spans: &mut [KeyedSpan],
            span_count: usize,
            stats: KeyedBatchStats,
        ) {
            assert!(span_count <= KEYED_BATCH);
            assert!(spans.len() >= span_count + KeyedSpan::PREFETCH_SLACK);
            assert!(self.prefetches >= self.pieces.len() + span_count);

            let mut expected_sum = 0u64;
            let mut expected_peak = 0usize;
            for span in &spans[..span_count] {
                let start = span.start as usize;
                let end = start + span.len as usize;
                assert!(end <= base.len());
                assert!(base.is_char_boundary(start) && base.is_char_boundary(end));
                assert_eq!(
                    span.key,
                    PieceKey::from_span(base.as_bytes(), start, span.len as usize)
                );
                let len = span.len as usize;
                expected_peak = expected_peak.max(len);
                expected_sum += len as u64 * (len as u64 + 1) / 2;
                self.pieces.push(base[start..end].to_owned());
            }
            if PROFILE {
                assert_eq!(stats.pieces, span_count);
                assert_eq!(stats.pretok_peak, expected_peak);
                assert_eq!(stats.pretok_sum, expected_sum);
            } else {
                assert_eq!(stats.pieces, 0);
                assert_eq!(stats.pretok_peak, 0);
                assert_eq!(stats.pretok_sum, 0);
            }
            self.batch_sizes.push(span_count);
        }
    }

    fn reference(enc: Encoding, input: &str) -> Vec<String> {
        let regex = fancy_regex::Regex::new(enc.split_pattern()).unwrap();
        regex
            .find_iter(input)
            .map(|piece| piece.unwrap().as_str().to_owned())
            .collect()
    }

    fn feed_at_most<const PROFILE: bool>(
        stream: &mut StreamPretokenizer,
        capture: &mut Capture<PROFILE>,
        input: &str,
        target: usize,
    ) {
        let mut start = 0;
        while start < input.len() {
            let mut end = (start + target).min(input.len());
            while end > start && !input.is_char_boundary(end) {
                end -= 1;
            }
            if end == start {
                end = input[start..]
                    .char_indices()
                    .nth(1)
                    .map_or(input.len(), |(offset, _)| start + offset);
            }
            stream.feed_keyed(&input[start..end], capture);
            start = end;
        }
        stream.finish_keyed(capture);
        assert!(capture.prefetches >= capture.pieces.len());
    }

    fn check(enc: Encoding, input: &str, targets: &[usize]) {
        let expected = reference(enc, input);
        for &target in targets {
            let mut stream = StreamPretokenizer::new(enc).unwrap();
            let mut capture = Capture::<true>::default();
            feed_at_most(&mut stream, &mut capture, input, target);
            assert_eq!(
                capture.pieces, expected,
                "{enc:?}, keyed chunk target {target}"
            );
            assert_eq!(capture.pieces.concat(), input);
        }
    }

    #[test]
    fn keyed_stream_matches_every_fast_encoding() {
        // More than 256 pieces, with every right-context-sensitive construct:
        // standalone and suffix contractions, 1/3-digit grouping, optional
        // prefix punctuation, CR/LF runs, and trailing horizontal whitespace.
        let ascii = concat!(
            "hello don't DON'T they'll WE'VE can'ts x'll'd 'sound ",
            "1234567 3'ts !word !!\r\n next  \n\tend | "
        )
        .repeat(48)
            + "trailing \n  \t";
        let mixed = concat!(
            "café CAFÉ 日本語 ١٢٣٤ １２３４ \u{a0}word ",
            "e\u{301}f ΑΒΓδε Привет Мир —dash\u{2028}\r\n x ",
            "中Éé \u{301}A中 "
        )
        .repeat(6)
            + " \n\u{a0}\t";

        for enc in [
            Encoding::R50k,
            Encoding::P50k,
            Encoding::Cl100k,
            Encoding::O200k,
            Encoding::Llama3,
            Encoding::Mistral,
            Encoding::Qwen3,
        ] {
            check(enc, &ascii, &[1, 2, 3, 7, 63, 64, 65, 257, ascii.len()]);
            check(enc, &mixed, &[1, 3, 31, 64, 257, mixed.len()]);

            let mut stream = StreamPretokenizer::new(enc).unwrap();
            let mut capture = Capture::<true>::default();
            stream.feed_keyed(&ascii, &mut capture);
            stream.finish_keyed(&mut capture);
            assert!(
                capture.batch_sizes.contains(&KEYED_BATCH),
                "{enc:?} never formed a full keyed batch"
            );

            let mut stream = StreamPretokenizer::new(enc).unwrap();
            let mut unprofiled = Capture::<false>::default();
            feed_at_most(&mut stream, &mut unprofiled, &ascii, 64);
            assert_eq!(
                unprofiled.pieces,
                reference(enc, &ascii),
                "{enc:?}, unprofiled keyed stream"
            );
        }
    }

    #[test]
    fn o200k_family_case_and_contraction_boundaries_cross_chunks() {
        let cases = [
            (Encoding::Mistral, "中Éé"),
            (Encoding::Mistral, "!中Éé"),
            (Encoding::O200k, "中Éé"),
            (Encoding::O200k, "É't"),
            (Encoding::O200k, "É're"),
            (Encoding::O200k, "中É're"),
        ];

        for (enc, input) in cases {
            let expected = reference(enc, input);
            for target in 1..=input.len() {
                let mut stream = StreamPretokenizer::new(enc).unwrap();
                let mut pieces = Vec::new();
                let mut start = 0;
                while start < input.len() {
                    let mut end = (start + target).min(input.len());
                    while end > start && !input.is_char_boundary(end) {
                        end -= 1;
                    }
                    if end == start {
                        end = input[start..]
                            .char_indices()
                            .nth(1)
                            .map_or(input.len(), |(offset, _)| start + offset);
                    }
                    stream.feed(&input[start..end], |piece| pieces.push(piece.to_owned()));
                    start = end;
                }
                stream.finish(|piece| pieces.push(piece.to_owned()));
                assert_eq!(pieces, expected, "{enc:?}, {input:?}, target {target}");

                let mut stream = StreamPretokenizer::new(enc).unwrap();
                let mut capture = Capture::<true>::default();
                feed_at_most(&mut stream, &mut capture, input, target);
                assert_eq!(
                    capture.pieces, expected,
                    "{enc:?}, keyed {input:?}, target {target}"
                );
            }
        }
    }

    const FAMILY_ENCODINGS: [Encoding; 7] = [
        Encoding::R50k,
        Encoding::P50k,
        Encoding::Cl100k,
        Encoding::O200k,
        Encoding::Llama3,
        Encoding::Mistral,
        Encoding::Qwen3,
    ];

    #[test]
    fn ordinary_family_chunks_retain_only_the_unsettled_suffix() {
        let chunk = "hello world 123! café 日本語 | ".repeat(2_048);
        let input = chunk.repeat(2);

        for enc in FAMILY_ENCODINGS {
            let mut stream = StreamPretokenizer::new(enc).unwrap();
            let mut pieces = Vec::new();
            for _ in 0..2 {
                stream.feed(&chunk, |piece| pieces.push(piece.to_owned()));
                let StreamImpl::Fast(fast) = &stream.inner else {
                    panic!("{enc:?} did not select the generic fast stream");
                };
                assert!(
                    fast.long.is_none(),
                    "{enc:?} activated the long-token fallback on ordinary text"
                );
                assert!(
                    fast.carry.len() < 256,
                    "{enc:?} retained {} bytes from a {}-byte chunk",
                    fast.carry.len(),
                    chunk.len()
                );
            }
            stream.finish(|piece| pieces.push(piece.to_owned()));
            assert_eq!(pieces, reference(enc, &input), "{enc:?}");
        }
    }

    #[test]
    fn multi_chunk_family_pretoken_uses_incremental_fallback() {
        let chunk = "a".repeat(FAST_FAMILY_LINEAR_THRESHOLD / 2);
        let chunks = 8usize;
        let expected = chunk.repeat(chunks);

        for enc in FAMILY_ENCODINGS {
            let mut stream = FastFamilyStream::new(FastFamily::from_encoding(enc));
            let mut pieces = Vec::new();
            for index in 0..chunks {
                stream.feed(&chunk, |piece| pieces.push(piece.to_owned()));
                assert!(pieces.is_empty(), "{enc:?} emitted the long word early");
                if index >= 1 {
                    let long = stream
                        .long
                        .as_ref()
                        .unwrap_or_else(|| panic!("{enc:?} did not activate linear carry"));
                    assert!(stream.carry.is_empty());
                    assert_eq!(long.buffered, (index + 1) * chunk.len());
                }
            }
            stream.finish(|piece| pieces.push(piece.to_owned()));
            assert_eq!(
                pieces.as_slice(),
                std::slice::from_ref(&expected),
                "{enc:?}"
            );
        }
    }

    #[test]
    fn keyed_multi_chunk_family_pretoken_uses_incremental_fallback() {
        let chunk = "a".repeat(FAST_FAMILY_LINEAR_THRESHOLD / 2);
        let chunks = 8usize;
        let expected = chunk.repeat(chunks);

        for enc in FAMILY_ENCODINGS {
            let mut stream = FastFamilyStream::new(FastFamily::from_encoding(enc));
            let mut capture = Capture::<true>::default();
            for index in 0..chunks {
                stream.feed_keyed(&chunk, &mut capture);
                assert!(
                    capture.pieces.is_empty(),
                    "{enc:?} emitted the long word early"
                );
                if index >= 1 {
                    let long = stream
                        .long
                        .as_ref()
                        .unwrap_or_else(|| panic!("{enc:?} did not activate linear carry"));
                    assert!(stream.carry.is_empty());
                    assert_eq!(long.buffered, (index + 1) * chunk.len());
                }
            }
            stream.finish_keyed(&mut capture);
            assert!(capture.prefetches >= capture.pieces.len());
            assert_eq!(
                capture.pieces.as_slice(),
                std::slice::from_ref(&expected),
                "{enc:?}"
            );
        }
    }

    #[test]
    fn long_family_bridge_hands_back_to_borrowed_chunk() {
        let head = "a".repeat(FAST_FAMILY_LINEAR_THRESHOLD);
        let tail = format!(
            "{} {}",
            "a".repeat(137),
            "hello don't 123! café 日本語 | ".repeat(128)
        );
        let input = format!("{head}{tail}");

        for enc in FAMILY_ENCODINGS {
            let mut stream = FastFamilyStream::new(FastFamily::from_encoding(enc));
            let mut pieces = Vec::new();
            stream.feed(&head[..head.len() / 2], |piece| {
                pieces.push(piece.to_owned())
            });
            stream.feed(&head[head.len() / 2..], |piece| {
                pieces.push(piece.to_owned())
            });
            assert!(stream.long.is_some(), "{enc:?}");

            stream.feed(&tail, |piece| pieces.push(piece.to_owned()));
            assert!(
                stream.long.is_none(),
                "{enc:?} did not leave the long-token fallback"
            );
            assert!(stream.carry.len() < 256, "{enc:?}");
            stream.finish(|piece| pieces.push(piece.to_owned()));
            assert_eq!(pieces, reference(enc, &input), "{enc:?}");

            let mut stream = FastFamilyStream::new(FastFamily::from_encoding(enc));
            let mut capture = Capture::<true>::default();
            stream.feed_keyed(&head[..head.len() / 2], &mut capture);
            stream.feed_keyed(&head[head.len() / 2..], &mut capture);
            assert!(stream.long.is_some(), "{enc:?}, keyed");
            stream.feed_keyed(&tail, &mut capture);
            assert!(
                stream.long.is_none(),
                "{enc:?} keyed path did not leave the long-token fallback"
            );
            assert!(stream.carry.len() < 256, "{enc:?}, keyed");
            stream.finish_keyed(&mut capture);
            assert!(capture.prefetches >= capture.pieces.len());
            assert_eq!(capture.pieces, reference(enc, &input), "{enc:?}, keyed");
        }
    }
}
