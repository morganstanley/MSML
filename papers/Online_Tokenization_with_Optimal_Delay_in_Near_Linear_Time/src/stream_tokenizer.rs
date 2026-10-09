//! Streaming tokenizer facade and optimized hardware engine.
//!
//! ```text
//!   byte chunks (file) | string (line)
//!     ▼  SpecialSegmenter          split on <|endoftext|>
//!     ▼  Normalizer    (optional)  NFC (qwen), carried incrementally across chunks
//!     ▼  Presplit      (optional)  isolate each digit (StarCoder)
//!     ▼  StreamPretokenizer        encoding-specific SIMD batches
//!     ▼  PieceCache                paired, prefetched cache probes
//!     ▼  FastBpeTables             specialized cold-miss merge kernel
//!   token ids
//! ```
//!
//! Input remains chunked and explicit output can be streamed to JSON or packed
//! little-endian `u32` pages. The optional pretoken cache grows with observed
//! distinct pieces, while input/output buffering stays bounded. The readable
//! byte-stepped DFA + MTC implementation is isolated in
//! `reference_stream_tokenizer.rs` and selected by the `_dfa` constructors.
//!
//! **Bounded memory in practice.** As implemented the pretokenizer buffers a
//! whole pre-token and BPE accumulates one pre-token's tokens, so those peaks
//! are O(pre-token), not the theoretical O(1)/O(T). On real text pre-tokens are
//! tiny, so the peaks stay small — see the `*_peak` fields of [`StreamStats`].

use std::{
    borrow::Borrow,
    sync::Arc,
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;

use crate::{
    AddedToken, Encoding, IncBpeTokenization, IncBpeTokenizer, Normalizer, Presplit,
    StreamPretokenizer, Vocab,
    fast_bpe::{FastBpeScratch, FastBpeTables},
    nfc::last_nfc_boundary,
    piece_cache::{
        PieceCache, PieceKey, ShortInsertSlot, packed_is_inline, packed_token_count,
        write_inline_lanes,
    },
    pretok::{KEYED_BATCH, KeyedBatchConsumer, KeyedBatchStats, KeyedSpan},
    reference_stream_tokenizer::ReferenceStreamEngine,
    runtime_vocab::RuntimeVocab,
    segment::{Seg, SegmenterConfig, SpecialSegmenter},
    stream_io::{
        ARRAY_DENSITY_SAMPLE_BYTES, CHUNK_SIZE, PACKED_U32_PAGE_IDS, PackedU32Page, TokenSink,
        read_chunks_while,
    },
};

/// Per-piece shortcut policy used by the streaming tokenizer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cache {
    /// Run the BPE merge for every piece.
    None,
    /// Return whole-piece vocabulary matches directly, then run BPE on misses.
    /// This is bounded by the immutable vocabulary.
    Lookup,
    /// Memoize each distinct pre-token's complete token sequence. Short keys
    /// and values use the packed cache; uncommon long entries spill to a map.
    Full,
}

/// Translate an internal merge-rank id to the model's output id.
///
/// Tiktoken models use rank == id. Converted Hugging Face models may provide a
/// relabel table; special-token ids bypass this helper and are emitted directly.
#[inline]
pub(crate) fn apply_relabel(rank_to_id: Option<&[u32]>, id: u32) -> u32 {
    match rank_to_id {
        Some(map) => map[id as usize],
        None => id,
    }
}

/// Detect a `rank_to_id` shaped by exactly one reserved rank, e.g.
/// `p50k_base` reserving `50256` for `<|endoftext|>`: identity below
/// `threshold`, then a constant `+ shift` above it.
///
/// This lets [`PieceEncoder::encode_keyed_batch_output`]'s branchless
/// direct-page path - otherwise limited to models with no relabel at all -
/// also cover this common single-gap case cheaply (a batched, branch-light
/// pass over just-written output) instead of falling back to the generic
/// per-span path for every pretoken. Multi-gap or otherwise irregular
/// relabels (arbitrary Hugging Face remaps) don't match this shape and
/// correctly fall through to `None`, leaving their behavior unchanged.
fn detect_affine_relabel(map: &[u32]) -> Option<(u32, u32)> {
    let threshold = map
        .iter()
        .enumerate()
        .find(|&(index, &rank)| index as u32 != rank)
        .map(|(index, _)| index as u32)?;
    let shift = map[threshold as usize] - threshold;
    map[threshold as usize..]
        .iter()
        .enumerate()
        .all(|(offset, &rank)| rank == threshold + offset as u32 + shift)
        .then_some((threshold, shift))
}

/// Initial capacity for a retained token vector. English-like r50k text
/// averages more than three input bytes per token; underestimates simply use
/// normal `Vec` growth for denser tokenizations.
#[inline]
fn token_capacity_hint(input_bytes: usize) -> usize {
    input_bytes.saturating_add(2) / 3
}

/// Per-run metrics returned by [`crate::StreamTokenizer::tokenize_file`].
pub struct StreamStats {
    pub total_tokens: usize,
    /// Elapsed time from the beginning of the chop operation until the first
    /// concrete token ID reaches the selected output sink.
    pub time_to_first_token: Option<Duration>,
    /// Number of `<|endoftext|>` special tokens emitted.
    pub specials: usize,
    pub total_bytes: usize,
    /// End-to-end wall time (read + segment + pretok + BPE).
    pub total: Duration,
    /// Peak bytes held back by added-token matching. Exact literals are bounded
    /// by the longest literal; `lstrip` may retain a trailing whitespace run.
    pub seg_peak: usize,
    /// Peak in-progress pre-token bytes buffered by the pretokenizer.
    pub pretok_peak: usize,
    /// Peak entries in the incremental-BPE state for a single pre-token.
    pub bpe_peak: usize,
    /// Average per-stage working set, as a **per-input-byte** time-average
    pub seg_avg: f64,
    pub pretok_avg: f64,
    pub bpe_avg: f64,
    /// Pre-tokens served from the cache, and the total pre-token count.
    pub cache_hits: usize,
    pub pieces: usize,
    /// Approximate heap footprint of `Cache::Full`, or zero for other modes.
    pub cache_bytes: usize,
}

/// Working-set/cache counters were useful during the original streaming-memory
/// investigation but distort first-token latency. Keep their instrumentation
/// available behind one compile-time switch while the public profile records
/// only TTFT and total output.
pub(crate) const PROFILE_WORKING_SET: bool = false;

#[derive(Default)]
struct Peaks {
    seg: usize,
    pretok: usize,
    bpe: usize,
}

/// Gigatoken's probe loop promotes cache lines from L2 to L1 this many
/// pretokens before use. The scanner has already issued the longer-distance
/// L2 prefetch while constructing the batch.
const CACHE_L1_PREFETCH_DISTANCE: usize = KeyedSpan::PREFETCH_SLACK;

struct PieceEncoder<T> {
    state: IncBpeTokenization<T>,
    fast_bpe: Option<Arc<FastBpeTables>>,
    fast_bpe_scratch: FastBpeScratch,
    piece_cache: Option<PieceCache>,
    scratch: Vec<u32>,
    emit_batch: Vec<u32>,
    cache: Cache,
    /// Hugging Face BPE semantic option, independent of the performance cache.
    ignore_merges: bool,
    cache_reserved_for: usize,
    peaks: Peaks,
    /// Per-input-byte integrals (numerators for the `*_avg` stats; `/ total_bytes`).
    pretok_sum: u64,
    bpe_sum: u64,
    cache_hits: usize,
    pieces: usize,
    /// Merge-rank -> output-id relabel (HF models remap ids); `None` = identity.
    /// Cached/looked-up ids store raw ranks and are relabeled on emit.
    rank_to_id: Option<Arc<[u32]>>,
    /// `rank_to_id`, when it is exactly a single-gap shift - see
    /// [`detect_affine_relabel`]. Lets the branchless direct-page path in
    /// `encode_keyed_batch_output` cover this case too.
    relabel_affine: Option<(u32, u32)>,
}

impl<T: Borrow<IncBpeTokenizer>> PieceEncoder<T> {
    fn new(
        state: IncBpeTokenization<T>,
        cache: Cache,
        vocab: &RuntimeVocab,
        rank_to_id: Option<Arc<[u32]>>,
        ignore_merges: bool,
        fast_bpe: Option<Arc<FastBpeTables>>,
    ) -> Self {
        let relabel_affine = rank_to_id.as_deref().and_then(detect_affine_relabel);
        Self {
            state,
            fast_bpe,
            fast_bpe_scratch: FastBpeScratch::new(),
            piece_cache: (cache == Cache::Full).then(|| PieceCache::seeded(vocab)),
            scratch: Vec::new(),
            emit_batch: Vec::with_capacity(KEYED_BATCH * 2),
            cache,
            ignore_merges,
            cache_reserved_for: 0,
            peaks: Peaks::default(),
            pretok_sum: 0,
            bpe_sum: 0,
            cache_hits: 0,
            pieces: 0,
            rank_to_id,
            relabel_affine,
        }
    }

    fn encode<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        sink: &mut TokenSink,
        piece: &str,
    ) {
        let key = if self.cache == Cache::Full {
            PieceKey::from_bytes(piece.as_bytes())
        } else {
            PieceKey::default()
        };
        self.encode_keyed::<PROFILE>(vocab, sink, piece, key);
    }

    #[inline]
    fn record_piece<const PROFILE: bool>(&mut self, piece_len: usize) {
        if PROFILE && PROFILE_WORKING_SET {
            self.pieces += 1;
            self.peaks.pretok = self.peaks.pretok.max(piece_len);
            let l = piece_len as u64;
            self.pretok_sum += l * (l + 1) / 2;
        }
    }

    #[inline]
    fn encode_keyed<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        sink: &mut TokenSink,
        piece: &str,
        key: PieceKey,
    ) {
        self.record_piece::<PROFILE>(piece.len());
        self.encode_keyed_unrecorded::<PROFILE>(vocab, sink, piece, key);
    }

    #[inline]
    fn encode_keyed_unrecorded<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        sink: &mut TokenSink,
        piece: &str,
        key: PieceKey,
    ) {
        let piece_bytes = piece.as_bytes();

        match self.cache {
            Cache::Full => {
                if key.is_short() {
                    let slot = match self
                        .piece_cache
                        .as_ref()
                        .expect("full cache must exist")
                        .get_or_short_slot(key)
                    {
                        Ok(cached) => {
                            if PROFILE && PROFILE_WORKING_SET {
                                self.cache_hits += 1;
                            }
                            sink.push_many_profiled::<PROFILE>(
                                cached.as_slice(),
                                self.rank_to_id.as_deref(),
                            );
                            return;
                        }
                        Err(slot) => slot,
                    };
                    self.encode_keyed_known_miss::<PROFILE>(vocab, sink, piece, key, Some(slot));
                    return;
                } else if let Some(cached) = self
                    .piece_cache
                    .as_ref()
                    .expect("full cache must exist")
                    .get_keyed(piece_bytes, key)
                {
                    if PROFILE && PROFILE_WORKING_SET {
                        self.cache_hits += 1;
                    }
                    sink.push_many_profiled::<PROFILE>(
                        cached.as_slice(),
                        self.rank_to_id.as_deref(),
                    );
                    return;
                }
            }
            Cache::Lookup => {
                if let Some(token) = vocab.lookup(piece_bytes) {
                    if PROFILE && PROFILE_WORKING_SET {
                        self.cache_hits += 1;
                    }
                    sink.push_profiled::<PROFILE>(apply_relabel(self.rank_to_id.as_deref(), token));
                    return;
                }
            }
            Cache::None => {
                if self.ignore_merges
                    && let Some(token) = vocab.lookup(piece_bytes)
                {
                    sink.push_profiled::<PROFILE>(apply_relabel(self.rank_to_id.as_deref(), token));
                    return;
                }
            }
        }

        self.encode_keyed_known_miss::<PROFILE>(vocab, sink, piece, key, None);
    }

    /// Encode a cache miss whose short-table insertion slot, when applicable,
    /// was already discovered by the exact cold lookup. Nothing in this body
    /// mutates the short table before that slot is consumed.
    #[inline(never)]
    fn encode_keyed_known_miss<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        sink: &mut TokenSink,
        piece: &str,
        key: PieceKey,
        short_slot: Option<ShortInsertSlot>,
    ) {
        let piece_bytes = piece.as_bytes();
        debug_assert!(self.cache != Cache::Full || key.is_short() == short_slot.is_some());

        self.scratch.clear();
        // Short vocabulary entries are seeded into the packed cache. Long
        // whole-vocabulary pieces take the existing lookup once and are then
        // memoized with the rest.
        if self.cache == Cache::Full
            && self.ignore_merges
            && !key.is_short()
            && let Some(token) = vocab.lookup(piece_bytes)
        {
            if PROFILE && PROFILE_WORKING_SET {
                self.cache_hits += 1;
            }
            self.scratch.push(token);
            sink.push_many_profiled::<PROFILE>(&self.scratch, self.rank_to_id.as_deref());
            let cache = self.piece_cache.as_mut().expect("full cache must exist");
            if let Some(slot) = short_slot {
                cache.insert_short_at(key, slot, &self.scratch);
            } else {
                cache.insert_long_known_absent(piece_bytes, &self.scratch);
            }
            return;
        }

        // Gigatoken's specialized pair-rank kernels win on English/code
        // pretokens. Hiriluk's MTC automaton is faster on multibyte text, so
        // retain it as the non-ASCII cold path.
        if piece_bytes.is_ascii()
            && let Some(fast_bpe) = self.fast_bpe.as_ref()
        {
            if PROFILE && PROFILE_WORKING_SET {
                let n = piece_bytes.len();
                self.peaks.bpe = self.peaks.bpe.max(n);
                let n = n as u64;
                self.bpe_sum += n * (n + 1) / 2;
            }
            let tokens = fast_bpe.encode(&mut self.fast_bpe_scratch, piece_bytes);
            sink.push_many_profiled::<PROFILE>(tokens, self.rank_to_id.as_deref());
            if self.cache == Cache::Full {
                let cache = self.piece_cache.as_mut().expect("full cache must exist");
                if let Some(slot) = short_slot {
                    cache.insert_short_at(key, slot, tokens);
                } else {
                    cache.insert_long_known_absent(piece_bytes, tokens);
                }
            }
            return;
        }

        self.state.reset();
        if PROFILE && PROFILE_WORKING_SET {
            let mut bpe_sum = 0u64;
            let mut bpe_peak = 0usize;
            for token_id in vocab.split_bytes_to_tokens_unchecked(piece_bytes) {
                // Drop bytes with no base token (incomplete byte-level vocab, e.g.
                // StarCoder); matches the batch path and the reference tokenizer.
                if token_id != crate::TokenId::MAX {
                    self.state.feed(token_id);
                    let frontier = self.state.inc_tokens().len();
                    bpe_peak = bpe_peak.max(frontier);
                    bpe_sum += frontier as u64;
                }
            }
            // Cache-hit pieces feed nothing and therefore contribute zero. Commit
            // the cold-path measurements once after the complete pretoken.
            self.peaks.bpe = self.peaks.bpe.max(bpe_peak);
            self.bpe_sum += bpe_sum;
        } else {
            for token_id in vocab.split_bytes_to_tokens_unchecked(piece_bytes) {
                if token_id != crate::TokenId::MAX {
                    self.state.feed(token_id);
                }
            }
        }
        // The chain iterator walks backward from the end, so collect then reverse.
        self.scratch.extend(
            self.state
                .current_token_chain()
                .token_ids()
                .map(|t| t.as_usize() as u32),
        );
        self.scratch.reverse();
        sink.push_many_profiled::<PROFILE>(&self.scratch, self.rank_to_id.as_deref());
        if self.cache == Cache::Full {
            let cache = self.piece_cache.as_mut().expect("full cache must exist");
            if let Some(slot) = short_slot {
                cache.insert_short_at(key, slot, &self.scratch);
            } else {
                cache.insert_long_known_absent(piece_bytes, &self.scratch);
            }
        }
    }

    /// Resolve one direct-output entry rejected by the branchless home-pair
    /// probe without leaving the current packed page.
    #[inline(never)]
    fn append_keyed_slow_to_page<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        page: &mut PackedU32Page<'_>,
        piece: &str,
        key: PieceKey,
    ) {
        if key.is_short() {
            match self
                .piece_cache
                .as_ref()
                .expect("keyed batches require the full cache")
                .get_or_short_slot(key)
            {
                Ok(cached) => {
                    page.extend_from_slice(cached.as_slice());
                    if PROFILE && PROFILE_WORKING_SET {
                        self.cache_hits += 1;
                    }
                }
                Err(slot) => self.append_keyed_known_miss_to_page::<PROFILE>(
                    vocab,
                    page,
                    piece,
                    key,
                    Some(slot),
                ),
            }
        } else if let Some(cached) = self
            .piece_cache
            .as_ref()
            .expect("keyed batches require the full cache")
            .get_keyed(piece.as_bytes(), key)
        {
            page.extend_from_slice(cached.as_slice());
            if PROFILE && PROFILE_WORKING_SET {
                self.cache_hits += 1;
            }
        } else {
            self.append_keyed_known_miss_to_page::<PROFILE>(vocab, page, piece, key, None);
        }
    }

    /// Direct-page counterpart of [`Self::encode_keyed_known_miss`].
    /// Identity-mapped encodings can append the cold result straight into the
    /// same bounded/retained u32 page used by cache hits.
    #[inline(never)]
    fn append_keyed_known_miss_to_page<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        page: &mut PackedU32Page<'_>,
        piece: &str,
        key: PieceKey,
        short_slot: Option<ShortInsertSlot>,
    ) {
        let piece_bytes = piece.as_bytes();
        debug_assert!(self.cache == Cache::Full);
        debug_assert_eq!(key.is_short(), short_slot.is_some());
        // Callers are the direct-page path's fast-hit miss handler and its
        // batched sibling, gated on identity or an affine (single-gap)
        // relabel; either way this writes raw compact ids, corrected in bulk
        // by the caller afterward.
        debug_assert!(self.rank_to_id.is_none() || self.relabel_affine.is_some());

        self.scratch.clear();
        if self.ignore_merges
            && !key.is_short()
            && let Some(token) = vocab.lookup(piece_bytes)
        {
            if PROFILE && PROFILE_WORKING_SET {
                self.cache_hits += 1;
            }
            self.scratch.push(token);
            page.extend_from_slice(&self.scratch);
            self.piece_cache
                .as_mut()
                .expect("full cache must exist")
                .insert_long_known_absent(piece_bytes, &self.scratch);
            return;
        }

        if piece_bytes.is_ascii()
            && let Some(fast_bpe) = self.fast_bpe.as_ref()
        {
            if PROFILE && PROFILE_WORKING_SET {
                let n = piece_bytes.len();
                self.peaks.bpe = self.peaks.bpe.max(n);
                let n = n as u64;
                self.bpe_sum += n * (n + 1) / 2;
            }
            let tokens = fast_bpe.encode(&mut self.fast_bpe_scratch, piece_bytes);
            page.extend_from_slice(tokens);
            let cache = self.piece_cache.as_mut().expect("full cache must exist");
            if let Some(slot) = short_slot {
                cache.insert_short_at(key, slot, tokens);
            } else {
                cache.insert_long_known_absent(piece_bytes, tokens);
            }
            return;
        }

        self.state.reset();
        if PROFILE && PROFILE_WORKING_SET {
            let mut bpe_sum = 0u64;
            let mut bpe_peak = 0usize;
            for token_id in vocab.split_bytes_to_tokens_unchecked(piece_bytes) {
                if token_id != crate::TokenId::MAX {
                    self.state.feed(token_id);
                    let frontier = self.state.inc_tokens().len();
                    bpe_peak = bpe_peak.max(frontier);
                    bpe_sum += frontier as u64;
                }
            }
            self.peaks.bpe = self.peaks.bpe.max(bpe_peak);
            self.bpe_sum += bpe_sum;
        } else {
            for token_id in vocab.split_bytes_to_tokens_unchecked(piece_bytes) {
                if token_id != crate::TokenId::MAX {
                    self.state.feed(token_id);
                }
            }
        }
        self.scratch.extend(
            self.state
                .current_token_chain()
                .token_ids()
                .map(|token| token.as_usize() as u32),
        );
        self.scratch.reverse();
        page.extend_from_slice(&self.scratch);
        let cache = self.piece_cache.as_mut().expect("full cache must exist");
        if let Some(slot) = short_slot {
            cache.insert_short_at(key, slot, &self.scratch);
        } else {
            cache.insert_long_known_absent(piece_bytes, &self.scratch);
        }
    }

    #[inline]
    fn record_keyed_batch<const PROFILE: bool>(&mut self, stats: KeyedBatchStats) {
        if PROFILE && PROFILE_WORKING_SET {
            self.pieces += stats.pieces;
            self.pretok_sum += stats.pretok_sum;
            self.peaks.pretok = self.peaks.pretok.max(stats.pretok_peak);
        }
    }

    /// Consume one scanner-produced batch. The first flat pass packs keys,
    /// hashes them, and stages their cache lines; the second probes only after
    /// the prefetches have had a full batch of independent work to complete.
    fn encode_keyed_batch<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        sink: &mut TokenSink,
        base: &str,
        spans: &mut [KeyedSpan],
        span_count: usize,
        stats: KeyedBatchStats,
    ) {
        debug_assert!(spans.len() >= span_count + CACHE_L1_PREFETCH_DISTANCE);
        self.record_keyed_batch::<PROFILE>(stats);
        self.encode_keyed_batch_output::<PROFILE>(vocab, sink, base, spans, span_count);
    }

    #[inline(never)]
    fn encode_keyed_batch_output<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        sink: &mut TokenSink,
        base: &str,
        spans: &mut [KeyedSpan],
        span_count: usize,
    ) {
        // r50k/tiktoken uses identity IDs. In packed-output mode, write cache
        // lanes straight into the sink's bounded page: this is Gigatoken's
        // flat probe-and-emit loop with a reusable 4 MiB destination instead
        // of an input-sized Vec. JSON and irregularly-relabeled HF models
        // retain the generic ordered path below; models relabeled only by a
        // single reserved rank (e.g. p50k_base's `50256`) still take this
        // path and get that shift applied in bulk afterward - seeing every
        // pretoken batch through the per-span path just for a one-rank
        // offset cost far more than the batched fix-up below.
        if (self.rank_to_id.is_none() || self.relabel_affine.is_some()) && span_count != 0 {
            let first = spans[0];
            let last = spans[span_count - 1];
            let batch_bytes = last.start as usize + last.len as usize - first.start as usize;
            // A byte-level BPE emits at most one token per input byte. Four
            // extra lanes cover the branchless probe's final unconditional
            // store even when the final pretoken emits fewer than four IDs.
            let required_spare = batch_bytes.saturating_add(4);
            if required_spare <= PACKED_U32_PAGE_IDS {
                let mut direct_hits = 0usize;
                let profile_start = sink.profile_start::<PROFILE>();
                let mut first_token_elapsed = None;
                let completed = sink.with_u32_page(required_spare, |page| {
                    let batch_start = page.len();
                    let mut cursor = batch_start;
                    let mut dst = page.as_mut_ptr();
                    let mut cache_view = self
                        .piece_cache
                        .as_ref()
                        .expect("keyed batches require the full cache")
                        .output_probe_view();
                    for index in 0..CACHE_L1_PREFETCH_DISTANCE {
                        // SAFETY: every keyed batch carries sixteen readable
                        // lookahead records after its logical prefix.
                        cache_view.prefetch_l1(unsafe { spans.get_unchecked(index).key });
                    }

                    for index in 0..span_count {
                        let prefetch_key =
                            unsafe { spans.get_unchecked(index + CACHE_L1_PREFETCH_DISTANCE).key };
                        cache_view.prefetch_l1(prefetch_key);
                        let span = unsafe { *spans.get_unchecked(index) };
                        let (val, ext, found) = cache_view.probe_pair(span.key);
                        let fast = found & packed_is_inline(val);
                        if fast
                            && PROFILE
                            && profile_start.is_some()
                            && first_token_elapsed.is_none()
                        {
                            first_token_elapsed = profile_start.map(|start| start.elapsed());
                        }
                        // A rejected probe leaves dead lanes at `cursor`;
                        // the outlined slow path overwrites them in place.
                        unsafe { write_inline_lanes(dst.add(cursor), val, ext) };
                        if fast {
                            cursor += packed_token_count(val);
                            direct_hits += usize::from(PROFILE && PROFILE_WORKING_SET);
                            continue;
                        }

                        // Synchronize the writable view before using its safe
                        // append operation. A miss may grow and move the cache,
                        // so refresh both raw snapshots afterwards.
                        unsafe { page.set_len(cursor) };
                        let start = span.start as usize;
                        let end = start + span.len as usize;
                        let output_start = page.len();
                        self.append_keyed_slow_to_page::<PROFILE>(
                            vocab,
                            page,
                            &base[start..end],
                            span.key,
                        );
                        if PROFILE
                            && profile_start.is_some()
                            && first_token_elapsed.is_none()
                            && page.len() > output_start
                        {
                            first_token_elapsed = profile_start.map(|start| start.elapsed());
                        }
                        cursor = page.len();
                        dst = page.as_mut_ptr();
                        cache_view = self
                            .piece_cache
                            .as_ref()
                            .expect("keyed batches require the full cache")
                            .output_probe_view();
                    }
                    if let Some((threshold, shift)) = self.relabel_affine {
                        // Every lane in `[batch_start, cursor)` is now a
                        // finished, valid compact id (dead speculative lanes
                        // from rejected probes always sit at or past
                        // `cursor`, never inside it), so the whole range can
                        // be corrected in one pass.
                        //
                        // Take a `&mut [u32]` rather than walking raw
                        // pointers, and add a masked `shift` unconditionally
                        // instead of storing under an `if`. Both matter: the
                        // slice gives LLVM the `noalias` it needs to
                        // vectorize, and an unconditional store keeps the
                        // body branchless. The raw-pointer conditional-store
                        // version of this loop measured 16.6% of total
                        // GitHub p50k throughput; this one is ~1%.
                        //
                        // SAFETY: `batch_start <= cursor <= capacity`, and
                        // every lane in that range was initialized above.
                        let lanes = unsafe {
                            std::slice::from_raw_parts_mut(
                                dst.add(batch_start),
                                cursor - batch_start,
                            )
                        };
                        for id in lanes {
                            *id += u32::from(*id >= threshold) * shift;
                        }
                    }
                    // SAFETY: `required_spare` covers every real output ID
                    // plus the final dead four-lane store.
                    unsafe { page.set_len(cursor) };
                });
                if completed.is_some() {
                    sink.record_first_token_elapsed::<PROFILE>(first_token_elapsed);
                    if PROFILE && PROFILE_WORKING_SET {
                        self.cache_hits += direct_hits;
                    }
                    return;
                }
            }
        }

        let mut hit_count = 0usize;
        self.emit_batch.clear();
        for span in spans[..span_count].iter() {
            let mut short_slot = None;
            let cached = if span.key.is_short() {
                match self
                    .piece_cache
                    .as_ref()
                    .expect("keyed batches require the full cache")
                    .get_or_short_slot(span.key)
                {
                    Ok(cached) => Some(cached),
                    Err(slot) => {
                        short_slot = Some(slot);
                        None
                    }
                }
            } else {
                let start = span.start as usize;
                let end = start + span.len as usize;
                self.piece_cache
                    .as_ref()
                    .expect("keyed batches require the full cache")
                    .get_keyed(&base.as_bytes()[start..end], span.key)
            };
            if let Some(cached) = cached {
                if !cached.as_slice().is_empty() {
                    // `emit_batch` defers the physical sink write so adjacent
                    // hits can be emitted in bulk. TTFT is the moment the
                    // first concrete ID is available, not the later batch
                    // flush.
                    sink.mark_first_token::<PROFILE>();
                }
                match self.rank_to_id.as_deref() {
                    Some(map) => self
                        .emit_batch
                        .extend(cached.as_slice().iter().map(|&id| map[id as usize])),
                    None => self.emit_batch.extend_from_slice(cached.as_slice()),
                }
                if PROFILE && PROFILE_WORKING_SET {
                    hit_count += 1;
                }
            } else {
                // Keep order while letting long runs of cache hits reach the
                // packed writer in one contiguous operation.
                sink.push_many_profiled::<PROFILE>(&self.emit_batch, None);
                self.emit_batch.clear();
                let start = span.start as usize;
                let end = start + span.len as usize;
                let piece = &base[start..end];
                self.encode_keyed_known_miss::<PROFILE>(vocab, sink, piece, span.key, short_slot);
            }
        }
        sink.push_many_profiled::<PROFILE>(&self.emit_batch, None);
        self.emit_batch.clear();
        if PROFILE && PROFILE_WORKING_SET {
            self.cache_hits += hit_count;
        }
    }

    #[inline]
    fn cache_bytes(&self) -> usize {
        let piece_cache = self
            .piece_cache
            .as_ref()
            .map_or(0, PieceCache::estimated_bytes);
        let merge_tables = self
            .fast_bpe
            .as_ref()
            .map_or(0, |tables| tables.estimated_bytes());
        piece_cache + merge_tables
    }

    fn reserve_cache_for_bytes(&mut self, expected_bytes: usize) {
        if self.cache == Cache::Full && expected_bytes > self.cache_reserved_for {
            self.piece_cache
                .as_mut()
                .expect("full cache must exist")
                .reserve_for_bytes(expected_bytes);
            self.cache_reserved_for = expected_bytes;
        }
    }
}

struct EncoderBatchConsumer<'a, 'v, T, const PROFILE: bool> {
    encoder: &'a mut PieceEncoder<T>,
    vocab: &'v RuntimeVocab,
    sink: &'a mut TokenSink,
}

impl<T: Borrow<IncBpeTokenizer>, const PROFILE: bool> KeyedBatchConsumer
    for EncoderBatchConsumer<'_, '_, T, PROFILE>
{
    const PROFILE: bool = PROFILE && PROFILE_WORKING_SET;

    #[inline(always)]
    fn prefetch(&mut self, key: PieceKey) {
        let cache = self
            .encoder
            .piece_cache
            .as_ref()
            .expect("keyed batches require the full cache");
        cache.prefetch_l2(key);
    }

    #[inline]
    fn consume(
        &mut self,
        base: &str,
        spans: &mut [KeyedSpan],
        span_count: usize,
        stats: KeyedBatchStats,
    ) {
        self.encoder
            .encode_keyed_batch::<PROFILE>(self.vocab, self.sink, base, spans, span_count, stats);
    }
}

#[inline]
fn feed_pretokens<T: Borrow<IncBpeTokenizer>, const PROFILE: bool>(
    text: &str,
    stream: &mut StreamPretokenizer,
    encoder: &mut PieceEncoder<T>,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    if encoder.cache == Cache::Full {
        let mut consumer = EncoderBatchConsumer::<_, PROFILE> {
            encoder,
            vocab,
            sink,
        };
        stream.feed_keyed(text, &mut consumer);
    } else {
        stream.feed(text, |piece| encoder.encode::<PROFILE>(vocab, sink, piece));
    }
}

#[inline]
fn finish_pretokens<T: Borrow<IncBpeTokenizer>, const PROFILE: bool>(
    stream: &mut StreamPretokenizer,
    encoder: &mut PieceEncoder<T>,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    if encoder.cache == Cache::Full {
        let mut consumer = EncoderBatchConsumer::<_, PROFILE> {
            encoder,
            vocab,
            sink,
        };
        stream.finish_keyed(&mut consumer);
    } else {
        stream.finish(|piece| encoder.encode::<PROFILE>(vocab, sink, piece));
    }
}

/// Feed one already-normalized text span through pre-splitting,
/// pretokenization, and BPE.
fn feed_text<T: Borrow<IncBpeTokenizer>, const PROFILE: bool>(
    text: &str,
    presplit: Presplit,
    stream: &mut StreamPretokenizer,
    encoder: &mut PieceEncoder<T>,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    match presplit {
        Presplit::None => feed_pretokens::<_, PROFILE>(text, stream, encoder, vocab, sink),
        Presplit::DigitsIndividual => presplit.split(text, |sub| {
            if sub.chars().next().is_some_and(char::is_numeric) {
                finish_pretokens::<_, PROFILE>(stream, encoder, vocab, sink);
                feed_pretokens::<_, PROFILE>(sub, stream, encoder, vocab, sink);
                finish_pretokens::<_, PROFILE>(stream, encoder, vocab, sink);
            } else {
                feed_pretokens::<_, PROFILE>(sub, stream, encoder, vocab, sink);
            }
        }),
    }
}

/// Normalize + emit any buffered NFC carry (a no-op when there is no carry, e.g.
/// [`Normalizer::None`]). Called at every hard NFC boundary — special tokens and
/// EOF — where the carry is a complete segment.
fn flush_carry<T: Borrow<IncBpeTokenizer>, const PROFILE: bool>(
    normalizer: Normalizer,
    carry: &mut String,
    presplit: Presplit,
    stream: &mut StreamPretokenizer,
    encoder: &mut PieceEncoder<T>,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    if !carry.is_empty() {
        let norm = normalizer.normalize(carry).into_owned();
        feed_text::<_, PROFILE>(&norm, presplit, stream, encoder, vocab, sink);
        carry.clear();
    }
}

/// Dispatch one segmenter event into the NFC → pre-split → pretokenizer → BPE
/// pipeline.
fn dispatch<T: Borrow<IncBpeTokenizer>, const PROFILE: bool>(
    ev: Seg,
    normalizer: Normalizer,
    carry: &mut String,
    presplit: Presplit,
    stream: &mut StreamPretokenizer,
    encoder: &mut PieceEncoder<T>,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    match ev {
        Seg::Text(t) => match normalizer {
            // Fast path: no normalization, feed the span straight through.
            Normalizer::None => feed_text::<_, PROFILE>(t, presplit, stream, encoder, vocab, sink),
            Normalizer::Nfc => {
                carry.push_str(t);
                let k = last_nfc_boundary(carry);
                if k > 0 {
                    let norm = normalizer.normalize(&carry[..k]).into_owned();
                    feed_text::<_, PROFILE>(&norm, presplit, stream, encoder, vocab, sink);
                    carry.drain(..k);
                }
            }
        },
        Seg::Special(id) => {
            flush_carry::<_, PROFILE>(normalizer, carry, presplit, stream, encoder, vocab, sink);
            finish_pretokens::<_, PROFILE>(stream, encoder, vocab, sink);
            sink.push_special_profiled::<PROFILE>(id);
        }
    }
}

/// Low-level streaming engine over one compiled encoding and BPE dictionary.
///
/// Most callers should use the named [`crate::StreamTokenizer`] facade. This
/// type remains available for benchmarks and tests that need explicit cache or
/// scanner construction.
#[doc(hidden)]
struct FastStreamEngine {
    vocab: Arc<RuntimeVocab>,
    /// Applied to each text span before pre-tokenization (e.g. NFC for qwen),
    /// incrementally across chunks. [`Normalizer::None`] is a pass-through.
    normalizer: Normalizer,
    /// Pre-split stage between the normalizer and the pattern pretokenizer (e.g.
    /// StarCoder's per-digit split). [`Presplit::None`] is a pass-through.
    presplit: Presplit,
    /// Reusable pretokenizer state.
    stream: StreamPretokenizer,
    seg: SpecialSegmenter,
    encoder: PieceEncoder<Arc<IncBpeTokenizer>>,
}

impl FastStreamEngine {
    /// Build a tokenizer with the encoding's built-in tiktoken specials and no
    /// id relabel (the tiktoken case: id == rank).
    ///
    /// * `enc` selects the streaming split pattern and the special tokens.
    /// * `vocab`/`tokenizer`/`ranks` are the loaded BPE dictionary.
    /// * `cache` is the per-piece shortcut strategy.
    pub fn new(
        enc: Encoding,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab = Arc::new(RuntimeVocab::from_legacy(vocab.as_ref(), ranks.as_ref()));
        Self::new_runtime(enc, vocab, tokenizer, cache)
    }

    /// Build directly from the compact immutable runtime vocabulary.
    pub(crate) fn new_runtime(
        enc: Encoding,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_specials_runtime(
            enc,
            enc.special_tokens(),
            None,
            Normalizer::None,
            Presplit::None,
            false,
            vocab,
            tokenizer,
            cache,
        )
    }

    /// Build with explicit model semantics and added-token configuration.
    /// `ignore_merges` is a tokenizer semantic, not a cache selection.
    #[allow(clippy::too_many_arguments)]
    pub fn with_specials(
        enc: Encoding,
        specials: &[(&str, u32)],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab = Arc::new(RuntimeVocab::from_legacy(vocab.as_ref(), ranks.as_ref()));
        Self::with_specials_runtime(
            enc,
            specials,
            rank_to_id.map(Arc::from),
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            cache,
        )
    }

    /// Build with explicit model semantics from the compact runtime
    /// vocabulary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_specials_runtime(
        enc: Encoding,
        specials: &[(&str, u32)],
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_segmenter_config_runtime(
            enc,
            SegmenterConfig::from_specials(specials),
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            cache,
        )
    }

    /// [`with_specials`](Self::with_specials) with full Hugging Face
    /// added-token boundary and whitespace semantics.
    #[allow(clippy::too_many_arguments)]
    pub fn with_added_tokens(
        enc: Encoding,
        added_tokens: &[AddedToken],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab = Arc::new(RuntimeVocab::from_legacy(vocab.as_ref(), ranks.as_ref()));
        Self::with_added_tokens_runtime(
            enc,
            added_tokens,
            rank_to_id.map(Arc::from),
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            cache,
        )
    }

    /// Build with Hugging Face added-token semantics from the compact runtime
    /// vocabulary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_added_tokens_runtime(
        enc: Encoding,
        added_tokens: &[AddedToken],
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_segmenter_config_runtime(
            enc,
            SegmenterConfig::from_added_tokens(added_tokens),
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            cache,
        )
    }

    /// Build from an added-token program compiled once during model loading.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_segmenter_config_runtime(
        enc: Encoding,
        segmenter_config: SegmenterConfig,
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_segmenter_runtime(
            enc,
            SpecialSegmenter::from_config(segmenter_config),
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_segmenter_runtime(
        enc: Encoding,
        seg: SpecialSegmenter,
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encoder = PieceEncoder::new(
            IncBpeTokenization::new(tokenizer),
            cache,
            vocab.as_ref(),
            rank_to_id,
            ignore_merges,
            None,
        );
        Ok(Self {
            vocab,
            normalizer,
            presplit,
            stream: StreamPretokenizer::new(enc)?,
            seg,
            encoder,
        })
    }

    /// Install immutable pair-rank tables shared by named optimized
    /// tokenizers. Low-level constructors retain MTC as their conservative
    /// fallback when a dictionary does not satisfy the fast-kernel invariants.
    pub(crate) fn install_fast_bpe(&mut self, tables: Option<Arc<FastBpeTables>>) {
        self.encoder.fast_bpe = tables;
    }

    /// Tokenize one in-memory string as an independent document, pushing its
    /// token ids into `sink` (which may be shared across many calls — counts and
    /// any configured output accumulates). The pre-tokenizer and BPE are
    /// flushed at the end, so each call is self-contained, but the built
    /// pipeline is reused. Profiling is disabled on this default entry point.
    pub fn tokenize_string(&mut self, text: &str, sink: &mut TokenSink) {
        self.tokenize_string_with_sink::<false>(text, sink);
    }

    /// Unprofiled string entry point for crate-owned output transports.
    pub(crate) fn tokenize_string_unprofiled_into(&mut self, text: &str, sink: &mut TokenSink) {
        self.tokenize_string_with_sink::<false>(text, sink);
    }

    fn tokenize_string_with_sink<const PROFILE: bool>(&mut self, text: &str, sink: &mut TokenSink) {
        self.encoder.reserve_cache_for_bytes(text.len());
        let output_tokens_start = sink.total();
        let capacity_sample_at = text.len().min(ARRAY_DENSITY_SAMPLE_BYTES);
        let Self {
            vocab,
            normalizer,
            presplit,
            stream,
            seg,
            encoder,
        } = self;
        // Split-borrow of `self`: copy the shared refs out (`&T` is `Copy`);
        // `stream`/`seg`/`encoder` stay `&mut`.
        let vocab = vocab.as_ref();
        let (normalizer, presplit) = (*normalizer, *presplit);
        let mut carry = String::new();
        let mut start = 0;
        while start < text.len() && !sink.is_cancelled() {
            let mut end = (start + CHUNK_SIZE).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            let chunk = &text[start..end];
            if PROFILE && PROFILE_WORKING_SET {
                seg.feed(chunk, |ev| {
                    dispatch::<_, PROFILE>(
                        ev, normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
                    )
                });
            } else {
                seg.feed_unprofiled(chunk, |ev| {
                    dispatch::<_, PROFILE>(
                        ev, normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
                    )
                });
            }
            if start < capacity_sample_at && end >= capacity_sample_at {
                sink.adapt_memory_u32_capacity(
                    end,
                    text.len(),
                    sink.total().saturating_sub(output_tokens_start),
                );
            }
            start = end;
        }
        if sink.is_cancelled() {
            return;
        }
        seg.finish(|ev| {
            dispatch::<_, PROFILE>(
                ev, normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
            )
        });
        flush_carry::<_, PROFILE>(
            normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
        );
        finish_pretokens::<_, PROFILE>(stream, encoder, vocab, sink);
    }

    pub(crate) fn tokenize_string_profiled_into(
        &mut self,
        text: &str,
        sink: &mut TokenSink,
    ) -> StreamStats {
        self.tokenize_string_with_sink::<true>(text, sink);

        StreamStats {
            total_tokens: sink.total(),
            time_to_first_token: sink.time_to_first_token(),
            specials: sink.specials(),
            total_bytes: text.len(),
            total: Duration::ZERO,
            seg_peak: 0,
            pretok_peak: 0,
            bpe_peak: 0,
            seg_avg: 0.0,
            pretok_avg: 0.0,
            bpe_avg: 0.0,
            cache_hits: 0,
            pieces: 0,
            cache_bytes: 0,
        }
    }

    /// Unprofiled file entry point for crate-owned output transports.
    pub(crate) fn tokenize_file_unprofiled_into(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        Ok(self
            .tokenize_file_into_sink::<false>(path, sink)?
            .total_tokens)
    }

    /// Profiled file entry point for crate-owned output transports.
    pub(crate) fn tokenize_file_profiled_into(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        self.tokenize_file_into_sink::<true>(path, sink)
    }

    fn tokenize_file_into_sink<const PROFILE: bool>(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        // Like Gigatoken's serial `fork_sized`, size the fast cache from the
        // known workload before the tokenizer's optional internal timer.
        // Whether the cache is reused is a caller policy; comparison
        // benchmarks construct a fresh tokenizer for every repetition.
        let expected_bytes = std::fs::metadata(path)?.len() as usize;
        self.encoder.reserve_cache_for_bytes(expected_bytes);

        let Self {
            vocab,
            normalizer,
            presplit,
            stream,
            seg,
            encoder,
        } = self;
        // Split-borrow of `self`: copy the shared refs out (`&T` is `Copy`);
        // `stream`/`seg`/`encoder` stay `&mut`.
        let vocab = vocab.as_ref();
        let (normalizer, presplit) = (*normalizer, *presplit);
        let pieces_start = if PROFILE && PROFILE_WORKING_SET {
            encoder.pieces
        } else {
            0
        };
        let cache_hits_start = if PROFILE && PROFILE_WORKING_SET {
            encoder.cache_hits
        } else {
            0
        };
        let pretok_sum_start = if PROFILE && PROFILE_WORKING_SET {
            encoder.pretok_sum
        } else {
            0
        };
        let bpe_sum_start = if PROFILE && PROFILE_WORKING_SET {
            encoder.bpe_sum
        } else {
            0
        };
        let seg_sum_start = if PROFILE && PROFILE_WORKING_SET {
            seg.held_sum()
        } else {
            0
        };
        if PROFILE && PROFILE_WORKING_SET {
            encoder.peaks = Peaks::default();
            seg.reset_peak();
        }
        // Trailing un-normalized combining sequence carried across chunks (NFC);
        // empty for `Normalizer::None`.
        let mut carry = String::new();
        let output_tokens_start = sink.total();
        let mut input_bytes_seen = 0usize;
        let capacity_sample_at = expected_bytes.min(ARRAY_DENSITY_SAMPLE_BYTES);

        let (total_bytes, completed) = read_chunks_while(path, |chunk| {
            if PROFILE && PROFILE_WORKING_SET {
                seg.feed(chunk, |ev| {
                    dispatch::<_, PROFILE>(
                        ev, normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
                    )
                });
            } else {
                seg.feed_unprofiled(chunk, |ev| {
                    dispatch::<_, PROFILE>(
                        ev, normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
                    )
                });
            }
            let previous_input_bytes = input_bytes_seen;
            input_bytes_seen = input_bytes_seen.saturating_add(chunk.len());
            if previous_input_bytes < capacity_sample_at && input_bytes_seen >= capacity_sample_at {
                sink.adapt_memory_u32_capacity(
                    input_bytes_seen,
                    expected_bytes,
                    sink.total().saturating_sub(output_tokens_start),
                );
            }
            !sink.is_cancelled()
        })?;
        if !completed {
            return Err(sink
                .take_error()
                .unwrap_or_else(|| {
                    std::io::Error::other("token output stopped without a recorded error")
                })
                .into());
        }

        // EOF: a well-formed file leaves no segmenter carry; any stray trailing
        // bytes are not a valid character, so the document ends here. Flush the
        // NFC carry (a complete segment at EOF), then finalize the pretokenizer.
        seg.finish(|ev| {
            dispatch::<_, PROFILE>(
                ev, normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
            )
        });
        flush_carry::<_, PROFILE>(
            normalizer, &mut carry, presplit, stream, encoder, vocab, sink,
        );
        finish_pretokens::<_, PROFILE>(stream, encoder, vocab, sink);

        if PROFILE && PROFILE_WORKING_SET {
            encoder.peaks.seg = seg.peak_held();
        }

        let denom = total_bytes.max(1) as f64;
        let stats = StreamStats {
            total_tokens: sink.total,
            time_to_first_token: sink.time_to_first_token(),
            specials: sink.specials,
            total_bytes,
            total: Duration::ZERO,
            seg_peak: if PROFILE && PROFILE_WORKING_SET {
                encoder.peaks.seg
            } else {
                0
            },
            pretok_peak: if PROFILE && PROFILE_WORKING_SET {
                encoder.peaks.pretok
            } else {
                0
            },
            bpe_peak: if PROFILE && PROFILE_WORKING_SET {
                encoder.peaks.bpe
            } else {
                0
            },
            seg_avg: if PROFILE && PROFILE_WORKING_SET {
                (seg.held_sum() - seg_sum_start) as f64 / denom
            } else {
                0.0
            },
            pretok_avg: if PROFILE && PROFILE_WORKING_SET {
                (encoder.pretok_sum - pretok_sum_start) as f64 / denom
            } else {
                0.0
            },
            bpe_avg: if PROFILE && PROFILE_WORKING_SET {
                (encoder.bpe_sum - bpe_sum_start) as f64 / denom
            } else {
                0.0
            },
            cache_hits: if PROFILE && PROFILE_WORKING_SET {
                encoder.cache_hits - cache_hits_start
            } else {
                0
            },
            pieces: if PROFILE && PROFILE_WORKING_SET {
                encoder.pieces - pieces_start
            } else {
                0
            },
            cache_bytes: if PROFILE && PROFILE_WORKING_SET {
                encoder.cache_bytes()
            } else {
                0
            },
        };
        Ok(stats)
    }

    /// Pre-tokens processed so far (accumulated across `tokenize_*` calls).
    pub fn pieces(&self) -> usize {
        self.encoder.pieces
    }
    /// Pre-tokens served by the single-token lookup / cache fast path.
    pub fn cache_hits(&self) -> usize {
        self.encoder.cache_hits
    }
    /// Peak bytes held back by the special-token segmenter (≤ longest special).
    pub fn seg_peak(&self) -> usize {
        self.seg.peak_held()
    }
    /// Peak in-progress pre-token bytes buffered by the pretokenizer.
    pub fn pretok_peak(&self) -> usize {
        self.encoder.peaks.pretok
    }
    /// Peak entries in the incremental-BPE state for a single pre-token.
    pub fn bpe_peak(&self) -> usize {
        self.encoder.peaks.bpe
    }
    /// Per-input-byte average working set of each stage
    pub fn seg_avg(&self) -> f64 {
        self.seg.avg_held()
    }
    pub fn pretok_avg(&self) -> f64 {
        self.encoder.pretok_sum as f64 / self.seg.bytes_seen().max(1) as f64
    }
    pub fn bpe_avg(&self) -> f64 {
        self.encoder.bpe_sum as f64 / self.seg.bytes_seen().max(1) as f64
    }
}

enum EngineImpl {
    Fast(FastStreamEngine),
    Reference(ReferenceStreamEngine),
}

/// Low-level streaming engine over one compiled encoding and BPE dictionary.
///
/// Ordinary constructors select the optimized SIMD engine. The `_dfa`
/// constructors select the isolated, readable DFA + MTC implementation in
/// `reference_stream_tokenizer.rs`.
#[doc(hidden)]
pub struct StreamEngine {
    inner: EngineImpl,
}

impl StreamEngine {
    pub fn new(
        enc: Encoding,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        FastStreamEngine::new(enc, vocab, tokenizer, ranks, cache).map(|engine| Self {
            inner: EngineImpl::Fast(engine),
        })
    }

    pub fn new_dfa(
        enc: Encoding,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        require_reference_cache(cache)?;
        ReferenceStreamEngine::new(enc, vocab, tokenizer, ranks).map(|engine| Self {
            inner: EngineImpl::Reference(engine),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_specials(
        enc: Encoding,
        specials: &[(&str, u32)],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        FastStreamEngine::with_specials(
            enc,
            specials,
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            ranks,
            cache,
        )
        .map(|engine| Self {
            inner: EngineImpl::Fast(engine),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_segmenter_config_runtime(
        enc: Encoding,
        segmenter_config: SegmenterConfig,
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        FastStreamEngine::with_segmenter_config_runtime(
            enc,
            segmenter_config,
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            cache,
        )
        .map(|engine| Self {
            inner: EngineImpl::Fast(engine),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_added_tokens(
        enc: Encoding,
        added_tokens: &[AddedToken],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        FastStreamEngine::with_added_tokens(
            enc,
            added_tokens,
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            ranks,
            cache,
        )
        .map(|engine| Self {
            inner: EngineImpl::Fast(engine),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_specials_dfa(
        enc: Encoding,
        specials: &[(&str, u32)],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        require_reference_cache(cache)?;
        ReferenceStreamEngine::with_specials(
            enc,
            specials,
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            ranks,
        )
        .map(|engine| Self {
            inner: EngineImpl::Reference(engine),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_added_tokens_dfa(
        enc: Encoding,
        added_tokens: &[AddedToken],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        require_reference_cache(cache)?;
        ReferenceStreamEngine::with_added_tokens(
            enc,
            added_tokens,
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
            ranks,
        )
        .map(|engine| Self {
            inner: EngineImpl::Reference(engine),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_segmenter_config_runtime_dfa(
        enc: Encoding,
        segmenter_config: SegmenterConfig,
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        cache: Cache,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        require_reference_cache(cache)?;
        ReferenceStreamEngine::with_segmenter_config_runtime(
            enc,
            segmenter_config,
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
        )
        .map(|engine| Self {
            inner: EngineImpl::Reference(engine),
        })
    }

    pub(crate) fn install_fast_bpe(&mut self, tables: Option<Arc<FastBpeTables>>) {
        if let EngineImpl::Fast(engine) = &mut self.inner {
            engine.install_fast_bpe(tables);
        }
    }

    pub fn tokenize_string(&mut self, text: &str, sink: &mut TokenSink) {
        match &mut self.inner {
            EngineImpl::Fast(engine) => engine.tokenize_string(text, sink),
            EngineImpl::Reference(engine) => engine.tokenize_string(text, sink),
        }
    }

    pub(crate) fn tokenize_string_unprofiled_into(&mut self, text: &str, sink: &mut TokenSink) {
        match &mut self.inner {
            EngineImpl::Fast(engine) => engine.tokenize_string_unprofiled_into(text, sink),
            EngineImpl::Reference(engine) => engine.tokenize_string_unprofiled_into(text, sink),
        }
    }

    pub(crate) fn tokenize_string_profiled_into(
        &mut self,
        text: &str,
        sink: &mut TokenSink,
    ) -> StreamStats {
        match &mut self.inner {
            EngineImpl::Fast(engine) => engine.tokenize_string_profiled_into(text, sink),
            EngineImpl::Reference(engine) => engine.tokenize_string_profiled_into(text, sink),
        }
    }

    pub fn tokenize_string_u32_le_unprofiled(
        &mut self,
        text: &str,
        dump_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut sink = TokenSink::new_u32_le(dump_path)?;
        self.tokenize_string_unprofiled_into(text, &mut sink);
        let total = sink.total();
        sink.finish()?;
        Ok(total)
    }

    pub fn tokenize_string_json_unprofiled(
        &mut self,
        text: &str,
        dump_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut sink = TokenSink::new_json(dump_path)?;
        self.tokenize_string_unprofiled_into(text, &mut sink);
        let total = sink.total();
        sink.finish()?;
        Ok(total)
    }

    pub fn tokenize_string_to_vec_unprofiled(
        &mut self,
        text: &str,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut sink = TokenSink::new_memory_u32(token_capacity_hint(text.len()));
        self.tokenize_string_unprofiled_into(text, &mut sink);
        Ok(sink.into_u32_vec()?)
    }

    pub fn tokenize_string_u32_le_profiled(
        &mut self,
        text: &str,
        dump_path: &str,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut sink = TokenSink::new_u32_le(dump_path)?;
        sink.start_profile(started);
        let mut stats = self.tokenize_string_profiled_into(text, &mut sink);
        sink.finish()?;
        stats.total = started.elapsed();
        Ok(stats)
    }

    pub fn tokenize_string_json_profiled(
        &mut self,
        text: &str,
        dump_path: &str,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut sink = TokenSink::new_json(dump_path)?;
        sink.start_profile(started);
        let mut stats = self.tokenize_string_profiled_into(text, &mut sink);
        sink.finish()?;
        stats.total = started.elapsed();
        Ok(stats)
    }

    pub fn tokenize_string_to_vec_profiled(
        &mut self,
        text: &str,
    ) -> Result<(Vec<u32>, StreamStats), Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut sink = TokenSink::new_memory_u32(token_capacity_hint(text.len()));
        sink.start_profile(started);
        let mut stats = self.tokenize_string_profiled_into(text, &mut sink);
        let output = sink.into_u32_vec()?;
        stats.total = started.elapsed();
        Ok((output, stats))
    }

    fn tokenize_file_profiled_into(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        match &mut self.inner {
            EngineImpl::Fast(engine) => engine.tokenize_file_profiled_into(path, sink),
            EngineImpl::Reference(engine) => engine.tokenize_file_profiled_into(path, sink),
        }
    }

    pub(crate) fn tokenize_file_unprofiled_into(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        match &mut self.inner {
            EngineImpl::Fast(engine) => engine.tokenize_file_unprofiled_into(path, sink),
            EngineImpl::Reference(engine) => engine.tokenize_file_unprofiled_into(path, sink),
        }
    }

    pub fn tokenize_file_u32_le(
        &mut self,
        path: &str,
        dump_path: &str,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut sink = TokenSink::new_u32_le(dump_path)?;
        sink.start_profile(started);
        let mut stats = self.tokenize_file_profiled_into(path, &mut sink)?;
        sink.finish()?;
        stats.total = started.elapsed();
        Ok(stats)
    }

    pub fn tokenize_file_u32_le_unprofiled(
        &mut self,
        path: &str,
        dump_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut sink = TokenSink::new_u32_le(dump_path)?;
        let total = self.tokenize_file_unprofiled_into(path, &mut sink)?;
        sink.finish()?;
        Ok(total)
    }

    pub fn tokenize_file_json_unprofiled(
        &mut self,
        path: &str,
        dump_path: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut sink = TokenSink::new_json(dump_path)?;
        let total = self.tokenize_file_unprofiled_into(path, &mut sink)?;
        sink.finish()?;
        Ok(total)
    }

    pub fn tokenize_file_to_vec_unprofiled(
        &mut self,
        path: &str,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let input_bytes = std::fs::metadata(path)?.len() as usize;
        let mut sink = TokenSink::new_memory_u32(token_capacity_hint(input_bytes));
        self.tokenize_file_unprofiled_into(path, &mut sink)?;
        Ok(sink.into_u32_vec()?)
    }

    pub fn tokenize_file_u32_le_profiled(
        &mut self,
        path: &str,
        dump_path: &str,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut sink = TokenSink::new_u32_le(dump_path)?;
        sink.start_profile(started);
        let mut stats = self.tokenize_file_profiled_into(path, &mut sink)?;
        sink.finish()?;
        stats.total = started.elapsed();
        Ok(stats)
    }

    pub fn tokenize_file_json_profiled(
        &mut self,
        path: &str,
        dump_path: &str,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut sink = TokenSink::new_json(dump_path)?;
        sink.start_profile(started);
        let mut stats = self.tokenize_file_profiled_into(path, &mut sink)?;
        sink.finish()?;
        stats.total = started.elapsed();
        Ok(stats)
    }

    pub fn tokenize_file_to_vec_profiled(
        &mut self,
        path: &str,
    ) -> Result<(Vec<u32>, StreamStats), Box<dyn std::error::Error>> {
        let started = Instant::now();
        let input_bytes = std::fs::metadata(path)?.len() as usize;
        let mut sink = TokenSink::new_memory_u32(token_capacity_hint(input_bytes));
        sink.start_profile(started);
        let mut stats = self.tokenize_file_profiled_into(path, &mut sink)?;
        let output = sink.into_u32_vec()?;
        stats.total = started.elapsed();
        Ok((output, stats))
    }

    pub fn pieces(&self) -> usize {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.pieces(),
            EngineImpl::Reference(engine) => engine.pieces(),
        }
    }

    pub fn cache_hits(&self) -> usize {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.cache_hits(),
            EngineImpl::Reference(engine) => engine.cache_hits(),
        }
    }

    pub fn seg_peak(&self) -> usize {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.seg_peak(),
            EngineImpl::Reference(engine) => engine.seg_peak(),
        }
    }

    pub fn pretok_peak(&self) -> usize {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.pretok_peak(),
            EngineImpl::Reference(engine) => engine.pretok_peak(),
        }
    }

    pub fn bpe_peak(&self) -> usize {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.bpe_peak(),
            EngineImpl::Reference(engine) => engine.bpe_peak(),
        }
    }

    pub fn seg_avg(&self) -> f64 {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.seg_avg(),
            EngineImpl::Reference(engine) => engine.seg_avg(),
        }
    }

    pub fn pretok_avg(&self) -> f64 {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.pretok_avg(),
            EngineImpl::Reference(engine) => engine.pretok_avg(),
        }
    }

    pub fn bpe_avg(&self) -> f64 {
        match &self.inner {
            EngineImpl::Fast(engine) => engine.bpe_avg(),
            EngineImpl::Reference(engine) => engine.bpe_avg(),
        }
    }
}

fn require_reference_cache(cache: Cache) -> Result<(), Box<dyn std::error::Error>> {
    if cache == Cache::Lookup {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the isolated DFA + MTC engine requires Cache::Lookup",
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::{Dictionary, NormalizedDict};

    #[test]
    fn ignore_merges_is_semantic_even_without_a_cache() {
        let mut tokens: Vec<Vec<u8>> = (0u16..=255).map(|byte| vec![byte as u8]).collect();
        tokens.push(b" hello".to_vec());
        let vocab = Arc::new(Vocab::new(tokens.clone()).unwrap());
        // Deliberately leave the whole-piece token without a merge rule. It is
        // reachable only through Hugging Face's ignore_merges lookup.
        let dict = Dictionary::new_from_id_pair(
            vocab.as_ref().clone(),
            std::iter::empty::<(usize, usize)>(),
        )
        .unwrap();
        let tokenizer = Arc::new(IncBpeTokenizer::new(
            NormalizedDict::new_in_bytes(dict).unwrap(),
        ));
        let ranks: Arc<FxHashMap<Vec<u8>, u32>> = Arc::new(
            tokens
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (bytes, id as u32))
                .collect(),
        );

        let mut stream = StreamEngine::with_specials(
            Encoding::R50k,
            &[],
            None,
            Normalizer::None,
            Presplit::None,
            true,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::None,
        )
        .unwrap();
        let mut sink = TokenSink::new_memory_u32(1);
        stream.tokenize_string(" hello", &mut sink);
        assert_eq!(sink.into_u32_vec().unwrap(), [256]);
    }

    #[test]
    fn detect_affine_relabel_recognizes_a_single_gap_and_rejects_irregular_maps() {
        // A single reserved rank at 100: identity below it, +1 above.
        let single_gap: Vec<u32> = (0..256u32)
            .map(|i| if i < 100 { i } else { i + 1 })
            .collect();
        assert_eq!(detect_affine_relabel(&single_gap), Some((100, 1)));

        // Two reserved ranks (10 and 20) is not a single shift.
        let mut two_gaps: Vec<u32> = (0..256u32).collect();
        for value in two_gaps.iter_mut().skip(10) {
            *value += 1;
        }
        for value in two_gaps.iter_mut().skip(20) {
            *value += 1;
        }
        assert_eq!(detect_affine_relabel(&two_gaps), None);

        // An arbitrary Hugging Face-style remap is not affine at all.
        let mut shuffled: Vec<u32> = (0..8u32).collect();
        shuffled.swap(2, 5);
        assert_eq!(detect_affine_relabel(&shuffled), None);
    }

    /// A model relabeled only by a single reserved rank (p50k_base's shape)
    /// must still take the branchless direct-page path in
    /// `encode_keyed_batch_output` and come out with exactly the ids the
    /// generic per-span path would have produced.
    #[test]
    fn keyed_batch_fast_path_applies_a_single_gap_relabel() {
        let byte_tokens: Vec<Vec<u8>> = (0u16..=255).map(|byte| vec![byte as u8]).collect();
        let vocab = Arc::new(Vocab::new(byte_tokens.clone()).unwrap());
        let dict = Dictionary::new_from_id_pair(
            vocab.as_ref().clone(),
            std::iter::empty::<(usize, usize)>(),
        )
        .unwrap();
        let tokenizer = Arc::new(IncBpeTokenizer::new(
            NormalizedDict::new_in_bytes(dict).unwrap(),
        ));
        let ranks: Arc<FxHashMap<Vec<u8>, u32>> = Arc::new(
            byte_tokens
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (bytes, id as u32))
                .collect(),
        );
        // A single reserved rank at 100, mirroring p50k_base's `50256` gap:
        // compact byte ids below it are unaffected, ids at or above it shift
        // up by one.
        let threshold = 100u32;
        let rank_to_id: Vec<u32> = (0..256u32)
            .map(|id| if id < threshold { id } else { id + 1 })
            .collect();

        let mut stream = StreamEngine::with_specials(
            Encoding::R50k,
            &[],
            Some(rank_to_id),
            Normalizer::None,
            Presplit::None,
            false,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();

        // Mixes bytes below and above the threshold, and repeats to also
        // cover the warm-cache path once the packed cache has seeded them.
        let input = "abcĀā abcĀā";
        let expected: Vec<u32> = input
            .bytes()
            .map(u32::from)
            .map(|id| if id < threshold { id } else { id + 1 })
            .collect();

        let mut sink = TokenSink::new_memory_u32(expected.len());
        stream.tokenize_string(input, &mut sink);
        assert_eq!(sink.into_u32_vec().unwrap(), expected);
    }

    /// The same relabel also has to survive the packed *file* sink, where
    /// `prepare_page` may flush and rotate before handing back a page, making
    /// the batch's start a post-rotation offset. A memory sink never rotates,
    /// so it cannot catch a mistake there. One page holds
    /// `PACKED_U32_PAGE_IDS` (65,536) ids, so this input deliberately spans
    /// several of them.
    #[test]
    fn keyed_batch_fast_path_relabels_across_packed_file_pages() {
        let byte_tokens: Vec<Vec<u8>> = (0u16..=255).map(|byte| vec![byte as u8]).collect();
        let vocab = Arc::new(Vocab::new(byte_tokens.clone()).unwrap());
        let dict = Dictionary::new_from_id_pair(
            vocab.as_ref().clone(),
            std::iter::empty::<(usize, usize)>(),
        )
        .unwrap();
        let tokenizer = Arc::new(IncBpeTokenizer::new(
            NormalizedDict::new_in_bytes(dict).unwrap(),
        ));
        let ranks: Arc<FxHashMap<Vec<u8>, u32>> = Arc::new(
            byte_tokens
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (bytes, id as u32))
                .collect(),
        );
        // Threshold 100 sits inside the ASCII lowercase range, so a plain
        // alphabetic input exercises both sides of the shift: 'a'..'c' stay
        // put while 'd'..'j' move up by one.
        let threshold = 100u32;
        let rank_to_id: Vec<u32> = (0..256u32)
            .map(|id| if id < threshold { id } else { id + 1 })
            .collect();

        let mut stream = StreamEngine::with_specials(
            Encoding::R50k,
            &[],
            Some(rank_to_id),
            Normalizer::None,
            Presplit::None,
            false,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();

        // This byte-only vocabulary emits one token per input byte, so well
        // over three pages' worth.
        let input = "abcdefghij ".repeat(20_000);
        let expected: Vec<u32> = input
            .bytes()
            .map(u32::from)
            .map(|id| if id < threshold { id } else { id + 1 })
            .collect();
        assert!(expected.len() > 3 * PACKED_U32_PAGE_IDS);

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "hiriluk-affine-file-pages-{}-{nonce}.u32le",
            std::process::id()
        ));
        let mut sink = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path")).unwrap();
        stream.tokenize_string(&input, &mut sink);
        assert_eq!(sink.total(), expected.len());
        sink.finish().unwrap();

        let packed = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let got: Vec<u32> = packed
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn cold_and_warm_cache_paths_materialize_exact_ids() {
        let byte_tokens: Vec<Vec<u8>> = (0u16..=255).map(|byte| vec![byte as u8]).collect();
        let vocab = Arc::new(Vocab::new(byte_tokens.clone()).unwrap());
        let dict = Dictionary::new_from_id_pair(
            vocab.as_ref().clone(),
            std::iter::empty::<(usize, usize)>(),
        )
        .unwrap();
        let tokenizer = Arc::new(IncBpeTokenizer::new(
            NormalizedDict::new_in_bytes(dict).unwrap(),
        ));
        let ranks: Arc<FxHashMap<Vec<u8>, u32>> = Arc::new(
            byte_tokens
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (bytes, id as u32))
                .collect(),
        );

        // The r50k scanner produces exactly three pretokens:
        //   " abc"               short key, four inline token lanes
        //   " abcde"             short key, six-token spill value
        //   " abcdefghijklmnop"  long-key fallback entry
        let input = " abc abcde abcdefghijklmnop";
        let expected: Vec<u32> = input.bytes().map(u32::from).collect();
        let mut stream = StreamEngine::new(
            Encoding::R50k,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();

        let mut cold_sink = TokenSink::new_memory_u32(expected.len());
        let cold_pieces_start = stream.pieces();
        let cold_hits_start = stream.cache_hits();
        stream.tokenize_string(input, &mut cold_sink);
        assert_eq!(cold_sink.total(), expected.len());
        assert_eq!(stream.pieces() - cold_pieces_start, 0);
        assert_eq!(stream.cache_hits() - cold_hits_start, 0);
        assert_eq!(cold_sink.into_u32_vec().unwrap(), expected);

        let mut warm_sink = TokenSink::new_memory_u32(expected.len());
        let warm_pieces_start = stream.pieces();
        let warm_hits_start = stream.cache_hits();
        stream.tokenize_string(input, &mut warm_sink);
        assert_eq!(warm_sink.total(), expected.len());
        assert_eq!(stream.pieces() - warm_pieces_start, 0);
        assert_eq!(stream.cache_hits() - warm_hits_start, 0);
        assert_eq!(warm_sink.into_u32_vec().unwrap(), expected);
    }

    #[test]
    fn retained_array_matches_packed_output_across_cache_growth() {
        let byte_tokens: Vec<Vec<u8>> = (0u16..=255).map(|byte| vec![byte as u8]).collect();
        let vocab = Arc::new(Vocab::new(byte_tokens.clone()).unwrap());
        let dict = Dictionary::new_from_id_pair(
            vocab.as_ref().clone(),
            std::iter::empty::<(usize, usize)>(),
        )
        .unwrap();
        let tokenizer = Arc::new(IncBpeTokenizer::new(
            NormalizedDict::new_in_bytes(dict).unwrap(),
        ));
        let ranks: Arc<FxHashMap<Vec<u8>, u32>> = Arc::new(
            byte_tokens
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (bytes, id as u32))
                .collect(),
        );
        let input = (0..200)
            .map(|value| format!("{value:03}\n"))
            .collect::<String>();

        let mut count_tokenizer = StreamEngine::new(
            Encoding::R50k,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();
        let mut count_sink = TokenSink::new_memory_u32(input.len());
        count_tokenizer.tokenize_string(&input, &mut count_sink);
        let count = count_sink.total();
        assert_eq!(
            count_sink.into_u32_vec().unwrap(),
            input.bytes().map(u32::from).collect::<Vec<_>>()
        );

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mtc-count-cache-growth-{}-{nonce}.u32le",
            std::process::id()
        ));
        let mut output_tokenizer = StreamEngine::new(
            Encoding::R50k,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();
        let mut output_sink =
            TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path")).unwrap();
        output_tokenizer.tokenize_string(&input, &mut output_sink);
        assert_eq!(output_sink.total(), count);
        output_sink.finish().unwrap();

        let packed = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let ids: Vec<u32> = packed
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(ids, input.bytes().map(u32::from).collect::<Vec<_>>());
        assert_eq!(count, ids.len());
        assert_eq!(count_tokenizer.pieces(), output_tokenizer.pieces());
        assert_eq!(
            count_tokenizer.pretok_peak(),
            output_tokenizer.pretok_peak()
        );
        assert_eq!(count_tokenizer.pretok_avg(), output_tokenizer.pretok_avg());
    }

    #[test]
    fn unprofiled_file_path_matches_profiled_output() {
        let byte_tokens: Vec<Vec<u8>> = (0u16..=255).map(|byte| vec![byte as u8]).collect();
        let vocab = Arc::new(Vocab::new(byte_tokens.clone()).unwrap());
        let dict = Dictionary::new_from_id_pair(
            vocab.as_ref().clone(),
            std::iter::empty::<(usize, usize)>(),
        )
        .unwrap();
        let tokenizer = Arc::new(IncBpeTokenizer::new(
            NormalizedDict::new_in_bytes(dict).unwrap(),
        ));
        let ranks: Arc<FxHashMap<Vec<u8>, u32>> = Arc::new(
            byte_tokens
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (bytes, id as u32))
                .collect(),
        );
        let input = "hello 123\ncan't stop <|endoftext|> café!\n".repeat(1000);

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp = std::env::temp_dir();
        let input_path = temp.join(format!(
            "mtc-unprofiled-input-{}-{nonce}.txt",
            std::process::id()
        ));
        let profiled_path = temp.join(format!(
            "mtc-profiled-output-{}-{nonce}.u32le",
            std::process::id()
        ));
        let unprofiled_path = temp.join(format!(
            "mtc-unprofiled-output-{}-{nonce}.u32le",
            std::process::id()
        ));
        std::fs::write(&input_path, input).unwrap();

        let input_str = input_path.to_str().unwrap();
        let profiled_str = profiled_path.to_str().unwrap();
        let unprofiled_str = unprofiled_path.to_str().unwrap();

        let mut profiled = StreamEngine::new(
            Encoding::R50k,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();
        let profiled_stats = profiled
            .tokenize_file_u32_le(input_str, profiled_str)
            .unwrap();
        assert!(profiled_stats.total > Duration::ZERO);
        assert!(
            profiled_stats
                .time_to_first_token
                .is_some_and(|ttft| ttft <= profiled_stats.total)
        );
        assert_eq!(profiled_stats.pieces, 0);

        let mut unprofiled = StreamEngine::new(
            Encoding::R50k,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();
        let unprofiled_count = unprofiled
            .tokenize_file_u32_le_unprofiled(input_str, unprofiled_str)
            .unwrap();
        assert_eq!(unprofiled_count, profiled_stats.total_tokens);
        assert_eq!(
            std::fs::read(&unprofiled_path).unwrap(),
            std::fs::read(&profiled_path).unwrap()
        );
        assert_eq!(unprofiled.pieces(), 0);
        assert_eq!(unprofiled.cache_hits(), 0);
        assert_eq!(unprofiled.seg_peak(), 0);
        assert_eq!(unprofiled.pretok_peak(), 0);
        assert_eq!(unprofiled.bpe_peak(), 0);

        let mut memory = StreamEngine::new(
            Encoding::R50k,
            Arc::clone(&vocab),
            Arc::clone(&tokenizer),
            Arc::clone(&ranks),
            Cache::Full,
        )
        .unwrap();
        let (array_ids, array_profile) = memory.tokenize_file_to_vec_profiled(input_str).unwrap();
        assert!(array_profile.total > Duration::ZERO);
        assert_eq!(array_profile.total_tokens, profiled_stats.total_tokens);
        assert_eq!(array_ids.len(), profiled_stats.total_tokens);
        assert_eq!(array_profile.specials, profiled_stats.specials);
        assert_eq!(array_profile.pieces, 0);
        assert!(
            array_profile
                .time_to_first_token
                .is_some_and(|ttft| ttft <= array_profile.total)
        );
        drop((vocab, tokenizer, ranks));

        std::fs::remove_file(input_path).unwrap();
        std::fs::remove_file(profiled_path).unwrap();
        std::fs::remove_file(unprofiled_path).unwrap();
    }
}
