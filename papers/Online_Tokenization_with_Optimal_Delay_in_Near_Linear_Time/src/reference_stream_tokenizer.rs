//! Readable reference streaming tokenizer.
//!
//! This is the compatibility pipeline from `main`, kept deliberately separate
//! from the hardware-oriented tokenizer:
//!
//! ```text
//! UTF-8 chunks
//!   -> added-token segmentation
//!   -> optional NFC normalization
//!   -> optional pre-split
//!   -> byte-stepped streaming DFA
//!   -> whole-piece vocabulary lookup
//!   -> MTC incremental BPE on a lookup miss
//!   -> TokenSink
//! ```
//!
//! The implementation has no packed pretoken keys, pretoken cache, keyed
//! batching, cache prefetching, SIMD scanner, or input-sized cache reservation.
//! It is intended to remain a compact correctness/reference implementation
//! while the optimized engine can evolve independently.

use std::{sync::Arc, time::Duration};

use rustc_hash::FxHashMap;

use crate::{
    AddedToken, Encoding, IncBpeTokenization, IncBpeTokenizer, Normalizer, Presplit, Vocab,
    nfc::last_nfc_boundary,
    pretok::DfaStream,
    runtime_vocab::RuntimeVocab,
    segment::{Seg, SegmenterConfig, SpecialSegmenter},
    stream_io::{CHUNK_SIZE, TokenSink, read_chunks_while},
    stream_tokenizer::{PROFILE_WORKING_SET, StreamStats, apply_relabel},
};

#[derive(Default)]
struct ReferencePeaks {
    pretok: usize,
    bpe: usize,
}

/// Whole-piece lookup followed by the original MTC incremental-BPE algorithm.
struct ReferencePieceEncoder {
    state: IncBpeTokenization<Arc<IncBpeTokenizer>>,
    scratch: Vec<u32>,
    rank_to_id: Option<Arc<[u32]>>,
    peaks: ReferencePeaks,
    pretok_sum: u64,
    bpe_sum: u64,
    cache_hits: usize,
    pieces: usize,
}

impl ReferencePieceEncoder {
    fn new(tokenizer: Arc<IncBpeTokenizer>, rank_to_id: Option<Arc<[u32]>>) -> Self {
        Self {
            state: IncBpeTokenization::new(tokenizer),
            scratch: Vec::new(),
            rank_to_id,
            peaks: ReferencePeaks::default(),
            pretok_sum: 0,
            bpe_sum: 0,
            cache_hits: 0,
            pieces: 0,
        }
    }

    fn encode<const PROFILE: bool>(
        &mut self,
        vocab: &RuntimeVocab,
        sink: &mut TokenSink,
        piece: &str,
    ) {
        if PROFILE && PROFILE_WORKING_SET {
            self.pieces += 1;
            self.peaks.pretok = self.peaks.pretok.max(piece.len());
            let len = piece.len() as u64;
            self.pretok_sum += len * (len + 1) / 2;
        }

        let bytes = piece.as_bytes();
        // This lookup is the reference path's only cache. It is bounded by the
        // immutable vocabulary and also implements HF `ignore_merges`: a whole
        // vocabulary token is returned before merge rules are consulted.
        if let Some(token) = vocab.lookup(bytes) {
            if PROFILE && PROFILE_WORKING_SET {
                self.cache_hits += 1;
            }
            sink.push_profiled::<PROFILE>(apply_relabel(self.rank_to_id.as_deref(), token));
            return;
        }

        self.state.reset();
        if PROFILE && PROFILE_WORKING_SET {
            let mut bpe_peak = 0usize;
            let mut bpe_sum = 0u64;
            for token in vocab.split_bytes_to_tokens_unchecked(bytes) {
                // Incomplete byte-level vocabularies can have uncovered bytes.
                // Dropping their sentinel matches the existing batch/reference
                // behavior.
                if token != crate::TokenId::MAX {
                    self.state.feed(token);
                    let frontier = self.state.inc_tokens().len();
                    bpe_peak = bpe_peak.max(frontier);
                    bpe_sum += frontier as u64;
                }
            }
            self.peaks.bpe = self.peaks.bpe.max(bpe_peak);
            self.bpe_sum += bpe_sum;
        } else {
            for token in vocab.split_bytes_to_tokens_unchecked(bytes) {
                if token != crate::TokenId::MAX {
                    self.state.feed(token);
                }
            }
        }

        // The MTC chain iterator walks backward from the end.
        self.scratch.clear();
        self.scratch.extend(
            self.state
                .current_token_chain()
                .token_ids()
                .map(|token| token.as_usize() as u32),
        );
        self.scratch.reverse();
        sink.push_many_profiled::<PROFILE>(&self.scratch, self.rank_to_id.as_deref());
    }
}

#[inline]
fn feed_text<const PROFILE: bool>(
    text: &str,
    presplit: Presplit,
    stream: &mut DfaStream,
    encoder: &mut ReferencePieceEncoder,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    let mut feed = |piece: &str| encoder.encode::<PROFILE>(vocab, sink, piece);
    match presplit {
        Presplit::None => stream.feed(text, &mut feed),
        Presplit::DigitsIndividual => presplit.split(text, |subspan| {
            if subspan.chars().next().is_some_and(char::is_numeric) {
                stream.finish(&mut feed);
                stream.feed(subspan, &mut feed);
                stream.finish(&mut feed);
            } else {
                stream.feed(subspan, &mut feed);
            }
        }),
    }
}

#[inline]
fn finish_pretokens<const PROFILE: bool>(
    stream: &mut DfaStream,
    encoder: &mut ReferencePieceEncoder,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    stream.finish(|piece| encoder.encode::<PROFILE>(vocab, sink, piece));
}

fn flush_normalizer<const PROFILE: bool>(
    normalizer: Normalizer,
    carry: &mut String,
    presplit: Presplit,
    stream: &mut DfaStream,
    encoder: &mut ReferencePieceEncoder,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    if carry.is_empty() {
        return;
    }
    let normalized = normalizer.normalize(carry).into_owned();
    feed_text::<PROFILE>(&normalized, presplit, stream, encoder, vocab, sink);
    carry.clear();
}

#[allow(clippy::too_many_arguments)]
fn dispatch<const PROFILE: bool>(
    event: Seg<'_>,
    normalizer: Normalizer,
    carry: &mut String,
    presplit: Presplit,
    stream: &mut DfaStream,
    encoder: &mut ReferencePieceEncoder,
    vocab: &RuntimeVocab,
    sink: &mut TokenSink,
) {
    match event {
        Seg::Text(text) => match normalizer {
            Normalizer::None => {
                feed_text::<PROFILE>(text, presplit, stream, encoder, vocab, sink);
            }
            Normalizer::Nfc => {
                carry.push_str(text);
                let boundary = last_nfc_boundary(carry);
                if boundary > 0 {
                    let normalized = normalizer.normalize(&carry[..boundary]).into_owned();
                    feed_text::<PROFILE>(&normalized, presplit, stream, encoder, vocab, sink);
                    carry.drain(..boundary);
                }
            }
        },
        Seg::Special(id) => {
            flush_normalizer::<PROFILE>(normalizer, carry, presplit, stream, encoder, vocab, sink);
            finish_pretokens::<PROFILE>(stream, encoder, vocab, sink);
            sink.push_special_profiled::<PROFILE>(id);
        }
    }
}

/// The isolated DFA + MTC streaming implementation.
///
/// Construction accepts the same semantic model components as the optimized
/// engine, but there is intentionally no cache/scanner mode argument.
pub(crate) struct ReferenceStreamEngine {
    vocab: Arc<RuntimeVocab>,
    normalizer: Normalizer,
    presplit: Presplit,
    stream: DfaStream,
    segmenter: SpecialSegmenter,
    encoder: ReferencePieceEncoder,
}

impl ReferenceStreamEngine {
    /// Compatibility constructor for callers that still hold the legacy
    /// vocabulary and rank-map pair.
    pub(crate) fn new(
        encoding: Encoding,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_runtime(
            encoding,
            Arc::new(RuntimeVocab::from_legacy(vocab.as_ref(), ranks.as_ref())),
            tokenizer,
        )
    }

    /// Construct directly from the compact immutable runtime vocabulary.
    pub(crate) fn new_runtime(
        encoding: Encoding,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_specials_runtime(
            encoding,
            encoding.special_tokens(),
            None,
            Normalizer::None,
            Presplit::None,
            false,
            vocab,
            tokenizer,
        )
    }

    /// Compatibility constructor for callers that still hold the legacy
    /// vocabulary and rank-map pair.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_specials(
        encoding: Encoding,
        specials: &[(&str, u32)],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_specials_runtime(
            encoding,
            specials,
            rank_to_id.map(Arc::from),
            normalizer,
            presplit,
            ignore_merges,
            Arc::new(RuntimeVocab::from_legacy(vocab.as_ref(), ranks.as_ref())),
            tokenizer,
        )
    }

    /// Build special-token semantics around an already compact runtime
    /// vocabulary, without reconstructing any immutable lookup state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_specials_runtime(
        encoding: Encoding,
        specials: &[(&str, u32)],
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_segmenter_config_runtime(
            encoding,
            SegmenterConfig::from_specials(specials),
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
        )
    }

    /// Compatibility constructor for callers that still hold the legacy
    /// vocabulary and rank-map pair.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_added_tokens(
        encoding: Encoding,
        added_tokens: &[AddedToken],
        rank_to_id: Option<Vec<u32>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<Vocab>,
        tokenizer: Arc<IncBpeTokenizer>,
        ranks: Arc<FxHashMap<Vec<u8>, u32>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_added_tokens_runtime(
            encoding,
            added_tokens,
            rank_to_id.map(Arc::from),
            normalizer,
            presplit,
            ignore_merges,
            Arc::new(RuntimeVocab::from_legacy(vocab.as_ref(), ranks.as_ref())),
            tokenizer,
        )
    }

    /// Build Hugging Face added-token semantics around an already compact
    /// runtime vocabulary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_added_tokens_runtime(
        encoding: Encoding,
        added_tokens: &[AddedToken],
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_segmenter_config_runtime(
            encoding,
            SegmenterConfig::from_added_tokens(added_tokens),
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
        )
    }

    /// Build a stream from an added-token program compiled during model load.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_segmenter_config_runtime(
        encoding: Encoding,
        segmenter_config: SegmenterConfig,
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_segmenter(
            encoding,
            SpecialSegmenter::from_config(segmenter_config),
            rank_to_id,
            normalizer,
            presplit,
            ignore_merges,
            vocab,
            tokenizer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_segmenter(
        encoding: Encoding,
        segmenter: SpecialSegmenter,
        rank_to_id: Option<Arc<[u32]>>,
        normalizer: Normalizer,
        presplit: Presplit,
        ignore_merges: bool,
        vocab: Arc<RuntimeVocab>,
        tokenizer: Arc<IncBpeTokenizer>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Cache::Lookup already returns every whole-vocabulary pretoken before
        // MTC, which is also the behavior requested by HF `ignore_merges`.
        let _ = ignore_merges;
        Ok(Self {
            vocab,
            normalizer,
            presplit,
            stream: DfaStream::new(encoding)?,
            segmenter,
            encoder: ReferencePieceEncoder::new(tokenizer, rank_to_id),
        })
    }

    /// Tokenize one string without profiling.
    pub(crate) fn tokenize_string(&mut self, text: &str, sink: &mut TokenSink) {
        self.tokenize_string_into::<false>(text, sink);
    }

    /// Tokenize one string without timing or working-set instrumentation.
    pub(crate) fn tokenize_string_unprofiled_into(&mut self, text: &str, sink: &mut TokenSink) {
        self.tokenize_string_into::<false>(text, sink);
    }

    /// Tokenize one string with opt-in chop profiling.
    pub(crate) fn tokenize_string_profiled_into(
        &mut self,
        text: &str,
        sink: &mut TokenSink,
    ) -> StreamStats {
        self.tokenize_string_into::<true>(text, sink);

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

    fn tokenize_string_into<const PROFILE: bool>(&mut self, text: &str, sink: &mut TokenSink) {
        let Self {
            vocab,
            normalizer,
            presplit,
            stream,
            segmenter,
            encoder,
        } = self;
        let vocab = vocab.as_ref();
        let (normalizer, presplit) = (*normalizer, *presplit);
        let mut normalization_carry = String::new();

        let mut start = 0usize;
        while start < text.len() && !sink.is_cancelled() {
            let mut end = (start + CHUNK_SIZE).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            let chunk = &text[start..end];
            if PROFILE && PROFILE_WORKING_SET {
                segmenter.feed(chunk, |event| {
                    dispatch::<PROFILE>(
                        event,
                        normalizer,
                        &mut normalization_carry,
                        presplit,
                        stream,
                        encoder,
                        vocab,
                        sink,
                    )
                });
            } else {
                segmenter.feed_unprofiled(chunk, |event| {
                    dispatch::<PROFILE>(
                        event,
                        normalizer,
                        &mut normalization_carry,
                        presplit,
                        stream,
                        encoder,
                        vocab,
                        sink,
                    )
                });
            }
            start = end;
        }
        if sink.is_cancelled() {
            return;
        }

        segmenter.finish(|event| {
            dispatch::<PROFILE>(
                event,
                normalizer,
                &mut normalization_carry,
                presplit,
                stream,
                encoder,
                vocab,
                sink,
            )
        });
        flush_normalizer::<PROFILE>(
            normalizer,
            &mut normalization_carry,
            presplit,
            stream,
            encoder,
            vocab,
            sink,
        );
        finish_pretokens::<PROFILE>(stream, encoder, vocab, sink);
    }

    /// Tokenize a UTF-8 file without timing or working-set instrumentation.
    pub(crate) fn tokenize_file_unprofiled_into(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        Ok(self.tokenize_file_into::<false>(path, sink)?.total_tokens)
    }

    /// Tokenize a UTF-8 file with opt-in chop profiling.
    pub(crate) fn tokenize_file_profiled_into(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        self.tokenize_file_into::<true>(path, sink)
    }

    fn tokenize_file_into<const PROFILE: bool>(
        &mut self,
        path: &str,
        sink: &mut TokenSink,
    ) -> Result<StreamStats, Box<dyn std::error::Error>> {
        let pieces_start = if PROFILE && PROFILE_WORKING_SET {
            self.encoder.pieces
        } else {
            0
        };
        let hits_start = if PROFILE && PROFILE_WORKING_SET {
            self.encoder.cache_hits
        } else {
            0
        };
        let pretok_sum_start = if PROFILE && PROFILE_WORKING_SET {
            self.encoder.pretok_sum
        } else {
            0
        };
        let bpe_sum_start = if PROFILE && PROFILE_WORKING_SET {
            self.encoder.bpe_sum
        } else {
            0
        };
        let segment_sum_start = if PROFILE && PROFILE_WORKING_SET {
            self.segmenter.held_sum()
        } else {
            0
        };
        if PROFILE && PROFILE_WORKING_SET {
            self.encoder.peaks = ReferencePeaks::default();
            self.segmenter.reset_peak();
        }

        let Self {
            vocab,
            normalizer,
            presplit,
            stream,
            segmenter,
            encoder,
        } = self;
        let vocab = vocab.as_ref();
        let (normalizer, presplit) = (*normalizer, *presplit);
        let mut normalization_carry = String::new();

        let (total_bytes, completed) = read_chunks_while(path, |chunk| {
            if PROFILE && PROFILE_WORKING_SET {
                segmenter.feed(chunk, |event| {
                    dispatch::<PROFILE>(
                        event,
                        normalizer,
                        &mut normalization_carry,
                        presplit,
                        stream,
                        encoder,
                        vocab,
                        sink,
                    )
                });
            } else {
                segmenter.feed_unprofiled(chunk, |event| {
                    dispatch::<PROFILE>(
                        event,
                        normalizer,
                        &mut normalization_carry,
                        presplit,
                        stream,
                        encoder,
                        vocab,
                        sink,
                    )
                });
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

        segmenter.finish(|event| {
            dispatch::<PROFILE>(
                event,
                normalizer,
                &mut normalization_carry,
                presplit,
                stream,
                encoder,
                vocab,
                sink,
            )
        });
        flush_normalizer::<PROFILE>(
            normalizer,
            &mut normalization_carry,
            presplit,
            stream,
            encoder,
            vocab,
            sink,
        );
        finish_pretokens::<PROFILE>(stream, encoder, vocab, sink);

        let denominator = total_bytes.max(1) as f64;
        Ok(StreamStats {
            total_tokens: sink.total(),
            time_to_first_token: sink.time_to_first_token(),
            specials: sink.specials(),
            total_bytes,
            total: Duration::ZERO,
            seg_peak: if PROFILE && PROFILE_WORKING_SET {
                segmenter.peak_held()
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
                (segmenter.held_sum() - segment_sum_start) as f64 / denominator
            } else {
                0.0
            },
            pretok_avg: if PROFILE && PROFILE_WORKING_SET {
                (encoder.pretok_sum - pretok_sum_start) as f64 / denominator
            } else {
                0.0
            },
            bpe_avg: if PROFILE && PROFILE_WORKING_SET {
                (encoder.bpe_sum - bpe_sum_start) as f64 / denominator
            } else {
                0.0
            },
            cache_hits: if PROFILE && PROFILE_WORKING_SET {
                encoder.cache_hits - hits_start
            } else {
                0
            },
            pieces: if PROFILE && PROFILE_WORKING_SET {
                encoder.pieces - pieces_start
            } else {
                0
            },
            cache_bytes: 0,
        })
    }

    pub(crate) fn pieces(&self) -> usize {
        self.encoder.pieces
    }

    pub(crate) fn cache_hits(&self) -> usize {
        self.encoder.cache_hits
    }

    pub(crate) fn seg_peak(&self) -> usize {
        self.segmenter.peak_held()
    }

    pub(crate) fn pretok_peak(&self) -> usize {
        self.encoder.peaks.pretok
    }

    pub(crate) fn bpe_peak(&self) -> usize {
        self.encoder.peaks.bpe
    }

    pub(crate) fn seg_avg(&self) -> f64 {
        self.segmenter.avg_held()
    }

    pub(crate) fn pretok_avg(&self) -> f64 {
        self.encoder.pretok_sum as f64 / self.segmenter.bytes_seen().max(1) as f64
    }

    pub(crate) fn bpe_avg(&self) -> f64 {
        self.encoder.bpe_sum as f64 / self.segmenter.bytes_seen().max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dictionary, NormalizedDict};

    fn byte_model() -> (
        Arc<Vocab>,
        Arc<IncBpeTokenizer>,
        Arc<FxHashMap<Vec<u8>, u32>>,
    ) {
        let tokens: Vec<Vec<u8>> = (0u16..=255).map(|byte| vec![byte as u8]).collect();
        let vocab = Arc::new(Vocab::new(tokens.clone()).unwrap());
        let dictionary = Dictionary::new_from_id_pair(
            vocab.as_ref().clone(),
            std::iter::empty::<(usize, usize)>(),
        )
        .unwrap();
        let tokenizer = Arc::new(IncBpeTokenizer::new(
            NormalizedDict::new_in_bytes(dictionary).unwrap(),
        ));
        let ranks = Arc::new(
            tokens
                .into_iter()
                .enumerate()
                .map(|(rank, bytes)| (bytes, rank as u32))
                .collect(),
        );
        (vocab, tokenizer, ranks)
    }

    #[test]
    fn dfa_stream_matches_reference_across_small_chunks() {
        let input = "hello can't 123 café 日本語!!!\r\n trailing \n\t";
        let regex = fancy_regex::Regex::new(Encoding::R50k.split_pattern()).unwrap();
        let expected: Vec<String> = regex
            .find_iter(input)
            .map(|piece| piece.unwrap().as_str().to_owned())
            .collect();

        for target in 1..=32 {
            let mut stream = DfaStream::new(Encoding::R50k).unwrap();
            let mut actual = Vec::new();
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
                stream.feed(&input[start..end], |piece| actual.push(piece.to_owned()));
                start = end;
            }
            stream.finish(|piece| actual.push(piece.to_owned()));
            assert_eq!(actual, expected, "chunk target {target}");
        }
    }

    #[test]
    fn reference_engine_materializes_exact_ids_and_reuses_state() {
        let (vocab, tokenizer, ranks) = byte_model();
        let relabel: Vec<u32> = (0..256).map(|id| id + 1_000).collect();
        let mut engine = ReferenceStreamEngine::with_specials(
            Encoding::R50k,
            &[("<x>", 42)],
            Some(relabel),
            Normalizer::None,
            Presplit::None,
            false,
            vocab,
            tokenizer,
            ranks,
        )
        .unwrap();

        let input = "ab<x>c";
        let expected = [1_097, 1_098, 42, 1_099];
        for _ in 0..2 {
            let mut sink = TokenSink::new_memory_u32(expected.len());
            engine.tokenize_string_unprofiled_into(input, &mut sink);
            assert_eq!(sink.total(), expected.len());
            assert_eq!(sink.specials(), 1);
            assert_eq!(sink.into_u32_vec().unwrap(), expected);
        }
    }
}
