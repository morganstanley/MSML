//! Load a tokenizer from its upstream runtime cache and compile its BPE
//! dictionary in memory.
//!
//! OpenAI encodings share Python tiktoken's URL-keyed cache. Hugging Face
//! tokenizers share the standard Hub snapshot cache. No converted vocabulary,
//! specials, relabel, or metadata files are written into the repository.

use std::sync::{Arc, Mutex, OnceLock};

use super::{
    ModelSource, SourceFormat, hf_json, resolve_source, source_for,
    vocab::{
        CompactEncoding, build_compact_from_ranks, parse_tiktoken_bpe, reconstruct_fast_merges,
    },
};
use crate::{
    Encoding, IncBpeTokenizer, Normalizer, Presplit, fast_bpe::FastBpeTables,
    runtime_vocab::RuntimeVocab, segment::SegmenterConfig,
};

type DynErr = Box<dyn std::error::Error>;

/// Models run when no `--model`/`--encoding` is given.
///
/// `llama31p` is intentionally absent: loading it requires properizing its merge
/// dictionary, which the runtime loader deliberately does not do yet. Selecting
/// it explicitly returns a focused error before any cache or network access.
pub const DEFAULT_MODELS: &[&str] = &[
    "r50k",
    "p50k",
    "cl100k",
    "o200k", // tiktoken encodings
    "gpt2",
    "roberta",
    "starcoder",
    "mistral",
    "llama4",
    "qwen",
    "gptoss",
];

/// A loaded tokenization target. Cached source bytes and construction-time
/// dictionaries have already been reduced to the immutable runtime structures
/// shared by its streaming engines.
pub struct Target {
    pub(crate) enc: Encoding,
    pub(crate) vocab: Arc<RuntimeVocab>,
    pub(crate) tokenizer: Arc<IncBpeTokenizer>,
    /// Added-token matching program compiled once and shared by every stream.
    pub(crate) segmenter: SegmenterConfig,
    pub(crate) rank_to_id: Option<Arc<[u32]>>,
    pub(crate) normalizer: Normalizer,
    pub(crate) presplit: Presplit,
    /// Hugging Face's whole-pretoken vocabulary lookup is semantically
    /// significant for models whose tokenizer JSON sets `ignore_merges`; the
    /// tokenizer constructors preserve it independently of the cache mode.
    pub(crate) ignore_merges: bool,
    /// Only native tiktoken encodings use the SIMD/full-cache execution path.
    /// Hugging Face targets always use the DFA + lookup + MTC engine.
    pub(crate) fast_path_allowed: bool,
    /// Construction-only merge triples retained until the tiktoken fast path
    /// is first prepared. DFA-first and Hugging Face targets never allocate
    /// this; a later DFA → fast switch reconstructs it from the compact arena.
    pub(crate) fast_bpe_source: Mutex<Option<Box<[(u32, u32, u32)]>>>,
    /// Immutable Gigatoken-style pair-rank tables. They are built only when
    /// the optimized engine is first requested, so the readable DFA + MTC
    /// path does not pay their initialization time or memory.
    pub(crate) fast_bpe: OnceLock<Option<Arc<FastBpeTables>>>,
}

impl Target {
    pub(crate) fn fast_bpe_tables(&self) -> Option<Arc<FastBpeTables>> {
        self.fast_bpe
            .get_or_init(|| {
                let merges = self
                    .fast_bpe_source
                    .lock()
                    .expect("fast BPE source lock poisoned")
                    .take()
                    .map(Ok)
                    .unwrap_or_else(|| reconstruct_fast_merges(&self.vocab, &self.tokenizer))
                    .ok()?;
                let byte_to_id = std::array::from_fn(|byte| self.vocab.byte_to_id()[byte].inner());
                FastBpeTables::build_from_parts(byte_to_id, &merges).map(Arc::new)
            })
            .clone()
    }

    /// A DFA engine cannot use construction-time fast merge triples. Drop
    /// them as soon as DFA becomes the first prepared mode; a later fast-mode
    /// request can reconstruct the same triples from the compact arena.
    pub(crate) fn discard_fast_bpe_source(&self) {
        self.fast_bpe_source
            .lock()
            .expect("fast BPE source lock poisoned")
            .take();
    }
}

/// The implemented main pre-tokenization pattern for a Hugging Face model.
pub fn pattern_for(model: &str) -> Result<Encoding, DynErr> {
    match model {
        "gpt2" | "roberta" | "starcoder" => Ok(Encoding::R50k),
        "llama3" | "llama31" => Ok(Encoding::Llama3),
        "llama4" | "gptoss" => Ok(Encoding::O200k),
        "mistral" => Ok(Encoding::Mistral),
        "qwen" | "qwen3" => Ok(Encoding::Qwen3),
        "llama31p" => Err(super::SourceError::Llama31pDisabled.into()),
        other => Err(format!("model {other:?} not yet wired (needs its own pre-tokenizer)").into()),
    }
}

fn omit_identity_relabel(rank_to_id: Vec<u32>) -> Option<Arc<[u32]>> {
    (!rank_to_id
        .iter()
        .enumerate()
        .all(|(rank, &id)| rank as u32 == id))
    .then(|| Arc::from(rank_to_id))
}

#[inline]
fn supports_fast_path(enc: Encoding) -> bool {
    matches!(
        enc,
        Encoding::R50k | Encoding::P50k | Encoding::Cl100k | Encoding::O200k
    )
}

/// Load a registered encoding/model by name.
///
/// OpenAI sources and immutable/offline Hugging Face revisions are cache-first.
/// Online mutable HF revisions are refreshed, then reuse or update the shared
/// snapshot cache.
pub fn load_target(name: &str) -> Result<Target, DynErr> {
    load_target_for_mode(name, false)
}

/// Load a target while optionally carrying construction-time merge triples
/// into immediate fast-engine preparation.
///
/// DFA-first construction omits them entirely. The retained compact model can
/// still reconstruct them if a later operation switches execution modes.
pub(crate) fn load_target_for_mode(name: &str, retain_fast_source: bool) -> Result<Target, DynErr> {
    let source = source_for(name)?;
    let enc = source.encoding().map_or_else(|| pattern_for(name), Ok)?;
    load_target_from_source_for_mode(enc, source, retain_fast_source)
}

/// Compile an explicit local/cached source with a selected pre-tokenization
/// pattern. This is the reusable core for a future Python `from_file` or
/// `from_pretrained` interface.
pub fn load_target_from_source(enc: Encoding, source: ModelSource) -> Result<Target, DynErr> {
    load_target_from_source_for_mode(enc, source, false)
}

fn load_target_from_source_for_mode(
    enc: Encoding,
    source: ModelSource,
    retain_fast_source: bool,
) -> Result<Target, DynErr> {
    let loaded = resolve_source(source)?;
    match loaded.source.format() {
        SourceFormat::Tiktoken => {
            let fast_path_allowed = supports_fast_path(enc);
            let ranked_tokens = parse_tiktoken_bpe(&loaded.bytes)?;
            drop(loaded);
            let mut compact =
                build_compact_from_ranks(ranked_tokens, fast_path_allowed && retain_fast_source)?;
            let rank_to_id = compact.rank_to_id.take().map(Arc::from);
            let segmenter = SegmenterConfig::from_specials(enc.special_tokens());
            finish_target(
                enc,
                compact,
                segmenter,
                rank_to_id,
                Normalizer::None,
                Presplit::None,
                false,
                fast_path_allowed,
            )
        }
        SourceFormat::HuggingFaceJson => {
            let parsed = hf_json::from_slice(&loaded.bytes)?;
            drop(loaded);
            let hf_json::ParsedTokenizer {
                ranked_tokens,
                added_tokens,
                rank_to_id,
                normalizer,
                presplit,
                split_pattern,
                ignore_merges,
            } = parsed;
            validate_split_pattern(enc, split_pattern.as_deref())?;
            drop(split_pattern);
            let segmenter = SegmenterConfig::from_added_tokens(&added_tokens);
            drop(added_tokens);
            let rank_to_id = omit_identity_relabel(rank_to_id);
            let compact = build_compact_from_ranks(ranked_tokens, false)?;
            finish_target(
                enc,
                compact,
                segmenter,
                rank_to_id,
                normalizer,
                presplit,
                ignore_merges,
                false,
            )
        }
    }
}

fn validate_split_pattern(enc: Encoding, split_pattern: Option<&str>) -> Result<(), DynErr> {
    match split_pattern {
        Some(pattern) if pattern == enc.split_pattern() => Ok(()),
        Some(_) => Err(format!(
            "tokenizer.json split regex does not match the registered {} pattern",
            enc.name()
        )
        .into()),
        None if matches!(enc, Encoding::R50k | Encoding::P50k) => Ok(()),
        None => Err(format!(
            "tokenizer.json uses ByteLevel's built-in GPT-2 regex, not the registered {} pattern",
            enc.name()
        )
        .into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_target(
    enc: Encoding,
    compact: CompactEncoding,
    segmenter: SegmenterConfig,
    rank_to_id: Option<Arc<[u32]>>,
    normalizer: Normalizer,
    presplit: Presplit,
    ignore_merges: bool,
    fast_bpe_allowed: bool,
) -> Result<Target, DynErr> {
    let fast_bpe_allowed = fast_bpe_allowed && supports_fast_path(enc);
    let CompactEncoding {
        vocab,
        tokenizer,
        fast_merges,
        ..
    } = compact;
    Ok(Target {
        enc,
        vocab: Arc::new(vocab),
        tokenizer: Arc::new(tokenizer),
        segmenter,
        rank_to_id,
        normalizer,
        presplit,
        ignore_merges,
        fast_path_allowed: fast_bpe_allowed,
        fast_bpe_source: Mutex::new(fast_merges),
        fast_bpe: OnceLock::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama31p_fails_before_source_resolution() {
        let error = match load_target("llama31p") {
            Ok(_) => panic!("llama31p must remain disabled"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("disabled"));
        assert!(error.contains("properization"));
    }

    #[test]
    fn identity_relabel_is_elided() {
        assert!(omit_identity_relabel(vec![0, 1, 2]).is_none());
        assert_eq!(
            omit_identity_relabel(vec![0, 9, 2]).as_deref(),
            Some([0, 9, 2].as_slice())
        );
    }
}
