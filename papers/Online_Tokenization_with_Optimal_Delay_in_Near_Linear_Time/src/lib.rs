pub mod benchmark_utils;
mod fast_bpe;
mod mtc_inc_bpe;
mod nfc;
mod piece_cache;
mod pretok;
mod reference_stream_tokenizer;
mod runtime_tokenizer;
mod runtime_vocab;
mod segment;
mod stream_io;
mod stream_tokenizer;
pub mod tiktoken;

// The MTC engine lives under `mtc_inc_bpe`; alias its submodules back to the
// crate root so the engine's own `crate::<module>` paths keep resolving without
// edits, and so the `pub use` block below can name them unqualified.
pub(crate) use mtc_inc_bpe::{
    aho_corasick, centroid, dict, inc_bpe, normalize, sp_impl, successor, suf_suc, typed_vec, vocab,
};

pub use crate::{
    dict::{DictBuildError, Dictionary, Rule, RuleId},
    inc_bpe::{IncBpeToken, IncBpeTokenChainIter, IncBpeTokenization, IncBpeTokenizer},
    nfc::Normalizer,
    normalize::{NormalizedDict, NormalizedDictBuildError},
    pretok::{
        CL100K_DFA_PATTERN, CL100K_SPLIT_PATTERN, Encoding, FastR50kPretokenizer, LazyPretokenizer,
        O200K_DFA_PATTERN, O200K_SPLIT_PATTERN, Presplit, Pretok, R50K_DFA_PATTERN,
        R50K_SPLIT_PATTERN, StreamPretokenizer,
    },
    runtime_tokenizer::{
        ChopOptions, ChopOutput, ChopProfile, ChopResult, OutputMode, StreamTokenizer,
        TokenizerError,
    },
    segment::{AddedToken, Seg, SpecialSegmenter},
    sp_impl::{bpe_with_heap, bpe_with_heap_last_merge},
    stream_io::TokenSink,
    stream_tokenizer::{Cache, StreamEngine, StreamStats},
    successor::SkipLen,
    tiktoken::{CoreBPE, Rank, byte_pair_encode, byte_pair_split},
    vocab::{MAX_TOKEN_LENGTH, Token, TokenId, Vocab, VocabBuildError},
};
pub use benchmark_utils::SourceError;

#[cfg(test)]
mod test_utils;
