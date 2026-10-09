//! The MTC incremental-BPE engine and its supporting data structures.
//!
//! This module groups the core algorithm — the incremental tokenizer
//! ([`inc_bpe`]) and its Aho-Corasick automaton, successor forest, suffix/
//! successor nodes, and centroid trees — together with the dictionary/vocab
//! representation it operates on ([`dict`], [`normalize`], [`vocab`],
//! [`typed_vec`]) and the reference heap BPE ([`sp_impl`]).
//!
//! The public surface is re-exported flat at the crate root (see `lib.rs`), so
//! external users still write `hiriluk::IncBpeTokenizer`; the private
//! `mtc_inc_bpe` module is only the internal organization of the engine. Its
//! submodules are `pub(crate)`
//! and aliased back to the crate root so existing `crate::<module>` paths inside
//! the engine keep resolving.

pub(crate) mod aho_corasick;
pub(crate) mod centroid;
pub(crate) mod dict;
pub(crate) mod inc_bpe;
pub(crate) mod normalize;
pub(crate) mod sp_impl;
pub(crate) mod successor;
pub(crate) mod suf_suc;
pub(crate) mod typed_vec;
pub(crate) mod vocab;
