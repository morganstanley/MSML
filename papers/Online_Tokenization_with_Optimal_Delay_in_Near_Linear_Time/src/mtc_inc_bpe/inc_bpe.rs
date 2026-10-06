use std::borrow::Borrow;

use bytes::Bytes;
use derive_more::{Constructor, Debug, Deref, From, Into};

use crate::{
    NormalizedDict, Rule, SkipLen, TokenId,
    aho_corasick::{AC_NODE_ROOT, ACAutomaton, ACNodeId, ACTransTable},
    centroid::SufSucCentroidTrees,
    normalize::{
        NormalizedDictBuildError, NormalizedView, normalize_parts, normalize_ranked_parts,
    },
    successor::{FOREST_VIRTUAL_ROOT, ForestNodeId, SucForest},
    suf_suc::{SufSucNode, SufSucNodeSet},
    typed_vec::TypedVec,
};

#[derive(Clone, Copy, Debug, Into, From, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct IncBpeToken {
    pub token_id: TokenId,
    pub skip_len: SkipLen,
}

#[derive(Debug, Deref)]
pub(crate) struct ForestData {
    #[deref]
    nodes: TypedVec<ForestNodeId, SufSucNode<TokenId>>,
    #[cfg(debug_assertions)]
    pub token_to_node_id: TypedVec<TokenId, ForestNodeId>,
    pub longest_token_node: TypedVec<ACNodeId, ForestNodeId>,
}

impl ForestData {
    fn extract(forest: SucForest, node_set: SufSucNodeSet) -> Self {
        let nodes = forest
            .keys()
            .map(|i| {
                let SufSucNode {
                    repr_id: _,
                    skip_len,
                    suc_skip_len,
                    valid_range,
                } = node_set[i];
                SufSucNode {
                    repr_id: forest[i].token_id,
                    skip_len,
                    suc_skip_len,
                    valid_range,
                }
            })
            .collect();
        Self {
            nodes,
            #[cfg(debug_assertions)]
            token_to_node_id: forest.token_to_node_id,
            longest_token_node: node_set.longest_token_node,
        }
    }
}

const TOKEN_SPAN_LEN_BITS: u32 = 16;
const TOKEN_SPAN_LEN_MASK: u64 = (1 << TOKEN_SPAN_LEN_BITS) - 1;

/// Canonical token bytes needed by the incremental automaton after construction.
///
/// A complete [`NormalizedDict`] also retains vocabulary lookup maps, merge-pair
/// maps, rules, and priorities. Tokenization only needs the canonical token
/// bytes addressed by token ID, so compact them into one flat arena. Each span
/// stores `(start << 16) | len`; zero denotes a noncanonical/empty token.
#[derive(Debug)]
pub(crate) struct CanonicalTokens {
    bytes: Bytes,
    spans: Box<[u64]>,
}

impl CanonicalTokens {
    pub(crate) fn from_view<D: NormalizedView + ?Sized>(dict: &D) -> Self {
        let total_bytes = (0..dict.num_of_tokens().as_usize())
            .map(TokenId::from)
            .filter(|&token_id| dict.is_canonical(token_id))
            .map(|token_id| dict.token(token_id).len())
            .sum();
        let mut bytes = Vec::with_capacity(total_bytes);
        let mut spans = Vec::with_capacity(dict.num_of_tokens().as_usize());

        for position in 0..dict.num_of_tokens().as_usize() {
            let token_id = TokenId::from(position);
            if !dict.is_canonical(token_id) {
                spans.push(0);
                continue;
            }
            let token = dict.token(token_id);
            debug_assert!(!token.is_empty());

            let start = u64::try_from(bytes.len()).expect("token arena length fits in u64");
            assert!(
                start <= u64::MAX >> TOKEN_SPAN_LEN_BITS,
                "canonical token arena exceeds packed-span capacity"
            );
            let len = u16::try_from(token.len())
                .expect("vocabulary construction bounds every token below u16::MAX");
            spans.push((start << TOKEN_SPAN_LEN_BITS) | u64::from(len));
            bytes.extend_from_slice(token);
        }

        debug_assert_eq!(bytes.len(), total_bytes);
        Self {
            bytes: Bytes::from(bytes),
            spans: spans.into_boxed_slice(),
        }
    }

    /// Pack rank-ordered canonical tokens into one shared allocation.
    ///
    /// The returned token views and this retained arena share the same bytes.
    /// A runtime lookup can therefore own the views while incremental BPE owns
    /// this arena without retaining two copies of every token payload.
    pub(crate) fn pack_ordered(tokens: &[Bytes]) -> (Self, Vec<Bytes>) {
        let total_bytes = tokens.iter().map(Bytes::len).sum();
        let mut bytes = Vec::with_capacity(total_bytes);
        let mut spans = Vec::with_capacity(tokens.len());
        for token in tokens {
            assert!(!token.is_empty(), "ranked runtime tokens are nonempty");
            let start = bytes.len();
            let len = u16::try_from(token.len())
                .expect("vocabulary construction bounds every token below u16::MAX");
            spans.push(
                (u64::try_from(start).expect("token arena length fits in u64")
                    << TOKEN_SPAN_LEN_BITS)
                    | u64::from(len),
            );
            bytes.extend_from_slice(token);
        }
        let bytes = Bytes::from(bytes);
        let views = spans
            .iter()
            .map(|&span| {
                let len = (span & TOKEN_SPAN_LEN_MASK) as usize;
                let start = (span >> TOKEN_SPAN_LEN_BITS) as usize;
                bytes.slice(start..start + len)
            })
            .collect();
        (
            Self {
                bytes,
                spans: spans.into_boxed_slice(),
            },
            views,
        )
    }

    #[inline(always)]
    fn get(&self, token_id: TokenId) -> Option<&[u8]> {
        let span = *self.spans.get(token_id.as_usize())?;
        let len = (span & TOKEN_SPAN_LEN_MASK) as usize;
        if len == 0 {
            return None;
        }
        let start = (span >> TOKEN_SPAN_LEN_BITS) as usize;
        // SAFETY: spans are constructed from live ranges in `bytes`.
        Some(unsafe { self.bytes.get_unchecked(start..start + len) })
    }

    #[inline(always)]
    fn is_canonical(&self, token_id: TokenId) -> bool {
        self.spans
            .get(token_id.as_usize())
            .is_some_and(|span| span & TOKEN_SPAN_LEN_MASK != 0)
    }

    #[inline(always)]
    fn len(&self) -> TokenId {
        TokenId::from(self.spans.len())
    }

    fn iter_canonical_or_empty(
        &self,
    ) -> impl DoubleEndedIterator<Item = &[u8]> + ExactSizeIterator {
        (0..self.spans.len()).map(|position| self.get(TokenId::from(position)).unwrap_or_default())
    }

    /// Recreate ordered token views without copying their payload bytes.
    ///
    /// Named runtime models use a fully canonical ranked vocabulary. This is
    /// only needed when a model was initialized for DFA and later switches to
    /// the optional fast engine, whose pair table is then built lazily.
    fn ordered_views(&self) -> Option<Vec<Bytes>> {
        self.spans
            .iter()
            .map(|&span| {
                let len = (span & TOKEN_SPAN_LEN_MASK) as usize;
                if len == 0 {
                    return None;
                }
                let start = (span >> TOKEN_SPAN_LEN_BITS) as usize;
                Some(self.bytes.slice(start..start + len))
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct IncBpeTokenizer {
    canonical_tokens: CanonicalTokens,
    pub(crate) trans_table: ACTransTable,
    pub(crate) forest_data: ForestData,
    pub(crate) trees: SufSucCentroidTrees,
}

#[derive(Clone, Debug, Constructor, Into, From)]
pub struct IncBpeTokenChainIter<S> {
    seq: S,
    pos: usize,
}

#[derive(Debug)]
pub struct IncBpeTokenization<T> {
    #[debug(ignore)]
    tokenizer: T,
    ac_state: ACNodeId,
    tokens: Vec<IncBpeToken>,
    forest_ids: Vec<ForestNodeId>,
}

impl IncBpeToken {
    #[inline(always)]
    pub const fn const_new(token_id: TokenId, skip_len: SkipLen) -> Self {
        Self { token_id, skip_len }
    }

    #[inline(always)]
    pub fn new<I: Into<TokenId>>(token_id: I, skip_len: SkipLen) -> Self {
        Self::const_new(token_id.into(), skip_len)
    }
}

impl IncBpeTokenizer {
    pub fn new(dict: NormalizedDict) -> Self {
        Self::new_from_normalized(&dict)
    }

    /// Build the retained incremental-BPE runtime directly from shared token
    /// bytes and merge triples.
    ///
    /// This performs the same byte normalization and improper-rule validation
    /// as [`NormalizedDict::new_in_bytes`] without constructing or retaining a
    /// [`crate::Vocab`], [`crate::Dictionary`], or [`NormalizedDict`].
    pub fn new_from_parts(
        tokens: &[Bytes],
        rules: &[Rule],
        byte_to_id: &[TokenId; 256],
    ) -> Result<Self, NormalizedDictBuildError> {
        let normalized = normalize_parts(tokens, rules, byte_to_id)?;
        Ok(Self::new_from_normalized(&normalized))
    }

    /// Ranked-parts construction with caller-supplied retained token storage.
    ///
    /// This seam lets the model loader share a token arena with another
    /// runtime index instead of copying canonical bytes into a second arena.
    pub(crate) fn new_from_ranked_parts_with_canonical(
        tokens: &[Bytes],
        rules: &[Rule],
        byte_to_id: &[TokenId; 256],
        canonical_tokens: CanonicalTokens,
    ) -> Result<Self, NormalizedDictBuildError> {
        let normalized = normalize_ranked_parts(tokens, rules, byte_to_id)?;
        Self::new_from_ranked_normalized(&normalized, canonical_tokens)
    }

    fn new_from_ranked_normalized<D: NormalizedView + ?Sized>(
        normalized: &D,
        canonical_tokens: CanonicalTokens,
    ) -> Result<Self, NormalizedDictBuildError> {
        if canonical_tokens.len() != normalized.num_of_tokens()
            || (0..normalized.num_of_tokens().as_usize())
                .map(TokenId::from)
                .any(|token_id| {
                    canonical_tokens.is_canonical(token_id) != normalized.is_canonical(token_id)
                })
        {
            return Err(NormalizedDictBuildError::InvalidParts {
                message: "canonical token storage does not match ranked merge parts".to_owned(),
            });
        }
        Ok(Self::new_from_normalized_with_canonical(
            normalized,
            canonical_tokens,
        ))
    }

    fn new_from_normalized<D: NormalizedView + ?Sized>(dict: &D) -> Self {
        let canonical_tokens = CanonicalTokens::from_view(dict);
        Self::new_from_normalized_with_canonical(dict, canonical_tokens)
    }

    fn new_from_normalized_with_canonical<D: NormalizedView + ?Sized>(
        dict: &D,
        canonical_tokens: CanonicalTokens,
    ) -> Self {
        let automaton = ACAutomaton::new(canonical_tokens.iter_canonical_or_empty());
        let forest = SucForest::new(dict);
        let node_set = SufSucNodeSet::new(&forest, &automaton);
        let trees = SufSucCentroidTrees::new(&node_set, &forest);
        Self {
            canonical_tokens,
            trans_table: automaton.trans_table,
            forest_data: ForestData::extract(forest, node_set),
            trees,
        }
    }

    /// Canonical token bytes used by the retained incremental automaton.
    ///
    /// Noncanonical, empty, and out-of-range token IDs return `None`.
    #[inline(always)]
    pub(crate) fn canonical_token(&self, token_id: TokenId) -> Option<&[u8]> {
        self.canonical_tokens.get(token_id)
    }

    /// Ordered zero-copy views of a fully canonical runtime vocabulary.
    pub(crate) fn canonical_token_views(&self) -> Option<Vec<Bytes>> {
        self.canonical_tokens.ordered_views()
    }

    #[inline(always)]
    pub fn is_canonical(&self, token_id: TokenId) -> bool {
        self.canonical_tokens.is_canonical(token_id)
    }

    #[inline(always)]
    pub fn num_of_tokens(&self) -> TokenId {
        self.canonical_tokens.len()
    }

    pub fn tokenize<I: IntoIterator<Item = TokenId>>(
        &self,
        token_ids: I,
    ) -> IncBpeTokenization<&Self> {
        let iter = token_ids.into_iter();
        let mut state = self.tokenization();
        state.reserve(iter.size_hint().0);
        for token_id in iter {
            state.feed(token_id);
        }
        state
    }

    #[inline(always)]
    pub fn tokenization(&self) -> IncBpeTokenization<&Self> {
        IncBpeTokenization::new(self)
    }
}

impl<S> IncBpeTokenChainIter<S> {
    #[inline(always)]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline(always)]
    pub fn seq(&self) -> &S {
        &self.seq
    }
}

impl<S: Borrow<[IncBpeToken]>> IncBpeTokenChainIter<S> {
    #[inline(always)]
    pub fn token_ids(self) -> impl Iterator<Item = TokenId> {
        self.map(|(_, t)| t.token_id)
    }
}

impl<S: Borrow<[IncBpeToken]>> Iterator for IncBpeTokenChainIter<S> {
    type Item = (usize, IncBpeToken);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let seq: &[IncBpeToken] = self.seq.borrow();
        let pos = self.pos;
        if pos >= seq.len() {
            return None;
        }
        let token = seq[pos];
        let skip_len = token.skip_len as usize;
        if skip_len <= pos {
            self.pos -= skip_len;
        } else {
            self.pos = seq.len();
        }
        Some((pos, token))
    }
}

impl<T: Borrow<IncBpeTokenizer>> IncBpeTokenization<T> {
    pub fn feed(&mut self, token_id: TokenId) -> IncBpeToken {
        let tokenizer: &IncBpeTokenizer = self.tokenizer.borrow();
        let (token, node_id) = if let Some(token) = tokenizer.canonical_token(token_id) {
            #[cfg(debug_assertions)]
            {
                let node_id = tokenizer.forest_data.token_to_node_id[token_id];
                debug_assert_eq!(tokenizer.forest_data[node_id].skip_len, 1);
            }
            self.ac_state = tokenizer.trans_table.feed(self.ac_state, token);
            let skip_to = |skip| {
                let len = self.forest_ids.len();
                if skip == 0 || skip > len {
                    FOREST_VIRTUAL_ROOT
                } else {
                    self.forest_ids[len - skip]
                }
            };
            let mut forest_id = tokenizer.forest_data.longest_token_node[self.ac_state];
            debug_assert_ne!(forest_id, FOREST_VIRTUAL_ROOT);
            let node = &tokenizer.forest_data[forest_id];
            if (node.skip_len as usize) <= self.tokens.len() && !node.verify(skip_to) {
                let tree = tokenizer.trees.get(forest_id);
                forest_id = tree.search(skip_to);
            }
            let node = &tokenizer.forest_data[forest_id];
            (
                IncBpeToken::const_new(node.repr_id, node.skip_len),
                forest_id,
            )
        } else {
            self.ac_state = AC_NODE_ROOT;
            (IncBpeToken::const_new(token_id, 1), FOREST_VIRTUAL_ROOT)
        };
        self.tokens.push(token);
        self.forest_ids.push(node_id);
        token
    }
}

impl<T> IncBpeTokenization<T> {
    #[inline(always)]
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            ac_state: AC_NODE_ROOT,
            tokens: Default::default(),
            forest_ids: Default::default(),
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.forest_ids.clear();
        self.ac_state = AC_NODE_ROOT;
    }

    #[inline(always)]
    pub fn reserve(&mut self, additional: usize) {
        self.tokens.reserve(additional);
        self.forest_ids.reserve(additional);
    }

    #[inline(always)]
    pub fn inc_tokens(&self) -> &[IncBpeToken] {
        &self.tokens
    }

    #[inline(always)]
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    #[inline(always)]
    pub fn into_inner(self) -> (T, Vec<IncBpeToken>) {
        (self.tokenizer, self.tokens)
    }

    #[inline(always)]
    pub fn into_inc_tokens(self) -> Vec<IncBpeToken> {
        self.tokens
    }

    #[inline(always)]
    pub fn token_chain(&self, end_pos: usize) -> IncBpeTokenChainIter<&[IncBpeToken]> {
        IncBpeTokenChainIter::new(&self.tokens, end_pos)
    }

    #[inline(always)]
    pub fn current_token_chain(&self) -> IncBpeTokenChainIter<&[IncBpeToken]> {
        self.token_chain(self.tokens.len().saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Borrow, sync::Arc};

    use crate::{
        Dictionary, IncBpeToken, IncBpeTokenChainIter, IncBpeTokenizer, NormalizedDict,
        NormalizedDictBuildError, TokenId, Vocab, bpe_with_heap,
        test_utils::{bytes_into_tokens, utf8_into_tokens},
    };

    fn inc_bpe_short_any_case(vocab: &[&str], rules: &[(&str, &str)], sequences: &[&str]) {
        inc_bpe_short_case::<true>(vocab, rules, sequences);
        inc_bpe_short_case::<false>(vocab, rules, sequences);
    }

    fn inc_bpe_short_case<const IN_BYTES: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        inc_bpe_case::<IN_BYTES, false>(vocab, rules, sequences);
    }

    fn inc_bpe_display_any_case(vocab: &[&str], rules: &[(&str, &str)], sequences: &[&str]) {
        inc_bpe_display_case::<true>(vocab, rules, sequences);
        inc_bpe_display_case::<false>(vocab, rules, sequences);
    }

    fn inc_bpe_display_case<const IN_BYTES: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        inc_bpe_case::<IN_BYTES, true>(vocab, rules, sequences);
    }

    fn validate(dict: &Dictionary, seq: &[TokenId], inc_res: &[IncBpeToken]) {
        for i in 0..seq.len() {
            let expected = bpe_with_heap::<false>(dict, &seq[0..i + 1]);
            let output = IncBpeTokenChainIter::new(inc_res, i).token_ids();
            let output = output.chain(std::iter::repeat(TokenId::MAX));
            assert!(expected.into_iter().rev().zip(output).all(|(i, j)| i == j));
        }
    }

    fn inc_bpe_case<const IN_BYTES: bool, const DISPLAY: bool>(
        vocab: &[&str],
        rules: &[(&str, &str)],
        sequences: &[&str],
    ) {
        let vocab = Vocab::new(vocab.iter().map(|&s| s.to_owned())).unwrap();
        let dict = Dictionary::new_from_token_pair(vocab.clone(), rules.iter().copied()).unwrap();
        let reference_dict = dict.clone();
        let tokenizer = IncBpeTokenizer::new(
            match if IN_BYTES {
                NormalizedDict::new_in_bytes
            } else {
                NormalizedDict::new_in_utf8
            }(dict)
            {
                Ok(dict) => dict,
                Err(NormalizedDictBuildError::ImproperDict { .. }) => {
                    return;
                }
                Err(e) => {
                    dbg!(e);
                    unreachable!();
                }
            },
        );

        let tokenize = |s: &str| {
            let atomic_tokens = if IN_BYTES {
                bytes_into_tokens(&vocab, s, 0usize)
            } else {
                utf8_into_tokens(&vocab, s, 0usize)
            };
            let res = tokenizer
                .tokenize(atomic_tokens.iter().copied())
                .into_inc_tokens();
            validate(&reference_dict, &atomic_tokens, &res);
            res
        };

        let display_res = |res: &[IncBpeToken]| {
            if DISPLAY {
                let msg = String::from_iter(res.iter().map(|t| {
                    let token = str::from_utf8(&vocab[t.token_id]).unwrap();
                    format!("[{token} ({})], ", t.token_id)
                }));
                println!("{msg}");
            }
        };

        for s in sequences {
            let res = tokenize(s);
            display_res(&res);
        }
    }

    #[test]
    fn test_inc_bpe_unk_tokens() {
        inc_bpe_display_any_case(
            &["", "a", "b", "ab", "ba", "aa"],
            &[("a", "b"), ("b", "a"), ("a", "a")],
            &["acbacbcabbacaaaaaacccabaccabca", "ccc", "c", ""],
        );
    }

    #[test]
    fn test_inc_bpe_short() {
        let vocab = [
            "", "a", "abc", "abcde", "abcdef", "b", "ba", "bc", "bcdef", "c", "cd", "cde", "cdefg",
            "d", "de", "def", "e", "ef", "efg", "f", "g",
        ];
        inc_bpe_display_any_case(
            &vocab,
            &[
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("b", "a"),
                ("a", "bc"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
            &["abcdefg", "babcdefg", "cdefg"],
        );
        inc_bpe_display_any_case(
            &vocab,
            &[
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("a", "bc"),
                ("b", "a"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
            &["abcdefg", "babcdefg", "cdefg"],
        );

        let vocab = ["", "a", "aa", "aaa", "aaaa", "aaaaa"];
        let rules = [("a", "a"), ("aa", "a"), ("aa", "aa"), ("aa", "aaa")];
        let seq = [
            "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa",
        ];
        inc_bpe_short_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a"), ("aa", "aa"), ("aa", "a"), ("aaaa", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a")];
        inc_bpe_display_any_case(&vocab, &rules, &seq);
        let rules = [("a", "a"), ("a", "aa")];
        inc_bpe_short_any_case(&vocab, &rules, &seq);

        for i in 1..6 {
            let mut vocab = vec!["<unk>".to_owned()];
            vocab.extend((0..i).map(|i| String::from_iter(std::iter::repeat_n("a", i + 1))));
            let vocab: Vec<_> = vocab.iter().map(|s| s.as_str()).collect();
            let all_rules: Vec<_> = vocab
                .iter()
                .skip(1)
                .flat_map(|s| (1..s.len()).map(|p| s.split_at(p)))
                .collect();
            assert!(all_rules.len() <= (1 << 10));
            for j in 0..(1 << all_rules.len()) {
                let rules: Vec<_> = all_rules
                    .iter()
                    .enumerate()
                    .filter_map(|(k, &v)| if (j & (1 << k)) != 0 { Some(v) } else { None })
                    .collect();
                inc_bpe_short_any_case(&vocab, &rules, &seq);
            }
        }

        let vocab = ["", "a", "aa", "aaa", "aaaa", "aaaaa"];
        let rules = [("a", "a"), ("aa", "a"), ("aa", "aa"), ("aa", "aaa")];
        let mut multiple_a_s: Vec<_> = [
            "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa",
        ]
        .map(|s| s.to_owned())
        .into_iter()
        .collect();
        for _ in 0..3 {
            for s in multiple_a_s.clone() {
                for k in 0..s.len() + 1 {
                    let (u, v) = s.split_at(k);
                    multiple_a_s.push(String::from_iter([u, "b", v]));
                }
            }
        }
        let multiple_a_s: Vec<_> = multiple_a_s.iter().map(|s| s.as_str()).collect();
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("aa", "aa"), ("aa", "a"), ("aaaa", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);
        let rules = [("a", "a"), ("a", "aa")];
        inc_bpe_short_any_case(&vocab, &rules, &multiple_a_s);

        let vocab = [
            "",
            "a",
            "b",
            "c",
            "d",
            "cd",
            "bcd",
            "abcd",
            "你",
            "好",
            "呀",
            "你好",
            "你好呀",
            "好你",
            "aa",
            "aaa",
        ];
        inc_bpe_short_any_case(
            &vocab,
            &[("c", "d"), ("b", "cd"), ("a", "bcd")],
            &["dcdbcdabcdab"],
        );
        inc_bpe_short_case::<false>(
            &vocab,
            &[("你", "好")],
            &["你好", "你好呀", "你好你好你好呀你好你好你"],
        );
        inc_bpe_short_case::<false>(
            &vocab,
            &[("你", "好"), ("你好", "呀")],
            &["你好", "你好呀", "你好你好你好呀你好你好你", "", "你"],
        );
        let seq = ["好你好你好呀你好你好你", "你好你好你好呀你好你好你"];
        for rules in [
            [("你", "好"), ("你好", "呀"), ("好", "你")],
            [("你", "好"), ("好", "你"), ("你好", "呀")],
            [("好", "你"), ("你", "好"), ("你好", "呀")],
        ] {
            inc_bpe_short_case::<false>(&vocab, &rules, &seq);
        }

        for rules in [
            &[("a", "a")] as &[_],
            &[("a", "a"), ("aa", "a")],
            &[("a", "a"), ("a", "aa")],
            &[("aa", "a"), ("a", "a")],
        ] {
            inc_bpe_short_any_case(&vocab, rules, &multiple_a_s);
        }
    }

    #[test]
    fn test_inc_bpe_non_longest() {
        let vocab = [
            "", "a", "b", "c", "d", "e", "f", "g", "h", "i", "ab", "ba", "bc", "cd", "de", "ef",
            "gh", "hi", "cde", "ghi", "fghi", "abcd", "fg", "efgh", "efghi", "bcd", "defgh",
            "bcde", "bcdef", "bcdefgh",
        ];
        let rules = [
            ("b", "a"),
            ("a", "b"),
            ("e", "f"),
            ("f", "g"),
            ("d", "e"),
            ("c", "de"),
            ("c", "d"),
            ("b", "cde"),
            ("b", "c"),
            ("b", "cd"),
            ("ab", "cd"),
            ("g", "h"),
            ("h", "i"),
            ("gh", "i"),
            ("ef", "gh"),
            ("d", "efgh"),
            ("bcd", "ef"),
            ("bcd", "efgh"),
            ("fg", "hi"),
            ("ef", "ghi"),
        ];
        let mut sequences = vec!["babcdefghi"];
        while sequences.last().unwrap().len() > 1 {
            sequences.push(&sequences.last().unwrap()[1..])
        }
        {
            let vocab = Vocab::new(vocab.iter().map(|&s| s.to_owned())).unwrap();
            let dict =
                Dictionary::new_from_token_pair(vocab.clone(), rules.iter().copied()).unwrap();
            let normalized = NormalizedDict::new_in_bytes(dict).unwrap();
            let mut expected: Vec<_> = normalized
                .canonical_rules
                .values()
                .map(|i| i.as_usize())
                .collect();
            expected.sort();
            assert_eq!(expected, (0..rules.len()).collect::<Vec<_>>());
            assert!(
                vocab
                    .tokens
                    .keys()
                    .skip(1)
                    .all(|id| normalized.is_canonical(id))
            );
        }
        inc_bpe_display_any_case(&vocab, &rules, &sequences);
    }

    fn inc_bpe_demo_case(rules: &[(&str, &str)]) {
        let vocab = Vocab::new([
            b"" as &[_],
            b"a",
            b"abc",
            b"abcde",
            b"abcdef",
            b"b",
            b"ba",
            b"bc",
            b"bcdef",
            b"c",
            b"cd",
            b"cde",
            b"cdefg",
            b"d",
            b"de",
            b"def",
            b"e",
            b"ef",
            b"efg",
            b"f",
            b"g",
        ])
        .unwrap();

        let dict = Dictionary::new_from_token_pair(vocab.clone(), rules.iter().copied()).unwrap();
        let tokenizer = IncBpeTokenizer::new(NormalizedDict::new_in_bytes(dict).unwrap());
        let tokenize = |s| {
            tokenizer
                .tokenize(bytes_into_tokens(&vocab, s, 0usize))
                .into_inc_tokens()
        };

        let display_res = |res: &[IncBpeToken]| {
            let msg = String::from_iter(res.iter().map(|t| {
                let token = str::from_utf8(&vocab[t.token_id]).unwrap();
                format!("[{token} ({})], ", t.token_id)
            }));
            println!("{msg}");
        };

        println!("{rules:?}");
        let res = tokenize("abcdefg");
        display_res(&res);
        let res = tokenize("babcdefg");
        display_res(&res);
        let res = tokenize("cdefg");
        display_res(&res);
    }

    #[test]
    fn test_inc_bpe_non_vocab_token() {
        let vocab = Vocab::new(["a", "aa"]).unwrap();
        let avail_token_ids = [0, 2, 3, TokenId::MAX.inner()].map(TokenId::new);
        for rules in [&[] as &[_], &[(0usize, 0usize)]] {
            let dict = Dictionary::new_from_id_pair(vocab.clone(), rules.iter().copied()).unwrap();
            let reference_dict = dict.clone();
            let tokenizer = IncBpeTokenizer::new(NormalizedDict::new_in_bytes(dict).unwrap());
            for len in 1..9 {
                for seq in 0..(1 << (len * 2)) {
                    let token_ids: Vec<_> = (0..len)
                        .map(|i| avail_token_ids[(seq >> (i * 2)) & 3])
                        .collect();
                    let res = tokenizer
                        .tokenize(token_ids.iter().copied())
                        .into_inc_tokens();
                    validate(&reference_dict, &token_ids, &res);
                }
            }
        }
    }

    #[test]
    fn test_inc_bpe_demo() {
        inc_bpe_demo_case(&[
            ("b", "c"),
            ("e", "f"),
            ("d", "e"),
            ("c", "d"),
            ("d", "ef"),
            ("b", "a"),
            ("a", "bc"),
            ("abc", "de"),
            ("abc", "def"),
            ("bc", "def"),
            ("c", "de"),
            ("ef", "g"),
            ("cd", "efg"),
        ]);
        inc_bpe_demo_case(&[
            ("b", "c"),
            ("e", "f"),
            ("d", "e"),
            ("c", "d"),
            ("d", "ef"),
            ("a", "bc"),
            ("b", "a"),
            ("abc", "de"),
            ("abc", "def"),
            ("bc", "def"),
            ("c", "de"),
            ("ef", "g"),
            ("cd", "efg"),
        ]);
    }

    fn verify_chain_iter(
        chain: impl Borrow<[IncBpeToken]> + Clone,
        seqs: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<TokenId>>>,
    ) {
        for (end_pos, seq) in seqs.into_iter().enumerate() {
            let expected: Vec<_> = seq.into_iter().map(|i| i.into()).collect();
            let iter = IncBpeTokenChainIter::new(chain.clone(), end_pos);
            let output: Vec<_> = iter.map(|(_, i)| i.token_id).collect();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn direct_parts_constructor_matches_compatibility_constructor() {
        let vocab = Vocab::new([
            b"a" as &[_],
            b"b",
            b"c",
            b"d",
            b"ab",
            b"bc",
            b"abc",
            b"cd",
            b"abcd",
        ])
        .unwrap();
        let dict = Dictionary::new_from_token_pair(
            vocab.clone(),
            [
                (b"a" as &[_], b"b" as &[_]),
                (b"b", b"c"),
                (b"ab", b"c"),
                (b"c", b"d"),
                (b"abc", b"d"),
            ],
        )
        .unwrap();
        let compatibility =
            IncBpeTokenizer::new(NormalizedDict::new_in_bytes(dict.clone()).unwrap());
        let byte_to_id = std::array::from_fn(|byte| vocab.find_by_byte_unchecked(byte as u8));
        let direct =
            IncBpeTokenizer::new_from_parts(vocab.tokens(), dict.rules(), &byte_to_id).unwrap();

        for len in 0usize..=8 {
            let cases = 4usize.pow(len as u32);
            for mut case in 0..cases {
                let mut input = Vec::with_capacity(len);
                for _ in 0..len {
                    input.push(TokenId::from(case & 3));
                    case >>= 2;
                }

                let compatibility_state = compatibility.tokenize(input.iter().copied());
                let compatibility_output: Vec<_> = compatibility_state
                    .current_token_chain()
                    .token_ids()
                    .collect();
                let direct_state = direct.tokenize(input.iter().copied());
                let direct_output: Vec<_> =
                    direct_state.current_token_chain().token_ids().collect();
                assert_eq!(direct_output, compatibility_output, "input={input:?}");
            }
        }
    }

    #[test]
    fn test_inc_bpe_chain_iter_box() {
        let chain_base = [
            IncBpeToken::new(1u32, 1),
            IncBpeToken::new(2u32, 1),
            IncBpeToken::new(3u32, 2),
            IncBpeToken::new(4u32, 2),
            IncBpeToken::new(5u32, 1),
            IncBpeToken::new(6u32, 3),
        ];

        let expected = [
            &[1u32] as &[u32],
            &[2, 1],
            &[3, 1],
            &[4, 2, 1],
            &[5, 4, 2, 1],
            &[6, 3, 1],
        ];
        let build_seq = || expected.iter().map(|&s| s.iter().copied());

        verify_chain_iter(&chain_base as &[IncBpeToken], build_seq());
        verify_chain_iter(chain_base, build_seq());

        let chain: Vec<IncBpeToken> = Vec::from(chain_base);
        verify_chain_iter(chain, build_seq());

        let chain: Box<[IncBpeToken]> = Box::from(chain_base);
        verify_chain_iter(chain, build_seq());

        let chain: Arc<[IncBpeToken]> = Arc::from(chain_base);
        verify_chain_iter(chain, build_seq());
    }

    #[test]
    fn test_inc_bpe_repeated() {
        let vocab: Vec<String> = ["".to_owned()]
            .into_iter()
            .chain((1..=32).map(|i| std::iter::repeat_n('a', i).collect()))
            .collect();
        let vocab_ref: Vec<_> = vocab.iter().map(|s| s.as_ref()).collect();
        inc_bpe_display_any_case(
            &vocab_ref[..18],
            &[
                ("a", "a"),
                ("aa", "a"),
                ("aa", "aa"),
                ("aaaa", "aaaa"),
                ("aaaa", "aa"),
                ("aa", "aaa"),
                ("aaaa", "aaa"),
                ("aaaaaaaa", "aaaaaaaa"),
            ],
            &vocab_ref[1..],
        );
    }
}
