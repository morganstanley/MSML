//! Parse rank-ordered BPE vocabularies and compile the compact structures used
//! by streaming tokenization. Internal token IDs follow rank order; gapped
//! sources such as p50k additionally retain a compact-id → output-rank relabel.

use base64::{Engine, engine::general_purpose::STANDARD};
use bytes::Bytes;
use rustc_hash::FxHashMap;

use crate::inc_bpe::CanonicalTokens;
#[cfg(test)]
use crate::{Dictionary, Vocab};
use crate::{IncBpeTokenizer, MAX_TOKEN_LENGTH, Rule, TokenId, runtime_vocab::RuntimeVocab};

// All encodings (r50k, cl100k, o200k) ship as the same `.tiktoken` base64
// format, so a single loader serves them all; the per-encoding split pattern is
// chosen elsewhere via [`Encoding`].

/// `(token_bytes, rank)` pairs sorted by rank — the common intermediate form
/// produced by the loader and consumed by [`build_compact_from_ranks`].
pub(crate) type RankedTokens = Vec<(Vec<u8>, u32)>;

/// What the loader returns: the MTC vocab + dictionary, the `byte_pair_encode`
/// encoder (`bytes → compacted vocab id`), and an optional `vocab id → tiktoken
/// rank` relabel.
///
/// The encoder/lookup map and the MTC path both speak the *compacted* id space
/// (the Vocab's sorted position), so they relabel uniformly. The relabel is
/// `Some` only when the ranks aren't contiguous — e.g. `p50k_base` reserves rank
/// 50256 for `<|endoftext|>`, leaving a gap, so every token above it sits one
/// position below its true rank and must be mapped back on output. For r50k /
/// cl100k / o200k (and dense HF in-memory ranks) it is `None`.
#[cfg(test)]
type LoadedEncoding = (Vocab, Dictionary, FxHashMap<Vec<u8>, u32>, Option<Vec<u32>>);

/// Compact data retained by the named streaming tokenizer.
///
/// Construction may use ordered token bytes and merge rules temporarily, but
/// the returned value contains only the whole-piece/byte lookup, the final MTC
/// automaton, the optional output-id relabel, and merge triples needed to build
/// the tiktoken-only specialized pair-rank table.
pub(crate) struct CompactEncoding {
    pub(crate) vocab: RuntimeVocab,
    pub(crate) tokenizer: IncBpeTokenizer,
    pub(crate) rank_to_id: Option<Vec<u32>>,
    pub(crate) fast_merges: Option<Box<[(u32, u32, u32)]>>,
    #[cfg(test)]
    pub(crate) complete: bool,
}

type IdMergeTable = FxHashMap<(u32, u32), (u32, u32)>;

/// Reconstruct the final merge of `token_bytes` using compact token ids.
///
/// The table contains every token-producing split. Replaying the lowest-ranked
/// adjacent merge until two symbols remain therefore matches byte-pair encoding
/// without repeatedly concatenating and hashing byte vectors. Values retain the
/// original rank separately from the compact id so gapped `.tiktoken`
/// vocabularies preserve their exact merge priority.
fn reconstruct_merge_pair_ids(
    token_bytes: &[u8],
    byte_token_ids: &[Option<u32>; 256],
    merges: &IdMergeTable,
) -> Option<(u32, u32)> {
    if token_bytes.len() <= 1 {
        return None;
    }

    let mut parts: Vec<u32> = token_bytes
        .iter()
        .map(|&byte| byte_token_ids[byte as usize])
        .collect::<Option<_>>()?;
    while parts.len() > 2 {
        let mut best: Option<(usize, u32, u32)> = None;
        for index in 0..parts.len() - 1 {
            let Some(&(merged_id, rank)) = merges.get(&(parts[index], parts[index + 1])) else {
                continue;
            };
            if best.is_none_or(|(best_index, _, best_rank)| {
                rank < best_rank || (rank == best_rank && index < best_index)
            }) {
                best = Some((index, merged_id, rank));
            }
        }

        let (index, merged_id, _) = best?;
        parts[index] = merged_id;
        parts.remove(index + 1);
    }
    Some((parts[0], parts[1]))
}

fn compact_rules(
    tokens: &[Bytes],
    original_ranks: Option<&[u32]>,
    vocab: &RuntimeVocab,
) -> Result<Vec<Rule>, Box<dyn std::error::Error>> {
    if original_ranks.is_some_and(|ranks| tokens.len() != ranks.len()) {
        return Err("token/rank length mismatch".into());
    }
    let rank_at = |token_id: usize| {
        original_ranks.map_or_else(
            || u32::try_from(token_id).expect("token count was validated"),
            |ranks| ranks[token_id],
        )
    };

    let byte_token_ids: [Option<u32>; 256] = std::array::from_fn(|byte| {
        let id = vocab.byte_to_id()[byte];
        (id != TokenId::MAX).then(|| id.inner())
    });

    // Map every possible token-id pair to the token it produces. A token can
    // have more than one valid split; reconstruction must see all of them to
    // make the same greedy choice as byte-pair encoding.
    let mut merges =
        IdMergeTable::with_capacity_and_hasher(tokens.len().saturating_mul(2), Default::default());
    for (token_id, token_bytes) in tokens.iter().enumerate() {
        if token_bytes.len() <= 1 {
            continue;
        }
        let rank = rank_at(token_id);
        let token_id = u32::try_from(token_id)?;
        for split in 1..token_bytes.len() {
            let (Some(left), Some(right)) = (
                vocab.lookup(&token_bytes[..split]),
                vocab.lookup(&token_bytes[split..]),
            ) else {
                continue;
            };
            let pair = (left, right);
            if let Some((previous_id, previous_rank)) = merges.insert(pair, (token_id, rank))
                && previous_id != token_id
            {
                return Err(format!(
                    "merge pair {pair:?} produces both token {previous_id} rank {previous_rank} \
                     and token {token_id} rank {rank}"
                )
                .into());
            }
        }
    }

    let mut rules = Vec::with_capacity(tokens.len());
    for (token_id, token_bytes) in tokens.iter().enumerate() {
        if token_bytes.len() <= 1 {
            continue;
        }
        let rank = rank_at(token_id);
        let pair =
            reconstruct_merge_pair_ids(token_bytes, &byte_token_ids, &merges).ok_or_else(|| {
                format!("could not reconstruct merge pair for token {token_bytes:?} rank {rank}")
            })?;
        let token_id = u32::try_from(token_id)?;
        if merges.get(&pair).map(|&(merged_id, _)| merged_id) != Some(token_id) {
            return Err(format!(
                "reconstructed pair {pair:?} does not produce token {token_id} rank {rank}"
            )
            .into());
        }
        rules.push(Rule {
            merged: TokenId::new(token_id),
            pre: TokenId::new(pair.0),
            suc: TokenId::new(pair.1),
        });
    }
    Ok(rules)
}

fn validate_compact_tokens(ranked_tokens: &RankedTokens) -> Result<(), Box<dyn std::error::Error>> {
    let _ = u32::try_from(ranked_tokens.len())
        .map_err(|_| "runtime vocabulary contains more than u32::MAX tokens")?;
    for (token_id, (token, _)) in ranked_tokens.iter().enumerate() {
        if token.is_empty() {
            return Err(format!("runtime vocabulary token {token_id} is empty").into());
        }
        if token.len() > MAX_TOKEN_LENGTH {
            return Err(format!(
                "runtime vocabulary token {token_id} exceeds length limit {MAX_TOKEN_LENGTH}"
            )
            .into());
        }
    }
    Ok(())
}

/// Compile rank-ordered source tokens directly into the compact streaming
/// runtime. It never constructs the legacy vocabulary/dictionary objects or
/// their duplicate lookup maps.
pub(crate) fn build_compact_from_ranks(
    ranked_tokens: RankedTokens,
    include_fast_merges: bool,
) -> Result<CompactEncoding, Box<dyn std::error::Error>> {
    validate_compact_tokens(&ranked_tokens)?;
    let rank_to_id = (!ranked_tokens
        .iter()
        .enumerate()
        .all(|(index, (_, rank))| index as u32 == *rank))
    .then(|| {
        ranked_tokens
            .iter()
            .map(|(_, rank)| *rank)
            .collect::<Vec<_>>()
    });
    let tokens = ranked_tokens
        .into_iter()
        .map(|(token, _)| Bytes::from(token))
        .collect::<Vec<_>>();
    let (canonical_tokens, tokens) = CanonicalTokens::pack_ordered(&tokens);
    let vocab = RuntimeVocab::try_from_ordered_tokens(&tokens)
        .ok_or("runtime vocabulary contains duplicates")?;
    let rules = compact_rules(&tokens, rank_to_id.as_deref(), &vocab)?;
    let tokenizer = IncBpeTokenizer::new_from_ranked_parts_with_canonical(
        &tokens,
        &rules,
        vocab.byte_to_id(),
        canonical_tokens,
    )?;
    let fast_merges = include_fast_merges.then(|| {
        rules
            .iter()
            .map(|rule| (rule.pre.inner(), rule.suc.inner(), rule.merged.inner()))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    #[cfg(test)]
    let complete = vocab
        .byte_to_id()
        .iter()
        .all(|&token_id| token_id != TokenId::MAX);

    Ok(CompactEncoding {
        vocab,
        tokenizer,
        rank_to_id,
        fast_merges,
        #[cfg(test)]
        complete,
    })
}

/// Reconstruct construction-only fast merge triples from the retained compact
/// runtime without copying token payloads.
///
/// This is the rare DFA-first → fast-mode path. Native fast-first construction
/// carries the already-built triples directly, while DFA-only models retain no
/// fast data at all.
pub(crate) fn reconstruct_fast_merges(
    vocab: &RuntimeVocab,
    tokenizer: &IncBpeTokenizer,
) -> Result<Box<[(u32, u32, u32)]>, Box<dyn std::error::Error>> {
    let tokens = tokenizer
        .canonical_token_views()
        .ok_or("fast BPE requires a fully canonical ranked vocabulary")?;
    let rules = compact_rules(&tokens, None, vocab)?;
    Ok(rules
        .into_iter()
        .map(|rule| (rule.pre.inner(), rule.suc.inner(), rule.merged.inner()))
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

/// Parse a `.tiktoken` file: each non-empty line is `<base64-token> <rank>`
/// (the format produced by tiktoken's `load_tiktoken_bpe`).
pub(crate) fn parse_tiktoken_bpe(
    contents: &[u8],
) -> Result<RankedTokens, Box<dyn std::error::Error>> {
    let contents = std::str::from_utf8(contents)?;
    let mut ranked: RankedTokens = Vec::new();
    for line in contents.lines() {
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let token_b64 = fields.next().ok_or("tiktoken line missing token")?;
        let rank: u32 = fields.next().ok_or("tiktoken line missing rank")?.parse()?;
        ranked.push((STANDARD.decode(token_b64)?, rank));
    }
    ranked.sort_by_key(|(_, rank)| *rank);
    Ok(ranked)
}

/// Build the MTC [`Vocab`] + [`Dictionary`] + tiktoken encoder from rank-sorted
/// tokens. Single-byte tokens are base tokens (no merge rule); longer tokens get
/// their merge pair reconstructed.
#[cfg(test)]
fn build_from_ranks(
    ranked_tokens: RankedTokens,
) -> Result<LoadedEncoding, Box<dyn std::error::Error>> {
    let token_bytes: Vec<Bytes> = ranked_tokens
        .iter()
        .map(|(token, _)| Bytes::from(token.clone()))
        .collect();
    let vocab = Vocab::new(token_bytes)?;

    // The encoder/lookup map uses the COMPACTED vocab id (the rank-sorted
    // position `i`), not the raw file rank, so `byte_pair_encode` and
    // `Cache::Lookup` feed the same id space as the MTC path.
    let encoder: FxHashMap<Vec<u8>, u32> = ranked_tokens
        .iter()
        .enumerate()
        .map(|(i, (token, _))| (token.clone(), i as u32))
        .collect();
    // Compacted vocab id `i` -> original tiktoken rank. `None` when it's the
    // identity (contiguous ranks); `Some` only for gapped vocabs like p50k_base.
    let original_ranks: Vec<u32> = ranked_tokens.iter().map(|(_, rank)| *rank).collect();
    let rank_to_id = if original_ranks
        .iter()
        .enumerate()
        .all(|(i, &r)| i as u32 == r)
    {
        None
    } else {
        Some(original_ranks)
    };

    let byte_token_ids: [Option<u32>; 256] = std::array::from_fn(|byte| {
        encoder
            .get([u8::try_from(byte).expect("byte index fits in u8")].as_slice())
            .copied()
    });

    // Map every possible token-id pair to the token it produces. A token can
    // have more than one valid split; reconstruction must see all of them to
    // make the same greedy choice as byte-pair encoding.
    let mut merges = IdMergeTable::with_capacity_and_hasher(
        ranked_tokens.len().saturating_mul(2),
        Default::default(),
    );
    for (token_id, (token_bytes, rank)) in ranked_tokens.iter().enumerate() {
        if token_bytes.len() <= 1 {
            continue;
        }
        let token_id = u32::try_from(token_id)?;
        for split in 1..token_bytes.len() {
            let (Some(&left), Some(&right)) = (
                encoder.get(&token_bytes[..split]),
                encoder.get(&token_bytes[split..]),
            ) else {
                continue;
            };
            let pair = (left, right);
            if let Some((previous_id, previous_rank)) = merges.insert(pair, (token_id, *rank))
                && previous_id != token_id
            {
                return Err(format!(
                    "merge pair {pair:?} produces both token {previous_id} rank {previous_rank} \
                     and token {token_id} rank {rank}"
                )
                .into());
            }
        }
    }

    let mut rules = Vec::with_capacity(ranked_tokens.len());
    for (token_id, (token_bytes, rank)) in ranked_tokens.iter().enumerate() {
        if token_bytes.len() <= 1 {
            continue;
        }
        let pair =
            reconstruct_merge_pair_ids(token_bytes, &byte_token_ids, &merges).ok_or_else(|| {
                format!("could not reconstruct merge pair for token {token_bytes:?} rank {rank}")
            })?;
        let token_id = u32::try_from(token_id)?;
        if merges.get(&pair).map(|&(merged_id, _)| merged_id) != Some(token_id) {
            return Err(format!(
                "reconstructed pair {pair:?} does not produce token {token_id} rank {rank}"
            )
            .into());
        }
        rules.push(pair);
    }
    let dict = Dictionary::new_from_id_pair(vocab.clone(), rules)?;

    Ok((vocab, dict, encoder, rank_to_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NormalizedDict;

    fn mtc_tokens(
        tokenizer: &IncBpeTokenizer,
        byte_tokens: impl IntoIterator<Item = TokenId>,
    ) -> Vec<u32> {
        let mut state = tokenizer.tokenization();
        for token_id in byte_tokens {
            if token_id != TokenId::MAX {
                state.feed(token_id);
            }
        }
        let mut tokens = state
            .current_token_chain()
            .token_ids()
            .map(TokenId::inner)
            .collect::<Vec<_>>();
        tokens.reverse();
        tokens
    }

    fn relabel(tokens: &[u32], rank_to_id: &Option<Vec<u32>>) -> Vec<u32> {
        match rank_to_id {
            Some(map) => tokens.iter().map(|&token| map[token as usize]).collect(),
            None => tokens.to_vec(),
        }
    }

    fn assert_compact_matches_legacy(ranked: RankedTokens, inputs: &[&[u8]]) {
        let compact = build_compact_from_ranks(ranked.clone(), false).unwrap();
        let generic_tokens = ranked
            .iter()
            .map(|(token, _)| Bytes::from(token.clone()))
            .collect::<Vec<_>>();
        let generic_ranks = ranked.iter().map(|(_, rank)| *rank).collect::<Vec<_>>();
        let generic_vocab = RuntimeVocab::try_from_ordered_tokens(&generic_tokens).unwrap();
        let generic_rules =
            compact_rules(&generic_tokens, Some(&generic_ranks), &generic_vocab).unwrap();
        let generic_direct = IncBpeTokenizer::new_from_parts(
            &generic_tokens,
            &generic_rules,
            generic_vocab.byte_to_id(),
        )
        .unwrap();
        let (legacy_vocab, legacy_dict, legacy_lookup, legacy_relabel) =
            build_from_ranks(ranked.clone()).unwrap();
        let legacy_tokenizer =
            IncBpeTokenizer::new(NormalizedDict::new_in_bytes(legacy_dict).unwrap());

        assert_eq!(compact.rank_to_id, legacy_relabel);
        assert_eq!(
            compact.complete,
            (0u8..=u8::MAX).all(|byte| legacy_vocab.find_by_byte(byte).is_some())
        );
        for (compact_id, (token, _)) in ranked.iter().enumerate() {
            assert_eq!(
                compact.vocab.lookup(token),
                legacy_lookup.get(token.as_slice()).copied()
            );
            assert_eq!(compact.vocab.lookup(token), Some(compact_id as u32));
            let lookup_bytes = compact
                .vocab
                .entries()
                .find_map(|(bytes, id)| (id == compact_id as u32).then_some(bytes))
                .unwrap();
            let mtc_bytes = compact
                .tokenizer
                .canonical_token(TokenId::from(compact_id))
                .unwrap();
            assert_eq!(
                lookup_bytes.as_ptr(),
                mtc_bytes.as_ptr(),
                "lookup and MTC should share the packed token arena"
            );
        }
        for byte in 0u8..=u8::MAX {
            assert_eq!(
                compact.vocab.byte_to_id()[byte as usize],
                legacy_vocab.find_by_byte_unchecked(byte)
            );
        }

        for &input in inputs {
            let compact_tokens = mtc_tokens(
                &compact.tokenizer,
                compact.vocab.split_bytes_to_tokens_unchecked(input),
            );
            let generic_tokens = mtc_tokens(
                &generic_direct,
                generic_vocab.split_bytes_to_tokens_unchecked(input),
            );
            let legacy_tokens = mtc_tokens(
                &legacy_tokenizer,
                legacy_vocab.split_bytes_to_tokens_unchecked(input),
            );
            assert_eq!(
                compact_tokens, generic_tokens,
                "ranked and generic direct normalization differ for input={input:?}"
            );
            assert_eq!(compact_tokens, legacy_tokens, "input={input:?}");
            assert_eq!(
                relabel(&compact_tokens, &compact.rank_to_id),
                relabel(&legacy_tokens, &legacy_relabel),
                "relabeled input={input:?}"
            );
        }
    }

    fn compact_error(ranked: RankedTokens) -> String {
        match build_compact_from_ranks(ranked, false) {
            Ok(_) => panic!("invalid compact vocabulary unexpectedly succeeded"),
            Err(error) => error.to_string(),
        }
    }

    fn complete_fixture() -> RankedTokens {
        let mut ranked = (0u8..=u8::MAX)
            .enumerate()
            .map(|(rank, byte)| (vec![byte], rank as u32))
            .collect::<RankedTokens>();
        ranked.extend([
            (b"ab".to_vec(), 256),
            (b"bc".to_vec(), 257),
            (b"abc".to_vec(), 258),
        ]);
        ranked
    }

    #[test]
    fn parses_bytes_and_preserves_gapped_output_ids() {
        // Rank 3 is intentionally reserved. `abc` must reuse compact token id
        // 3 for `ab`, rather than its raw rank 4, when reconstructing its rule.
        let ranked = parse_tiktoken_bpe(b"YQ== 0\nYg== 1\nYw== 2\nYWI= 4\nYWJj 5\n").unwrap();
        let (vocab, dict, ranks, relabel) = build_from_ranks(ranked).unwrap();

        assert_eq!(vocab.num_of_tokens().as_usize(), 5);
        assert_eq!(ranks.get(b"a".as_slice()), Some(&0));
        assert_eq!(ranks.get(b"ab".as_slice()), Some(&3));
        assert_eq!(dict.rules()[1].pre.as_usize(), 3);
        assert_eq!(dict.rules()[1].suc.as_usize(), 2);
        assert_eq!(relabel, Some(vec![0, 1, 2, 4, 5]));
    }

    #[test]
    fn reconstructs_complete_vocab_merges_in_token_id_space() {
        let ranked = complete_fixture();

        let (vocab, dict, ranks, relabel) = build_from_ranks(ranked).unwrap();
        assert_eq!(vocab.num_of_tokens().as_usize(), 259);
        assert_eq!(dict.num_of_rules().as_usize(), 3);
        assert_eq!(ranks.get(b"abc".as_slice()), Some(&258));
        assert_eq!(relabel, None);
    }

    #[test]
    fn reconstructs_incomplete_valid_byte_vocab() {
        let ranked = vec![(b"a".to_vec(), 0), (b"b".to_vec(), 1), (b"ab".to_vec(), 2)];

        let (vocab, dict, ranks, relabel) = build_from_ranks(ranked).unwrap();
        assert_eq!(vocab.num_of_tokens().as_usize(), 3);
        assert_eq!(dict.num_of_rules().as_usize(), 1);
        assert_eq!(ranks.get(b"ab".as_slice()), Some(&2));
        assert_eq!(relabel, None);
    }

    #[test]
    fn reconstructs_using_the_lowest_ranked_adjacent_merge() {
        let mut byte_token_ids = [None; 256];
        byte_token_ids[b'a' as usize] = Some(0);
        byte_token_ids[b'b' as usize] = Some(1);
        byte_token_ids[b'c' as usize] = Some(2);

        let mut merges = IdMergeTable::default();
        merges.insert((0, 1), (3, 3));
        merges.insert((1, 2), (4, 4));
        assert_eq!(
            reconstruct_merge_pair_ids(b"abc", &byte_token_ids, &merges),
            Some((3, 2))
        );

        merges.insert((0, 1), (3, 5));
        assert_eq!(
            reconstruct_merge_pair_ids(b"abc", &byte_token_ids, &merges),
            Some((0, 4))
        );
    }

    #[test]
    fn rejects_a_token_whose_byte_leaf_is_missing() {
        let error = build_from_ranks(vec![(b"a".to_vec(), 0), (b"ab".to_vec(), 1)]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("could not reconstruct merge pair")
        );
    }

    #[test]
    fn compact_builder_matches_legacy_for_complete_vocab() {
        assert_compact_matches_legacy(
            complete_fixture(),
            &[b"", b"a", b"abc", b"zabc", &[0, 1, 2, 255]],
        );
    }

    #[test]
    fn compact_builder_matches_legacy_for_gapped_vocab() {
        assert_compact_matches_legacy(
            vec![
                (b"a".to_vec(), 0),
                (b"b".to_vec(), 1),
                (b"c".to_vec(), 2),
                (b"ab".to_vec(), 4),
                (b"abc".to_vec(), 5),
            ],
            &[b"", b"a", b"abc", b"cabc"],
        );
    }

    #[test]
    fn deferred_fast_merges_match_construction_time_merges() {
        let mut ranked = complete_fixture();
        for (_, rank) in &mut ranked[256..] {
            *rank += 1;
        }
        let compact = build_compact_from_ranks(ranked, true).unwrap();
        let rebuilt = reconstruct_fast_merges(&compact.vocab, &compact.tokenizer).unwrap();
        assert_eq!(rebuilt.as_ref(), compact.fast_merges.unwrap().as_ref());
    }

    #[test]
    fn compact_builder_matches_legacy_for_incomplete_vocab() {
        assert_compact_matches_legacy(
            vec![(b"a".to_vec(), 0), (b"b".to_vec(), 1), (b"ab".to_vec(), 2)],
            &[b"", b"a", b"ab", b"cabca"],
        );
    }

    #[test]
    fn compact_builder_rejects_invalid_source_tokens() {
        let duplicate = compact_error(vec![(b"a".to_vec(), 0), (b"a".to_vec(), 1)]);
        assert!(duplicate.contains("duplicates"));

        let empty = compact_error(vec![(Vec::new(), 0)]);
        assert!(empty.contains("empty"));

        let too_long = compact_error(vec![(vec![b'a'; MAX_TOKEN_LENGTH + 1], 0)]);
        assert!(too_long.contains("exceeds length limit"));
    }
}
