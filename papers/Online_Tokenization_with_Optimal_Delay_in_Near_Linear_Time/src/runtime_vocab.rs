//! Compact immutable vocabulary data retained by streaming engines.
//!
//! Construction uses [`Bytes`] keys so an ordered build-time token slice and
//! the whole-token lookup can share token-byte allocations. The runtime keeps
//! only that lookup and the fixed raw-byte-to-token table.

use bytes::Bytes;
use rustc_hash::FxHashMap;
use std::iter::FusedIterator;

use crate::{TokenId, Vocab};

/// The vocabulary operations required while chopping byte streams.
pub(crate) struct RuntimeVocab {
    lookup: FxHashMap<Bytes, u32>,
    byte_to_id: [TokenId; 256],
}

impl RuntimeVocab {
    /// Build from unique tokens in compact-ID order.
    ///
    /// Cloning a [`Bytes`] value retains its shared backing allocation; token
    /// payloads are not copied into the lookup table. Returns `None` if the
    /// source contains duplicate byte strings.
    pub(crate) fn try_from_ordered_tokens(tokens: &[Bytes]) -> Option<Self> {
        let mut lookup = FxHashMap::with_capacity_and_hasher(tokens.len(), Default::default());
        let mut byte_to_id = [TokenId::MAX; 256];

        for (index, token) in tokens.iter().enumerate() {
            let token_id = TokenId::from(index);
            let previous = lookup.insert(token.clone(), token_id.inner());
            if previous.is_some() {
                return None;
            }
            if let [byte] = token.as_ref() {
                byte_to_id[*byte as usize] = token_id;
            }
        }

        Some(Self { lookup, byte_to_id })
    }

    /// Convert today's full vocabulary/rank representations without copying
    /// token payloads. This is a migration adapter, not the eventual direct
    /// runtime-model construction path.
    pub(crate) fn from_legacy(vocab: &Vocab, ranks: &FxHashMap<Vec<u8>, u32>) -> Self {
        let mut lookup = FxHashMap::with_capacity_and_hasher(ranks.len(), Default::default());
        for token in vocab.tokens() {
            let &id = ranks
                .get(token.as_ref())
                .expect("legacy rank map must contain every vocabulary token");
            let previous = lookup.insert(token.clone(), id);
            debug_assert!(previous.is_none(), "legacy runtime tokens must be unique");
        }
        debug_assert_eq!(
            lookup.len(),
            ranks.len(),
            "legacy rank map and vocabulary must contain the same tokens"
        );

        Self {
            lookup,
            byte_to_id: std::array::from_fn(|byte| vocab.find_by_byte_unchecked(byte as u8)),
        }
    }

    /// Look up a complete token's compact internal ID.
    #[inline(always)]
    pub(crate) fn lookup(&self, bytes: &[u8]) -> Option<u32> {
        self.lookup.get(bytes).copied()
    }

    /// Split raw bytes into their compact atomic token IDs.
    ///
    /// Missing byte tokens are represented by [`TokenId::MAX`], matching the
    /// former [`Vocab::split_bytes_to_tokens_unchecked`] contract.
    #[inline(always)]
    pub(crate) fn split_bytes_to_tokens_unchecked<'a>(
        &'a self,
        bytes: &'a [u8],
    ) -> impl DoubleEndedIterator<Item = TokenId> + ExactSizeIterator + FusedIterator + 'a {
        bytes
            .iter()
            .map(move |&byte| self.byte_to_id[byte as usize])
    }

    /// Fixed raw-byte-to-compact-token map.
    #[inline(always)]
    pub(crate) fn byte_to_id(&self) -> &[TokenId; 256] {
        &self.byte_to_id
    }

    /// Whole-token entries used to seed the adaptive pretoken cache.
    #[inline]
    pub(crate) fn entries(&self) -> impl ExactSizeIterator<Item = (&[u8], u32)> + '_ {
        self.lookup.iter().map(|(bytes, &id)| (bytes.as_ref(), id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_tokens_share_bytes_and_build_lookup_and_byte_map() {
        let tokens = vec![
            Bytes::from_static(b"a"),
            Bytes::from_static(b"bc"),
            Bytes::from(vec![0xff]),
        ];
        let vocab = RuntimeVocab::try_from_ordered_tokens(&tokens).unwrap();

        assert_eq!(vocab.lookup(b"a"), Some(0));
        assert_eq!(vocab.lookup(b"bc"), Some(1));
        assert_eq!(vocab.lookup(&[0xff]), Some(2));
        assert_eq!(vocab.lookup(b"missing"), None);
        assert_eq!(vocab.byte_to_id()[b'a' as usize], TokenId::new(0));
        assert_eq!(vocab.byte_to_id()[0xff], TokenId::new(2));
        assert_eq!(vocab.byte_to_id()[b'b' as usize], TokenId::MAX);

        for token in &tokens {
            let stored = vocab
                .entries()
                .find_map(|(bytes, _)| (bytes == token.as_ref()).then_some(bytes))
                .unwrap();
            assert_eq!(stored.as_ptr(), token.as_ptr());
        }
    }

    #[test]
    fn byte_split_preserves_missing_sentinel() {
        let vocab = RuntimeVocab::try_from_ordered_tokens(&[
            Bytes::from_static(b"a"),
            Bytes::from_static(b"b"),
            Bytes::from_static(b"ab"),
        ])
        .unwrap();
        assert_eq!(
            vocab
                .split_bytes_to_tokens_unchecked(b"abc")
                .collect::<Vec<_>>(),
            [TokenId::new(0), TokenId::new(1), TokenId::MAX]
        );
    }

    #[test]
    fn legacy_adapter_preserves_rank_ids_without_copying_payloads() {
        let legacy = Vocab::new([Bytes::from_static(b"a"), Bytes::from_static(b"ab")]).unwrap();
        let ranks = FxHashMap::from_iter([(b"a".to_vec(), 7), (b"ab".to_vec(), 11)]);
        let runtime = RuntimeVocab::from_legacy(&legacy, &ranks);

        assert_eq!(runtime.lookup(b"a"), Some(7));
        assert_eq!(runtime.lookup(b"ab"), Some(11));
        for token in legacy.tokens() {
            let stored = runtime
                .entries()
                .find_map(|(bytes, _)| (bytes == token.as_ref()).then_some(bytes))
                .unwrap();
            assert_eq!(stored.as_ptr(), token.as_ptr());
        }
    }
}
