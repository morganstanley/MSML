use std::{hash::Hash, ops::Index};

use bytes::BytesMut;
use derive_more::{Deref, From, Into};
use rapidhash::{HashMapExt, RapidHashMap};
use thiserror::Error;
use tinyvec::TinyVec;

use crate::{
    Token, TokenId, Vocab,
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub RuleId, u32);

pub(crate) type RuleIdVec = TinyVec<[RuleId; 6]>;
const _: () = {
    assert!(std::mem::size_of::<RuleIdVec>() == 32);
};

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Into, From)]
pub struct Rule {
    pub merged: TokenId,
    pub pre: TokenId,
    pub suc: TokenId,
}

#[derive(Clone, Debug, Deref)]
pub struct Dictionary {
    #[deref]
    vocab: Vocab,
    pub(crate) rules: TypedVec<RuleId, Rule>,
    pair_to_rule_id: RapidHashMap<(TokenId, TokenId), RuleId>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DictBuildError {
    #[error("rule {rule_id} uses an unknown token")]
    UnknownToken { rule_id: RuleId, token: Token },
    #[error("rule {rule_id} uses token id {token_id} which exceeds vocab size")]
    InvalidTokenId { rule_id: RuleId, token_id: TokenId },
    #[error("rule {rule_id} uses an empty or special token with id {token_id}")]
    EmptyToken { rule_id: RuleId, token_id: TokenId },
}

impl Dictionary {
    fn from_rules(vocab: Vocab, rules: TypedVec<RuleId, Rule>) -> Self {
        let mut pair_to_rule_id = RapidHashMap::with_capacity(rules.len().as_usize());
        for (id, rule) in rules.enumerate() {
            pair_to_rule_id.insert((rule.pre, rule.suc), id);
        }
        Self {
            vocab,
            rules,
            pair_to_rule_id,
        }
    }

    pub fn new_from_id_pair<T: Into<TokenId>, R: IntoIterator<Item = (T, T)>>(
        vocab: Vocab,
        rule_iter: R,
    ) -> Result<Self, DictBuildError> {
        let rule_iter = rule_iter.into_iter();
        let mut rules = TypedVec::with_capacity(RuleId::from(rule_iter.size_hint().0));
        let get_token = |rule_id, token_id| {
            vocab
                .get_token(token_id)
                .ok_or(DictBuildError::InvalidTokenId { rule_id, token_id })
                .and_then(|t| {
                    if t.is_empty() {
                        Err(DictBuildError::EmptyToken { rule_id, token_id })
                    } else {
                        Ok(t)
                    }
                })
        };
        for (pos, (left, right)) in rule_iter.map(|(i, j)| (i.into(), j.into())).enumerate() {
            let rule_id = RuleId::from(pos);
            let token = {
                let mut buf = BytesMut::from(get_token(rule_id, left)?.clone());
                buf.extend_from_slice(get_token(rule_id, right)?);
                buf.freeze()
            };
            let merged = vocab
                .find_token_id(&token)
                .ok_or(DictBuildError::UnknownToken { rule_id, token })?;
            rules.push(Rule {
                merged,
                pre: left,
                suc: right,
            });
        }
        Ok(Self::from_rules(vocab, rules))
    }

    pub fn new_from_token_pair<T: AsRef<[u8]>, R: IntoIterator<Item = (T, T)>>(
        vocab: Vocab,
        rule_iter: R,
    ) -> Result<Self, DictBuildError> {
        let rule_iter = rule_iter.into_iter();
        let mut rules = TypedVec::with_capacity(RuleId::from(rule_iter.size_hint().0));
        let get_id = |pos, token: &[u8]| {
            vocab
                .find_token_id(token)
                .ok_or(DictBuildError::UnknownToken {
                    rule_id: pos,
                    token: token.to_owned().into(),
                })
        };
        for (pos, (left, right)) in rule_iter.enumerate() {
            let (left, right) = (left.as_ref(), right.as_ref());
            let pos = RuleId::from(pos);
            let left_id = get_id(pos, left)?;
            let right_id = get_id(pos, right)?;
            let token = {
                let mut buf = BytesMut::from(left);
                buf.extend_from_slice(right);
                buf.freeze()
            };
            let merged = get_id(pos, &token)?;
            rules.push(Rule {
                merged,
                pre: left_id,
                suc: right_id,
            });
        }
        Ok(Self::from_rules(vocab, rules))
    }

    #[inline(always)]
    pub fn rules(&self) -> &[Rule] {
        self.rules.as_slice()
    }

    #[inline(always)]
    pub fn get_rule(&self, rule_id: RuleId) -> Option<&Rule> {
        self.rules.get(rule_id)
    }

    #[inline(always)]
    pub fn num_of_rules(&self) -> RuleId {
        self.rules.len()
    }

    #[inline(always)]
    pub fn find_rule(&self, left: TokenId, right: TokenId) -> Option<RuleId> {
        self.pair_to_rule_id.get(&(left, right)).copied()
    }
}

impl Index<RuleId> for Dictionary {
    type Output = Rule;

    #[inline(always)]
    fn index(&self, index: RuleId) -> &Self::Output {
        self.rules.index(index)
    }
}

impl Index<TokenId> for Dictionary {
    type Output = Token;

    #[inline(always)]
    fn index(&self, index: TokenId) -> &Self::Output {
        self.vocab.index(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Dictionary, Vocab};

    fn build_dict<T: AsRef<[u8]>, R: IntoIterator<Item = (T, T)>>(
        vocab: &Vocab,
        rules: R,
    ) -> Dictionary {
        Dictionary::new_from_token_pair(vocab.clone(), rules).unwrap()
    }

    #[test]
    fn test_dict() {
        let vocab = Vocab::new([
            b"a" as &[_],
            b"b",
            b"c",
            b"d",
            b"cd",
            b"bcd",
            b"abcd",
            "你".as_bytes(),
            "好".as_bytes(),
            "呀".as_bytes(),
            "你好".as_bytes(),
            "你好呀".as_bytes(),
            b"\xe4",
            b"\xbd",
            b"\xa0",
            b"\xbd\xa0",
        ])
        .unwrap();

        assert!(Dictionary::new_from_token_pair(vocab.clone(), [("c", "d")]).is_ok());
        assert!(Dictionary::new_from_token_pair(vocab.clone(), [("a", "b")]).is_err());
        assert!(Dictionary::new_from_id_pair(vocab.clone(), [(2usize, 3)]).is_ok());
        assert!(Dictionary::new_from_id_pair(vocab.clone(), [(0usize, 1)]).is_err());

        build_dict(&vocab, [("c", "d"), ("b", "cd"), ("a", "bcd")]);
        build_dict(&vocab, [("b", "cd")]);
        build_dict(
            &vocab,
            [(b"\xbd" as &[_], b"\xa0" as &[_]), (b"\xe4", b"\xbd\xa0")],
        );
        build_dict(&vocab, [("你", "好")]);
        build_dict(&vocab, [("你", "好"), ("你好", "呀")]);
        build_dict(&vocab, [("你好", "呀"), ("你", "好")]);
    }
}
