use std::{hash::Hash, iter::FusedIterator, ops::Index};

use bytes::Bytes;
use rapidhash::RapidHashMap;
use thiserror::Error;
use tinyvec::TinyVec;

use crate::typed_vec::{TypedVec, typed_vec_index};

typed_vec_index!(pub TokenId, u32);

pub(crate) type TokenIdVec = TinyVec<[TokenId; 6]>;
const _: () = {
    assert!(std::mem::size_of::<TokenIdVec>() == 32);
};

pub type Token = Bytes;

pub const MAX_TOKEN_LENGTH: usize = (1 << 14) - 1;

const _: () = {
    assert!(MAX_TOKEN_LENGTH < u16::MAX as usize);
};

#[derive(Clone, Debug)]
pub struct Vocab {
    pub(crate) tokens: TypedVec<TokenId, Token>,
    token_to_id: RapidHashMap<Token, TokenId>,
    u8_to_id: Box<[TokenId; 1 << 8]>,
    char_to_id: RapidHashMap<char, TokenId>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum VocabBuildError {
    #[error("duplicated tokens with ids {a} and {b}")]
    Duplicated { a: TokenId, b: TokenId },
    /// Token length is limited by [`MAX_TOKEN_LENGTH`].
    #[error("token {token_id} exceeds length limit {MAX_TOKEN_LENGTH}")]
    TokenTooLong { token_id: TokenId },
}

#[inline(always)]
fn utf8_char_token(token: &[u8]) -> Option<char> {
    if token.is_empty() || token.len() > 4 {
        return None;
    }
    let Ok(s) = str::from_utf8(token) else {
        return None;
    };
    debug_assert!(!s.is_empty());
    let mut iter = s.chars();
    let res = iter.next().unwrap();
    if iter.next().is_none() {
        Some(res)
    } else {
        None
    }
}

impl Vocab {
    pub fn new<T: Into<Token>, I: IntoIterator<Item = T>>(
        iter: I,
    ) -> Result<Self, VocabBuildError> {
        let mut token_to_id = RapidHashMap::default();
        let mut u8_to_id = Box::new([TokenId::MAX; _]);
        let mut char_to_id = RapidHashMap::default();

        let convert_token = |(k, token): (usize, T)| {
            let token = token.into();
            let token_id = TokenId::from(k);
            if token.len() == 1 {
                u8_to_id[token.as_ref()[0] as usize] = token_id;
            }
            if let Some(c) = utf8_char_token(&token) {
                char_to_id.insert(c, token_id);
            }
            if token.len() > MAX_TOKEN_LENGTH {
                Err(VocabBuildError::TokenTooLong { token_id })
            } else if !token.is_empty()
                && let Some(other) = token_to_id.insert(token.clone(), token_id)
            {
                Err(VocabBuildError::Duplicated {
                    a: other,
                    b: token_id,
                })
            } else {
                Ok(token)
            }
        };

        let tokens: TypedVec<_, _> = iter
            .into_iter()
            .enumerate()
            .map(convert_token)
            .collect::<Result<_, _>>()?;
        debug_assert!(tokens.as_slice().len() >= token_to_id.len());

        Ok(Self {
            tokens,
            token_to_id,
            u8_to_id,
            char_to_id,
        })
    }

    #[inline(always)]
    pub fn find_token_id<T: AsRef<[u8]>>(&self, token: T) -> Option<TokenId> {
        self.token_to_id.get(token.as_ref()).copied()
    }

    #[inline(always)]
    pub fn get_token<T: Into<TokenId>>(&self, token_id: T) -> Option<&Token> {
        self.tokens.get(token_id.into())
    }

    #[inline(always)]
    pub fn num_of_tokens(&self) -> TokenId {
        self.tokens.len()
    }

    #[inline(always)]
    pub fn tokens(&self) -> &[Token] {
        self.tokens.as_slice()
    }

    #[inline(always)]
    pub fn token_to_id_map(&self) -> &RapidHashMap<Token, TokenId> {
        &self.token_to_id
    }

    #[inline(always)]
    pub fn find_by_byte_unchecked(&self, b: u8) -> TokenId {
        self.u8_to_id[b as usize]
    }

    #[inline(always)]
    pub fn find_by_byte(&self, b: u8) -> Option<TokenId> {
        Some(self.find_by_byte_unchecked(b)).filter(|&i| i != TokenId::MAX)
    }

    #[inline(always)]
    pub fn find_by_char(&self, c: char) -> Option<TokenId> {
        self.char_to_id.get(&c).copied()
    }

    #[inline(always)]
    pub fn split_bytes_to_tokens_unchecked(
        &self,
        seq: &[u8],
    ) -> impl DoubleEndedIterator<Item = TokenId> + ExactSizeIterator + FusedIterator {
        seq.iter().map(|&b| self.find_by_byte_unchecked(b))
    }

    #[inline(always)]
    pub fn split_bytes_to_tokens(
        &self,
        seq: &[u8],
    ) -> impl DoubleEndedIterator<Item = Option<TokenId>> + ExactSizeIterator + FusedIterator {
        seq.iter().map(|&b| self.find_by_byte(b))
    }

    #[inline(always)]
    pub fn split_utf8_to_tokens(
        &self,
        seq: &str,
    ) -> impl DoubleEndedIterator<Item = Option<TokenId>> + FusedIterator {
        seq.chars().map(|c| self.find_by_char(c))
    }
}

impl Index<TokenId> for Vocab {
    type Output = Token;

    #[inline(always)]
    fn index(&self, index: TokenId) -> &Self::Output {
        self.tokens.index(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        TokenId, Vocab,
        test_utils::{bytes_into_tokens, utf8_into_tokens},
    };

    #[test]
    fn test_vocab() {
        assert!(Vocab::new([b"abc" as &[_], b"abcd"]).is_ok());
        assert!(Vocab::new([b"" as &[_], b"abc", b""]).is_ok());

        let vocab = Vocab::new([b"a" as &[_], b"b", b"c", b"d", b"cd", b"bcd", b"abcd"]).unwrap();

        assert_eq!(vocab.num_of_tokens().0, 7);

        assert_eq!(vocab.find_token_id(b"a"), Some(TokenId::new(0)));
        assert_eq!(vocab.find_token_id(b"b"), Some(TokenId::new(1)));
        assert_eq!(vocab.find_token_id(b"cd"), Some(TokenId::new(4)));
        assert_eq!(vocab.find_token_id(b"abcd"), Some(TokenId::new(6)));
        assert_eq!(vocab.find_token_id(b""), None);
        assert_eq!(vocab.find_token_id(b"e"), None);
        assert_eq!(vocab.find_token_id(b"random"), None);

        let check_token = |id: u32, e: &str| {
            let token = vocab.get_token(id).map(|b| b.as_ref());
            assert_eq!(token, Some(e.as_bytes()));
        };
        check_token(0, "a");
        check_token(3, "d");
        check_token(6, "abcd");
        assert!(vocab.get_token(7u32).is_none());
    }

    #[test]
    fn test_pre_tokenize() {
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

        let expected = [12, 13, 14, u32::MAX, u32::MAX, 13];
        let output = bytes_into_tokens(&vocab, "你好", u32::MAX);
        assert_eq!(output, expected.map(TokenId::new));

        let output = utf8_into_tokens(&vocab, "你好", u32::MAX);
        assert_eq!(output, [7, 8].map(TokenId::new));
    }
}
