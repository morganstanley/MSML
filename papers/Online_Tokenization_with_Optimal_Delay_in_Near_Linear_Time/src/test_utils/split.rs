use crate::{TokenId, Vocab};

pub fn bytes_into_tokens<S: AsRef<[u8]>, I: Into<TokenId>>(
    vocab: &Vocab,
    seq: S,
    unk_id: I,
) -> Vec<TokenId> {
    let unk_id = unk_id.into();
    vocab
        .split_bytes_to_tokens(seq.as_ref())
        .map(|i| i.unwrap_or(unk_id))
        .collect()
}

pub fn utf8_into_tokens<S: AsRef<str>, I: Into<TokenId>>(
    vocab: &Vocab,
    seq: S,
    unk_id: I,
) -> Vec<TokenId> {
    let unk_id = unk_id.into();
    vocab
        .split_utf8_to_tokens(seq.as_ref())
        .map(|i| i.unwrap_or(unk_id))
        .collect()
}
