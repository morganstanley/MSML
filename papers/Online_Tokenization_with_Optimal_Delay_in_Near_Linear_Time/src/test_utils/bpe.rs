use crate::{
    Dictionary, RuleId, TokenId,
    test_utils::heap::AdjustableHeap,
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub(super) InputTextPos, u32);

type Heap = AdjustableHeap<InputTextPos, RuleId>;

pub fn bpe_with_heap<const ALLOW_IMPROPER_RULES: bool>(
    dict: &Dictionary,
    seq: impl Into<Vec<TokenId>>,
) -> Vec<TokenId> {
    let seq = seq.into();
    let seq_len = seq.len();
    if seq_len <= 1 {
        return seq;
    }

    let mut next_split_pos: TypedVec<InputTextPos, _> =
        (1..seq_len + 1).map(InputTextPos::from).collect();
    debug_assert!(next_split_pos.enumerate_copied().all(|(i, j)| i < j));
    let mut prev_split_pos =
        TypedVec::new_with(InputTextPos::ZERO, InputTextPos::from(seq_len + 1));
    for (cur, next) in next_split_pos.enumerate_copied() {
        if next < prev_split_pos.len() {
            prev_split_pos[next] = cur;
        }
    }

    let mut seq = TypedVec::from(seq);
    let mut heap = Heap::new(
        seq.len(),
        seq.as_slice()
            .windows(2)
            .enumerate()
            .flat_map(|(i, window)| {
                let left = window[0];
                let right = window[1];
                dict.find_rule(left, right)
                    .map(|rule_id| (InputTextPos::from(i + 1), rule_id))
            }),
    );

    while let Some((pos, rule_id)) = heap.pop() {
        let rule = &dict[rule_id];

        debug_assert!(
            InputTextPos::ZERO < pos && pos < prev_split_pos.len() && pos < next_split_pos.len()
        );

        let prev_pos = prev_split_pos[pos];
        let next_pos = next_split_pos[pos];

        debug_assert!(prev_pos < pos && pos < next_pos);

        debug_assert_eq!(seq[prev_pos], rule.pre);
        debug_assert_eq!(seq[pos], rule.suc);

        seq[prev_pos] = rule.merged;

        debug_assert!(prev_pos < next_split_pos.len());
        debug_assert!(InputTextPos::ZERO < next_pos && next_pos < prev_split_pos.len());

        next_split_pos[prev_pos] = next_pos;
        prev_split_pos[next_pos] = prev_pos;

        if prev_pos > InputTextPos::ZERO {
            if let Some(new_rule_id) = dict.find_rule(seq[prev_split_pos[prev_pos]], rule.merged)
                && (ALLOW_IMPROPER_RULES || new_rule_id > rule_id)
            {
                heap.set(prev_pos, new_rule_id);
            } else {
                heap.remove(prev_pos);
            }
        }

        if next_pos < seq.len() {
            if let Some(new_rule_id) = dict.find_rule(rule.merged, seq[next_pos])
                && (ALLOW_IMPROPER_RULES || new_rule_id > rule_id)
            {
                heap.set(next_pos, new_rule_id);
            } else {
                heap.remove(next_pos);
            }
        }
    }

    let mut res = Vec::with_capacity(seq.len().as_usize());
    let mut cur = InputTextPos::ZERO;
    while cur < next_split_pos.len() {
        res.push(seq[cur]);
        cur = next_split_pos[cur];
    }
    res
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, TokenId, Vocab,
        test_utils::{bpe_with_heap, bytes_into_tokens, utf8_into_tokens},
    };

    fn build_dict<T: AsRef<[u8]>, R: IntoIterator<Item = (T, T)>>(
        vocab: &Vocab,
        rules: R,
    ) -> Dictionary {
        Dictionary::new_from_token_pair(vocab.clone(), rules).unwrap()
    }

    fn check_in_bytes<S: AsRef<[u8]>, I: Into<TokenId>, T: IntoIterator<Item = I>>(
        dict: &Dictionary,
        seq: S,
        tokens: T,
    ) {
        let seq = seq.as_ref();
        let tokens: Vec<_> = tokens.into_iter().map(I::into).collect();
        let inputs = bytes_into_tokens(dict, seq, 0usize);
        assert_eq!(bpe_with_heap::<true>(dict, inputs), tokens);
        check_properly_in_bytes(dict, seq, tokens);
    }

    fn check_in_utf8<S: AsRef<str>, I: Into<TokenId>, T: IntoIterator<Item = I>>(
        dict: &Dictionary,
        seq: S,
        tokens: T,
    ) {
        let seq = seq.as_ref();
        let tokens: Vec<_> = tokens.into_iter().map(I::into).collect();
        let inputs = utf8_into_tokens(dict, seq, 0usize);
        assert_eq!(bpe_with_heap::<true>(dict, inputs), tokens);
        check_properly_in_utf8(dict, seq, tokens);
    }

    fn check_properly_in_bytes<S: AsRef<[u8]>, I: Into<TokenId>, T: IntoIterator<Item = I>>(
        dict: &Dictionary,
        seq: S,
        tokens: T,
    ) {
        let tokens: Vec<_> = tokens.into_iter().map(I::into).collect();
        let inputs = bytes_into_tokens(dict, seq, 0usize);
        assert_eq!(bpe_with_heap::<false>(dict, inputs), tokens);
    }

    fn check_properly_in_utf8<S: AsRef<str>, I: Into<TokenId>, T: IntoIterator<Item = I>>(
        dict: &Dictionary,
        seq: S,
        tokens: T,
    ) {
        let tokens: Vec<_> = tokens.into_iter().map(I::into).collect();
        let inputs = utf8_into_tokens(dict, seq, 0usize);
        assert_eq!(bpe_with_heap::<false>(dict, inputs), tokens);
    }

    #[test]
    fn test_bpe_with_heap() {
        let vocab = Vocab::new([
            b"" as &[_],
            b"a",
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
            "好你".as_bytes(),
            b"\xe4",
            b"\xbd",
            b"\xa0",
            b"\xbd\xa0",
            b"aa",
            b"aaa",
        ])
        .unwrap();

        let dict = build_dict(&vocab, [("c", "d"), ("b", "cd"), ("a", "bcd")]);
        check_in_bytes(&dict, "dcdbcdabcdab", [4u32, 5, 6, 7, 1, 2]);
        check_in_utf8(&dict, "dcdbcdabcdab", [4u32, 5, 6, 7, 1, 2]);

        let dict = build_dict(
            &vocab,
            [(b"\xbd" as &[_], b"\xa0" as &[_]), (b"\xe4", b"\xbd\xa0")],
        );
        check_in_bytes(&dict, "你好", [8u32, 0, 0, 15]);
        check_properly_in_utf8(&dict, "你好", [8u32, 9]);
        let output = utf8_into_tokens(&dict, "你好", 0usize);
        assert_eq!(output, [8, 9].map(TokenId::new));

        let dict = build_dict(&vocab, [("你", "好")]);
        check_in_utf8(&dict, "你好", [11u32]);
        check_in_utf8(&dict, "你好呀", [11u32, 10]);
        check_in_utf8(
            &dict,
            "你好你好你好呀你好你好你",
            [11u32, 11, 11, 10, 11, 11, 8],
        );

        let dict = build_dict(&vocab, [("你", "好"), ("你好", "呀")]);
        check_in_utf8(&dict, "你好", [11u32]);
        check_in_utf8(&dict, "你好呀", [12u32]);
        check_in_utf8(
            &dict,
            "你好你好你好呀你好你好你",
            [11u32, 11, 12, 11, 11, 8],
        );

        let verify = |rules: &[(&str, &str)], seq: &str, tokens: &[u32]| {
            let dict = build_dict(&vocab, rules.iter().copied());
            check_in_utf8(&dict, seq, tokens.iter().copied());
        };

        verify(&[("你", "好"), ("你好", "呀")], "", &[]);
        verify(&[("你", "好"), ("你好", "呀")], "你", &[8]);

        let long_case = "好你好你好呀你好你好你";
        verify(
            &[("你", "好"), ("你好", "呀"), ("好", "你")],
            long_case,
            &[9, 11, 12, 11, 11, 8],
        );
        verify(
            &[("你", "好"), ("好", "你"), ("你好", "呀")],
            long_case,
            &[9, 11, 12, 11, 11, 8],
        );
        verify(
            &[("好", "你"), ("你", "好"), ("你好", "呀")],
            long_case,
            &[13, 13, 9, 10, 8, 13, 13],
        );

        let long_case = "你好你好你好呀你好你好你";
        verify(
            &[("你", "好"), ("你好", "呀"), ("好", "你")],
            long_case,
            &[11, 11, 12, 11, 11, 8],
        );
        verify(
            &[("你", "好"), ("好", "你"), ("你好", "呀")],
            long_case,
            &[11, 11, 12, 11, 11, 8],
        );
        verify(
            &[("好", "你"), ("你", "好"), ("你好", "呀")],
            long_case,
            &[8, 13, 13, 9, 10, 8, 13, 13],
        );

        verify(&[("a", "a")], "aaaaa", &[18, 18, 1]);
        verify(&[("a", "a")], "aaaaaa", &[18, 18, 18]);

        verify(&[("a", "a"), ("aa", "a")], "aaaaa", &[18, 19]);
        verify(&[("a", "a"), ("aa", "a")], "aaaaaa", &[18, 18, 18]);
        verify(&[("a", "a"), ("aa", "a")], "aaaaaaa", &[18, 18, 19]);

        verify(&[("a", "a"), ("a", "aa")], "aaaaa", &[18, 18, 1]);
        verify(&[("a", "a"), ("a", "aa")], "aaaaaa", &[18, 18, 18]);
        verify(&[("a", "a"), ("a", "aa")], "aaaaaaa", &[18, 18, 18, 1]);

        let check_properly = |rules: &[(&str, &str)], seq: &str, tokens: &[u32]| {
            let dict = build_dict(&vocab, rules.iter().copied());
            check_properly_in_utf8(&dict, seq, tokens.iter().copied());
        };

        let long_case = "好你好你好呀你好你好你";
        check_properly(
            &[("你好", "呀"), ("你", "好"), ("好", "你")],
            long_case,
            &[9, 11, 11, 10, 11, 11, 8],
        );
        check_properly(
            &[("你好", "呀"), ("好", "你"), ("你", "好")],
            long_case,
            &[13, 13, 9, 10, 8, 13, 13],
        );
        check_properly(
            &[("好", "你"), ("你好", "呀"), ("你", "好")],
            long_case,
            &[13, 13, 9, 10, 8, 13, 13],
        );

        let long_case = "你好你好你好呀你好你好你";
        check_properly(
            &[("你好", "呀"), ("你", "好"), ("好", "你")],
            long_case,
            &[11, 11, 11, 10, 11, 11, 8],
        );
        check_properly(
            &[("你好", "呀"), ("好", "你"), ("你", "好")],
            long_case,
            &[8, 13, 13, 9, 10, 8, 13, 13],
        );
        check_properly(
            &[("好", "你"), ("你好", "呀"), ("你", "好")],
            long_case,
            &[8, 13, 13, 9, 10, 8, 13, 13],
        );

        check_properly(&[("aa", "a"), ("a", "a")], "aaaaa", &[18, 18, 1]);
        check_properly(&[("aa", "a"), ("a", "a")], "aaaaaa", &[18, 18, 18]);
        check_properly(&[("aa", "a"), ("a", "a")], "aaaaaaa", &[18, 18, 18, 1]);
    }
}
