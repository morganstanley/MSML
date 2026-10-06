use super::{AddedToken, ConfigMatcher, Seg, SegmenterConfig, SpecialSegmenter};
use crate::Encoding;

#[derive(Debug, PartialEq)]
enum Ev {
    Txt(String),
    Sp(u32),
}

fn segment_specials(specials: &[(&str, u32)], chunks: &[&str]) -> Vec<Ev> {
    let mut segmenter = SpecialSegmenter::from_specials(specials);
    collect_segments(&mut segmenter, chunks)
}

fn segment_added(tokens: &[AddedToken], chunks: &[&str]) -> Vec<Ev> {
    let mut segmenter = SpecialSegmenter::from_added_tokens(tokens);
    collect_segments(&mut segmenter, chunks)
}

fn collect_segments(segmenter: &mut SpecialSegmenter, chunks: &[&str]) -> Vec<Ev> {
    let mut output = Vec::new();
    let mut push = |segment: Seg| match segment {
        Seg::Text(text) => match output.last_mut() {
            Some(Ev::Txt(previous)) => previous.push_str(text),
            _ => output.push(Ev::Txt(text.to_owned())),
        },
        Seg::Special(id) => output.push(Ev::Sp(id)),
    };
    for chunk in chunks {
        segmenter.feed(chunk, &mut push);
    }
    segmenter.finish(&mut push);
    output
}

#[test]
fn compiled_config_is_shared_but_stream_state_is_independent() {
    let config = SegmenterConfig::from_specials(&[("<special>", 7)]);
    let cloned = config.clone();
    match (&config.matcher, &cloned.matcher) {
        (ConfigMatcher::Exact(left), ConfigMatcher::Exact(right)) => {
            assert!(std::sync::Arc::ptr_eq(left, right));
        }
        _ => panic!("exact literals must compile to the exact matcher"),
    }

    let mut first = SpecialSegmenter::from_config(config);
    let mut second = SpecialSegmenter::from_config(cloned);
    assert_eq!(
        collect_segments(&mut first, &["a<spec", "ial>b"]),
        vec![Ev::Txt("a".into()), Ev::Sp(7), Ev::Txt("b".into())]
    );
    assert_eq!(
        collect_segments(&mut second, &["x<special>y"]),
        vec![Ev::Txt("x".into()), Ev::Sp(7), Ev::Txt("y".into())]
    );
}

fn segment(encoding: Encoding, chunks: &[&str]) -> Vec<Ev> {
    segment_specials(encoding.special_tokens(), chunks)
}

#[test]
fn multibyte_starts() {
    use Ev::*;
    const SPECIALS: &[(&str, u32)] = &[("<s>", 1), ("[INST]", 2), ("{tool}", 3)];
    assert_eq!(
        segment_specials(SPECIALS, &["a<s>b[INST]c{tool}d"]),
        vec![
            Txt("a".into()),
            Sp(1),
            Txt("b".into()),
            Sp(2),
            Txt("c".into()),
            Sp(3),
            Txt("d".into())
        ]
    );
    assert_eq!(
        segment_specials(SPECIALS, &["x[y<s>"]),
        vec![Txt("x[y".into()), Sp(1)]
    );
    assert_eq!(
        segment_specials(SPECIALS, &["p[IN", "ST]q"]),
        vec![Txt("p".into()), Sp(2), Txt("q".into())]
    );
    assert_eq!(
        segment_specials(SPECIALS, &["plain text only"]),
        vec![Txt("plain text only".into())]
    );
}

#[test]
fn exact_false_prefixes_do_not_fragment_text_events() {
    use Ev::*;

    let input = "if (a < b) { x << 1; y <|not-a-special|>; }".repeat(16);
    let mut segmenter = SpecialSegmenter::from_specials(&[("<|endoftext|>", 50_256)]);
    let mut events = Vec::new();
    segmenter.feed_unprofiled(&input, |segment| match segment {
        Seg::Text(text) => events.push(Txt(text.to_owned())),
        Seg::Special(id) => events.push(Sp(id)),
    });
    segmenter.finish(|segment| match segment {
        Seg::Text(text) => events.push(Txt(text.to_owned())),
        Seg::Special(id) => events.push(Sp(id)),
    });

    assert_eq!(events, [Txt(input.to_owned())]);
}

#[test]
fn exact_many_false_prefixes_preserve_cross_chunk_specials() {
    use Ev::*;

    let code = "if (a < b) { x << 1; }".repeat(16);
    let left = format!("{code}before<|endof");
    let right = format!("text|>after{code}");
    assert_eq!(
        segment_specials(&[("<|endoftext|>", 50_256)], &[&left, &right]),
        vec![
            Txt(format!("{code}before")),
            Sp(50_256),
            Txt(format!("after{code}")),
        ]
    );
}

#[test]
fn r50k_endoftext() {
    use Ev::*;
    let encoding = Encoding::R50k;
    assert_eq!(segment(encoding, &["abc"]), vec![Txt("abc".into())]);
    assert_eq!(
        segment(encoding, &["a<|endoftext|>b"]),
        vec![Txt("a".into()), Sp(50_256), Txt("b".into())]
    );
    assert_eq!(
        segment(encoding, &["a<|endofte"]),
        vec![Txt("a<|endofte".into())]
    );
    assert_eq!(
        segment(encoding, &["<|<|endoftext|>"]),
        vec![Txt("<|".into()), Sp(50_256)]
    );
    assert_eq!(
        segment(encoding, &["<|endoftext|x"]),
        vec![Txt("<|endoftext|x".into())]
    );
    assert_eq!(segment(encoding, &["<|end", "oftext|>"]), vec![Sp(50_256)]);
    assert_eq!(
        segment(encoding, &["before<", "|endoftext|>after"]),
        vec![Txt("before".into()), Sp(50_256), Txt("after".into()),]
    );
    assert_eq!(
        segment(encoding, &["a<|endof", "text|>b"]),
        vec![Txt("a".into()), Sp(50_256), Txt("b".into())]
    );
    assert_eq!(
        segment(encoding, &["<|fim_prefix|>"]),
        vec![Txt("<|fim_prefix|>".into())]
    );
}

#[test]
fn cl100k_multi() {
    use Ev::*;
    let encoding = Encoding::Cl100k;
    assert_eq!(
        segment(
            encoding,
            &["<|fim_prefix|>x<|fim_middle|>y<|fim_suffix|>z<|endofprompt|>"]
        ),
        vec![
            Sp(100_258),
            Txt("x".into()),
            Sp(100_259),
            Txt("y".into()),
            Sp(100_260),
            Txt("z".into()),
            Sp(100_276),
        ]
    );
    assert_eq!(segment(encoding, &["<|endoftext|>"]), vec![Sp(100_257)]);
    assert_eq!(segment(encoding, &["<|endofprompt|>"]), vec![Sp(100_276)]);
    assert_eq!(
        segment(encoding, &["<|endofX"]),
        vec![Txt("<|endofX".into())]
    );
    assert_eq!(
        segment(encoding, &["<|endof", "prompt|>"]),
        vec![Sp(100_276)]
    );
    assert_eq!(
        segment(encoding, &["<|fim_", "suffix|>"]),
        vec![Sp(100_260)]
    );
}

#[test]
fn o200k_endofprompt() {
    use Ev::*;
    let encoding = Encoding::O200k;
    assert_eq!(segment(encoding, &["<|endofprompt|>"]), vec![Sp(200_018)]);
    assert_eq!(segment(encoding, &["<|endoftext|>"]), vec![Sp(199_999)]);
    assert_eq!(
        segment(encoding, &["<|fim_prefix|>"]),
        vec![Txt("<|fim_prefix|>".into())]
    );
}

#[test]
fn roberta_mask_lstrip_across_chunks() {
    use Ev::*;
    let mask = AddedToken {
        content: "<mask>".into(),
        id: 50_264,
        single_word: false,
        lstrip: true,
        rstrip: false,
        normalized: false,
        special: true,
    };

    assert_eq!(
        segment_added(std::slice::from_ref(&mask), &["hello   <ma", "sk>world"]),
        vec![Txt("hello".into()), Sp(50_264), Txt("world".into())]
    );
    assert_eq!(
        segment_added(
            std::slice::from_ref(&mask),
            &["hello ", " \t", "<", "mask>", "world"]
        ),
        vec![Txt("hello".into()), Sp(50_264), Txt("world".into())]
    );
    assert_eq!(
        segment_added(std::slice::from_ref(&mask), &["hello ", " \t"]),
        vec![Txt("hello  \t".into())]
    );
}

#[test]
fn rstrip_waits_for_whitespace_run_to_end() {
    use Ev::*;
    let token = AddedToken {
        content: "[X]".into(),
        id: 7,
        single_word: false,
        lstrip: false,
        rstrip: true,
        normalized: false,
        special: true,
    };
    assert_eq!(
        segment_added(
            std::slice::from_ref(&token),
            &["before[X]", " \t", "\u{2000}", "after"]
        ),
        vec![Txt("before".into()), Sp(7), Txt("after".into())]
    );
    assert_eq!(
        segment_added(std::slice::from_ref(&token), &["[X]", "  "]),
        vec![Sp(7)]
    );
}

#[test]
fn single_word_uses_unicode_boundaries_across_chunks() {
    use Ev::*;
    let ing = AddedToken {
        content: "ing".into(),
        id: 9,
        single_word: true,
        lstrip: false,
        rstrip: false,
        normalized: true,
        special: false,
    };
    assert_eq!(
        segment_added(
            std::slice::from_ref(&ing),
            &["morn", "ing ing", " ing", "ot"]
        ),
        vec![Txt("morning ".into()), Sp(9), Txt(" ingot".into())]
    );
    assert_eq!(
        segment_added(std::slice::from_ref(&ing), &["ing", "x"]),
        vec![Txt("ingx".into())]
    );
    assert_eq!(
        segment_added(std::slice::from_ref(&ing), &["ing"]),
        vec![Sp(9)]
    );

    let mark = "\u{0300}";
    assert_eq!(
        segment_added(std::slice::from_ref(&ing), &[&format!("{mark}ing")]),
        vec![Txt(format!("{mark}ing"))]
    );
}

#[test]
fn modifier_match_is_leftmost_longest() {
    use Ev::*;
    let tokens = [
        AddedToken::exact("<x>", 1),
        AddedToken {
            content: "<x>long".into(),
            id: 2,
            single_word: false,
            lstrip: false,
            rstrip: true,
            normalized: false,
            special: true,
        },
    ];
    assert_eq!(
        segment_added(&tokens, &["a<x>", "long ", "b"]),
        vec![Txt("a".into()), Sp(2), Txt("b".into())]
    );

    let exact = [AddedToken::exact("<x>", 1), AddedToken::exact("<x>long", 2)];
    assert_eq!(
        segment_added(&exact, &["a<x>", "longb"]),
        vec![Txt("a".into()), Sp(2), Txt("b".into())]
    );
}

#[test]
fn exact_unicode_added_token_is_safe_across_chunks() {
    use Ev::*;
    let token = AddedToken::exact("你好", 42);
    assert_eq!(
        segment_added(std::slice::from_ref(&token), &["a你", "好b"]),
        vec![Txt("a".into()), Sp(42), Txt("b".into())]
    );
    assert_eq!(
        segment_added(std::slice::from_ref(&token), &["a你", "呀b"]),
        vec![Txt("a你呀b".into())]
    );
}
