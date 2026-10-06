//! Parse a Hugging Face `tokenizer.json` for the byte-level BPE models used by
//! the benchmarks.
//!
//! This is deliberately only a format/semantics adapter. It does not download a
//! model and it does not build an MTC dictionary; callers can feed
//! [`ParsedTokenizer::ranked_tokens`] to the common vocabulary builder.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

use serde_json::{Map, Value};

use crate::{AddedToken, Normalizer, Presplit};

/// The in-memory equivalent of the converted vocabulary/metadata sidecars used
/// by the old repository-local model workflow.
#[derive(Clone, Debug)]
pub(crate) struct ParsedTokenizer {
    /// `(raw token bytes, dense merge rank)`, sorted by rank.
    pub(crate) ranked_tokens: Vec<(Vec<u8>, u32)>,
    /// Literal added token and its Hugging Face token id, sorted by id.
    ///
    /// Hugging Face recognizes added tokens even when `special=false`; that
    /// flag controls bookkeeping, not whether the literal is extracted.
    pub(crate) added_tokens: Vec<AddedToken>,
    /// Dense merge rank to the original Hugging Face token id.
    pub(crate) rank_to_id: Vec<u32>,
    pub(crate) normalizer: Normalizer,
    pub(crate) presplit: Presplit,
    /// The custom main split regex, or `None` for ByteLevel's built-in GPT-2
    /// regex. The model registry verifies this against the selected Encoding.
    pub(crate) split_pattern: Option<String>,
    /// Hugging Face's whole-piece-vocabulary shortcut. Callers must preserve
    /// this behavior rather than silently treating it as ordinary BPE.
    pub(crate) ignore_merges: bool,
}

/// A tokenizer JSON error with enough context to diagnose an unsupported model
/// without exposing serde's internal representation in the public API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HfJsonError(String);

impl HfJsonError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for HfJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for HfJsonError {}

/// Parse a complete Hugging Face `tokenizer.json` directly from its cached
/// bytes.
pub(crate) fn from_slice(json: &[u8]) -> Result<ParsedTokenizer, HfJsonError> {
    let root: Value = serde_json::from_slice(json)
        .map_err(|e| HfJsonError::new(format!("invalid tokenizer.json: {e}")))?;
    parse_root(&root)
}

fn parse_root(root: &Value) -> Result<ParsedTokenizer, HfJsonError> {
    let root = object(root, "tokenizer.json root")?;
    let model = root
        .get("model")
        .ok_or_else(|| HfJsonError::new("tokenizer.json is missing `model`"))
        .and_then(|v| object(v, "`model`"))?;

    // Old GPT-2 tokenizer.json files predate the tagged model representation
    // and omit `model.type`; their `vocab` + `merges` shape is unambiguously BPE.
    if let Some(model_type) = model.get("type") {
        let model_type = string(model_type, "`model.type`")?;
        if model_type != "BPE" {
            return Err(HfJsonError::new(format!(
                "unsupported model type {model_type:?}; only byte-level BPE is supported"
            )));
        }
    }
    validate_model_options(model)?;

    let normalizer = parse_normalizer(root.get("normalizer"))?;
    let (presplit, split_pattern) = parse_pretokenizer(root.get("pre_tokenizer"))?;
    let (added_tokens, added_ids) = parse_added_tokens(root.get("added_tokens"), normalizer)?;
    let vocab = parse_vocab(model, &added_ids)?;
    let merges = parse_merges(model.get("merges"))?;
    validate_merge_outputs(&vocab, &merges)?;
    let ignore_merges = optional_bool(model, "ignore_merges", false, "`model`")?;

    let id_candidate = id_order(&vocab)?;
    let chosen = if ignore_merges && !merges.is_empty() {
        let merge_candidate = merge_order(&vocab, &merges, ignore_merges)?;
        if valid_tree(&merge_candidate.ranked_tokens) {
            merge_candidate
        } else {
            return Err(HfJsonError::new(
                "merge-order ranks do not form a valid merge tree; \
                 this vocabulary requires rank reconstruction/properization",
            ));
        }
    } else if valid_tree(&id_candidate.ranked_tokens) {
        id_candidate
    } else if !merges.is_empty() {
        let merge_candidate = merge_order(&vocab, &merges, ignore_merges)?;
        if valid_tree(&merge_candidate.ranked_tokens) {
            merge_candidate
        } else {
            return Err(HfJsonError::new(
                "neither id- nor merge-order ranks form a valid merge tree; \
                 this vocabulary requires rank reconstruction/properization",
            ));
        }
    } else {
        return Err(HfJsonError::new(
            "id-order ranks do not form a valid merge tree and `model.merges` is empty",
        ));
    };

    Ok(ParsedTokenizer {
        ranked_tokens: chosen.ranked_tokens,
        added_tokens,
        rank_to_id: chosen.rank_to_id,
        normalizer,
        presplit,
        split_pattern,
        ignore_merges,
    })
}

fn validate_model_options(model: &Map<String, Value>) -> Result<(), HfJsonError> {
    if let Some(dropout) = model.get("dropout")
        && !dropout.is_null()
        && dropout.as_f64() != Some(0.0)
    {
        return Err(HfJsonError::new(
            "unsupported `model.dropout`; BPE dropout must be null or zero",
        ));
    }

    for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
        if let Some(value) = model.get(key)
            && !value.is_null()
            && value.as_str() != Some("")
        {
            return Err(HfJsonError::new(format!(
                "unsupported non-empty `model.{key}`"
            )));
        }
    }

    if let Some(unk) = model.get("unk_token")
        && !unk.is_null()
    {
        return Err(HfJsonError::new(
            "unsupported `model.unk_token`; only byte-complete BPE without unknown fallback is supported",
        ));
    }
    for key in ["fuse_unk", "byte_fallback"] {
        if optional_bool(model, key, false, "`model`")? {
            return Err(HfJsonError::new(format!("unsupported `model.{key}=true`")));
        }
    }
    // Validate the retained option's type here, even though its value is read
    // again when constructing the result.
    let _ = optional_bool(model, "ignore_merges", false, "`model`")?;
    Ok(())
}

fn parse_normalizer(value: Option<&Value>) -> Result<Normalizer, HfJsonError> {
    let Some(value) = value else {
        return Ok(Normalizer::None);
    };
    if value.is_null() {
        return Ok(Normalizer::None);
    }
    let normalizer = object(value, "`normalizer`")?;
    match required_string(normalizer, "type", "`normalizer`")? {
        "NFC" => Ok(Normalizer::Nfc),
        "Sequence" => {
            let stages = match normalizer.get("normalizers") {
                None => &[][..],
                Some(value) => array(value, "`normalizer.normalizers`")?,
            };
            let mut result = Normalizer::None;
            for stage in stages {
                match parse_normalizer(Some(stage))? {
                    Normalizer::None => {}
                    Normalizer::Nfc => result = Normalizer::Nfc,
                }
            }
            Ok(result)
        }
        kind => Err(HfJsonError::new(format!(
            "unsupported normalizer {kind:?}; only NFC (or none) is implemented"
        ))),
    }
}

#[derive(Default)]
struct PretokenizerState {
    byte_level: bool,
    byte_level_uses_regex: bool,
    split_pattern: Option<String>,
    digits: bool,
}

fn parse_pretokenizer(value: Option<&Value>) -> Result<(Presplit, Option<String>), HfJsonError> {
    let Some(value) = value else {
        return Err(HfJsonError::new(
            "missing `pre_tokenizer`; a ByteLevel pre-tokenizer is required",
        ));
    };
    if value.is_null() {
        return Err(HfJsonError::new(
            "null `pre_tokenizer`; a ByteLevel pre-tokenizer is required",
        ));
    }

    let mut state = PretokenizerState::default();
    visit_pretokenizer(value, &mut state)?;
    if !state.byte_level {
        return Err(HfJsonError::new(
            "unsupported pre-tokenizer: no ByteLevel stage",
        ));
    }
    if state.split_pattern.is_some() && state.byte_level_uses_regex {
        return Err(HfJsonError::new(
            "unsupported pre-tokenizer: Split followed by ByteLevel with `use_regex=true`",
        ));
    }
    if state.split_pattern.is_none() && !state.byte_level_uses_regex {
        return Err(HfJsonError::new(
            "unsupported pre-tokenizer: ByteLevel `use_regex=false` requires a registered Split stage",
        ));
    }
    let presplit = if state.digits {
        Presplit::DigitsIndividual
    } else {
        Presplit::None
    };
    Ok((presplit, state.split_pattern))
}

fn visit_pretokenizer(value: &Value, state: &mut PretokenizerState) -> Result<(), HfJsonError> {
    let stage = object(value, "pre-tokenizer stage")?;
    let kind = required_string(stage, "type", "pre-tokenizer stage")?;
    match kind {
        "Sequence" => {
            let stages = stage
                .get("pretokenizers")
                .ok_or_else(|| {
                    HfJsonError::new("Sequence pre-tokenizer is missing `pretokenizers`")
                })
                .and_then(|v| array(v, "`pre_tokenizer.pretokenizers`"))?;
            if stages.is_empty() {
                return Err(HfJsonError::new("unsupported empty Sequence pre-tokenizer"));
            }
            for child in stages {
                visit_pretokenizer(child, state)?;
            }
        }
        "Digits" => {
            ensure_before_byte_level(state, "Digits")?;
            if state.digits {
                return Err(HfJsonError::new(
                    "unsupported duplicate Digits pre-tokenizer stage",
                ));
            }
            if !optional_bool(stage, "individual_digits", false, "Digits pre-tokenizer")? {
                return Err(HfJsonError::new(
                    "unsupported grouped Digits pre-tokenizer; `individual_digits` must be true",
                ));
            }
            state.digits = true;
        }
        "Split" => {
            ensure_before_byte_level(state, "Split")?;
            if state.split_pattern.is_some() {
                return Err(HfJsonError::new(
                    "unsupported duplicate Split pre-tokenizer stage",
                ));
            }
            if required_string(stage, "behavior", "Split pre-tokenizer")? != "Isolated" {
                return Err(HfJsonError::new(
                    "unsupported Split behavior; only `Isolated` is implemented",
                ));
            }
            if optional_bool(stage, "invert", false, "Split pre-tokenizer")? {
                return Err(HfJsonError::new(
                    "unsupported Split pre-tokenizer with `invert=true`",
                ));
            }
            let pattern = stage
                .get("pattern")
                .ok_or_else(|| HfJsonError::new("Split pre-tokenizer is missing `pattern`"))
                .and_then(|v| object(v, "`Split.pattern`"))?;
            let regex = pattern
                .get("Regex")
                .ok_or_else(|| {
                    HfJsonError::new(
                        "unsupported Split pattern; a registered `Regex` pattern is required",
                    )
                })
                .and_then(|v| string(v, "`Split.pattern.Regex`"))?;
            if regex.is_empty() {
                return Err(HfJsonError::new("unsupported empty Split regex pattern"));
            }
            state.split_pattern = Some(regex.to_owned());
        }
        "ByteLevel" => {
            if state.byte_level {
                return Err(HfJsonError::new(
                    "unsupported duplicate ByteLevel pre-tokenizer stage",
                ));
            }
            if optional_bool(stage, "add_prefix_space", false, "ByteLevel pre-tokenizer")? {
                return Err(HfJsonError::new(
                    "unsupported ByteLevel `add_prefix_space=true`",
                ));
            }
            // Offset trimming does not affect token ids, but reject malformed
            // values instead of silently treating them as a default.
            let _ = optional_bool(stage, "trim_offsets", true, "ByteLevel pre-tokenizer")?;
            state.byte_level_uses_regex =
                optional_bool(stage, "use_regex", true, "ByteLevel pre-tokenizer")?;
            state.byte_level = true;
        }
        other => {
            return Err(HfJsonError::new(format!(
                "unsupported pre-tokenizer stage {other:?}; expected Digits, Split, or ByteLevel"
            )));
        }
    }
    Ok(())
}

fn ensure_before_byte_level(state: &PretokenizerState, stage: &str) -> Result<(), HfJsonError> {
    if state.byte_level {
        Err(HfJsonError::new(format!(
            "unsupported pre-tokenizer order: {stage} appears after ByteLevel"
        )))
    } else {
        Ok(())
    }
}

type AddedTokens = (Vec<AddedToken>, HashSet<u32>);

fn parse_added_tokens(
    value: Option<&Value>,
    normalizer: Normalizer,
) -> Result<AddedTokens, HfJsonError> {
    let Some(value) = value else {
        return Ok((Vec::new(), HashSet::new()));
    };
    if value.is_null() {
        return Ok((Vec::new(), HashSet::new()));
    }
    let added = array(value, "`added_tokens`")?;
    let mut literals = Vec::new();
    let mut added_ids = HashSet::new();
    let mut contents = HashSet::new();
    for (index, token) in added.iter().enumerate() {
        let context = format!("`added_tokens[{index}]`");
        let token = object(token, &context)?;
        // Both ordinary and special added tokens are extracted before BPE.
        // `special` controls decoding/bookkeeping, not whether the literal is
        // recognized during encoding.
        let special = optional_bool(token, "special", false, &context)?;
        let single_word = optional_bool(token, "single_word", false, &context)?;
        let lstrip = optional_bool(token, "lstrip", false, &context)?;
        let rstrip = optional_bool(token, "rstrip", false, &context)?;
        let normalized = optional_bool(token, "normalized", false, &context)?;
        if normalized && normalizer != Normalizer::None {
            return Err(HfJsonError::new(format!(
                "unsupported {context}.normalized=true with an active normalizer"
            )));
        }
        let content = required_string(token, "content", &context)?;
        if content.is_empty() {
            return Err(HfJsonError::new(format!(
                "{context}.content must not be empty"
            )));
        }
        let id = required_u32(token, "id", &context)?;
        if !added_ids.insert(id) {
            return Err(HfJsonError::new(format!("duplicate added-token id {id}")));
        }
        if !contents.insert(content.to_owned()) {
            return Err(HfJsonError::new(format!(
                "duplicate added-token literal {content:?}"
            )));
        }
        literals.push(AddedToken {
            content: content.to_owned(),
            id,
            single_word,
            lstrip,
            rstrip,
            normalized,
            special,
        });
    }
    literals.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.content.cmp(&b.content)));
    Ok((literals, added_ids))
}

#[derive(Clone)]
struct VocabEntry {
    id: u32,
    /// Added-token ids are not mergeable and need not use the ByteLevel alphabet.
    raw: Option<Vec<u8>>,
}

fn parse_vocab(
    model: &Map<String, Value>,
    added_ids: &HashSet<u32>,
) -> Result<Vec<VocabEntry>, HfJsonError> {
    let vocab = model
        .get("vocab")
        .ok_or_else(|| HfJsonError::new("BPE model is missing `vocab`"))
        .and_then(|v| object(v, "`model.vocab`"))?;
    if vocab.is_empty() {
        return Err(HfJsonError::new("BPE `model.vocab` is empty"));
    }

    let alphabet = ByteAlphabet::new();
    let mut seen_ids = HashMap::<u32, &str>::new();
    let mut entries = Vec::with_capacity(vocab.len());
    for (encoded, value) in vocab {
        let id = u32_value(value, &format!("vocabulary id for {encoded:?}"))?;
        if let Some(previous) = seen_ids.insert(id, encoded) {
            return Err(HfJsonError::new(format!(
                "duplicate vocabulary id {id} for {previous:?} and {encoded:?}"
            )));
        }
        let raw = if added_ids.contains(&id) {
            None
        } else {
            Some(alphabet.decode(encoded)?)
        };
        entries.push(VocabEntry { id, raw });
    }
    if entries.iter().all(|entry| entry.raw.is_none()) {
        return Err(HfJsonError::new(
            "BPE vocabulary contains no mergeable tokens after removing specials",
        ));
    }
    Ok(entries)
}

type Merge = (Vec<u8>, Vec<u8>);

fn validate_merge_outputs(vocab: &[VocabEntry], merges: &[Merge]) -> Result<(), HfJsonError> {
    let vocab_tokens: HashSet<&[u8]> = vocab
        .iter()
        .filter_map(|entry| entry.raw.as_deref())
        .collect();
    for (index, (left, right)) in merges.iter().enumerate() {
        let mut merged = Vec::with_capacity(left.len() + right.len());
        merged.extend_from_slice(left);
        merged.extend_from_slice(right);
        if !vocab_tokens.contains(merged.as_slice()) {
            return Err(HfJsonError::new(format!(
                "`model.merges[{index}]` produces a token absent from `model.vocab`"
            )));
        }
    }
    Ok(())
}

fn parse_merges(value: Option<&Value>) -> Result<Vec<Merge>, HfJsonError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let merges = array(value, "`model.merges`")?;
    let alphabet = ByteAlphabet::new();
    let mut result = Vec::with_capacity(merges.len());
    for (index, merge) in merges.iter().enumerate() {
        let context = format!("`model.merges[{index}]`");
        let (left, right) = match merge {
            Value::String(pair) => pair.split_once(' ').ok_or_else(|| {
                HfJsonError::new(format!(
                    "{context} must contain two tokens separated by a space"
                ))
            })?,
            Value::Array(pair) if pair.len() == 2 => (
                string(&pair[0], &format!("{context}[0]"))?,
                string(&pair[1], &format!("{context}[1]"))?,
            ),
            Value::Array(_) => {
                return Err(HfJsonError::new(format!(
                    "{context} must be a two-string array"
                )));
            }
            _ => {
                return Err(HfJsonError::new(format!(
                    "{context} must be a merge string or a two-string array"
                )));
            }
        };
        if left.is_empty() || right.is_empty() {
            return Err(HfJsonError::new(format!(
                "{context} contains an empty merge operand"
            )));
        }
        result.push((alphabet.decode(left)?, alphabet.decode(right)?));
    }
    Ok(result)
}

struct RankCandidate {
    ranked_tokens: Vec<(Vec<u8>, u32)>,
    rank_to_id: Vec<u32>,
}

fn id_order(vocab: &[VocabEntry]) -> Result<RankCandidate, HfJsonError> {
    let mut entries: Vec<&VocabEntry> = vocab.iter().filter(|entry| entry.raw.is_some()).collect();
    entries.sort_by_key(|entry| entry.id);
    let mut ranked_tokens = Vec::with_capacity(entries.len());
    let mut rank_to_id = Vec::with_capacity(entries.len());
    for (index, entry) in entries.into_iter().enumerate() {
        let rank = checked_rank(index)?;
        ranked_tokens.push((entry.raw.clone().expect("filtered above"), rank));
        rank_to_id.push(entry.id);
    }
    Ok(RankCandidate {
        ranked_tokens,
        rank_to_id,
    })
}

fn merge_order(
    vocab: &[VocabEntry],
    merges: &[Merge],
    ignore_merges: bool,
) -> Result<RankCandidate, HfJsonError> {
    let alphabet = ByteAlphabet::new();
    let mut ranked_tokens = Vec::with_capacity(256 + merges.len());
    let mut ranks = HashMap::<Vec<u8>, u32>::with_capacity(256 + merges.len());
    let vocab_ids: HashMap<&[u8], u32> = vocab
        .iter()
        .filter_map(|entry| entry.raw.as_deref().map(|raw| (raw, entry.id)))
        .collect();

    // Hugging Face/GPT-2 assigns the first 256 ranks in ByteLevel alphabet
    // order, which is deliberately not numeric byte order. Some valid models
    // have an incomplete byte vocabulary, so only retain bytes that actually
    // have an output id rather than inventing synthetic entries.
    for &byte in &alphabet.rank_order {
        if vocab_ids.contains_key([byte].as_slice()) {
            insert_rank(&mut ranked_tokens, &mut ranks, vec![byte])?;
        }
    }
    for (index, (left, right)) in merges.iter().enumerate() {
        let mut merged = Vec::with_capacity(left.len() + right.len());
        merged.extend_from_slice(left);
        merged.extend_from_slice(right);
        if !vocab_ids.contains_key(merged.as_slice()) {
            return Err(HfJsonError::new(format!(
                "`model.merges[{index}]` produces a token absent from `model.vocab`"
            )));
        }
        insert_rank(&mut ranked_tokens, &mut ranks, merged)?;
    }

    if ignore_merges {
        // A whole-piece token that has no real merge rule needs a separate
        // semantic lookup table; adding it to the merge dictionary would make
        // it mergeable inside larger pieces and change Hugging Face behavior.
        // None of the currently registered ignore_merges models require that
        // extension, so reject such a source until the two maps are distinct.
        for entry in vocab.iter().filter(|entry| entry.raw.is_some()) {
            let raw = entry.raw.as_deref().expect("filtered above");
            if !ranks.contains_key(raw) {
                return Err(HfJsonError::new(format!(
                    "`model.ignore_merges=true` token id {} has no producing merge; \
                     separate whole-piece-only vocabulary entries are not yet supported",
                    entry.id
                )));
            }
        }
    }

    let rank_to_id = ranked_tokens
        .iter()
        .map(|(raw, _)| {
            vocab_ids.get(raw.as_slice()).copied().ok_or_else(|| {
                HfJsonError::new("internal error: ranked token has no Hugging Face output id")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RankCandidate {
        ranked_tokens,
        rank_to_id,
    })
}

fn insert_rank(
    ranked_tokens: &mut Vec<(Vec<u8>, u32)>,
    ranks: &mut HashMap<Vec<u8>, u32>,
    token: Vec<u8>,
) -> Result<(), HfJsonError> {
    if ranks.contains_key(token.as_slice()) {
        return Ok(());
    }
    let rank = checked_rank(ranked_tokens.len())?;
    ranks.insert(token.clone(), rank);
    ranked_tokens.push((token, rank));
    Ok(())
}

fn valid_tree(ranked_tokens: &[(Vec<u8>, u32)]) -> bool {
    let ranks: HashMap<&[u8], u32> = ranked_tokens
        .iter()
        .map(|(token, rank)| (token.as_slice(), *rank))
        .collect();
    ranked_tokens.iter().all(|(token, rank)| {
        token.len() <= 1
            || (1..token.len()).any(|split| {
                ranks.get(&token[..split]).is_some_and(|left| left < rank)
                    && ranks.get(&token[split..]).is_some_and(|right| right < rank)
            })
    })
}

struct ByteAlphabet {
    rank_order: Vec<u8>,
    char_to_byte: HashMap<char, u8>,
}

impl ByteAlphabet {
    fn new() -> Self {
        let mut rank_order = Vec::with_capacity(256);
        rank_order.extend(b'!'..=b'~');
        rank_order.extend(0xa1..=0xac);
        rank_order.extend(0xae..=0xff);

        let mut present = [false; 256];
        for &byte in &rank_order {
            present[byte as usize] = true;
        }
        for byte in 0u8..=u8::MAX {
            if !present[byte as usize] {
                rank_order.push(byte);
            }
        }

        let mut char_to_byte = HashMap::with_capacity(256);
        let mut extra = 0u32;
        for &byte in &rank_order {
            let ch = if (b'!'..=b'~').contains(&byte)
                || (0xa1..=0xac).contains(&byte)
                || (0xae..=0xff).contains(&byte)
            {
                char::from(byte)
            } else {
                let ch = char::from_u32(256 + extra).expect("ByteLevel codepoint is valid");
                extra += 1;
                ch
            };
            char_to_byte.insert(ch, byte);
        }
        debug_assert_eq!(rank_order.len(), 256);
        debug_assert_eq!(char_to_byte.len(), 256);
        Self {
            rank_order,
            char_to_byte,
        }
    }

    fn decode(&self, encoded: &str) -> Result<Vec<u8>, HfJsonError> {
        let mut raw = Vec::with_capacity(encoded.len());
        for ch in encoded.chars() {
            let byte = self.char_to_byte.get(&ch).copied().ok_or_else(|| {
                HfJsonError::new(format!(
                    "token {encoded:?} contains non-ByteLevel character {ch:?}"
                ))
            })?;
            raw.push(byte);
        }
        if raw.is_empty() {
            return Err(HfJsonError::new(
                "empty tokens are unsupported in a BPE vocabulary",
            ));
        }
        Ok(raw)
    }
}

fn checked_rank(rank: usize) -> Result<u32, HfJsonError> {
    u32::try_from(rank).map_err(|_| HfJsonError::new("BPE vocabulary exceeds u32 ranks"))
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, HfJsonError> {
    value
        .as_object()
        .ok_or_else(|| HfJsonError::new(format!("{context} must be a JSON object")))
}

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a [Value], HfJsonError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| HfJsonError::new(format!("{context} must be a JSON array")))
}

fn string<'a>(value: &'a Value, context: &str) -> Result<&'a str, HfJsonError> {
    value
        .as_str()
        .ok_or_else(|| HfJsonError::new(format!("{context} must be a string")))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a str, HfJsonError> {
    object
        .get(key)
        .ok_or_else(|| HfJsonError::new(format!("{context} is missing `{key}`")))
        .and_then(|value| string(value, &format!("{context}.{key}")))
}

fn u32_value(value: &Value, context: &str) -> Result<u32, HfJsonError> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| HfJsonError::new(format!("{context} must be a u32 integer")))
}

fn required_u32(object: &Map<String, Value>, key: &str, context: &str) -> Result<u32, HfJsonError> {
    object
        .get(key)
        .ok_or_else(|| HfJsonError::new(format!("{context} is missing `{key}`")))
        .and_then(|value| u32_value(value, &format!("{context}.{key}")))
}

fn optional_bool(
    object: &Map<String, Value>,
    key: &str,
    default: bool,
    context: &str,
) -> Result<bool, HfJsonError> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| HfJsonError::new(format!("{context}.{key} must be a boolean"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_legacy_bpe_string_merges_and_stages() {
        let parsed = from_slice(
            br#"{
                "added_tokens": [
                    {"id": 99, "content": "<z>", "special": true},
                    {
                        "id": 98,
                        "content": "<a>",
                        "single_word": true,
                        "lstrip": true,
                        "rstrip": true,
                        "normalized": false,
                        "special": true
                    },
                    {"id": 77, "content": "ordinary", "special": false}
                ],
                "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}]},
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": [
                        {"type": "Digits", "individual_digits": true},
                        {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
                    ]
                },
                "model": {
                    "vocab": {"a": 0, "b": 1, "ab": 2},
                    "merges": ["a b"],
                    "ignore_merges": true
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            parsed.ranked_tokens,
            vec![(b"a".to_vec(), 0), (b"b".to_vec(), 1), (b"ab".to_vec(), 2)]
        );
        assert_eq!(parsed.rank_to_id, vec![0, 1, 2]);
        assert_eq!(
            parsed.added_tokens,
            vec![
                AddedToken {
                    content: "ordinary".into(),
                    id: 77,
                    single_word: false,
                    lstrip: false,
                    rstrip: false,
                    normalized: false,
                    special: false,
                },
                AddedToken {
                    content: "<a>".into(),
                    id: 98,
                    single_word: true,
                    lstrip: true,
                    rstrip: true,
                    normalized: false,
                    special: true,
                },
                AddedToken::exact("<z>", 99),
            ]
        );
        assert_eq!(parsed.normalizer, Normalizer::Nfc);
        assert_eq!(parsed.presplit, Presplit::DigitsIndividual);
        assert_eq!(parsed.split_pattern, None);
        assert!(parsed.ignore_merges);
    }

    #[test]
    fn accepts_pair_merges_and_falls_back_to_merge_order() {
        let parsed = from_slice(
            br#"{
                "pre_tokenizer": {"type": "ByteLevel", "use_regex": true},
                "model": {
                    "type": "BPE",
                    "vocab": {"ab": 0, "a": 1, "b": 2},
                    "merges": [["a", "b"]]
                }
            }"#,
        )
        .unwrap();

        assert_eq!(parsed.ranked_tokens.len(), 3);
        let a_rank = parsed
            .ranked_tokens
            .iter()
            .position(|(token, _)| token == b"a")
            .unwrap();
        let ab_rank = parsed
            .ranked_tokens
            .iter()
            .position(|(token, _)| token == b"ab")
            .unwrap();
        assert_eq!(ab_rank, 2);
        assert_eq!(parsed.rank_to_id[a_rank], 1);
        assert_eq!(parsed.rank_to_id[ab_rank], 0);
    }

    #[test]
    fn merge_fallback_never_invents_missing_vocab_ids() {
        let error = from_slice(
            br#"{
                "pre_tokenizer": {"type": "ByteLevel", "use_regex": true},
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1},
                    "merges": [["a", "b"]]
                }
            }"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("absent from `model.vocab`"));
    }

    #[test]
    fn ignore_merges_rejects_whole_piece_only_entries_until_maps_are_separate() {
        let error = from_slice(
            br#"{
                "pre_tokenizer": {"type": "ByteLevel", "use_regex": true},
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "ab": 2, "extra": 3},
                    "merges": [["a", "b"]],
                    "ignore_merges": true
                }
            }"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("whole-piece-only"));
    }

    #[test]
    fn accepts_registered_split_shape() {
        let parsed = from_slice(
            br#"{
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": [
                        {
                            "type": "Split",
                            "pattern": {"Regex": "\\p{N}"},
                            "behavior": "Isolated",
                            "invert": false
                        },
                        {"type": "ByteLevel", "use_regex": false}
                    ]
                },
                "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
            }"#,
        )
        .unwrap();
        assert_eq!(parsed.presplit, Presplit::None);
        assert_eq!(parsed.split_pattern.as_deref(), Some("\\p{N}"));
    }

    #[test]
    fn rejects_unsupported_semantics() {
        let cases = [
            (
                r#"{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"WordPiece","vocab":{"a":0}}}"#,
                "model type",
            ),
            (
                r#"{"normalizer":{"type":"NFKC"},"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"a":0}}}"#,
                "normalizer",
            ),
            (
                r#"{"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":true},"model":{"type":"BPE","vocab":{"a":0}}}"#,
                "add_prefix_space",
            ),
            (
                r#"{"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":false},{"type":"ByteLevel"}]},"model":{"type":"BPE","vocab":{"a":0}}}"#,
                "grouped Digits",
            ),
            (
                r#"{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","dropout":0.1,"vocab":{"a":0}}}"#,
                "dropout",
            ),
        ];
        for (json, expected) in cases {
            let error = from_slice(json.as_bytes()).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in error {error:?}"
            );
        }
    }
}
