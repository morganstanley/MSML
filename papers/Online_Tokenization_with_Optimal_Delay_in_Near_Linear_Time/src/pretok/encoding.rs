//! Tokenizer split patterns and their DFA-compatible rewrites.
//!
//! Many encoding (e.g. `r50k_base`/gpt2, `cl100k_base`, `o200k_base`) splits
//! text with a regex that contains features a DFA cannot handle directly —
//! possessive quantifiers, the end-anchored whitespace branch (`\s++` then end),
//! and the `\s+(?!\S)` negative lookahead. For each we keep:
//!   * the exact reference pattern (run by fancy-regex's backtracking VM), and
//!   * a DFA-compatible rewrite (possessives → greedy, lookahead/anchor dropped)
//!     plus a small boundary [`Encoding::apply_fixup`] that recovers the dropped
//!     branches by post-processing each match's end offset.
//!
//! The fixups come in two flavors, recovering the dropped branches:
//!   * **extend** — replaces the end-anchored whitespace branch: snap a
//!     pure-whitespace piece ending in a newline out to end-of-string when the
//!     remainder is horizontal whitespace.
//!   * **trim** — replaces `\s+(?!\S)`: drop one trailing whitespace char that
//!     belongs to the next piece.
//!
//! Equivalence of each DFA pattern + fixup to its reference is checked
//! exhaustively over a character-class alphabet in `examples/fst_equiv.rs`.

// ----- r50k_base / gpt2 / p50k_base ---------------------------------------

/// Exact gpt2 / r50k_base / p50k_base split pattern
pub const R50K_SPLIT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

/// DFA-compatible r50k pattern
pub const R50K_DFA_PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

// ----- cl100k_base --------------------------------------------------------

/// Exact cl100k_base split pattern
pub const CL100K_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

/// DFA-compatible cl100k pattern.
pub const CL100K_DFA_PATTERN: &str = concat!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]",
    r"|[^\S\r\n]+",
);

// ----- o200k_base ---------------------------------------------------------

/// Exact o200k_base split pattern.
pub const O200K_SPLIT_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// DFA-compatible o200k pattern:
pub const O200K_DFA_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// ----- llama3 (Meta-Llama-3, byte-level BPE) ------------------------------

/// Exact Llama 3 split pattern.
pub const LLAMA3_SPLIT_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// DFA-compatible Llama 3 pattern
pub const LLAMA3_DFA_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// ----- mistral (Mistral-NeMo "Tekken", byte-level BPE) --------------------

/// Exact Mistral-NeMo split pattern
pub const MISTRAL_SPLIT_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// DFA-compatible Mistral pattern
pub const MISTRAL_DFA_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// ----- qwen3 (Qwen-3, byte-level BPE + NFC) --------------------------------

/// Exact Qwen-3 split pattern. Qwen-3 also applies NFC
/// normalization, which is handled separately by [`Normalizer`](crate::Normalizer).
pub const QWEN3_SPLIT_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// DFA-compatible Qwen-3 pattern
pub const QWEN3_DFA_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+",
);

// ----- encoding selector --------------------------------------------------

/// Which tiktoken split pattern (and matching fixup) a pretokenizer uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// gpt2 / r50k_base.
    R50k,
    /// p50k_base / p50k_edit
    P50k,
    /// cl100k_base (GPT-3.5/4).
    Cl100k,
    /// o200k_base (GPT-4o).
    O200k,
    /// Meta-Llama-3 byte-level BPE.
    Llama3,
    /// Mistral-NeMo ("Tekken") byte-level BPE.
    Mistral,
    /// Qwen-3 byte-level BPE.
    Qwen3,
}

impl Encoding {
    /// Parse an encoding name. Accepts `r50k`/`gpt2`, `p50k`, `cl100k`, `o200k`
    /// (and the `*_base` spellings).
    pub fn from_name(name: &str) -> Option<Encoding> {
        match name {
            "r50k" | "r50k_base" | "gpt2" => Some(Encoding::R50k),
            "p50k" | "p50k_base" | "p50k_edit" => Some(Encoding::P50k),
            "cl100k" | "cl100k_base" => Some(Encoding::Cl100k),
            "o200k" | "o200k_base" => Some(Encoding::O200k),
            "llama3" => Some(Encoding::Llama3),
            "mistral" => Some(Encoding::Mistral),
            "qwen3" => Some(Encoding::Qwen3),
            _ => None,
        }
    }

    /// Short canonical name, used in benchmark labels and dump filenames.
    pub fn name(self) -> &'static str {
        match self {
            Encoding::R50k => "r50k",
            Encoding::P50k => "p50k",
            Encoding::Cl100k => "cl100k",
            Encoding::O200k => "o200k",
            Encoding::Llama3 => "llama3",
            Encoding::Mistral => "mistral",
            Encoding::Qwen3 => "qwen3",
        }
    }

    pub fn special_tokens(self) -> &'static [(&'static str, u32)] {
        match self {
            // p50k_base shares r50k's single `<|endoftext|>` at 50256 (p50k_edit
            // adds FIM tokens, not handled here).
            Encoding::R50k | Encoding::P50k => &[("<|endoftext|>", 50256)],
            Encoding::Cl100k => &[
                ("<|endoftext|>", 100257),
                ("<|fim_prefix|>", 100258),
                ("<|fim_middle|>", 100259),
                ("<|fim_suffix|>", 100260),
                ("<|endofprompt|>", 100276),
            ],
            Encoding::O200k => &[("<|endoftext|>", 199999), ("<|endofprompt|>", 200018)],
            // HuggingFace models carry their specials in tokenizer.json (loaded at
            // runtime and passed to the segmenter explicitly), not here.
            Encoding::Llama3 | Encoding::Mistral | Encoding::Qwen3 => &[],
        }
    }

    /// The exact reference split pattern (for the fancy-regex VM).
    pub fn split_pattern(self) -> &'static str {
        match self {
            Encoding::R50k | Encoding::P50k => R50K_SPLIT_PATTERN,
            Encoding::Cl100k => CL100K_SPLIT_PATTERN,
            Encoding::O200k => O200K_SPLIT_PATTERN,
            Encoding::Llama3 => LLAMA3_SPLIT_PATTERN,
            Encoding::Mistral => MISTRAL_SPLIT_PATTERN,
            Encoding::Qwen3 => QWEN3_SPLIT_PATTERN,
        }
    }

    /// The DFA-compatible split pattern (combine with `Self::apply_fixup`).
    pub fn dfa_pattern(self) -> &'static str {
        match self {
            Encoding::R50k | Encoding::P50k => R50K_DFA_PATTERN,
            Encoding::Cl100k => CL100K_DFA_PATTERN,
            Encoding::O200k => O200K_DFA_PATTERN,
            Encoding::Llama3 => LLAMA3_DFA_PATTERN,
            Encoding::Mistral => MISTRAL_DFA_PATTERN,
            Encoding::Qwen3 => QWEN3_DFA_PATTERN,
        }
    }

    /// Recover the true boundary from a DFA match `[pos..end]` over
    /// [`Self::dfa_pattern`], given the full input of length `n`.
    #[inline]
    pub(crate) fn apply_fixup(self, input: &str, pos: usize, end: usize, n: usize) -> usize {
        match self {
            // cl100k: extend a trailing whitespace/newline piece to EOF, then
            // trim a dangling non-newline whitespace char.
            Encoding::Cl100k => {
                let end = extend(input, pos, end, n);
                trim(input, pos, end, n, /* skip_newline */ true)
            }
            // o200k / llama3 / mistral: trim only (their \s*[\r\n]+ branch peels
            // newline runs, so the trim is non-newline); no extend.
            Encoding::O200k | Encoding::Llama3 | Encoding::Mistral | Encoding::Qwen3 => {
                trim(input, pos, end, n, /* skip_newline */ true)
            }
            // r50k / p50k: trim any trailing whitespace (no \s*[\r\n] branch); no extend.
            Encoding::R50k | Encoding::P50k => {
                trim(input, pos, end, n, /* skip_newline */ false)
            }
        }
    }
}

/// True for whitespace that is neither `\r` nor `\n` (the `[^\S\r\n]` class).
#[inline]
pub(crate) fn is_horizontal_ws(c: char) -> bool {
    c.is_whitespace() && !matches!(c, '\r' | '\n')
}

/// Decision for the *extend* fixup
#[inline]
pub(crate) fn should_extend(
    last_char: Option<char>,
    pretoken_all_ws: bool,
    tail_all_horizontal_ws: bool,
) -> bool {
    matches!(last_char, Some('\r') | Some('\n')) && pretoken_all_ws && tail_all_horizontal_ws
}

/// Decision for the *trim* fixup (mirrors `\s+(?!\S)`):
#[inline]
pub(crate) fn should_trim(
    last_char: char,
    has_more: bool,
    next_char: Option<char>,
    skip_newline: bool,
) -> bool {
    last_char.is_whitespace()
        && (!skip_newline || !matches!(last_char, '\r' | '\n'))
        && has_more
        && matches!(next_char, Some(c) if !c.is_whitespace())
}

/// Extend fixup over the whole matched slice `[pos..end]` (remainder `[end..n]`).
#[inline]
fn extend(input: &str, pos: usize, end: usize, n: usize) -> usize {
    let pretoken = &input[pos..end];
    let last = pretoken.chars().next_back();
    // Gate the two full-slice whitespace scans on the cheap newline check
    // (should_extend's first conjunct): a piece not ending in a newline can never
    // extend, so `&&` short-circuits before either scan runs. Without this gate
    // the scans would run on every match (eager argument evaluation).
    if matches!(last, Some('\r') | Some('\n'))
        && should_extend(
            last,
            pretoken.chars().all(char::is_whitespace),
            input[end..n].chars().all(is_horizontal_ws),
        )
    {
        n
    } else {
        end
    }
}

/// Trim fixup over the whole matched slice `[pos..end]` (remainder `[end..n]`).
#[inline]
fn trim(input: &str, pos: usize, end: usize, n: usize, skip_newline: bool) -> usize {
    if end <= pos {
        return end;
    }
    let last = input[pos..end].chars().next_back().unwrap();
    let lastlen = last.len_utf8();
    if should_trim(
        last,
        end - pos > lastlen,
        input[end..n].chars().next(),
        skip_newline,
    ) {
        end - lastlen
    } else {
        end
    }
}
