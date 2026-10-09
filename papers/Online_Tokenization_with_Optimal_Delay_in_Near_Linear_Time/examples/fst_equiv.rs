//! Exhaustive equivalence check between each tiktoken encoding's exact
//! `fancy-regex` split pattern and its DFA-compatible pattern composed with
//! `apply_fixup`.
//!
//! The DFA pattern drops the branches a DFA cannot express (`\s+(?!\S)`, `\s++$`)
//! and relaxes possessive quantifiers to greedy ones; `apply_fixup` recovers the
//! boundaries those branches would have produced, given the complete input. Each
//! enumerated string is therefore fed whole, which is the premise the fixups'
//! contract assumes. Streaming-truncation behaviour is a separate property,
//! covered by the chunk sweeps in `pretok::stream_pretok`'s unit tests.
//!
//! Usage: cargo run --release --example fst_equiv -- [flags] [max_len] [encoding]
//!   (default max_len 13 and all four tiktoken encodings; the optional encoding
//!    is r50k, p50k, cl100k, or o200k)
//!   Flags:
//!     --quick        bounded per-encoding depth
//!
//! The enumeration runs on the rayon global thread pool (one engine bundle per
//! worker). It scales ~linearly with cores; set `RAYON_NUM_THREADS` to control the
//! count (under SLURM, pin it to `$SLURM_CPUS_PER_TASK`, as the sbatch does).

use std::cell::RefCell;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};

use fancy_regex::Regex;
use hiriluk::{Encoding, StreamPretokenizer};
use rayon::prelude::*;

/// One test case and its character-class representatives.
struct Case {
    enc: Encoding,
    alphabet: &'static [char],
}

// The literal contraction fragments every split pattern recognizes
const CONTRACTIONS: &[&str] = &["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
// Cell depth for the contraction pass: enough to place a fragment between two
// class neighbours on each side (its boundary decision is local).
const CONTRACTION_CELLS: usize = 4;

const CASES: &[Case] = &[
    Case {
        enc: Encoding::R50k,
        alphabet: &['a', '1', '\n', ' ', '\t', '.'],
    },
    Case {
        enc: Encoding::P50k,
        alphabet: &['a', '1', '\n', ' ', '\t', '.'],
    },
    Case {
        enc: Encoding::Cl100k,
        alphabet: &['a', '1', '\n', ' ', '\t', '.'],
    },
    Case {
        enc: Encoding::O200k,
        alphabet: &['A', 'a', '1', '\n', ' ', '\t', '.', '/'],
    },
];

/// Reference piece-length sequence: fancy-regex `find_iter` over the full split
/// pattern. Comparing length sequences suffices — both segmenters tile the string
/// left-to-right with no gaps, so equal lengths <=> equal pieces.
fn reference_lens(re: &Regex, s: &str, out: &mut Vec<usize>) {
    out.clear();
    out.extend(
        re.find_iter(s)
            .filter_map(Result::ok)
            .map(|m| m.as_str().len())
            .filter(|&n| n != 0),
    );
}

/// DFA piece-lengths: feed `s` whole, then `finish` (which resets the instance
/// for reuse).
fn dfa_lens(stream: &mut StreamPretokenizer, s: &str, out: &mut Vec<usize>) {
    out.clear();
    stream.feed(s, |p| out.push(p.len()));
    stream.finish(|p| out.push(p.len()));
}

/// Reconstruct the pieces a length sequence implies, for mismatch reports. Both
/// segmenters tile gaplessly, so slicing `s` by successive lengths recovers them;
/// the `.min` clamps defensively if a real bug made the lengths overshoot.
fn pieces_from_lens<'a>(s: &'a str, lens: &[usize]) -> Vec<&'a str> {
    let mut out = Vec::with_capacity(lens.len());
    let mut i = 0;
    for &l in lens {
        let end = (i + l).min(s.len());
        out.push(&s[i..end]);
        i = end;
    }
    out
}

/// Report a disagreement between the DFA and the reference, capped at the first
/// 30 globally.
fn report(
    enc: Encoding,
    label: &str,
    s: &str,
    expected: &[usize],
    got: &[usize],
    mismatches: &AtomicU64,
) {
    if got != expected && mismatches.fetch_add(1, Ordering::Relaxed) < 30 {
        println!("  MISMATCH [{}/{label}] on {s:?}", enc.name());
        println!("    reference: {:?}", pieces_from_lens(s, expected));
        println!("    dfa+fixup: {:?}", pieces_from_lens(s, got));
    }
}

/// Per-thread bundle containing the exact reference, the DFA-plus-fixup
/// pretokenizer, and reusable scratch buffers.
struct Engines {
    reference: Regex,
    dfa: StreamPretokenizer,
    expected: Vec<usize>,
    got: Vec<usize>,
}

impl Engines {
    fn build(enc: Encoding) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            reference: Regex::new(enc.split_pattern())?,
            dfa: StreamPretokenizer::new_with_mode(enc, true)?,
            expected: Vec::new(),
            got: Vec::new(),
        })
    }

    /// Check one string against the reference, reporting (and counting) a
    /// disagreement.
    fn check(&mut self, enc: Encoding, label: &str, s: &str, mismatches: &AtomicU64) {
        let Engines {
            reference,
            dfa,
            expected,
            got,
        } = self;
        reference_lens(reference, s, expected);
        dfa_lens(dfa, s, got);
        report(enc, label, s, expected, got, mismatches);
    }
}

fn parallel_check(
    enc: Encoding,
    label: &str,
    mismatches: &AtomicU64,
    total: u64,
    decode: impl Fn(u64, &mut String) + Sync + Send,
) {
    thread_local! {
        static WORKER: RefCell<Option<(Encoding, Engines, String)>> = const { RefCell::new(None) };
    }
    (0..total).into_par_iter().for_each(|code| {
        WORKER.with_borrow_mut(|slot| {
            if !matches!(slot, Some((e, _, _)) if *e == enc) {
                let eng = Engines::build(enc).expect("build validated by caller");
                *slot = Some((enc, eng, String::new()));
            }
            let (_, eng, s) = slot.as_mut().unwrap();
            decode(code, s);
            eng.check(enc, label, s.as_str(), mismatches);
        });
    });
}

/// Enumerate every string over `alphabet` up to `max_len` and check it. Returns
/// the number of strings checked. `label` tags the pass in progress / mismatches.
///
/// Space (U+0020) is admitted only at position 0, shrinking the interior alphabet.
/// Its sole class-distinct behavior is the leading ` ?` absorption — a position-0
/// phenomenon.
fn enumerate(
    enc: Encoding,
    label: &str,
    alphabet: &[char],
    max_len: usize,
    mismatches: &AtomicU64,
) -> u64 {
    let interior: Vec<char> = alphabet.iter().copied().filter(|&c| c != ' ').collect();
    let n0 = alphabet.len() as u64;
    let ni = interior.len() as u64;
    let mut checked: u64 = 0;
    for len in 1..=max_len {
        // total = n0 * ni^(len-1)
        let total = n0 * ni.pow((len - 1) as u32);
        parallel_check(enc, label, mismatches, total, |code, s| {
            s.clear();
            let mut c = code;
            s.push(alphabet[(c % n0) as usize]);
            c /= n0;
            for _ in 1..len {
                s.push(interior[(c % ni) as usize]);
                c /= ni;
            }
        });
        checked += total;
        println!(
            "  [{label} len {len:>2}] cumulative: checked {checked}, {} mismatches",
            mismatches.load(Ordering::Relaxed)
        );
    }
    checked
}

fn enumerate_contractions(
    enc: Encoding,
    alphabet: &[char],
    max_cells: usize,
    mismatches: &AtomicU64,
) -> u64 {
    // Symbols: each alphabet char as a 1-char string, then the contraction tokens.
    let singles: Vec<String> = alphabet.iter().map(|c| c.to_string()).collect();
    let symbols: Vec<&str> = singles
        .iter()
        .map(String::as_str)
        .chain(CONTRACTIONS.iter().copied())
        .collect();
    let nsym = symbols.len() as u64;
    let mut checked: u64 = 0;
    for cells in 1..=max_cells {
        let total = nsym.pow(cells as u32);
        parallel_check(enc, "contraction", mismatches, total, |code, s| {
            s.clear();
            let mut c = code;
            for _ in 0..cells {
                s.push_str(symbols[(c % nsym) as usize]);
                c /= nsym;
            }
        });
        checked += total;
        println!(
            "  [contraction cells {cells:>2}] cumulative: checked {checked}, {} mismatches",
            mismatches.load(Ordering::Relaxed)
        );
    }
    checked
}

fn run_case(case: &Case, max_len: usize) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let enc = case.enc;

    println!("=== encoding {} ===", enc.name());
    println!("  class alphabet: {:?}", case.alphabet);

    // Validate the engine build once up front so a genuine construction error
    // surfaces here as a returned Err rather than panicking inside a worker thread.
    drop(Engines::build(enc)?);

    let mismatches = AtomicU64::new(0);
    let mut checked = enumerate(enc, "ascii", case.alphabet, max_len, &mismatches);

    // Contraction pass: inject the whole `'s`/`'ll`/… fragments in every local
    // context, without adding their letters to the alphabet.
    println!("  contractions: {CONTRACTIONS:?}  (up to {CONTRACTION_CELLS} cells)");
    checked += enumerate_contractions(enc, case.alphabet, CONTRACTION_CELLS, &mismatches);

    Ok((checked, mismatches.into_inner()))
}

/// Per-encoding enumeration depth used by `--quick`. r50k's decisions saturate by
/// length 8; cl100k's digit/whitespace grouping needs a few more symbols (12) and
/// o200k's case-split word + longer whitespace grouping needs the full 13. A
/// positional `max_len` still caps this (so `--quick 4` remains a fast smoke test).
fn quick_len(enc: Encoding) -> usize {
    match enc {
        Encoding::R50k | Encoding::P50k => 8,
        Encoding::Cl100k => 12,
        Encoding::O200k => 13,
        Encoding::Llama3 | Encoding::Mistral | Encoding::Qwen3 => unreachable!(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let has = |f: &str| args.iter().any(|a| a == f);
    let quick = has("--quick");

    for arg in args.iter().filter(|arg| arg.starts_with("--")) {
        if arg != "--quick" {
            return Err(format!("unknown flag {arg:?} (use --quick)").into());
        }
    }

    let pos: Vec<&str> = args
        .iter()
        .map(String::as_str)
        .filter(|a| !a.starts_with("--"))
        .collect();
    let (max_len, only_name) = match pos.as_slice() {
        [] => (13, None),
        [value] => match value.parse() {
            Ok(max_len) => (max_len, None),
            Err(_) => (13, Some(*value)),
        },
        [max_len, encoding] => (max_len.parse()?, Some(*encoding)),
        _ => {
            return Err("usage: fst_equiv [--quick] [max_len] [encoding]".into());
        }
    };
    let only = only_name
        .map(|name| {
            let encoding = Encoding::from_name(name)
                .ok_or_else(|| format!("unknown tiktoken encoding {name:?}"))?;
            if !matches!(
                encoding,
                Encoding::R50k | Encoding::P50k | Encoding::Cl100k | Encoding::O200k
            ) {
                return Err(format!("{name:?} is not a tiktoken encoding"));
            }
            Ok(encoding)
        })
        .transpose()?;

    println!(
        "Exhaustive DFA+fixup-vs-reference equivalence over tiktoken class alphabets ({} threads)",
        rayon::current_num_threads()
    );
    if quick {
        println!(
            "--quick: per-encoding depth r50k/p50k=8 cl100k=12 o200k=13 (capped at {max_len})\n"
        );
    } else {
        println!("Lengths 1..={max_len}\n");
    }

    let mut total_checked: u64 = 0;
    let mut total_mismatches: u64 = 0;
    for case in CASES {
        if only.is_some_and(|enc| enc != case.enc) {
            continue;
        }
        let case_max_len = if quick {
            quick_len(case.enc).min(max_len)
        } else {
            max_len
        };
        let (checked, mismatches) = run_case(case, case_max_len)?;
        total_checked += checked;
        total_mismatches += mismatches;
        println!();
    }

    if total_mismatches == 0 {
        println!("EQUIVALENT: all {total_checked} class-strings agree with their references.");
        Ok(())
    } else {
        println!("NOT EQUIVALENT: {total_mismatches} mismatches over {total_checked} strings.");
        std::process::exit(1);
    }
}
