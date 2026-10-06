//! Streaming matcher for exact ASCII literals.
//!
//! Complete candidates are checked locally inside each input chunk. A false
//! prefix therefore remains part of the surrounding text run and never
//! creates a downstream callback boundary. Only a literal prefix at the end
//! of a chunk is retained.

use super::Seg;
use std::sync::Arc;

const ROOT: usize = 0;

struct Node {
    children: Vec<(u8, usize)>,
    /// Representative literal used to return a held prefix after a mismatch.
    repr: usize,
    terminal: Option<u32>,
}

/// Immutable trie shared by every stream created from one loaded target.
pub(super) struct ExactConfig {
    nodes: Vec<Node>,
    /// Each literal is stored once. Trie nodes keep only a compact index into
    /// this array instead of cloning the full literal at every byte.
    literals: Vec<Box<str>>,
    root_bytes: Box<[u8]>,
    /// Shared two-byte prefix, when every literal has one. Tiktoken's exact
    /// specials all begin with `<|`; scanning for that pair avoids entering
    /// the trie for the many ordinary `<` bytes found in source code.
    common_prefix: Option<[u8; 2]>,
}

pub(super) struct ExactMatcher {
    config: Arc<ExactConfig>,
    /// Trie state for the one candidate suffix held across a chunk boundary.
    node: usize,
    depth: usize,
    peak: usize,
    held_sum: u64,
    bytes_seen: u64,
}

impl ExactConfig {
    pub(super) fn new(specials: &[(&str, u32)]) -> Self {
        let mut nodes = vec![Node {
            children: Vec::new(),
            repr: 0,
            terminal: None,
        }];
        let literals: Vec<Box<str>> = specials
            .iter()
            .map(|(literal, _)| Box::<str>::from(*literal))
            .collect();
        for (literal_index, &(literal, id)) in specials.iter().enumerate() {
            let mut current = ROOT;
            for &byte in literal.as_bytes() {
                let next = nodes[current]
                    .children
                    .iter()
                    .find(|(candidate, _)| *candidate == byte)
                    .map(|(_, child)| *child);
                current = match next {
                    Some(child) => child,
                    None => {
                        let child = nodes.len();
                        nodes.push(Node {
                            children: Vec::new(),
                            repr: literal_index,
                            terminal: None,
                        });
                        nodes[current].children.push((byte, child));
                        child
                    }
                };
                nodes[current].repr = literal_index;
            }
            nodes[current].terminal = Some(id);
        }

        let mut root_bytes: Vec<u8> = nodes[ROOT].children.iter().map(|(b, _)| *b).collect();
        root_bytes.sort_unstable();
        root_bytes.dedup();
        let common_prefix = literals.first().and_then(|first| {
            let prefix: [u8; 2] = first.as_bytes().get(..2)?.try_into().ok()?;
            literals
                .iter()
                .all(|literal| literal.as_bytes().starts_with(&prefix))
                .then_some(prefix)
        });
        Self {
            nodes,
            literals,
            root_bytes: root_bytes.into_boxed_slice(),
            common_prefix,
        }
    }
}

impl ExactMatcher {
    pub(super) fn from_config(config: Arc<ExactConfig>) -> Self {
        Self {
            config,
            node: ROOT,
            depth: 0,
            peak: 0,
            held_sum: 0,
            bytes_seen: 0,
        }
    }

    pub(super) fn peak_held(&self) -> usize {
        self.peak
    }

    pub(super) fn held_sum(&self) -> u64 {
        self.held_sum
    }

    pub(super) fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }

    pub(super) fn reset_peak(&mut self) {
        debug_assert_eq!(self.depth, 0, "segmenter must be idle between runs");
        self.peak = 0;
    }

    #[inline]
    fn next_candidate(&self, haystack: &[u8]) -> Option<usize> {
        if let Some(prefix) = self.config.common_prefix {
            // `memmem` skips false first-byte candidates in one vectorized
            // scan. A trailing first byte is still returned so the normal
            // trie state can retain it across the streaming chunk boundary.
            return memchr::memmem::find(haystack, &prefix).or_else(|| {
                haystack
                    .last()
                    .is_some_and(|byte| *byte == prefix[0])
                    .then(|| haystack.len() - 1)
            });
        }
        match self.config.root_bytes.as_ref() {
            [] => None,
            [a] => memchr::memchr(*a, haystack),
            [a, b] => memchr::memchr2(*a, *b, haystack),
            [a, b, c] => memchr::memchr3(*a, *b, *c, haystack),
            set => haystack.iter().position(|byte| set.contains(byte)),
        }
    }

    #[inline(always)]
    fn child(&self, node: usize, byte: u8) -> Option<usize> {
        self.config.nodes[node]
            .children
            .iter()
            .find(|(candidate, _)| *candidate == byte)
            .map(|(_, child)| *child)
    }

    #[inline(always)]
    fn record<const PROFILE: bool>(&mut self, depth: usize) {
        if PROFILE {
            self.peak = self.peak.max(depth);
            self.held_sum += depth as u64;
        }
    }

    #[inline(never)]
    pub(super) fn feed<const PROFILE: bool>(&mut self, chunk: &str, sink: &mut impl FnMut(Seg)) {
        let bytes = chunk.as_bytes();
        if PROFILE {
            self.bytes_seen += bytes.len() as u64;
        }

        let mut index = 0usize;
        let mut run_start = 0usize;

        // Resolve the only state carried between chunks. On a mismatch, emit
        // the old literal prefix but do not consume the mismatching byte: it
        // may itself begin a new literal.
        while self.node != ROOT && index < bytes.len() {
            let Some(child) = self.child(self.node, bytes[index]) else {
                let repr = self.config.nodes[self.node].repr;
                sink(Seg::Text(&self.config.literals[repr][..self.depth]));
                self.node = ROOT;
                self.depth = 0;
                run_start = index;
                break;
            };
            self.node = child;
            self.depth += 1;
            self.record::<PROFILE>(self.depth);
            index += 1;
            if let Some(id) = self.config.nodes[self.node].terminal {
                sink(Seg::Special(id));
                self.node = ROOT;
                self.depth = 0;
                run_start = index;
                break;
            }
        }
        if self.node != ROOT {
            return;
        }

        while index < bytes.len() {
            let Some(offset) = self.next_candidate(&bytes[index..]) else {
                break;
            };
            let candidate = index + offset;
            let mut node = self
                .child(ROOT, bytes[candidate])
                .expect("candidate byte must be a root child");
            let mut depth = 1usize;
            self.record::<PROFILE>(depth);

            loop {
                // Assumption: special tokens are prefix-free so shortest terminal suffices.
                if let Some(id) = self.config.nodes[node].terminal {
                    if run_start < candidate {
                        sink(Seg::Text(&chunk[run_start..candidate]));
                    }
                    sink(Seg::Special(id));
                    index = candidate + depth;
                    run_start = index;
                    break;
                }

                let next_index = candidate + depth;
                if next_index == bytes.len() {
                    // All remaining bytes are a real literal prefix. Emit the
                    // settled text before it and retain only this suffix.
                    if run_start < candidate {
                        sink(Seg::Text(&chunk[run_start..candidate]));
                    }
                    self.node = node;
                    self.depth = depth;
                    return;
                }

                match self.child(node, bytes[next_index]) {
                    Some(child) => {
                        node = child;
                        depth += 1;
                        self.record::<PROFILE>(depth);
                    }
                    None => {
                        // False prefix: keep it inside the current text run.
                        // Advance one byte, not `depth`, so overlapping starts
                        // remain discoverable.
                        index = candidate + 1;
                        break;
                    }
                }
            }
        }

        if run_start < bytes.len() {
            sink(Seg::Text(&chunk[run_start..]));
        }
    }

    pub(super) fn finish(&mut self, sink: &mut impl FnMut(Seg)) {
        if self.node != ROOT {
            let repr = self.config.nodes[self.node].repr;
            sink(Seg::Text(&self.config.literals[repr][..self.depth]));
            self.node = ROOT;
            self.depth = 0;
        }
    }
}
