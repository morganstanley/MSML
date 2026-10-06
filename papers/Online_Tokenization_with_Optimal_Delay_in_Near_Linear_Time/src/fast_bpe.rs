//! Specialized cold-miss BPE for the SIMD/full-cache streaming engine.
//!
//! This module is adapted from Gigatoken's MIT-licensed byte-level BPE miss
//! path. It deliberately knows nothing about pretokenization, the pretoken
//! cache, or output sinks:
//!
//! * [`FastBpeTables`] is immutable model data: a raw-byte-to-compact-ID map
//!   and a cache-friendly pair-rank table.
//! * [`FastBpeScratch`] is reusable per-engine state. Pretokens of at most
//!   fifteen bytes stay entirely on the stack; longer pretokens reuse vectors.
//!
//! The merge kernels rely on one important Hiriluk invariant: internal compact
//! token IDs increase with merge priority. [`FastBpeTables::build_from_parts`]
//! checks that invariant and returns `None` when a dictionary cannot safely use
//! this path. Callers retain MTC as the correctness fallback.
//!
//! Upstream:
//! <https://github.com/marcelroed/gigatoken/blob/main/src/bpe/mod.rs>
//! <https://github.com/marcelroed/gigatoken/blob/main/src/bpe/tiktoken.rs>

use std::{cmp::Reverse, collections::BinaryHeap};

const PAIR_ID_BITS: u32 = 21;
const PAIR_ID_LIMIT: u32 = 1 << PAIR_ID_BITS;
/// A 1024×1024 grid is 4 MiB: large enough to cover every initial byte pair and
/// the hottest early merges, without evicting the streaming pretoken cache.
///
/// Gigatoken uses a 2048×2048 (16 MiB) grid tuned for the large per-CCX caches
/// on its Zen benchmark machines, and this previously matched that on non-Apple
/// targets. Measured on one bound core of a Xeon Platinum 8462Y+ (2 MiB private
/// L2, 60 MiB L3 shared by 32 cores), 4 MiB is the better trade there too:
/// r50k English gains 1.9%, GitHub is unchanged within noise at -0.2%, and
/// Chinese gains 0.2%. Alongside the pretoken cache, which `reserve_for_bytes`
/// sizes to 32 MiB for a 128 MiB input, the 16 MiB grid simply did not fit.
///
/// A 512×512 (1 MiB) grid was also measured. It keeps most of the English gain
/// but costs GitHub 0.8%, because more of its pairs fall through to the sparse
/// table, so it is not the better default.
const DEFAULT_DENSE_LOG2: u32 = 10;
const EMPTY_SLOT: u64 = u64::MAX;
const SHORT_MERGE_CAPACITY: usize = 16;
const SMALL_MERGE_MAX: usize = 32;
const NONE: u32 = u32::MAX;

/// Immutable lookup tables shared by fast BPE engines for one model.
pub(crate) struct FastBpeTables {
    byte_to_id: [u32; 256],
    pair_ranks: PairRankTable,
}

impl FastBpeTables {
    /// Build directly from the compact data needed by the merge kernels.
    ///
    /// `byte_to_id` maps every raw byte to its compact internal token ID.
    /// `merges` must be in priority order and contain `(left, right, merged)`
    /// compact IDs. This entry point lets runtime model construction skip the
    /// full [`Vocab`] and [`Dictionary`] representations once it can produce
    /// these two inputs directly.
    ///
    /// Returns `None` when an ID cannot be packed, priorities are not
    /// monotone, a pair is duplicated, or the sparse table clusters
    /// pathologically.
    pub(crate) fn build_from_parts(
        byte_to_id: [u32; 256],
        merges: &[(u32, u32, u32)],
    ) -> Option<Self> {
        if byte_to_id.iter().any(|&id| id >= PAIR_ID_LIMIT) {
            return None;
        }

        let mut previous_merged = None;
        for &(pre, suc, merged) in merges {
            if (pre | suc | merged) >= PAIR_ID_LIMIT {
                return None;
            }
            // The kernels use the merged compact ID itself as the priority.
            // Strict monotonicity makes that equivalent to merge-list order,
            // including vocabularies whose external IDs contain gaps.
            if previous_merged.is_some_and(|previous| merged <= previous) {
                return None;
            }
            previous_merged = Some(merged);
        }

        Some(Self {
            byte_to_id,
            pair_ranks: PairRankTable::build(merges, DEFAULT_DENSE_LOG2)?,
        })
    }

    /// Encode one complete pretoken into compact internal token IDs.
    ///
    /// The returned slice borrows `scratch` and remains valid until its next
    /// mutation. Input/output relabeling belongs to the caller.
    #[inline]
    pub(crate) fn encode<'a>(&self, scratch: &'a mut FastBpeScratch, bytes: &[u8]) -> &'a [u32] {
        if bytes.len() < SHORT_MERGE_CAPACITY {
            let n = bytes.len();
            for (dst, &byte) in scratch.short[..n].iter_mut().zip(bytes) {
                *dst = self.byte_to_id[byte as usize];
            }
            let merged = match n {
                0 | 1 => n,
                _ => merge_short(&self.pair_ranks, &mut scratch.short, n),
            };
            return &scratch.short[..merged];
        }

        let FastBpeScratch { symbols, merge, .. } = scratch;
        symbols.clear();
        symbols.extend(bytes.iter().map(|&byte| self.byte_to_id[byte as usize]));
        merge_long(&self.pair_ranks, symbols, merge);
        symbols
    }

    /// Heap bytes owned by immutable fast BPE tables.
    pub(crate) fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.pair_ranks.dense.len() * std::mem::size_of::<u32>()
            + self.pair_ranks.slots.len() * std::mem::size_of::<u64>()
    }
}

/// Reusable mutable state for cold misses in one streaming engine.
#[derive(Default)]
pub(crate) struct FastBpeScratch {
    short: [u32; SHORT_MERGE_CAPACITY],
    symbols: Vec<u32>,
    merge: MergeScratch,
}

impl FastBpeScratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Two-level pair lookup replacing a general hash map in merge loops.
///
/// The dense 2048×2048 grid covers all byte-token pairs and the most frequent
/// early merge IDs. A complete packed open-address table handles every pair.
struct PairRankTable {
    dense: Box<[u32]>,
    dense_log2: u32,
    slots: Box<[u64]>,
    mask: usize,
    shift: u32,
}

impl PairRankTable {
    fn build(merges: &[(u32, u32, u32)], dense_log2: u32) -> Option<Self> {
        if !(8..=11).contains(&dense_log2) {
            return None;
        }
        let mut dense = vec![u32::MAX; 1usize << (2 * dense_log2)].into_boxed_slice();
        let n_slots = merges
            .len()
            .max(1)
            .checked_mul(2)?
            .next_power_of_two()
            .max(64);
        let mask = n_slots - 1;
        let shift = 64 - n_slots.trailing_zeros();
        let mut slots = vec![EMPTY_SLOT; n_slots].into_boxed_slice();

        for &(a, b, merged) in merges {
            if (a | b | merged) >= PAIR_ID_LIMIT {
                return None;
            }
            if (a | b) >> dense_log2 == 0 {
                dense[((a as usize) << dense_log2) | b as usize] = merged;
            }

            let key = pair_key(a, b);
            let mut index = pair_index(key, shift);
            let mut displacement = 0usize;
            loop {
                let slot = slots[index];
                if slot == EMPTY_SLOT {
                    slots[index] = (key << PAIR_ID_BITS) | merged as u64;
                    break;
                }
                // A dictionary must define at most one merge for a pair.
                if slot >> PAIR_ID_BITS == key {
                    return None;
                }
                index = (index + 1) & mask;
                displacement += 1;
                if displacement > 64 {
                    return None;
                }
            }
        }

        Some(Self {
            dense,
            dense_log2,
            slots,
            mask,
            shift,
        })
    }

    /// Return the merged compact ID, which is also the merge priority, or MAX.
    #[inline(always)]
    fn rank(&self, a: u32, b: u32) -> u32 {
        debug_assert!((a | b) < PAIR_ID_LIMIT);
        if (a | b) >> self.dense_log2 == 0 {
            let index = ((a as usize) << self.dense_log2) | b as usize;
            // SAFETY: both IDs are below 2^dense_log2.
            return unsafe { *self.dense.get_unchecked(index) };
        }

        let key = pair_key(a, b);
        let mut index = pair_index(key, self.shift);
        loop {
            // SAFETY: the initial index is below slots.len(); subsequent
            // indexes are masked.
            let slot = unsafe { *self.slots.get_unchecked(index) };
            if slot >> PAIR_ID_BITS == key {
                return (slot & ((1 << PAIR_ID_BITS) - 1)) as u32;
            }
            if slot == EMPTY_SLOT {
                return u32::MAX;
            }
            index = (index + 1) & self.mask;
        }
    }

    /// Request the first lookup line while list surgery is still in flight.
    #[inline(always)]
    fn prefetch_rank(&self, a: u32, b: u32) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let address = if (a | b) >> self.dense_log2 == 0 {
                let index = ((a as usize) << self.dense_log2) | b as usize;
                self.dense.as_ptr().add(index).cast::<u8>()
            } else {
                let index = pair_index(pair_key(a, b), self.shift);
                self.slots.as_ptr().add(index).cast::<u8>()
            };
            core::arch::asm!(
                "prfm pldl1keep, [{address}]",
                address = in(reg) address,
                options(nostack, preserves_flags, readonly)
            );
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the address calculation is identical to `rank`; both
        // branches remain within a live allocation, and prefetch does not
        // dereference the pointer.
        unsafe {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let address = if (a | b) >> self.dense_log2 == 0 {
                let index = ((a as usize) << self.dense_log2) | b as usize;
                self.dense.as_ptr().add(index).cast::<i8>()
            } else {
                let index = pair_index(pair_key(a, b), self.shift);
                self.slots.as_ptr().add(index).cast::<i8>()
            };
            _mm_prefetch(address, _MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = (a, b);
    }
}

#[inline(always)]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << PAIR_ID_BITS) | b as u64
}

#[inline(always)]
fn pair_index(key: u64, shift: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize
}

#[inline(always)]
fn merge_short(
    table: &PairRankTable,
    symbols: &mut [u32; SHORT_MERGE_CAPACITY],
    n: usize,
) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        merge_short_neon(table, symbols, n)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        merge_short_scalar(table, symbols, n)
    }
}

/// Stack-resident short merge used on non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
fn merge_short_scalar(
    table: &PairRankTable,
    symbols: &mut [u32; SHORT_MERGE_CAPACITY],
    n: usize,
) -> usize {
    debug_assert!((2..SHORT_MERGE_CAPACITY).contains(&n));
    let mut next = [0u8; SHORT_MERGE_CAPACITY];
    let mut prev = [0u8; SHORT_MERGE_CAPACITY];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }

    let mut ranks = [u32::MAX; SHORT_MERGE_CAPACITY];
    for i in 0..n - 1 {
        ranks[i] = table.rank(symbols[i], symbols[i + 1]);
    }

    loop {
        let mut best = u32::MAX;
        let mut best_index = 0usize;
        for (index, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_index = index;
            }
        }
        if best == u32::MAX {
            break;
        }

        let index = best_index;
        let dead = next[index] as usize;
        let new_right = next[dead] as usize;
        let left = prev[index] as usize;
        if new_right < n {
            table.prefetch_rank(best, symbols[new_right]);
        }
        if left < n {
            table.prefetch_rank(symbols[left], best);
        }

        symbols[index] = best;
        next[index] = new_right as u8;
        ranks[dead] = u32::MAX;
        if new_right < n {
            prev[new_right] = index as u8;
            ranks[index] = table.rank(symbols[index], symbols[new_right]);
        } else {
            ranks[index] = u32::MAX;
        }
        if left < n {
            ranks[left] = table.rank(symbols[left], symbols[index]);
        }
    }

    compact_short(symbols, &next, n)
}

#[cfg(target_arch = "aarch64")]
fn merge_short_neon(
    table: &PairRankTable,
    symbols: &mut [u32; SHORT_MERGE_CAPACITY],
    n: usize,
) -> usize {
    use core::arch::aarch64::{vld1q_u32, vminq_u32, vminvq_u32};

    debug_assert!((2..SHORT_MERGE_CAPACITY).contains(&n));
    const NO_MERGE_FLOOR: u32 = u32::MAX << 8;
    let pack = |rank: u32, index: usize| (rank << 8) | index as u32;

    let mut next = [0u8; SHORT_MERGE_CAPACITY];
    let mut prev = [0u8; SHORT_MERGE_CAPACITY];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }

    let mut packed_ranks = [u32::MAX; SHORT_MERGE_CAPACITY];
    for i in 0..n - 1 {
        packed_ranks[i] = pack(table.rank(symbols[i], symbols[i + 1]), i);
    }
    let narrow = n <= 8;

    loop {
        // SAFETY: packed_ranks contains sixteen contiguous u32 lanes.
        let best = unsafe {
            let ptr = packed_ranks.as_ptr();
            let first = vminq_u32(vld1q_u32(ptr), vld1q_u32(ptr.add(4)));
            let all = if narrow {
                first
            } else {
                let second = vminq_u32(vld1q_u32(ptr.add(8)), vld1q_u32(ptr.add(12)));
                vminq_u32(first, second)
            };
            vminvq_u32(all)
        };
        if best >= NO_MERGE_FLOOR {
            break;
        }

        let index = (best & 0xFF) as usize;
        let merged = best >> 8;
        let dead = next[index] as usize;
        let new_right = next[dead] as usize;
        let left = prev[index] as usize;
        if new_right < n {
            table.prefetch_rank(merged, symbols[new_right]);
        }
        if left < n {
            table.prefetch_rank(symbols[left], merged);
        }

        symbols[index] = merged;
        next[index] = new_right as u8;
        packed_ranks[dead] = u32::MAX;
        if new_right < n {
            prev[new_right] = index as u8;
            packed_ranks[index] = pack(table.rank(symbols[index], symbols[new_right]), index);
        } else {
            packed_ranks[index] = u32::MAX;
        }
        if left < n {
            packed_ranks[left] = pack(table.rank(symbols[left], symbols[index]), left);
        }
    }

    compact_short(symbols, &next, n)
}

#[inline]
fn compact_short(
    symbols: &mut [u32; SHORT_MERGE_CAPACITY],
    next: &[u8; SHORT_MERGE_CAPACITY],
    n: usize,
) -> usize {
    let mut write = 0usize;
    let mut index = 0usize;
    while index < n {
        symbols[write] = symbols[index];
        write += 1;
        index = next[index] as usize;
    }
    write
}

#[derive(Default)]
struct MergeScratch {
    next: Vec<u32>,
    prev: Vec<u32>,
    heap: Vec<Reverse<u64>>,
}

#[inline(never)]
fn merge_long(table: &PairRankTable, symbols: &mut Vec<u32>, scratch: &mut MergeScratch) {
    let n = symbols.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_MERGE_MAX {
        merge_small(table, symbols);
        return;
    }
    let n_u32 = u32::try_from(n).expect("a pretoken cannot contain 2^32 symbols");

    scratch.next.clear();
    scratch.next.extend(1..n_u32);
    scratch.next.push(NONE);
    scratch.prev.clear();
    scratch.prev.push(NONE);
    scratch.prev.extend(0..n_u32 - 1);

    let mut seeds = std::mem::take(&mut scratch.heap);
    seeds.clear();
    for index in 0..n - 1 {
        let merged = table.rank(symbols[index], symbols[index + 1]);
        if merged != u32::MAX {
            seeds.push(Reverse(pack_merge_entry(merged, index as u32)));
        }
    }
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::from(seeds);

    while let Some(Reverse(entry)) = heap.pop() {
        let index = entry as u32 as usize;
        let expected = (entry >> 32) as u32;
        let right = scratch.next[index];
        if right == NONE {
            continue;
        }
        let right = right as usize;
        let merged = table.rank(symbols[index], symbols[right]);
        if merged != expected {
            continue;
        }

        symbols[index] = merged;
        let new_right = scratch.next[right];
        let left = scratch.prev[index];
        if new_right != NONE {
            table.prefetch_rank(merged, symbols[new_right as usize]);
        }
        if left != NONE {
            table.prefetch_rank(symbols[left as usize], merged);
        }
        scratch.next[index] = new_right;
        if new_right != NONE {
            scratch.prev[new_right as usize] = index as u32;
        }
        scratch.next[right] = NONE;

        if left != NONE {
            let candidate = table.rank(symbols[left as usize], symbols[index]);
            if candidate != u32::MAX {
                heap.push(Reverse(pack_merge_entry(candidate, left)));
            }
        }
        if scratch.next[index] != NONE {
            let candidate = table.rank(symbols[index], symbols[scratch.next[index] as usize]);
            if candidate != u32::MAX {
                heap.push(Reverse(pack_merge_entry(candidate, index as u32)));
            }
        }
    }
    scratch.heap = heap.into_vec();

    compact_long(symbols, &scratch.next);
}

fn merge_small(table: &PairRankTable, symbols: &mut Vec<u32>) {
    let n = symbols.len();
    debug_assert!((2..=SMALL_MERGE_MAX).contains(&n));
    let mut next = [0u8; SMALL_MERGE_MAX];
    let mut prev = [0u8; SMALL_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }

    let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
    for i in 0..n - 1 {
        ranks[i] = table.rank(symbols[i], symbols[i + 1]);
    }
    loop {
        let mut best = u32::MAX;
        let mut best_index = 0usize;
        for (index, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_index = index;
            }
        }
        if best == u32::MAX {
            break;
        }

        let index = best_index;
        let dead = next[index] as usize;
        let new_right = next[dead] as usize;
        let left = prev[index] as usize;
        if new_right < n {
            table.prefetch_rank(best, symbols[new_right]);
        }
        if left < n {
            table.prefetch_rank(symbols[left], best);
        }

        symbols[index] = best;
        next[index] = new_right as u8;
        ranks[dead] = u32::MAX;
        if new_right < n {
            prev[new_right] = index as u8;
            ranks[index] = table.rank(symbols[index], symbols[new_right]);
        } else {
            ranks[index] = u32::MAX;
        }
        if left < n {
            ranks[left] = table.rank(symbols[left], symbols[index]);
        }
    }

    let mut write = 0usize;
    let mut index = 0usize;
    while index < n {
        symbols[write] = symbols[index];
        write += 1;
        index = next[index] as usize;
    }
    symbols.truncate(write);
}

fn compact_long(symbols: &mut Vec<u32>, next: &[u32]) {
    let mut write = 0usize;
    let mut index = 0usize;
    loop {
        symbols[write] = symbols[index];
        write += 1;
        if next[index] == NONE {
            break;
        }
        index = next[index] as usize;
    }
    symbols.truncate(write);
}

#[inline(always)]
fn pack_merge_entry(merged: u32, position: u32) -> u64 {
    ((merged as u64) << 32) | position as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dictionary, Vocab};

    fn complete_dictionary() -> (Vocab, Dictionary) {
        let mut tokens: Vec<Vec<u8>> = (0u8..=u8::MAX).map(|byte| vec![byte]).collect();
        tokens.extend([
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"bc".to_vec(),
            b"abcd".to_vec(),
        ]);
        let vocab = Vocab::new(tokens).unwrap();
        let dictionary = Dictionary::new_from_id_pair(
            vocab.clone(),
            [
                (b'a' as usize, b'b' as usize),
                (256usize, b'c' as usize),
                (b'b' as usize, b'c' as usize),
                (257usize, b'd' as usize),
            ],
        )
        .unwrap();
        (vocab, dictionary)
    }

    fn reference_encode(vocab: &Vocab, dictionary: &Dictionary, bytes: &[u8]) -> Vec<u32> {
        let mut symbols: Vec<u32> = bytes
            .iter()
            .map(|&byte| vocab.find_by_byte_unchecked(byte).inner())
            .collect();
        loop {
            let mut best: Option<(usize, usize, u32)> = None;
            for index in 0..symbols.len().saturating_sub(1) {
                let Some(rule_id) = dictionary.find_rule(
                    crate::TokenId::new(symbols[index]),
                    crate::TokenId::new(symbols[index + 1]),
                ) else {
                    continue;
                };
                let priority = rule_id.as_usize();
                let merged = dictionary[rule_id].merged.inner();
                if best.is_none_or(|(best_priority, best_index, _)| {
                    (priority, index) < (best_priority, best_index)
                }) {
                    best = Some((priority, index, merged));
                }
            }
            let Some((_, index, merged)) = best else {
                break;
            };
            symbols[index] = merged;
            symbols.remove(index + 1);
        }
        symbols
    }

    fn compact_parts(vocab: &Vocab, dictionary: &Dictionary) -> ([u32; 256], Vec<(u32, u32, u32)>) {
        let byte_to_id =
            std::array::from_fn(|byte| vocab.find_by_byte_unchecked(byte as u8).inner());
        let merges = dictionary
            .rules()
            .iter()
            .map(|rule| (rule.pre.inner(), rule.suc.inner(), rule.merged.inner()))
            .collect();
        (byte_to_id, merges)
    }

    #[test]
    fn short_and_long_misses_match_dictionary_order() {
        let (vocab, dictionary) = complete_dictionary();
        let (byte_to_id, merges) = compact_parts(&vocab, &dictionary);
        let tables = FastBpeTables::build_from_parts(byte_to_id, &merges).unwrap();
        let mut scratch = FastBpeScratch::new();

        for input in [
            b"".as_slice(),
            b"a",
            b"abc",
            b"zabcx",
            b"abcabcabcabcabcabcabcabcabcabc",
            b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc",
        ] {
            assert_eq!(
                tables.encode(&mut scratch, input),
                reference_encode(&vocab, &dictionary, input)
            );
        }
    }

    /// Broad randomized cover for the short-merge kernel, including the runs of
    /// repeated symbols where equal ranks make tie-breaking observable.
    ///
    /// A packed-reduction kernel for x86, mirroring `merge_short_neon`, was
    /// measured against this and was correct but slower: AVX2 -4.4% and AVX-512
    /// -4.9% on r50k English. Both reload all sixteen lanes right after a scalar
    /// store to one of them, so every merge iteration pays a store-forwarding
    /// stall that the narrow scalar loop avoids. AVX2 regressing as much as
    /// AVX-512 rules out frequency licensing as the cause. Keep this test if
    /// that idea is revisited.
    #[test]
    fn short_merge_matches_the_dictionary_across_random_inputs() {
        let (vocab, dictionary) = complete_dictionary();
        let (byte_to_id, merges) = compact_parts(&vocab, &dictionary);
        let tables = FastBpeTables::build_from_parts(byte_to_id, &merges).unwrap();
        let mut scratch = FastBpeScratch::new();

        let alphabet = b"abcdz";
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for length in 2..SHORT_MERGE_CAPACITY {
            for _ in 0..512 {
                let mut bytes = Vec::with_capacity(length);
                for _ in 0..length {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    bytes.push(alphabet[(state % alphabet.len() as u64) as usize]);
                }
                assert_eq!(
                    tables.encode(&mut scratch, &bytes),
                    reference_encode(&vocab, &dictionary, &bytes),
                    "short merge disagreed on {:?}",
                    String::from_utf8_lossy(&bytes),
                );
            }
        }
    }

    #[test]
    fn compact_parts_builder_rejects_invalid_ids_and_priorities() {
        let (vocab, dictionary) = complete_dictionary();
        let (byte_to_id, merges) = compact_parts(&vocab, &dictionary);

        let mut missing_byte = byte_to_id;
        missing_byte[0] = u32::MAX;
        assert!(FastBpeTables::build_from_parts(missing_byte, &merges).is_none());

        let mut oversized_merge = merges.clone();
        oversized_merge[0].2 = PAIR_ID_LIMIT;
        assert!(FastBpeTables::build_from_parts(byte_to_id, &oversized_merge).is_none());

        let mut nonmonotone = merges.clone();
        nonmonotone.swap(0, 1);
        assert!(FastBpeTables::build_from_parts(byte_to_id, &nonmonotone).is_none());

        let duplicate_pair = [(1, 2, 256), (1, 2, 257)];
        assert!(FastBpeTables::build_from_parts(byte_to_id, &duplicate_pair).is_none());
    }

    #[test]
    fn rejects_incomplete_byte_vocabulary() {
        let mut byte_to_id = [0u32; 256];
        byte_to_id[0] = u32::MAX;
        assert!(FastBpeTables::build_from_parts(byte_to_id, &[]).is_none());
    }

    #[test]
    fn rejects_nonmonotone_compact_priority() {
        let byte_to_id = std::array::from_fn(|byte| byte as u32);
        let merges = [
            (b'b' as u32, b'c' as u32, 257),
            (b'a' as u32, b'b' as u32, 256),
        ];
        assert!(FastBpeTables::build_from_parts(byte_to_id, &merges).is_none());
    }

    #[test]
    fn pair_rank_table_covers_dense_sparse_hits_and_misses() {
        let table = PairRankTable::build(
            &[(1, 2, 256), (3_000, 4_000, 5_000), (4_000, 3_000, 5_001)],
            DEFAULT_DENSE_LOG2,
        )
        .unwrap();
        assert_eq!(table.rank(1, 2), 256);
        assert_eq!(table.rank(3_000, 4_000), 5_000);
        assert_eq!(table.rank(4_000, 3_000), 5_001);
        assert_eq!(table.rank(1, 3), u32::MAX);
        assert_eq!(table.rank(3_000, 4_001), u32::MAX);
    }
}
