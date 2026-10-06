//! Cache-line-oriented memoization for complete pretoken encodings.
//!
//! Pretokens up to 15 bytes are packed into a `u128` and stored in a linear
//! probing table with two entries per 64-byte bucket. Encodings of up to four
//! token IDs stay inline; larger encodings spill into an append-only arena.
//! Longer pretokens use a conventional hash map, but are rare on natural text.
//!
//! This layout follows the cache hierarchy described by Gigatoken's
//! MIT-licensed `ShortPretokenCache`:
//! <https://github.com/marcelroed/gigatoken/blob/main/src/bpe/pretoken_cache.rs>

use rustc_hash::FxHashMap;
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ptr::NonNull;

use crate::runtime_vocab::RuntimeVocab;

const SPILL: u64 = 0x80;
const EMPTY_KEY: u128 = 0;

/// Cache lookup metadata derived while the pretoken scanner already has the
/// span hot. `packed == 0` routes long (>15-byte) pieces to the fallback map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct PieceKey {
    packed_lo: u64,
    packed_hi: u64,
    hash: u64,
}

const _: () = assert!(std::mem::size_of::<PieceKey>() == 24);

impl PieceKey {
    #[inline(always)]
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        let packed = pack_key(bytes).unwrap_or(0);
        Self::from_packed(packed)
    }

    /// Pack a known in-buffer span. Almost every call can take one checked
    /// 16-byte load; only a short span within the final 15 bytes needs the
    /// careful slice packer.
    #[inline(always)]
    pub(crate) fn from_span(bytes: &[u8], start: usize, len: usize) -> Self {
        if len == 0 {
            return Self::default();
        }
        if len > 15 {
            return Self::default();
        }
        let end = start.checked_add(len).expect("pretoken span overflow");
        assert!(end <= bytes.len(), "pretoken span exceeds backing bytes");
        let packed = if bytes.len() - start >= 16 {
            let word = u128::from_le(unsafe {
                (bytes.as_ptr().add(start) as *const u128).read_unaligned()
            });
            pack_loaded_table(word, len)
        } else {
            pack_key(&bytes[start..end]).expect("short span must pack")
        };
        Self::from_packed(packed)
    }

    #[inline(always)]
    fn from_packed(packed: u128) -> Self {
        Self::from_packed_hash(packed, key_hash(packed))
    }

    #[inline(always)]
    fn from_packed_hash(packed: u128, hash: u64) -> Self {
        Self {
            packed_lo: packed as u64,
            packed_hi: (packed >> 64) as u64,
            hash,
        }
    }

    #[inline(always)]
    pub(crate) fn packed(self) -> u128 {
        self.packed_lo as u128 | ((self.packed_hi as u128) << 64)
    }

    #[inline(always)]
    pub(crate) fn is_short(self) -> bool {
        self.packed_lo != 0 || self.packed_hi != 0
    }
}

/// Batch-local short-key packer. Pinning the mask-table base once avoids
/// rematerializing its address for every pretoken in the phase-B loop.
#[derive(Clone, Copy)]
pub(crate) struct PieceKeyPacker {
    masks: *const [u64; 2],
}

impl PieceKeyPacker {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            masks: std::hint::black_box(PACK_MASK_TABLE.as_ptr()),
        }
    }

    /// Construct a key in a fill body whose x86 CRC/fold arm was selected
    /// once before the per-span loop. Off x86 the const parameter is ignored.
    ///
    /// The `X86_CRC = true` instantiation may only be reached from an
    /// SSE4.2-gated fill wrapper (see [`fill_key_hash`]).
    #[inline(always)]
    pub(crate) fn key_from_loaded_fill<const X86_CRC: bool>(
        self,
        word: u128,
        len: usize,
    ) -> PieceKey {
        let packed = self.pack_loaded(word, len);
        PieceKey::from_packed_hash(packed, fill_key_hash::<X86_CRC>(packed))
    }

    /// Careful counterpart of [`Self::key_from_loaded_fill`] for a span whose
    /// 16-byte wide load would cross the end of its backing slice.
    #[inline(always)]
    pub(crate) fn key_from_span_fill<const X86_CRC: bool>(
        self,
        bytes: &[u8],
        start: usize,
        len: usize,
    ) -> PieceKey {
        if len == 0 || len > 15 {
            return PieceKey::default();
        }
        let end = start.checked_add(len).expect("pretoken span overflow");
        assert!(end <= bytes.len(), "pretoken span exceeds backing bytes");
        let packed = if bytes.len() - start >= 16 {
            let word = u128::from_le(unsafe {
                (bytes.as_ptr().add(start) as *const u128).read_unaligned()
            });
            self.pack_loaded(word, len)
        } else {
            pack_key(&bytes[start..end]).expect("short span must pack")
        };
        PieceKey::from_packed_hash(packed, fill_key_hash::<X86_CRC>(packed))
    }

    #[inline(always)]
    fn pack_loaded(self, word: u128, len: usize) -> u128 {
        debug_assert!(len != 0);
        let packed_len = len.min(15);
        let [lo_mask, hi_mask] = unsafe { *self.masks.add(packed_len) };
        let packed = ((word as u64 & lo_mask) as u128)
            | (((((word >> 64) as u64 & hi_mask) | ((packed_len as u64) << 56)) as u128) << 64);
        packed & ((len <= 15) as u128).wrapping_neg()
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u128,
    val: u64,
    ext: u64,
}

const _: () = assert!(std::mem::size_of::<Entry>() == 32);

/// Zeroed, cache-line aligned storage for the short-key table.
///
/// Two adjacent entries form one 64-byte probe pair. On Linux, tables at
/// least one huge page in size are 2 MiB aligned and marked `MADV_HUGEPAGE`
/// before first touch, matching Gigatoken's dTLB-friendly allocation. Other
/// platforms retain the alignment and simply ignore the page hint.
struct Slots {
    ptr: NonNull<Entry>,
    cap: usize,
}

impl Slots {
    const HUGE_PAGE: usize = 2 * 1024 * 1024;

    fn new_zeroed(cap: usize) -> Self {
        debug_assert!(cap.is_power_of_two() && cap >= 2);
        let layout = Self::layout(cap);
        // SAFETY: `layout` is non-zero because `cap >= 2`.
        let raw = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(raw.cast::<Entry>()) else {
            handle_alloc_error(layout);
        };
        madvise_hugepage(raw, layout.size());
        // Hint before first touch so Linux can fault the table through THP.
        // SAFETY: the allocation contains exactly `layout.size()` bytes.
        unsafe { std::ptr::write_bytes(raw, 0, layout.size()) };
        Self { ptr, cap }
    }

    fn layout(cap: usize) -> Layout {
        let size = cap
            .checked_mul(std::mem::size_of::<Entry>())
            .expect("short pretoken cache allocation overflow");
        let align = Self::HUGE_PAGE.min(size.next_power_of_two()).max(64);
        Layout::from_size_align(size, align).expect("short pretoken cache layout overflow")
    }

    #[inline(always)]
    unsafe fn get(&self, index: usize) -> &Entry {
        debug_assert!(index < self.cap);
        // SAFETY: guaranteed by the caller.
        unsafe { &*self.ptr.as_ptr().add(index) }
    }

    #[inline(always)]
    unsafe fn get_mut(&mut self, index: usize) -> &mut Entry {
        debug_assert!(index < self.cap);
        // SAFETY: guaranteed by the caller.
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

impl Drop for Slots {
    fn drop(&mut self) {
        // SAFETY: allocated using this exact layout in `new_zeroed`.
        unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), Self::layout(self.cap)) };
    }
}

// SAFETY: `Slots` exclusively owns its allocation, just like `Box<[Entry]>`.
unsafe impl Send for Slots {}
unsafe impl Sync for Slots {}

/// One-table, one-cache-line short-pretoken cache. Home positions are even
/// slot indexes; linear probing advances one pair at a time.
struct ShortCache {
    slots: Slots,
    mask: usize,
    len: usize,
}

impl ShortCache {
    fn with_entries(entries: usize) -> Self {
        let mut cap = (1usize << 16).max(entries.next_power_of_two());
        while (entries + 1) * 4 > cap * 3 {
            cap *= 2;
        }
        Self {
            slots: Slots::new_zeroed(cap),
            mask: cap - 1,
            len: 0,
        }
    }

    #[cfg(test)]
    fn with_pow2_capacity(cap: usize) -> Self {
        Self {
            slots: Slots::new_zeroed(cap),
            mask: cap - 1,
            len: 0,
        }
    }

    #[inline(always)]
    fn home_pair(&self, hash: u64) -> usize {
        hash as usize & self.mask & !1
    }

    #[inline(always)]
    fn pair_ptr(&self, hash: u64) -> *const Entry {
        // SAFETY: the masked even index is at most `cap - 2`.
        unsafe { self.slots.ptr.as_ptr().add(self.home_pair(hash)) }
    }

    /// Exact lookup. A miss also returns the first empty insertion slot, so a
    /// cold BPE path can avoid walking the just-touched probe chain twice.
    #[inline]
    fn get_or_slot(&self, key: u128, hash: u64) -> Result<(u64, u64), usize> {
        debug_assert_ne!(key, EMPTY_KEY);
        let mut index = self.home_pair(hash);
        loop {
            // SAFETY: `index` is masked and even, so both slots are live.
            let (a, b) = unsafe { (self.slots.get(index), self.slots.get(index + 1)) };
            if a.key == key {
                return Ok((a.val, a.ext));
            }
            if b.key == key {
                return Ok((b.val, b.ext));
            }
            if a.key == EMPTY_KEY {
                return Err(index);
            }
            if b.key == EMPTY_KEY {
                return Err(index + 1);
            }
            index = (index + 2) & self.mask;
        }
    }

    #[inline]
    fn get(&self, key: u128, hash: u64) -> Option<(u64, u64)> {
        self.get_or_slot(key, hash).ok()
    }

    fn first_empty(&self, hash: u64) -> usize {
        let mut index = self.home_pair(hash);
        loop {
            // SAFETY: `index` is masked and even, so both slots are live.
            unsafe {
                if self.slots.get(index).key == EMPTY_KEY {
                    return index;
                }
                if self.slots.get(index + 1).key == EMPTY_KEY {
                    return index + 1;
                }
            }
            index = (index + 2) & self.mask;
        }
    }

    /// Insert or replace. Generic callers retain the former cache semantics;
    /// optimized miss paths can use `insert_at` to skip the duplicate walk.
    fn insert(&mut self, key: u128, hash: u64, val: u64, ext: u64) {
        debug_assert_ne!(key, EMPTY_KEY);
        match self.get_or_slot(key, hash) {
            Ok(_) => {
                let mut index = self.home_pair(hash);
                loop {
                    // SAFETY: pair indexes are in bounds.
                    unsafe {
                        if self.slots.get(index).key == key {
                            *self.slots.get_mut(index) = Entry { key, val, ext };
                            return;
                        }
                        if self.slots.get(index + 1).key == key {
                            *self.slots.get_mut(index + 1) = Entry { key, val, ext };
                            return;
                        }
                    }
                    index = (index + 2) & self.mask;
                }
            }
            Err(slot) => self.insert_at(slot, key, hash, val, ext),
        }
    }

    /// Insert a key known absent at the slot returned by `get_or_slot`.
    fn insert_at(&mut self, mut slot: usize, key: u128, hash: u64, val: u64, ext: u64) {
        if (self.len + 1) * 4 > self.slots.cap * 3 {
            self.grow();
            slot = self.first_empty(hash);
        }
        debug_assert_eq!(slot, self.first_empty(hash));
        // SAFETY: `get_or_slot`/`first_empty` return an in-bounds slot.
        unsafe { *self.slots.get_mut(slot) = Entry { key, val, ext } };
        self.len += 1;
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        self.rehash(self.slots.cap * 2);
    }

    /// Replace the table with one allocation of exactly `new_cap` slots and
    /// reinsert every live entry once.
    #[cold]
    #[inline(never)]
    fn rehash(&mut self, new_cap: usize) {
        debug_assert!(new_cap.is_power_of_two());
        debug_assert!(new_cap > self.slots.cap);
        let old = std::mem::replace(&mut self.slots, Slots::new_zeroed(new_cap));
        self.mask = new_cap - 1;
        for index in 0..old.cap {
            // SAFETY: `index < old.cap`.
            let entry = *unsafe { old.get(index) };
            if entry.key != EMPTY_KEY {
                let dst = self.first_empty(key_hash(entry.key));
                // SAFETY: `first_empty` returns an in-bounds slot.
                unsafe { *self.slots.get_mut(dst) = entry };
            }
        }
    }

    /// Grow once to the cache capacity predicted for a known input workload.
    /// Gigatoken sizes its serial worker before encoding for the same reason:
    /// keeping nearly every short key in its home pair is substantially
    /// cheaper than taking the displaced cold path on a warmed corpus.
    fn reserve_slots(&mut self, min_slots: usize) {
        let target = min_slots.max(1 << 16).next_power_of_two();
        if self.slots.cap < target {
            self.rehash(target);
        }
    }

    #[inline(always)]
    fn prefetch_l2(&self, hash: u64) {
        prefetch_line::<false>(self.pair_ptr(hash));
    }

    #[inline(always)]
    fn probe_view(&self) -> RawProbeView {
        RawProbeView {
            base: self.slots.ptr.as_ptr(),
            pair_mask: self.mask & !1,
        }
    }
}

/// Copyable raw snapshot of the current short-cache allocation.
///
/// A cache insertion may grow and move the table. Callers must refresh this
/// snapshot immediately after any operation that can mutate the cache.
#[derive(Clone, Copy)]
struct RawProbeView {
    base: *const Entry,
    pair_mask: usize,
}

impl RawProbeView {
    #[inline(always)]
    fn pair_ptr(self, hash: u64) -> *const Entry {
        // SAFETY: masked even index is at most capacity - 2.
        unsafe { self.base.add(hash as usize & self.pair_mask) }
    }

    /// Promote the home pair from L2 to L1 a short, fixed distance before its
    /// probe. Scanner fill already requested the same line into L2.
    #[inline(always)]
    fn prefetch_l1(self, key: PieceKey) {
        // Long spans carry hash zero. Prefetching that masked, in-bounds line
        // is harmless and keeps the overwhelmingly common short-key path free
        // of a per-pretoken branch, matching Gigatoken's probe view.
        prefetch_line::<true>(self.pair_ptr(key.hash));
    }

    /// Branchlessly select either slot of the home pair. Displaced hits and
    /// genuine misses both return `found == false`; they belong on the cold
    /// exact-probe path. For key zero, empty slots compare equal, so callers
    /// must include `key.is_short()` in their fast predicate.
    #[inline(always)]
    fn probe_pair(self, key: PieceKey) -> (u64, u64, bool) {
        let ptr = self.pair_ptr(key.hash);
        // SAFETY: `pair_ptr` returns the first slot of a live aligned pair.
        let (a, b) = unsafe { (&*ptr, &*ptr.add(1)) };
        let match_a = a.key == key.packed();
        let match_b = b.key == key.packed();

        #[cfg(target_arch = "aarch64")]
        let (val, ext) = {
            let (mut val, mut ext) = (a.val, a.ext);
            // Register-value selects avoid LLVM producing a selected address
            // followed by a dependent value load.
            unsafe {
                core::arch::asm!(
                    "cmp {matched}, #0",
                    "csel {val}, {val}, {bval}, ne",
                    "csel {ext}, {ext}, {bext}, ne",
                    matched = in(reg) match_a as u64,
                    val = inout(reg) val,
                    ext = inout(reg) ext,
                    bval = in(reg) b.val,
                    bext = in(reg) b.ext,
                    options(pure, nomem, nostack),
                );
            }
            (val, ext)
        };
        #[cfg(target_arch = "x86_64")]
        let (val, ext) = {
            let (mut val, mut ext) = (b.val, b.ext);
            unsafe {
                core::arch::asm!(
                    "test {matched}, {matched}",
                    "cmovne {val}, {aval}",
                    "cmovne {ext}, {aext}",
                    matched = in(reg) match_a as u64,
                    val = inout(reg) val,
                    ext = inout(reg) ext,
                    aval = in(reg) a.val,
                    aext = in(reg) a.ext,
                    options(pure, nomem, nostack),
                );
            }
            (val, ext)
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let (val, ext) = {
            let select_a = (match_a as u64).wrapping_neg();
            (
                (a.val & select_a) | (b.val & !select_a),
                (a.ext & select_a) | (b.ext & !select_a),
            )
        };
        (val, ext, match_a | match_b)
    }
}

/// Output projection of a raw table view. Its home-pair probe returns the two
/// packed token words directly for Gigatoken-style unconditional lane stores.
#[derive(Clone, Copy)]
pub(crate) struct OutputProbeView(RawProbeView);

impl OutputProbeView {
    /// Promote the home pair L2 -> L1 shortly before probing it.
    #[inline(always)]
    pub(crate) fn prefetch_l1(self, key: PieceKey) {
        self.0.prefetch_l1(key);
    }

    /// Branchless home-pair packed-value probe. The caller's fast predicate is
    /// `found && packed_is_inline(val) && key.is_short()`.
    #[inline(always)]
    pub(crate) fn probe_pair(self, key: PieceKey) -> (u64, u64, bool) {
        let (val, ext, found) = self.0.probe_pair(key);
        (val, ext, found & key.is_short())
    }
}

/// Token count encoded in a packed cache value.
#[inline(always)]
pub(crate) fn packed_token_count(val: u64) -> usize {
    (val & 0x7f) as usize
}

/// Whether all token IDs are resident in the packed value's four lanes.
#[inline(always)]
pub(crate) fn packed_is_inline(val: u64) -> bool {
    val & SPILL == 0
}

/// Decode the four resident token lanes. Only the first
/// `packed_token_count(val)` lanes are part of the encoding.
#[inline(always)]
pub(crate) fn unpack_inline_lanes(val: u64, ext: u64) -> [u32; 4] {
    [
        (val >> 8) as u32 & 0x00ff_ffff,
        (val >> 32) as u32,
        ext as u32,
        (ext >> 32) as u32,
    ]
}

/// Store all four inline token lanes as two packed writes. Callers may advance
/// their logical cursor by only `packed_token_count(val)`; later stores can
/// overwrite the unused tail lanes, as in Gigatoken's flat emit loop.
///
/// # Safety
///
/// `dst` must have room for four writable `u32`s.
#[inline(always)]
pub(crate) unsafe fn write_inline_lanes(dst: *mut u32, val: u64, ext: u64) {
    let first_two = ((val >> 8) & 0x00ff_ffff) | (val & 0xffff_ffff_0000_0000);
    // SAFETY: guaranteed by the caller; unaligned writes support arbitrary
    // `Vec<u32>` cursor alignment.
    unsafe {
        dst.cast::<u64>().write_unaligned(first_two);
        dst.add(2).cast::<u64>().write_unaligned(ext);
    }
}

#[derive(Clone, Copy)]
struct ArenaValue {
    offset: u32,
    len: u32,
}

/// A cache hit, either decoded from an inline value or borrowed from the
/// append-only spill arena.
pub(crate) enum CachedTokens<'a> {
    Inline { lanes: [u32; 4], len: usize },
    Arena(&'a [u32]),
}

/// First empty short-table slot returned by an exact failed lookup.
///
/// The field is deliberately private and the value is not `Copy`: it may be
/// consumed only by [`PieceCache::insert_short_at`], with no intervening
/// short-table insertion. Arena growth and BPE/output work do not invalidate
/// it. If consuming the slot itself crosses the table's load threshold,
/// `ShortCache::insert_at` grows first and recomputes the destination.
pub(crate) struct ShortInsertSlot(usize);

impl CachedTokens<'_> {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u32] {
        match self {
            CachedTokens::Inline { lanes, len } => &lanes[..*len],
            CachedTokens::Arena(tokens) => tokens,
        }
    }
}

/// Full-pretoken cache used by `Cache::Full`.
pub(crate) struct PieceCache {
    short: ShortCache,
    long: FxHashMap<Box<[u8]>, ArenaValue>,
    long_key_bytes: usize,
    arena: Vec<u32>,
}

impl PieceCache {
    /// Seed every short vocabulary entry as a one-token encoding. This moves
    /// the overwhelmingly common whole-vocabulary-word hit out of the generic
    /// whole-token lookup before the timed encode loop starts.
    pub(crate) fn seeded(vocab: &RuntimeVocab) -> Self {
        let short_count = vocab
            .entries()
            .filter(|(key, _)| !key.is_empty() && key.len() <= 15)
            .count();
        let mut cache = Self {
            short: ShortCache::with_entries(short_count),
            long: FxHashMap::default(),
            long_key_bytes: 0,
            arena: Vec::new(),
        };
        for (bytes, id) in vocab.entries() {
            if let Some(key) = pack_key(bytes)
                && key != 0
                && let Some((val, ext)) = cache_value(&mut cache.arena, &[id])
            {
                cache.short.insert(key, key_hash(key), val, ext);
            }
        }
        cache
    }

    /// Pre-size cache storage for a serial workload of `expected_bytes`.
    ///
    /// The short-key estimate is Gigatoken's OWT-calibrated Heaps-law model
    /// (`distinct ≈ 3.45·n^0.62`) at the table's 3/4 load threshold with 1.4×
    /// headroom. This is only a capacity hint: streaming input and output stay
    /// bounded, and every cache structure can still grow when the corpus is
    /// more diverse than the estimate.
    pub(crate) fn reserve_for_bytes(&mut self, expected_bytes: usize) {
        let distinct = 3.45 * (expected_bytes as f64).powf(0.62);
        let cache_slots = ((distinct * (4.0 / 3.0) * 1.4) as usize)
            .clamp(1 << 16, 1 << 22)
            .next_power_of_two();
        self.short.reserve_slots(cache_slots);

        let arena_cap = (expected_bytes / 256).min(1 << 24);
        if self.arena.capacity() < arena_cap {
            self.arena.reserve(arena_cap - self.arena.capacity());
        }
        let long_cap = (expected_bytes / 8192).min(1 << 20);
        if self.long.capacity() < long_cap {
            self.long.reserve(long_cap - self.long.capacity());
        }
    }

    #[inline(always)]
    pub(crate) fn get_keyed(&self, bytes: &[u8], piece_key: PieceKey) -> Option<CachedTokens<'_>> {
        if piece_key.is_short() {
            let (val, ext) = self.short.get(piece_key.packed(), piece_key.hash)?;
            return Some(self.decode_short_value(val, ext));
        }

        let value = self.long.get(bytes)?;
        let start = value.offset as usize;
        Some(CachedTokens::Arena(
            &self.arena[start..start + value.len as usize],
        ))
    }

    /// Resolve a short-key home-pair rejection exactly. A hit borrows its
    /// cached encoding; a miss returns the insertion slot discovered by the
    /// same probe walk so the cold BPE path does not walk the chain again.
    #[inline]
    pub(crate) fn get_or_short_slot(
        &self,
        piece_key: PieceKey,
    ) -> Result<CachedTokens<'_>, ShortInsertSlot> {
        debug_assert!(piece_key.is_short());
        match self.short.get_or_slot(piece_key.packed(), piece_key.hash) {
            Ok((val, ext)) => Ok(self.decode_short_value(val, ext)),
            Err(slot) => Err(ShortInsertSlot(slot)),
        }
    }

    #[inline(always)]
    fn decode_short_value(&self, val: u64, ext: u64) -> CachedTokens<'_> {
        let len = packed_token_count(val);
        if packed_is_inline(val) {
            CachedTokens::Inline {
                lanes: unpack_inline_lanes(val, ext),
                len,
            }
        } else {
            let offset = (val >> 32) as usize;
            CachedTokens::Arena(&self.arena[offset..offset + len])
        }
    }

    /// Insert a long key after the caller has already proved it absent.
    ///
    /// Fast-path callers arrive directly from a failed exact lookup. Skipping
    /// `contains_key` avoids hashing a long code pretoken twice on its first
    /// occurrence.
    #[inline]
    pub(crate) fn insert_long_known_absent(&mut self, bytes: &[u8], tokens: &[u32]) {
        debug_assert!(bytes.len() > 15);
        debug_assert!(!self.long.contains_key(bytes));
        let Ok(offset) = u32::try_from(self.arena.len()) else {
            return;
        };
        let Ok(len) = u32::try_from(tokens.len()) else {
            return;
        };
        self.arena.extend_from_slice(tokens);
        self.long.insert(bytes.into(), ArenaValue { offset, len });
        self.long_key_bytes += bytes.len();
    }

    /// Insert a short key at the first-empty slot returned by
    /// [`Self::get_or_short_slot`].
    ///
    /// No short-table insertion may occur between producing and consuming
    /// `slot`. The token arena may grow freely. A growth caused by this insert
    /// is handled inside `ShortCache::insert_at`, which recomputes the slot
    /// against the new table.
    #[inline]
    pub(crate) fn insert_short_at(
        &mut self,
        piece_key: PieceKey,
        slot: ShortInsertSlot,
        tokens: &[u32],
    ) {
        debug_assert!(piece_key.is_short());
        if let Some((val, ext)) = cache_value(&mut self.arena, tokens) {
            self.short
                .insert_at(slot.0, piece_key.packed(), piece_key.hash, val, ext);
        }
    }

    /// Start the single home-pair cache line toward L2 during scanner fill.
    #[inline(always)]
    pub(crate) fn prefetch_l2(&self, piece_key: PieceKey) {
        // A long-key PieceKey has hash zero; touching that arbitrary live line
        // is cheaper than branching on every ordinary short pretoken.
        self.short.prefetch_l2(piece_key.hash);
    }

    /// Snapshot the table base/mask for one packed-output probe batch.
    #[inline(always)]
    pub(crate) fn output_probe_view(&self) -> OutputProbeView {
        OutputProbeView(self.short.probe_view())
    }

    /// Approximate bytes owned by the full-pretoken cache.
    ///
    /// The paired short table, token arena, long-map entry payloads, and
    /// heap-allocated long keys are included. Hash-table control metadata is
    /// implementation-specific, so this remains an estimate.
    pub(crate) fn estimated_bytes(&self) -> usize {
        let long_entry_bytes = self.long.capacity()
            * (std::mem::size_of::<Box<[u8]>>() + std::mem::size_of::<ArenaValue>() + 1);
        std::mem::size_of::<Self>()
            + self.short.slots.cap * std::mem::size_of::<Entry>()
            + self.arena.capacity() * std::mem::size_of::<u32>()
            + long_entry_bytes
            + self.long_key_bytes
    }
}

/// Request one cache line into L1 (`L1 = true`) or L2. Prefetch instructions
/// are non-faulting hints; the pointer always addresses live cache storage.
#[inline(always)]
fn prefetch_line<const L1: bool>(ptr: *const Entry) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        if L1 {
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags, readonly)
            );
        } else {
            core::arch::asm!(
                "prfm pldl2keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};
        if L1 {
            _mm_prefetch(ptr.cast(), _MM_HINT_T0);
        } else {
            _mm_prefetch(ptr.cast(), _MM_HINT_T1);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = ptr;
}

#[inline]
fn madvise_hugepage(ptr: *mut u8, len: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let _ = libc::madvise(ptr.cast(), len, libc::MADV_HUGEPAGE);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (ptr, len);
}

/// Bytes occupy the low 15 lanes; the top byte tags the length. Zero remains
/// the short table's empty sentinel.
#[inline(always)]
fn pack_key(bytes: &[u8]) -> Option<u128> {
    let n = bytes.len();
    if n > 15 {
        return None;
    }
    if n == 0 {
        return Some(0);
    }

    // Callers that can prove 16 readable bytes use `PieceKey::from_span`.
    // A standalone slice carries no such provenance, even when the following
    // bytes happen to reside on the same mapped page, so copy its exact extent.
    let mut lanes = [0u8; 16];
    lanes[..n].copy_from_slice(bytes);
    let low = u128::from_le_bytes(lanes);
    Some(low | ((n as u128) << 120))
}

/// Phase-B mask rows: one L1-resident 16-byte load replaces two independent
/// variable-shift chains for the issue-width-bound span pack loop.
static PACK_MASK_TABLE: [[u64; 2]; 16] = {
    let mut table = [[0u64; 2]; 16];
    let mut n = 1usize;
    while n <= 15 {
        let bits = (n * 8) as u32;
        table[n] = [
            if n < 8 {
                u64::MAX >> (64u32.wrapping_sub(bits) & 63)
            } else {
                u64::MAX
            },
            if n > 8 {
                u64::MAX >> (128u32.wrapping_sub(bits) & 63)
            } else {
                0
            },
        ];
        n += 1;
    }
    table
};

#[inline(always)]
fn pack_loaded_table(word: u128, n: usize) -> u128 {
    debug_assert!((1..=15).contains(&n));
    let [lo_mask, hi_mask] = unsafe { *PACK_MASK_TABLE.get_unchecked(n) };
    ((word as u64 & lo_mask) as u128)
        | (((((word >> 64) as u64 & hi_mask) | ((n as u64) << 56)) as u128) << 64)
}

#[inline(always)]
fn key_hash(key: u128) -> u64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "crc"))]
    {
        use std::arch::aarch64::__crc32d;
        // CRC is part of the compile-time target on Apple Silicon, so this
        // stays in the hot function without per-key feature detection or an
        // out-of-line target-feature call.
        unsafe { __crc32d(__crc32d(0, key as u64), (key >> 64) as u64) as u64 }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crc_hash_selected() {
            // SAFETY: guarded by the runtime SSE4.2 check.
            return unsafe { key_hash_crc32(key) };
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "crc")))]
    {
        key_hash_fold(key)
    }
}

#[allow(dead_code)]
#[inline(always)]
fn key_hash_fold(key: u128) -> u64 {
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    let mut hash = (lo ^ hi.rotate_right(25)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash ^= hash >> 32;
    hash
}

/// Whether x86 pretoken keys use the SSE4.2 CRC32C hash arm. The result is
/// process-immutable; fast fill loops dispatch on it once per batch and use
/// [`fill_key_hash`] inside their per-span loop.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn crc_hash_selected() -> bool {
    std::arch::is_x86_feature_detected!("sse4.2")
}

/// Hash a packed key after the x86 hash arm has been selected outside the
/// per-span loop. `X86_CRC = true` is valid only inside an SSE4.2-gated fill
/// wrapper. Off x86 the parameter is ignored and the normal compile-time arm
/// is used.
#[inline(always)]
fn fill_key_hash<const X86_CRC: bool>(key: u128) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if X86_CRC {
            debug_assert!(crc_hash_selected());
            // SAFETY: the true instantiation is only reached through an
            // SSE4.2-gated fill wrapper.
            unsafe { key_hash_crc32(key) }
        } else {
            key_hash_fold(key)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = X86_CRC;
        key_hash(key)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[inline]
// `_mm_crc32_u64` requires `unsafe` on our Rust 1.89 MSRV but is safe on
// newer compilers. Keep the block for the MSRV without warning there or here.
#[allow(unused_unsafe)]
unsafe fn key_hash_crc32(key: u128) -> u64 {
    use std::arch::x86_64::_mm_crc32_u64;
    unsafe { _mm_crc32_u64(_mm_crc32_u64(0, key as u64), (key >> 64) as u64) }
}

#[inline(always)]
fn pack_inline(tokens: &[u32]) -> Option<(u64, u64)> {
    match tokens {
        [a] if *a < 1 << 24 => Some((1 | ((*a as u64) << 8), 0)),
        [a, b] if *a < 1 << 24 => Some((2 | ((*a as u64) << 8) | ((*b as u64) << 32), 0)),
        [a, b, c] if *a < 1 << 24 => {
            Some((3 | ((*a as u64) << 8) | ((*b as u64) << 32), *c as u64))
        }
        [a, b, c, d] if *a < 1 << 24 => Some((
            4 | ((*a as u64) << 8) | ((*b as u64) << 32),
            *c as u64 | ((*d as u64) << 32),
        )),
        _ => None,
    }
}

#[inline]
fn cache_value(arena: &mut Vec<u32>, tokens: &[u32]) -> Option<(u64, u64)> {
    if let Some(value) = pack_inline(tokens) {
        return Some(value);
    }
    assert!(
        tokens.len() < 128,
        "a <=15-byte pretoken cannot produce 128 tokens"
    );
    let offset = u32::try_from(arena.len()).ok()?;
    arena.extend_from_slice(tokens);
    Some((SPILL | tokens.len() as u64 | ((offset as u64) << 32), 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn runtime_vocab(entries: &[(&[u8], u32)]) -> RuntimeVocab {
        let tokens = entries
            .iter()
            .map(|(bytes, _)| Bytes::copy_from_slice(bytes))
            .collect::<Vec<_>>();
        let legacy = crate::Vocab::new(tokens).unwrap();
        let ranks = entries
            .iter()
            .map(|(bytes, id)| (bytes.to_vec(), *id))
            .collect();
        RuntimeVocab::from_legacy(&legacy, &ranks)
    }

    fn empty_runtime_vocab() -> RuntimeVocab {
        RuntimeVocab::try_from_ordered_tokens(&[]).unwrap()
    }

    fn get<'a>(cache: &'a PieceCache, bytes: &[u8]) -> Option<CachedTokens<'a>> {
        cache.get_keyed(bytes, PieceKey::from_bytes(bytes))
    }

    fn insert(cache: &mut PieceCache, bytes: &[u8], tokens: &[u32]) {
        let key = PieceKey::from_bytes(bytes);
        if key.is_short() {
            let slot = match cache.get_or_short_slot(key) {
                Ok(_) => panic!("test key unexpectedly present"),
                Err(slot) => slot,
            };
            cache.insert_short_at(key, slot, tokens);
        } else {
            assert!(cache.get_keyed(bytes, key).is_none());
            cache.insert_long_known_absent(bytes, tokens);
        }
    }

    #[test]
    fn seeded_inline_and_spilled_roundtrip() {
        let vocab = runtime_vocab(&[(b"hello", 42), (b"123456789012345", 99)]);
        let mut cache = PieceCache::seeded(&vocab);
        assert_eq!(get(&cache, b"hello").unwrap().as_slice(), &[42]);
        assert_eq!(get(&cache, b"123456789012345").unwrap().as_slice(), &[99]);

        insert(&mut cache, b"miss", &[1, 2, 3, 4]);
        assert_eq!(get(&cache, b"miss").unwrap().as_slice(), &[1, 2, 3, 4]);
        insert(&mut cache, b"five", &[1, 2, 3, 4, 5]);
        assert_eq!(get(&cache, b"five").unwrap().as_slice(), &[1, 2, 3, 4, 5]);

        let long = b"this pretoken is definitely longer than fifteen bytes";
        insert(&mut cache, long, &[7, 8, 9]);
        assert_eq!(get(&cache, long).unwrap().as_slice(), &[7, 8, 9]);
    }

    #[test]
    fn seeded_large_token_id_spills_without_panicking() {
        let token_id = 1 << 24;
        let vocab = runtime_vocab(&[(b"large-id", token_id)]);
        let cache = PieceCache::seeded(&vocab);
        assert_eq!(get(&cache, b"large-id").unwrap().as_slice(), &[token_id]);
    }

    #[test]
    fn span_key_matches_bounded_slice_key() {
        let bytes: Vec<u8> = (0..96).map(|i| (i as u8).wrapping_mul(37)).collect();
        for start in 0..bytes.len() {
            for len in 0..=(bytes.len() - start).min(32) {
                assert_eq!(
                    PieceKey::from_span(&bytes, start, len),
                    PieceKey::from_bytes(&bytes[start..start + len]),
                    "start={start} len={len}",
                );
            }
        }

        // The same selected bytes must not depend on readable trailing lanes.
        let mut a = b"short-tail-1234xxxxxxxxxxxxxxxx".to_vec();
        let mut b = a.clone();
        b[15..].fill(0xa5);
        assert_eq!(
            PieceKey::from_span(&a, 0, 15),
            PieceKey::from_span(&b, 0, 15)
        );
        a.truncate(15);
        assert_eq!(PieceKey::from_span(&a, 0, 15), PieceKey::from_bytes(&a));
    }

    /// Find three distinct short keys sharing one home pair, so that inserting
    /// all three forces the last one onto the displaced probe path.
    ///
    /// The search must use keys wider than two bytes. A two-byte key varies over
    /// exactly sixteen packed bits, and every hash arm here is affine in those
    /// bits: CRC32C, the aarch64 CRC, and the multiply/xor fold all reduce to a
    /// bijection on the sixteen bits the home index is taken from. All 65,535
    /// two-byte candidates therefore land exactly two per home pair and no third
    /// collision exists to find, so the original two-byte search could only ever
    /// exhaust its space and panic.
    fn colliding_short_keys(cache: &PieceCache) -> Vec<(Vec<u8>, PieceKey)> {
        let mut groups: FxHashMap<usize, Vec<(Vec<u8>, PieceKey)>> = FxHashMap::default();
        for value in 1u32..=u32::MAX {
            let bytes = value.to_le_bytes().to_vec();
            let key = PieceKey::from_bytes(&bytes);
            let home = key.hash as usize & cache.short.mask & !1;
            let group = groups.entry(home).or_default();
            group.push((bytes, key));
            if group.len() == 3 {
                return std::mem::take(group);
            }
        }
        panic!("must find three colliding short keys");
    }

    #[test]
    fn exact_lookup_handles_home_lanes_displacement_and_spill() {
        let mut cache = PieceCache::seeded(&empty_runtime_vocab());
        let keys = colliding_short_keys(&cache);

        insert(&mut cache, &keys[0].0, &[1]);
        insert(&mut cache, &keys[1].0, &[1, 2, 3, 4, 5]);
        insert(&mut cache, &keys[2].0, &[1, 2, 3]);
        assert_eq!(
            cache
                .get_keyed(&keys[0].0, keys[0].1)
                .expect("first cached key")
                .as_slice(),
            &[1]
        );
        assert_eq!(
            cache
                .get_keyed(&keys[1].0, keys[1].1)
                .expect("second cached key")
                .as_slice(),
            &[1, 2, 3, 4, 5]
        );
        assert_eq!(
            cache
                .get_keyed(&keys[2].0, keys[2].1)
                .expect("displaced cached key")
                .as_slice(),
            &[1, 2, 3]
        );
    }

    #[test]
    fn exact_miss_slot_is_recomputed_when_its_insert_grows() {
        let mut cache = PieceCache {
            short: ShortCache::with_pow2_capacity(64),
            long: FxHashMap::default(),
            long_key_bytes: 0,
            arena: Vec::new(),
        };
        for value in 1u16..=48 {
            let bytes = value.to_le_bytes();
            insert(&mut cache, &bytes, &[value as u32]);
        }
        assert_eq!(cache.short.slots.cap, 64);

        let bytes = 49u16.to_le_bytes();
        let key = PieceKey::from_bytes(&bytes);
        let slot = match cache.get_or_short_slot(key) {
            Ok(_) => panic!("new key unexpectedly present"),
            Err(slot) => slot,
        };
        cache.insert_short_at(key, slot, &[7, 11, 13, 17, 19]);

        assert_eq!(cache.short.slots.cap, 128);
        let cached = match cache.get_or_short_slot(key) {
            Ok(cached) => cached,
            Err(_) => panic!("inserted key missing after growth"),
        };
        assert_eq!(cached.as_slice(), &[7, 11, 13, 17, 19]);
    }

    #[test]
    fn reserve_slots_rehashes_directly_to_target_and_preserves_entries() {
        let mut cache = ShortCache::with_pow2_capacity(64);
        let mut entries = Vec::new();
        for value in 1u16..=40 {
            let bytes = value.to_le_bytes();
            let key = PieceKey::from_bytes(&bytes);
            let val = value as u64 | ((value as u64 + 1) << 32);
            let ext = !(value as u64);
            cache.insert(key.packed(), key.hash, val, ext);
            entries.push((key, val, ext));
        }

        cache.reserve_slots(1 << 17);

        assert_eq!(cache.slots.cap, 1 << 17);
        assert_eq!(cache.mask, (1 << 17) - 1);
        assert_eq!(cache.len, entries.len());
        for (key, val, ext) in entries {
            assert_eq!(cache.get(key.packed(), key.hash), Some((val, ext)));
        }

        // Reservation is monotone and never rebuilds a sufficiently large
        // table.
        let allocation = cache.slots.ptr;
        cache.reserve_slots(1 << 16);
        assert_eq!(cache.slots.ptr, allocation);
    }

    #[test]
    fn output_probe_exposes_packed_lanes_for_direct_emit() {
        let mut cache = PieceCache::seeded(&empty_runtime_vocab());
        let key = PieceKey::from_bytes(b"four");
        insert(&mut cache, b"four", &[17, 29, 41, 53]);

        let view = cache.output_probe_view();
        view.prefetch_l1(key);
        let (val, ext, found) = view.probe_pair(key);
        assert!(found);
        assert!(packed_is_inline(val));
        assert_eq!(packed_token_count(val), 4);
        assert_eq!(unpack_inline_lanes(val, ext), [17, 29, 41, 53]);

        let mut direct = [0u32; 4];
        // SAFETY: `direct` has four writable lanes.
        unsafe { write_inline_lanes(direct.as_mut_ptr(), val, ext) };
        assert_eq!(direct, [17, 29, 41, 53]);
    }

    #[test]
    fn long_key_never_hits_empty_short_slots() {
        let cache = PieceCache::seeded(&empty_runtime_vocab());
        let key = PieceKey::from_bytes(b"this key is longer than fifteen bytes");
        assert!(!key.is_short());
        assert!(!cache.output_probe_view().probe_pair(key).2);
    }
}
