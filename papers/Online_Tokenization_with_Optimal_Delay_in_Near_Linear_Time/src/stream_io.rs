//! Shared I/O for [`StreamTokenizer`](crate::StreamTokenizer): a fixed-chunk
//! UTF-8 reader and token sinks for a retained array, a streamed JSON array, or
//! a packed little-endian `u32` file.

use std::{
    fs::File,
    io::{self, Read, Write},
    marker::PhantomData,
    ptr::NonNull,
    time::{Duration, Instant},
};

/// Bytes read (and fed through the pipeline) per step. Output is identical at any
/// size; 256 KiB amortizes UTF-8 validation and stream handoff while keeping
/// the working buffer comfortably bounded.
pub(crate) const CHUNK_SIZE: usize = 256 * 1024;

/// Packed file output uses one reusable, bounded heap page.
///
/// This page does two jobs, and 256 KiB is comfortably past the knee for both.
/// Write amortization is a function of syscall count alone and flattens here;
/// measured at 96.4 M IDs, 16 KiB costs 6.94 ns/ID, 64 KiB 6.56, 256 KiB 6.47
/// and 1 MiB 6.43, a consistent ~1.4 us per `write`. The page also bounds the
/// direct cache-emission reservation, which is `batch_bytes + 4` for one keyed
/// batch; measured `batch_bytes` maxima are 2,761 (English r50k), 5,780 (GitHub
/// r50k), 7,906 (Chinese r50k) and 12,148 (Chinese o200k), so 65,536 IDs leaves
/// better than 5x headroom before a batch would fall back to the ordered path.
/// The former 4 MiB provided 86x more headroom than any real input needs.
const PACKED_U32_PAGE_BYTES: usize = 256 * 1024;
pub(crate) const PACKED_U32_PAGE_IDS: usize = PACKED_U32_PAGE_BYTES / size_of::<u32>();
/// JSON output stages token IDs before converting them, so the packed cache
/// emitter can write into it directly exactly as the compact target does.
///
/// This also bounds the direct-page reservation the JSON sink accepts, which is
/// `batch_bytes + 4` for one keyed batch. Measured `batch_bytes` maxima are
/// 2,761 (English r50k), 5,780 (GitHub r50k), 7,906 (Chinese r50k) and 12,148
/// (Chinese o200k), so this leaves better than 2.5x headroom while staying small
/// enough that staging and its byte page remain L2-resident between the emitter
/// that fills them and the converter that drains them.
const JSON_STAGE_IDS: usize = 32 * 1024;
/// Reusable output page for converted decimal bytes. Write amortization is
/// entirely a function of syscall count and flattens here: measured at 96.4 M
/// IDs, 16 KiB costs 6.94 ns/ID, 64 KiB 6.56, 256 KiB 6.47 and 1 MiB 6.43,
/// a consistent ~1.4 us per `write`. Past this point a larger page buys
/// throughput below measurement noise and costs resident memory.
const JSON_PAGE_BYTES: usize = 64 * 1024;
/// Widest single JSON element: one separator plus `u32::MAX`'s ten digits.
const JSON_MAX_ELEMENT_BYTES: usize = 11;
/// Enough input to make token density stable on typical text/code while still
/// reserving long before the initial array estimate can fill.
pub(crate) const ARRAY_DENSITY_SAMPLE_BYTES: usize = 1024 * 1024;

/// Read `path` in `CHUNK_SIZE`-byte steps, passing each maximal valid-UTF-8
/// prefix to `process`. Incomplete trailing bytes (≤ 3) are carried to the next
/// read, so `process` always gets whole characters. The returned boolean is
/// `false` when `process` requests early termination after consuming the
/// reported number of bytes.
pub(crate) fn read_chunks_while<F: FnMut(&str) -> bool>(
    path: &str,
    mut process: F,
) -> io::Result<(usize, bool)> {
    let mut reader = File::open(path)?;
    let mut buf = vec![0u8; CHUNK_SIZE];
    let mut carry: Vec<u8> = Vec::new();
    let mut total_bytes = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }

        // The overwhelmingly common case is a complete UTF-8 chunk. Feed it
        // directly from the read buffer instead of copying the entire chunk
        // through `carry`; only a split multibyte suffix needs buffering.
        if carry.is_empty() {
            match std::str::from_utf8(&buf[..n]) {
                Ok(chunk) => {
                    total_bytes += n;
                    if !process(chunk) {
                        return Ok((total_bytes, false));
                    }
                    continue;
                }
                Err(error) if error.error_len().is_none() => {
                    let valid = error.valid_up_to();
                    if valid != 0 {
                        total_bytes += valid;
                        if !process(unsafe { std::str::from_utf8_unchecked(&buf[..valid]) }) {
                            return Ok((total_bytes, false));
                        }
                    }
                    carry.extend_from_slice(&buf[valid..n]);
                    continue;
                }
                Err(error) => return Err(invalid_utf8_error(total_bytes, error)),
            }
        }

        carry.extend_from_slice(&buf[..n]);
        let valid = match std::str::from_utf8(&carry) {
            Ok(_) => carry.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(error) => return Err(invalid_utf8_error(total_bytes, error)),
        };
        total_bytes += valid;
        let chunk = unsafe { std::str::from_utf8_unchecked(&carry[..valid]) };
        if valid != 0 && !process(chunk) {
            return Ok((total_bytes, false));
        }
        carry.drain(..valid);
    }
    if !carry.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "input ends with {} byte(s) of an incomplete UTF-8 sequence at byte {total_bytes}",
                carry.len()
            ),
        ));
    }
    Ok((total_bytes, true))
}

fn invalid_utf8_error(offset: usize, error: std::str::Utf8Error) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "invalid UTF-8 at byte {}",
            offset.saturating_add(error.valid_up_to())
        ),
    )
}

/// Token consumer for in-memory arrays or streaming file output.
///
/// The file targets each use bounded, reusable pages. The memory target
/// intentionally retains one contiguous, input-sized token array.
///
/// A single sink can be driven across many tokenization calls: JSON output and
/// counts accumulate until [`finish`](TokenSink::finish) closes it.
pub struct TokenSink {
    pub(crate) total: usize,
    pub(crate) specials: usize,
    writer: Option<PackedU32Writer>,
    error: Option<io::Error>,
    first_token_timer: Option<FirstTokenTimer>,
}

struct FirstTokenTimer {
    started: Instant,
    elapsed: Option<Duration>,
}

/// Page-oriented writer for packed token IDs.
///
/// File output fills a reusable heap page and writes each completed page in
/// bulk. JSON stages IDs the same way before converting them to decimal. The
/// memory target grows one contiguous `Vec<u32>` for zero-copy conversion into
/// a NumPy array.
struct PackedU32Writer {
    target: PackedU32Target,
}

enum PackedU32Target {
    File(PackedFileWriter),
    Memory(MemoryU32Writer),
    Json(JsonU32Writer),
}

/// A writable view over initialized output followed by spare storage.
///
/// It intentionally exposes only the operations required by the flat cache
/// emitter. Unlike `Vec`, it cannot allocate or move the backing page.
pub(crate) struct PackedU32Page<'a> {
    ptr: NonNull<u32>,
    len: usize,
    capacity: usize,
    _borrow: PhantomData<&'a mut [u32]>,
}

impl PackedU32Page<'_> {
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u32 {
        self.ptr.as_ptr()
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn push(&mut self, id: u32) {
        assert!(self.len < self.capacity, "packed output page is full");
        // SAFETY: `len < capacity`, and the page is exclusively borrowed.
        unsafe { self.ptr.as_ptr().add(self.len).write(id) };
        self.len += 1;
    }

    #[inline]
    pub(crate) fn extend_from_slice(&mut self, ids: &[u32]) {
        assert!(
            ids.len() <= self.capacity - self.len,
            "packed output page has insufficient spare capacity"
        );
        // SAFETY: the source is initialized, the destination range is within
        // the exclusively borrowed page, and the two allocations do not
        // overlap in all supported call sites.
        unsafe {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), self.ptr.as_ptr().add(self.len), ids.len())
        };
        self.len += ids.len();
    }

    /// Mark the first `new_len` lanes initialized after the caller wrote them.
    ///
    /// # Safety
    ///
    /// Every newly included lane must have been initialized as a valid `u32`.
    #[inline(always)]
    pub(crate) unsafe fn set_len(&mut self, new_len: usize) {
        assert!(
            new_len >= self.len && new_len <= self.capacity,
            "invalid packed output page length"
        );
        self.len = new_len;
    }
}

impl PackedU32Writer {
    fn new_file(file: File) -> Self {
        Self {
            target: PackedU32Target::File(PackedFileWriter::new(file)),
        }
    }

    fn new_memory(capacity_hint: usize) -> Self {
        Self {
            target: PackedU32Target::Memory(MemoryU32Writer::new(capacity_hint)),
        }
    }

    fn new_json(file: File) -> Self {
        Self {
            target: PackedU32Target::Json(JsonU32Writer::new(file)),
        }
    }

    #[inline]
    fn push(&mut self, id: u32) -> io::Result<()> {
        match &mut self.target {
            PackedU32Target::File(writer) => writer.push(id),
            PackedU32Target::Memory(writer) => {
                writer.push(id);
                Ok(())
            }
            PackedU32Target::Json(writer) => writer.push(id),
        }
    }

    #[inline]
    fn push_many(&mut self, ids: &[u32]) -> io::Result<()> {
        match &mut self.target {
            PackedU32Target::File(writer) => writer.push_many(ids),
            PackedU32Target::Memory(writer) => {
                writer.push_many(ids);
                Ok(())
            }
            PackedU32Target::Json(writer) => writer.push_many(ids),
        }
    }

    #[inline]
    fn push_many_mapped(&mut self, ids: &[u32], rank_to_id: &[u32]) -> io::Result<()> {
        match &mut self.target {
            PackedU32Target::File(writer) => writer.push_many_mapped(ids, rank_to_id),
            PackedU32Target::Memory(writer) => {
                writer.push_many_mapped(ids, rank_to_id);
                Ok(())
            }
            PackedU32Target::Json(writer) => writer.push_many_mapped(ids, rank_to_id),
        }
    }

    #[inline]
    fn prepare_page(&mut self, required_spare: usize) -> io::Result<PackedU32Page<'_>> {
        match &mut self.target {
            PackedU32Target::File(writer) => writer.prepare_page(required_spare),
            PackedU32Target::Memory(writer) => Ok(writer.prepare_page(required_spare)),
            PackedU32Target::Json(writer) => writer.prepare_page(required_spare),
        }
    }

    #[inline]
    fn commit_page(&mut self, new_len: usize) {
        match &mut self.target {
            PackedU32Target::File(writer) => writer.commit_page(new_len),
            PackedU32Target::Memory(writer) => writer.commit_page(new_len),
            PackedU32Target::Json(writer) => writer.commit_page(new_len),
        }
    }

    #[inline(always)]
    fn supports_direct_page(&self, required_spare: usize) -> bool {
        // The emitter packs two token lanes into one `u64` store, so every
        // direct-page target depends on little-endian lane order, including the
        // JSON target that later reformats those lanes as decimal.
        cfg!(target_endian = "little")
            && match &self.target {
                PackedU32Target::File(_) => required_spare <= PACKED_U32_PAGE_IDS,
                PackedU32Target::Memory(_) => true,
                PackedU32Target::Json(_) => required_spare <= JSON_STAGE_IDS,
            }
    }

    fn finish(self) -> io::Result<()> {
        match self.target {
            PackedU32Target::File(writer) => writer.finish(),
            PackedU32Target::Memory(_) => Ok(()),
            PackedU32Target::Json(writer) => writer.finish(),
        }
    }
}

type PackedFileWriter = BufferedFileU32Writer;

/// Serial, bounded heap-page writer. Completed pages are copied to the kernel
/// in one `write_all`, then immediately reused.
struct BufferedFileU32Writer {
    file: File,
    page: Vec<u32>,
    failed: bool,
}

impl BufferedFileU32Writer {
    fn new(file: File) -> Self {
        Self {
            file,
            page: Vec::with_capacity(PACKED_U32_PAGE_IDS),
            failed: false,
        }
    }

    fn push(&mut self, id: u32) -> io::Result<()> {
        if self.page.len() == PACKED_U32_PAGE_IDS {
            self.flush_page()?;
        }
        self.page.push(id);
        Ok(())
    }

    fn push_many(&mut self, mut ids: &[u32]) -> io::Result<()> {
        while !ids.is_empty() {
            let take = ids
                .len()
                .min(PACKED_U32_PAGE_IDS.saturating_sub(self.page.len()));
            self.page.extend_from_slice(&ids[..take]);
            ids = &ids[take..];
            if self.page.len() == PACKED_U32_PAGE_IDS {
                self.flush_page()?;
            }
        }
        Ok(())
    }

    fn push_many_mapped(&mut self, ids: &[u32], rank_to_id: &[u32]) -> io::Result<()> {
        for &rank in ids {
            self.push(rank_to_id[rank as usize])?;
        }
        Ok(())
    }

    fn prepare_page(&mut self, required_spare: usize) -> io::Result<PackedU32Page<'_>> {
        assert!(required_spare <= PACKED_U32_PAGE_IDS);
        if PACKED_U32_PAGE_IDS - self.page.len() < required_spare {
            self.flush_page()?;
        }
        Ok(PackedU32Page {
            ptr: NonNull::new(self.page.as_mut_ptr()).expect("Vec allocation is non-null"),
            len: self.page.len(),
            capacity: self.page.capacity(),
            _borrow: PhantomData,
        })
    }

    fn commit_page(&mut self, new_len: usize) {
        // SAFETY: `with_u32_page` validates that the callback initialized only
        // reserved lanes before committing the new length.
        unsafe { self.page.set_len(new_len) };
    }

    fn flush_page(&mut self) -> io::Result<()> {
        if self.failed || self.page.is_empty() {
            return Ok(());
        }
        if let Err(error) = write_u32s_le(&mut self.file, &self.page) {
            // `write_all` may already have committed a prefix. Never retry the
            // same page from `finish` or `Drop`, which could duplicate tokens.
            self.failed = true;
            self.page.clear();
            return Err(error);
        }
        self.page.clear();
        Ok(())
    }

    fn finish(mut self) -> io::Result<()> {
        self.flush_page()?;
        self.file.flush()
    }
}

impl Drop for BufferedFileU32Writer {
    fn drop(&mut self) {
        let _ = self.flush_page();
    }
}

/// Streaming JSON integer array over two bounded, reusable pages.
///
/// Token IDs land first in a `u32` staging page, so the packed cache emitter can
/// append into it directly through [`TokenSink::with_u32_page`] exactly as the
/// compact target does. Each completed staging page is converted to canonical
/// decimal bytes in one bulk pass over a contiguous slice and appended to a byte
/// page, which is handed to the kernel whenever it can no longer accept a
/// maximal-width element. Neither page grows with the token count.
struct JsonU32Writer {
    file: File,
    stage: Vec<u32>,
    page: Vec<u8>,
    first: bool,
    failed: bool,
}

impl JsonU32Writer {
    fn new(file: File) -> Self {
        let mut page = Vec::with_capacity(JSON_PAGE_BYTES);
        // The opening bracket is buffered rather than written eagerly: the file
        // was already truncated by `File::create`, and a small output then costs
        // one `write` instead of two.
        page.push(b'[');
        Self {
            file,
            stage: Vec::with_capacity(JSON_STAGE_IDS),
            page,
            first: true,
            failed: false,
        }
    }

    #[inline]
    fn push(&mut self, id: u32) -> io::Result<()> {
        if self.stage.len() == JSON_STAGE_IDS {
            self.flush_stage()?;
        }
        self.stage.push(id);
        Ok(())
    }

    fn push_many(&mut self, mut ids: &[u32]) -> io::Result<()> {
        while !ids.is_empty() {
            let take = ids
                .len()
                .min(JSON_STAGE_IDS.saturating_sub(self.stage.len()));
            self.stage.extend_from_slice(&ids[..take]);
            ids = &ids[take..];
            if self.stage.len() == JSON_STAGE_IDS {
                self.flush_stage()?;
            }
        }
        Ok(())
    }

    fn push_many_mapped(&mut self, ids: &[u32], rank_to_id: &[u32]) -> io::Result<()> {
        for &rank in ids {
            self.push(rank_to_id[rank as usize])?;
        }
        Ok(())
    }

    fn prepare_page(&mut self, required_spare: usize) -> io::Result<PackedU32Page<'_>> {
        assert!(required_spare <= JSON_STAGE_IDS);
        if JSON_STAGE_IDS - self.stage.len() < required_spare {
            self.flush_stage()?;
        }
        Ok(PackedU32Page {
            ptr: NonNull::new(self.stage.as_mut_ptr()).expect("Vec allocation is non-null"),
            len: self.stage.len(),
            capacity: self.stage.capacity(),
            _borrow: PhantomData,
        })
    }

    fn commit_page(&mut self, new_len: usize) {
        // SAFETY: `with_u32_page` validates that the callback initialized only
        // reserved lanes before committing the new length.
        unsafe { self.stage.set_len(new_len) };
    }

    /// Convert one completed staging page to decimal bytes.
    fn flush_stage(&mut self) -> io::Result<()> {
        if self.failed {
            self.stage.clear();
            return Ok(());
        }
        if self.stage.is_empty() {
            return Ok(());
        }
        let Self {
            file,
            stage,
            page,
            first,
            failed,
        } = self;
        let result = append_decimal_ids(file, page, first, stage.as_slice());
        stage.clear();
        if let Err(error) = result {
            // `write_all` may already have committed a prefix. Never retry the
            // same page from `finish` or `Drop`, which could duplicate tokens.
            *failed = true;
            page.clear();
            return Err(error);
        }
        Ok(())
    }

    fn flush_page(&mut self) -> io::Result<()> {
        if self.failed || self.page.is_empty() {
            return Ok(());
        }
        if let Err(error) = self.file.write_all(&self.page) {
            self.failed = true;
            self.page.clear();
            return Err(error);
        }
        self.page.clear();
        Ok(())
    }

    fn finish(mut self) -> io::Result<()> {
        self.flush_stage()?;
        if !self.failed {
            // The conversion loop reserves before each element, so the page can
            // be exactly full here; make room before appending the trailer.
            if self.page.len() == self.page.capacity() {
                self.flush_page()?;
            }
            self.page.push(b']');
        }
        self.flush_page()?;
        self.file.flush()
    }
}

impl Drop for JsonU32Writer {
    fn drop(&mut self) {
        // An abandoned sink flushes what it buffered and does not synthesize a
        // closing bracket, matching the `BufWriter` behavior this replaced: only
        // `finish` completes the array. After `finish` both pages are empty, so
        // this is a no-op rather than a second trailer.
        let _ = self.flush_stage();
        let _ = self.flush_page();
    }
}

/// Two ASCII digits per index, so decimal conversion consumes two digits per
/// iteration without a division-by-ten chain per digit.
static DECIMAL_PAIRS: [u8; 200] = {
    let mut table = [0u8; 200];
    let mut value = 0usize;
    while value < 100 {
        table[value * 2] = b'0' + (value / 10) as u8;
        table[value * 2 + 1] = b'0' + (value % 10) as u8;
        value += 1;
    }
    table
};

#[inline(always)]
fn decimal_width(id: u32) -> usize {
    if id == 0 { 1 } else { id.ilog10() as usize + 1 }
}

/// Append `id`'s canonical decimal form with no intermediate buffer.
///
/// Deriving the width up front lets the digits be filled backwards straight into
/// the page. Formatting into a stack buffer and copying out, which is the shape
/// `itoa` requires, measured 6.42 ns/ID against 5.97 for this at 96.4 M IDs.
///
/// # Safety
///
/// The page must have spare capacity for this ID's digits, which is at most
/// [`JSON_MAX_ELEMENT_BYTES`] less the separator the caller already appended.
#[inline(always)]
unsafe fn push_decimal_u32(page: &mut Vec<u8>, id: u32) {
    let width = decimal_width(id);
    debug_assert!(page.capacity() - page.len() >= width);
    let start = page.len();
    // SAFETY: the caller guarantees the spare capacity, and every one of the
    // `width` bytes below is written before the length is committed.
    unsafe {
        let base = page.as_mut_ptr().add(start);
        let mut remaining = id;
        let mut offset = width;
        while remaining >= 100 {
            let pair = ((remaining % 100) * 2) as usize;
            remaining /= 100;
            offset -= 2;
            base.add(offset).write(*DECIMAL_PAIRS.get_unchecked(pair));
            base.add(offset + 1)
                .write(*DECIMAL_PAIRS.get_unchecked(pair + 1));
        }
        if remaining >= 10 {
            let pair = (remaining * 2) as usize;
            offset -= 2;
            base.add(offset).write(*DECIMAL_PAIRS.get_unchecked(pair));
            base.add(offset + 1)
                .write(*DECIMAL_PAIRS.get_unchecked(pair + 1));
        } else {
            offset -= 1;
            base.add(offset).write(b'0' + remaining as u8);
        }
        debug_assert_eq!(offset, 0);
        page.set_len(start + width);
    }
}

/// Convert a contiguous run of staged IDs into the byte page, writing completed
/// pages through to `file`. `first` tracks the array's leading element across
/// calls so the separator placement is independent of staging boundaries.
fn append_decimal_ids(
    file: &mut File,
    page: &mut Vec<u8>,
    first: &mut bool,
    ids: &[u32],
) -> io::Result<()> {
    for &id in ids {
        // Reserving a separator plus ten digits keeps every element whole within
        // one page, so no token can straddle a write.
        if page.len() + JSON_MAX_ELEMENT_BYTES > JSON_PAGE_BYTES {
            file.write_all(page)?;
            page.clear();
        }
        if *first {
            *first = false;
        } else {
            page.push(b',');
        }
        // SAFETY: the page has at least `JSON_MAX_ELEMENT_BYTES` spare capacity
        // from the check above, less the one separator byte just pushed.
        unsafe { push_decimal_u32(page, id) };
    }
    Ok(())
}

/// Contiguous packed output returned to Python for `output="array"`.
///
/// Unlike the bounded file targets this intentionally grows
/// with the token count. Direct cache emission writes into its spare capacity,
/// avoiding an intermediate token representation or per-token Python calls.
struct MemoryU32Writer {
    tokens: Vec<u32>,
    adaptive_reserve_done: bool,
}

impl MemoryU32Writer {
    fn new(capacity_hint: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity_hint),
            adaptive_reserve_done: false,
        }
    }

    /// Correct the coarse initial capacity estimate once, using token density
    /// observed over the beginning of the actual input. The 12.5% margin
    /// absorbs normal density drift without the large copies caused by `Vec`'s
    /// geometric growth on code and CJK corpora.
    fn adapt_capacity(
        &mut self,
        input_bytes_seen: usize,
        total_input_bytes: usize,
        output_tokens_seen: usize,
    ) {
        if self.adaptive_reserve_done
            || input_bytes_seen == 0
            || input_bytes_seen < total_input_bytes.min(ARRAY_DENSITY_SAMPLE_BYTES)
        {
            return;
        }
        self.adaptive_reserve_done = true;

        let projected = total_input_bytes
            .saturating_mul(output_tokens_seen)
            .saturating_add(input_bytes_seen - 1)
            / input_bytes_seen;
        let target = projected.saturating_add(projected / 8).saturating_add(1024);
        if target > self.tokens.capacity() {
            self.tokens
                .reserve_exact(target.saturating_sub(self.tokens.len()));
        }
        advise_huge_pages(self.tokens.as_mut_ptr(), self.tokens.capacity());
    }

    #[inline]
    fn push(&mut self, id: u32) {
        let grows = self.tokens.len() == self.tokens.capacity();
        self.tokens.push(id);
        self.readvise_after_growth(grows);
    }

    #[inline]
    fn push_many(&mut self, ids: &[u32]) {
        let grows = ids.len() > self.tokens.capacity() - self.tokens.len();
        self.tokens.extend_from_slice(ids);
        self.readvise_after_growth(grows);
    }

    #[inline]
    fn push_many_mapped(&mut self, ids: &[u32], rank_to_id: &[u32]) {
        let grows = ids.len() > self.tokens.capacity() - self.tokens.len();
        self.tokens
            .extend(ids.iter().map(|&rank| rank_to_id[rank as usize]));
        self.readvise_after_growth(grows);
    }

    #[inline]
    fn prepare_page(&mut self, required_spare: usize) -> PackedU32Page<'_> {
        let grows = required_spare > self.tokens.capacity() - self.tokens.len();
        self.tokens.reserve(required_spare);
        self.readvise_after_growth(grows);
        PackedU32Page {
            ptr: NonNull::new(self.tokens.as_mut_ptr()).expect("Vec allocation is non-null"),
            len: self.tokens.len(),
            // Expose only the space requested by the caller even when the Vec
            // allocator granted more capacity.
            capacity: self.tokens.len() + required_spare,
            _borrow: PhantomData,
        }
    }

    #[inline]
    fn commit_page(&mut self, new_len: usize) {
        // SAFETY: the direct-page callback initialized every newly committed
        // lane, and `prepare_page` reserved at least that much storage.
        unsafe { self.tokens.set_len(new_len) };
    }

    fn finish(self) -> Vec<u32> {
        self.tokens
    }

    #[inline]
    fn readvise_after_growth(&mut self, grew: bool) {
        if grew && self.adaptive_reserve_done {
            advise_huge_pages(self.tokens.as_mut_ptr(), self.tokens.capacity());
        }
    }
}

/// Hint that Linux may back the retained output with transparent huge pages.
/// This changes neither allocation nor ownership; unsupported kernels simply
/// ignore the best-effort advice.
#[cfg(target_os = "linux")]
fn advise_huge_pages(tokens: *mut u32, capacity: usize) {
    if capacity == 0 {
        return;
    }
    // SAFETY: `sysconf` has no pointer preconditions. `madvise` receives a
    // page-aligned range covering the Vec allocation and does not access it.
    unsafe {
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if page_size <= 0 {
            return;
        }
        let page_size = page_size as usize;
        let allocation_start = tokens as usize;
        let allocation_end =
            allocation_start.saturating_add(capacity.saturating_mul(size_of::<u32>()));
        // Round inward so the advice never names allocator metadata or an
        // adjacent allocation that happens to share an edge page.
        let advised_start = allocation_start.saturating_add(page_size - 1) / page_size * page_size;
        if advised_start >= allocation_end {
            return;
        }
        let _ = libc::madvise(
            advised_start as *mut libc::c_void,
            allocation_end - advised_start,
            libc::MADV_HUGEPAGE,
        );
    }
}

#[cfg(not(target_os = "linux"))]
#[inline]
fn advise_huge_pages(_tokens: *mut u32, _capacity: usize) {}

impl TokenSink {
    /// Build a sink that streams token IDs as one human-readable JSON array.
    pub fn new_json(dump_path: &str) -> io::Result<Self> {
        Ok(Self {
            total: 0,
            specials: 0,
            writer: Some(PackedU32Writer::new_json(File::create(dump_path)?)),
            error: None,
            first_token_timer: None,
        })
    }

    /// Build a sink that streams each token ID as one packed little-endian
    /// `u32`, with no header, delimiter, or trailer.
    pub fn new_u32_le(dump_path: &str) -> io::Result<Self> {
        let file = File::create(dump_path)?;
        Ok(Self {
            total: 0,
            specials: 0,
            writer: Some(PackedU32Writer::new_file(file)),
            error: None,
            first_token_timer: None,
        })
    }

    /// Build an in-memory packed-u32 sink. This mode intentionally retains the
    /// complete output and is used by the Python `output="array"` API.
    pub fn new_memory_u32(capacity_hint: usize) -> Self {
        Self {
            total: 0,
            specials: 0,
            writer: Some(PackedU32Writer::new_memory(capacity_hint)),
            error: None,
            first_token_timer: None,
        }
    }

    /// Finalize the retained array's capacity from early observed token
    /// density. Other output modes deliberately do nothing.
    pub(crate) fn adapt_memory_u32_capacity(
        &mut self,
        input_bytes_seen: usize,
        total_input_bytes: usize,
        output_tokens_seen: usize,
    ) {
        if let Some(PackedU32Writer {
            target: PackedU32Target::Memory(writer),
        }) = self.writer.as_mut()
        {
            writer.adapt_capacity(input_bytes_seen, total_input_bytes, output_tokens_seen);
        }
    }

    /// Begin opt-in per-chop timing. Ordinary sinks never call this, and their
    /// emission methods instantiate with `PROFILE = false`, so the compiler
    /// removes all first-token bookkeeping from the production hot path.
    pub(crate) fn start_profile(&mut self, started: Instant) {
        debug_assert_eq!(self.total, 0);
        self.first_token_timer = Some(FirstTokenTimer {
            started,
            elapsed: None,
        });
    }

    #[inline(always)]
    pub(crate) fn mark_first_token<const PROFILE: bool>(&mut self) {
        if PROFILE
            && let Some(timer) = self.first_token_timer.as_mut()
            && timer.elapsed.is_none()
        {
            timer.elapsed = Some(timer.started.elapsed());
        }
    }

    #[inline(always)]
    pub(crate) fn profile_start<const PROFILE: bool>(&self) -> Option<Instant> {
        if PROFILE {
            self.first_token_timer
                .as_ref()
                .and_then(|timer| timer.elapsed.is_none().then_some(timer.started))
        } else {
            None
        }
    }

    #[inline(always)]
    pub(crate) fn record_first_token_elapsed<const PROFILE: bool>(
        &mut self,
        elapsed: Option<Duration>,
    ) {
        if PROFILE
            && let Some(elapsed) = elapsed
            && let Some(timer) = self.first_token_timer.as_mut()
            && timer.elapsed.is_none()
        {
            timer.elapsed = Some(elapsed);
        }
    }

    pub(crate) fn time_to_first_token(&self) -> Option<Duration> {
        self.first_token_timer
            .as_ref()
            .and_then(|timer| timer.elapsed)
    }

    /// Whether output failed, so tokenization can stop at a safe boundary.
    #[inline(always)]
    pub(crate) fn is_cancelled(&self) -> bool {
        self.error.is_some()
    }

    /// Take a concrete output error recorded by a non-fallible emission hook.
    pub(crate) fn take_error(&mut self) -> Option<io::Error> {
        self.error.take()
    }

    #[inline]
    pub(crate) fn push(&mut self, id: u32) {
        self.total += 1;
        if self.error.is_some() {
            return;
        }
        let result = match &mut self.writer {
            Some(writer) => writer.push(id),
            None => Ok(()),
        };
        if let Err(error) = result {
            self.error = Some(error);
        }
    }

    #[inline(always)]
    pub(crate) fn push_profiled<const PROFILE: bool>(&mut self, id: u32) {
        self.mark_first_token::<PROFILE>();
        self.push(id);
    }

    /// Push a complete pretoken encoding, materializing every ID.
    #[inline]
    pub(crate) fn push_many(&mut self, ids: &[u32], rank_to_id: Option<&[u32]>) {
        self.total += ids.len();
        if self.error.is_some() {
            return;
        }
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        let result = match rank_to_id {
            Some(map) => writer.push_many_mapped(ids, map),
            None => writer.push_many(ids),
        };
        if let Err(error) = result {
            self.error = Some(error);
        }
    }

    #[inline(always)]
    pub(crate) fn push_many_profiled<const PROFILE: bool>(
        &mut self,
        ids: &[u32],
        rank_to_id: Option<&[u32]>,
    ) {
        if !ids.is_empty() {
            self.mark_first_token::<PROFILE>();
        }
        self.push_many(ids, rank_to_id);
    }

    /// Append directly into the reusable packed-u32 output page.
    ///
    /// This is the cache fast path: `append` may add at most
    /// `required_spare` IDs to `page`. The page is flushed before the callback
    /// when necessary, so that much spare capacity is guaranteed. The sink
    /// derives its token count from the length change and preserves ordering
    /// with [`push`](Self::push), [`push_many`](Self::push_many), and
    /// [`push_special`](Self::push_special).
    ///
    /// The file and JSON targets use fixed bounded pages; the memory
    /// target exposes spare capacity in its contiguous vector. Callers cannot
    /// allocate or move the storage. Returns `None` when the target cannot
    /// accept a reservation this large, leaving the caller's ordered path
    /// responsible for the batch.
    #[inline]
    pub(crate) fn with_u32_page<R>(
        &mut self,
        required_spare: usize,
        append: impl FnOnce(&mut PackedU32Page<'_>) -> R,
    ) -> Option<R> {
        if self.error.is_some() {
            return None;
        }
        let writer = match self.writer.as_mut() {
            Some(writer) if writer.supports_direct_page(required_spare) => writer,
            _ => return None,
        };
        let append_result = (|| {
            let (result, added, new_len) = {
                let mut page = writer.prepare_page(required_spare)?;
                let old_len = page.len();
                let result = append(&mut page);
                let added = page
                    .len()
                    .checked_sub(old_len)
                    .expect("direct output callback may only append");
                assert!(
                    added <= required_spare,
                    "direct output callback exceeded its requested page space"
                );
                (result, added, page.len())
            };
            writer.commit_page(new_len);
            Ok::<_, io::Error>((result, added))
        })();
        match append_result {
            Ok((result, added)) => {
                self.total += added;
                Some(result)
            }
            Err(error) => {
                self.error = Some(error);
                None
            }
        }
    }

    /// Emit a special token id and count it.
    #[cfg(test)]
    #[inline]
    pub(crate) fn push_special(&mut self, id: u32) {
        self.push(id);
        self.specials += 1;
    }

    #[inline(always)]
    pub(crate) fn push_special_profiled<const PROFILE: bool>(&mut self, id: u32) {
        self.push_profiled::<PROFILE>(id);
        self.specials += 1;
    }

    /// Total token ids pushed (across every `tokenize_string` call driven into
    /// this sink).
    #[inline]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Of [`total`](Self::total), how many were special tokens.
    #[inline]
    pub fn specials(&self) -> usize {
        self.specials
    }

    /// Finish the output and flush. JSON receives its closing `]`; packed-u32
    /// output has no trailer. Call once after the last tokenization operation.
    pub fn finish(mut self) -> io::Result<()> {
        let finish_result = match self.writer.take() {
            Some(writer) => writer.finish(),
            None => Ok(()),
        };
        if let Some(error) = self.error.take() {
            let _ = finish_result;
            Err(error)
        } else {
            finish_result
        }
    }

    /// Finish an in-memory packed-u32 sink and return its complete token
    /// buffer.
    pub fn into_u32_vec(mut self) -> io::Result<Vec<u32>> {
        let output = match self.writer.take() {
            Some(PackedU32Writer {
                target: PackedU32Target::Memory(writer),
            }) => Ok(writer.finish()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "token sink is not an in-memory u32 sink",
            )),
        };
        if let Some(error) = self.error.take() {
            let _ = output;
            Err(error)
        } else {
            output
        }
    }
}

#[inline]
fn write_u32s_le(writer: &mut File, ids: &[u32]) -> io::Result<()> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: every initialized `u32` occupies exactly four bytes, and a
        // byte slice may view storage at any alignment. The host representation
        // is the requested little-endian representation under this cfg.
        let bytes = unsafe {
            std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), std::mem::size_of_val(ids))
        };
        writer.write_all(bytes)
    }
    #[cfg(target_endian = "big")]
    {
        for id in ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };

    use super::{
        ARRAY_DENSITY_SAMPLE_BYTES, JSON_PAGE_BYTES, JSON_STAGE_IDS, MemoryU32Writer,
        PACKED_U32_PAGE_IDS, TokenSink, read_chunks_while,
    };

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("mtc-{label}-{}-{nonce}.bin", std::process::id()))
    }

    #[test]
    fn read_chunks_rejects_invalid_utf8() {
        let path = temp_path("invalid-utf8");
        std::fs::write(&path, b"abc\xffdef").unwrap();
        let error = read_chunks_while(path.to_str().unwrap(), |_| true).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("byte 3"));
    }

    #[test]
    fn read_chunks_rejects_truncated_utf8_at_eof() {
        let path = temp_path("truncated-utf8");
        std::fs::write(&path, [b'a', 0xe2, 0x82]).unwrap();
        let error = read_chunks_while(path.to_str().unwrap(), |_| true).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("incomplete UTF-8"));
    }

    #[test]
    fn packed_u32_sink_is_headerless_little_endian() -> std::io::Result<()> {
        let path = temp_path("u32-sink-test");

        let mut sink = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path"))?;
        sink.push(0x0102_0304);
        sink.push_many(&[5, u32::MAX], None);
        sink.push_many(&[0, 2], Some(&[9, 8, 7]));
        assert_eq!(sink.total(), 5);
        sink.finish()?;

        let got = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let expected: Vec<_> = [0x0102_0304_u32, 5, u32::MAX, 9, 7]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(got, expected);
        Ok(())
    }

    #[test]
    fn json_sink_streams_valid_decimal_ids_and_empty_arrays() -> std::io::Result<()> {
        let path = temp_path("json-sink-test");

        let mut sink = TokenSink::new_json(path.to_str().expect("UTF-8 temp path"))?;
        sink.push(0x0102_0304);
        sink.push_many(&[5, u32::MAX], None);
        sink.push_many(&[0, 2], Some(&[9, 8, 7]));
        assert_eq!(sink.total(), 5);
        sink.finish()?;
        assert_eq!(
            std::fs::read_to_string(&path)?,
            "[16909060,5,4294967295,9,7]"
        );

        let empty = TokenSink::new_json(path.to_str().expect("UTF-8 temp path"))?;
        empty.finish()?;
        assert_eq!(std::fs::read_to_string(&path)?, "[]");
        std::fs::remove_file(path)?;
        Ok(())
    }

    /// Deterministic mixed-width IDs plus every decimal-width boundary.
    fn json_probe_ids(count: usize) -> Vec<u32> {
        let mut ids = vec![
            0,
            1,
            9,
            10,
            99,
            100,
            999,
            1_000,
            9_999,
            10_000,
            99_999,
            100_000,
            999_999,
            1_000_000,
            999_999_999,
            1_000_000_000,
            u32::MAX - 1,
            u32::MAX,
        ];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        while ids.len() < count {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ids.push((state % 50_257) as u32);
        }
        ids
    }

    fn expected_json(ids: &[u32]) -> String {
        let body: Vec<String> = ids.iter().map(u32::to_string).collect();
        format!("[{}]", body.join(","))
    }

    #[test]
    fn json_sink_preserves_bytes_across_staging_and_page_boundaries() -> std::io::Result<()> {
        let path = temp_path("json-page-boundary");
        // Enough IDs to rotate the byte page many times and the staging page
        // more than twice, with slice lengths coprime to both.
        let ids = json_probe_ids(JSON_STAGE_IDS * 2 + 4_099);

        let mut sink = TokenSink::new_json(path.to_str().expect("UTF-8 temp path"))?;
        for chunk in ids.chunks(257) {
            sink.push_many(chunk, None);
        }
        assert_eq!(sink.total(), ids.len());
        sink.finish()?;

        let got = std::fs::read_to_string(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(got, expected_json(&ids));
        Ok(())
    }

    #[test]
    fn json_sink_mixes_scalar_bulk_mapped_and_direct_page_emission() -> std::io::Result<()> {
        let path = temp_path("json-mixed-emission");
        let map = [7_u32, 4_294_967_295, 0, 50_256];
        let mut expected: Vec<u32> = Vec::new();

        let mut sink = TokenSink::new_json(path.to_str().expect("UTF-8 temp path"))?;
        // A leading scalar push proves the separator state is owned by the
        // writer rather than by any one emission method.
        sink.push(0);
        expected.push(0);

        let direct = [1_u32, 999, 1_000, u32::MAX];
        assert_eq!(
            sink.with_u32_page(direct.len(), |page| {
                page.extend_from_slice(&direct);
                "appended"
            }),
            Some("appended"),
            "JSON must accept the direct cache page"
        );
        expected.extend_from_slice(&direct);

        sink.push_many(&[10, 99], None);
        expected.extend_from_slice(&[10, 99]);

        sink.push_many(&[0, 1, 3], Some(&map));
        expected.extend([map[0], map[1], map[3]]);

        assert_eq!(sink.total(), expected.len());
        sink.finish()?;

        let got = std::fs::read_to_string(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(got, expected_json(&expected));
        Ok(())
    }

    #[test]
    fn json_sink_rejects_reservations_larger_than_its_staging_page() -> std::io::Result<()> {
        let path = temp_path("json-oversized-reservation");
        let mut sink = TokenSink::new_json(path.to_str().expect("UTF-8 temp path"))?;

        // Oversized batches must fall back to the ordered path, not truncate.
        assert_eq!(
            sink.with_u32_page(JSON_STAGE_IDS + 1, |page| {
                page.push(1);
            }),
            None
        );
        assert_eq!(sink.total(), 0);
        assert!(
            sink.with_u32_page(JSON_STAGE_IDS, |_| ()).is_some(),
            "a reservation of exactly the staging capacity must be accepted"
        );

        sink.push(42);
        sink.finish()?;
        let got = std::fs::read_to_string(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(got, "[42]");
        Ok(())
    }

    #[test]
    fn json_sink_truncates_a_reused_output_file() -> std::io::Result<()> {
        let path = temp_path("json-reused-output");
        let ids = json_probe_ids(JSON_STAGE_IDS + 31);

        let mut first = TokenSink::new_json(path.to_str().expect("UTF-8 temp path"))?;
        first.push_many(&ids, None);
        first.finish()?;
        assert!(std::fs::metadata(&path)?.len() > JSON_PAGE_BYTES as u64);

        let mut second = TokenSink::new_json(path.to_str().expect("UTF-8 temp path"))?;
        second.push(42);
        second.finish()?;

        let got = std::fs::read_to_string(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(got, "[42]");
        Ok(())
    }

    /// `/dev/full` opens for writing and fails every write with `ENOSPC`, which
    /// exercises the real `File` failure path without a generic writer.
    #[cfg(target_os = "linux")]
    #[test]
    fn json_sink_surfaces_a_failing_write_once() {
        let mut sink = TokenSink::new_json("/dev/full").expect("/dev/full opens for writing");
        // Enough IDs to force at least one page flush inside the conversion.
        sink.push_many(&json_probe_ids(JSON_STAGE_IDS * 2), None);
        assert!(
            sink.is_cancelled(),
            "a failed page write must cancel the sink"
        );
        let error = sink.take_error().expect("stored output error");
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        // A cancelled sink must not resurrect the failed page from `finish`.
        assert!(sink.finish().is_ok());
    }

    #[test]
    fn packed_u32_sink_handles_page_boundaries_and_frequent_slices() -> std::io::Result<()> {
        let path = temp_path("u32-page-boundary-test");
        let ids: Vec<u32> = (0..PACKED_U32_PAGE_IDS as u32 + 1_019).collect();

        let mut sink = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path"))?;
        for chunk in ids.chunks(257) {
            sink.push_many(chunk, None);
        }
        assert_eq!(sink.total(), ids.len());
        sink.finish()?;

        let got = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(got.len(), ids.len() * size_of::<u32>());
        for (bytes, expected) in got.chunks_exact(4).zip(ids) {
            assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), expected);
        }
        Ok(())
    }

    #[test]
    fn packed_u32_sink_truncates_a_reused_output_file() -> std::io::Result<()> {
        let path = temp_path("u32-reused-output-test");
        let ids = vec![17_u32; PACKED_U32_PAGE_IDS + 31];

        let mut first = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path"))?;
        first.push_many(&ids, None);
        first.finish()?;

        let mut second = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path"))?;
        second.push(42);
        second.finish()?;

        let got = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(got, 42_u32.to_le_bytes());
        Ok(())
    }

    #[test]
    fn packed_u32_sink_remaps_ranks_across_page_boundary() -> std::io::Result<()> {
        let path = temp_path("u32-remapped-page-test");
        let map = [101_u32, 7, u32::MAX, 42];
        let ranks: Vec<u32> = (0..PACKED_U32_PAGE_IDS + 31)
            .map(|index| (index % map.len()) as u32)
            .collect();

        let mut sink = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path"))?;
        for chunk in ranks.chunks(509) {
            sink.push_many(chunk, Some(&map));
        }
        sink.finish()?;

        let got = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let expected = ranks.into_iter().map(|rank| map[rank as usize]);
        for (bytes, expected) in got.chunks_exact(4).zip(expected) {
            assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), expected);
        }
        assert_eq!(got.len(), (PACKED_U32_PAGE_IDS + 31) * size_of::<u32>());
        Ok(())
    }

    #[test]
    fn direct_page_appends_preserve_order_and_count() -> std::io::Result<()> {
        let path = temp_path("u32-direct-page-test");
        let mut sink = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path"))?;
        sink.push(3);
        let result = sink.with_u32_page(3, |page| {
            page.extend_from_slice(&[5, 8, 13]);
            "appended"
        });
        assert_eq!(result, Some("appended"));
        sink.push_special(21);
        assert_eq!(sink.total(), 5);
        assert_eq!(sink.specials(), 1);
        sink.finish()?;

        let got = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let ids: Vec<_> = got
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(ids, [3, 5, 8, 13, 21]);

        let mut memory_sink = TokenSink::new_memory_u32(1);
        assert_eq!(
            memory_sink.with_u32_page(1, |page| {
                page.push(34);
                "appended"
            }),
            Some("appended")
        );
        assert_eq!(memory_sink.total(), 1);
        assert_eq!(memory_sink.into_u32_vec().unwrap(), [34]);
        Ok(())
    }

    #[test]
    fn direct_file_page_reservation_rotates_without_losing_ids() -> std::io::Result<()> {
        let path = temp_path("u32-direct-page-rotation-test");
        let prefix: Vec<u32> = (0..PACKED_U32_PAGE_IDS as u32 - 2).collect();
        let direct = [0x0102_0304, 7, u32::MAX, PACKED_U32_PAGE_IDS as u32];
        let ranks = [2_u32, 0, 1];
        let rank_to_id = [91_u32, 83, 79];

        let mut sink = TokenSink::new_u32_le(path.to_str().expect("UTF-8 temp path"))?;
        sink.push_many(&prefix, None);

        // Only two lanes remain in the first 4 MiB page, so reserving four
        // forces the file-backed direct page to flush before the callback.
        assert_eq!(
            sink.with_u32_page(4, |page| {
                page.extend_from_slice(&direct);
                "rotated"
            }),
            Some("rotated")
        );
        sink.push_many(&ranks, Some(&rank_to_id));

        let expected_count = prefix.len() + direct.len() + ranks.len();
        assert_eq!(sink.total(), expected_count);
        sink.finish()?;

        let got = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(got.len(), expected_count * size_of::<u32>());

        let mut expected = prefix;
        expected.extend_from_slice(&direct);
        expected.extend(ranks.map(|rank| rank_to_id[rank as usize]));
        for (bytes, expected) in got.chunks_exact(4).zip(expected) {
            assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), expected);
        }
        Ok(())
    }

    #[test]
    fn memory_sink_returns_exact_direct_and_scalar_ids() {
        let mut sink = TokenSink::new_memory_u32(6);
        sink.push(7);
        sink.push_many(&[11, 13], None);
        sink.with_u32_page(3, |page| {
            page.extend_from_slice(&[17, 19, 23]);
        })
        .unwrap();
        assert_eq!(sink.total(), 6);
        assert_eq!(sink.into_u32_vec().unwrap(), [7, 11, 13, 17, 19, 23]);
    }

    #[test]
    fn memory_writer_adapts_capacity_once_without_changing_output() {
        let expected: Vec<u32> = (0..10).collect();
        let mut writer = MemoryU32Writer::new(4);
        writer.push_many(&expected);
        writer.adapt_capacity(
            ARRAY_DENSITY_SAMPLE_BYTES,
            ARRAY_DENSITY_SAMPLE_BYTES * 4,
            expected.len(),
        );
        let adapted_capacity = writer.tokens.capacity();
        assert!(adapted_capacity >= 46);

        // Later density observations cannot move the array a second time.
        writer.adapt_capacity(
            ARRAY_DENSITY_SAMPLE_BYTES * 2,
            ARRAY_DENSITY_SAMPLE_BYTES * 8,
            1000,
        );
        assert_eq!(writer.tokens.capacity(), adapted_capacity);
        assert_eq!(writer.finish(), expected);
    }

    #[test]
    fn opt_in_profile_records_only_the_first_emitted_token() {
        let mut sink = TokenSink::new_memory_u32(3);
        sink.start_profile(Instant::now());
        assert_eq!(sink.time_to_first_token(), None);

        sink.push_profiled::<true>(7);
        let first = sink
            .time_to_first_token()
            .expect("the first emitted token should set TTFT");
        sink.push_many_profiled::<true>(&[11, 13], None);

        assert_eq!(sink.time_to_first_token(), Some(first));
        assert_eq!(sink.into_u32_vec().unwrap(), [7, 11, 13]);
    }

    #[test]
    fn profiled_empty_output_has_no_first_token() {
        let mut sink = TokenSink::new_memory_u32(0);
        sink.start_profile(Instant::now());
        sink.push_many_profiled::<true>(&[], None);
        assert_eq!(sink.time_to_first_token(), None);
    }

    #[test]
    fn concrete_output_error_marks_sink_cancelled() {
        let mut failed = TokenSink::new_memory_u32(0);
        failed.error = Some(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "output denied",
        ));
        assert!(failed.is_cancelled());
        let error = failed.take_error().expect("stored output error");
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    }
}
