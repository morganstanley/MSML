//! Reference byte-stepped DFA streaming pretokenizer.
//!
//! This module deliberately contains no SIMD or encoding-specific fast-path
//! composition. It is the compatibility implementation selected by `--dfa`.

use regex_automata::{
    Anchored, Input,
    hybrid::{
        LazyStateID,
        dfa::{Cache, DFA},
    },
    util::syntax,
};

use super::Encoding;

/// Byte length of the first `char` of `s` (0 if empty).
#[inline]
fn first_char_len(s: &str) -> usize {
    s.chars().next().map_or(0, |c| c.len_utf8())
}

/// Generic streaming lazy-DFA pretokenizer.
///
/// Completed pretokens are emitted immediately. Only the final run whose
/// boundary may still move is retained between calls to [`Self::feed`].
pub(crate) struct DfaStream {
    dfa: DFA,
    cache: Cache,
    enc: Encoding,
    buf: String,
    start: usize,
    state: Option<LazyStateID>,
    scanned: usize,
    last_accept: Option<usize>,
}

impl DfaStream {
    pub(crate) fn new(enc: Encoding) -> Result<Self, Box<dyn std::error::Error>> {
        let dfa = DFA::builder()
            .syntax(
                syntax::Config::new()
                    .unicode(true)
                    .utf8(true)
                    .case_insensitive(false),
            )
            .build(enc.dfa_pattern())?;
        let cache = dfa.create_cache();
        Ok(Self {
            dfa,
            cache,
            enc,
            buf: String::new(),
            start: 0,
            state: None,
            scanned: 0,
            last_accept: None,
        })
    }

    /// Feed the next chunk and emit every pretoken whose boundary is settled.
    pub(crate) fn feed<F: FnMut(&str)>(&mut self, chunk: &str, mut emit: F) {
        if self.start > 0 {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        self.buf.push_str(chunk);
        self.pump(false, &mut emit);
    }

    /// Signal end-of-stream and emit the final unsettled pretoken.
    pub(crate) fn finish<F: FnMut(&str)>(&mut self, mut emit: F) {
        self.pump(true, &mut emit);
        debug_assert!(
            self.start >= self.buf.len(),
            "finish must fully consume the buffer"
        );
        self.buf.clear();
        self.start = 0;
        self.reset_run();
    }

    /// `is_final` distinguishes end-of-chunk from end-of-stream.
    fn pump<F: FnMut(&str)>(&mut self, is_final: bool, emit: &mut F) {
        loop {
            let start = self.start;

            if self.state.is_none() {
                if start >= self.buf.len() {
                    return;
                }
                let input = Input::new(&self.buf.as_bytes()[start..]).anchored(Anchored::Yes);
                match self.dfa.start_state_forward(&mut self.cache, &input) {
                    Ok(state) => self.state = Some(state),
                    Err(_) => {
                        self.start += first_char_len(&self.buf[start..]).max(1);
                        continue;
                    }
                }
                self.scanned = 0;
                self.last_accept = None;
            }

            let mut state = self.state.expect("DFA state initialized above");
            let mut dead = false;
            while start + self.scanned < self.buf.len() {
                let byte = self.buf.as_bytes()[start + self.scanned];
                match self.dfa.next_state(&mut self.cache, state, byte) {
                    Ok(next) => state = next,
                    Err(_) => {
                        dead = true;
                        break;
                    }
                }
                self.scanned += 1;
                if state.is_match() {
                    // regex-automata reports this delayed by one byte.
                    self.last_accept = Some(self.scanned - 1);
                }
                if state.is_dead() || state.is_quit() {
                    dead = true;
                    break;
                }
            }
            self.state = Some(state);

            if !dead {
                if !is_final {
                    return;
                }
                if let Ok(eoi) = self.dfa.next_eoi_state(&mut self.cache, state)
                    && eoi.is_match()
                {
                    self.last_accept = Some(self.scanned);
                }
            }

            match self.last_accept {
                Some(end) => {
                    let actual =
                        self.enc
                            .apply_fixup(&self.buf[start..], 0, end, self.buf.len() - start);
                    emit(&self.buf[start..start + actual]);
                    self.start += actual;
                    self.reset_run();
                }
                None => {
                    // All supported patterns are total. Keep this guard so a
                    // future non-total pattern cannot make the stream spin.
                    if start >= self.buf.len() {
                        return;
                    }
                    self.start += first_char_len(&self.buf[start..]).max(1);
                    self.reset_run();
                }
            }
        }
    }

    #[inline]
    fn reset_run(&mut self) {
        self.state = None;
        self.scanned = 0;
        self.last_accept = None;
    }
}
