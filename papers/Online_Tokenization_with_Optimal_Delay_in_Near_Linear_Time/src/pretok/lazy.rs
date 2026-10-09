use regex_automata::{
    Anchored, Input,
    hybrid::dfa::{Cache, DFA},
    util::syntax,
};

use super::Encoding;
/// For genuine chunked streaming see [`StreamPretokenizer`](super::StreamPretokenizer).
pub struct LazyPretokenizer {
    dfa: DFA,
    cache: Cache,
    enc: Encoding,
}

impl LazyPretokenizer {
    pub fn new(enc: Encoding) -> Result<Self, Box<dyn std::error::Error>> {
        let dfa = DFA::builder()
            .syntax(
                syntax::Config::new()
                    .unicode(true)
                    .utf8(true)
                    .case_insensitive(false),
            )
            .build(enc.dfa_pattern())?;
        let cache = dfa.create_cache();
        Ok(Self { dfa, cache, enc })
    }

    pub fn split<'a, F>(&mut self, input: &'a str, mut emit: F)
    where
        F: FnMut(&'a str),
    {
        let bytes = input.as_bytes();
        let n = bytes.len();
        let mut pos = 0;
        while pos < n {
            let start_input = Input::new(bytes).range(pos..n).anchored(Anchored::Yes);
            let start_state = match self.dfa.start_state_forward(&mut self.cache, &start_input) {
                Ok(s) => s,
                Err(_) => {
                    pos += 1;
                    continue;
                }
            };
            let mut state = start_state;
            let mut last_accept: Option<usize> = None;
            let mut j = pos;
            while j < n {
                state = match self.dfa.next_state(&mut self.cache, state, bytes[j]) {
                    Ok(s) => s,
                    Err(_) => break,
                };
                j += 1;
                if state.is_match() {
                    last_accept = Some(j - 1);
                }
                if state.is_dead() || state.is_quit() {
                    break;
                }
            }
            if j == n && !state.is_dead() && !state.is_quit() {
                if let Ok(eoi) = self.dfa.next_eoi_state(&mut self.cache, state) {
                    if eoi.is_match() {
                        last_accept = Some(n);
                    }
                }
            }
            match last_accept {
                Some(end) => {
                    let actual_end = self.enc.apply_fixup(input, pos, end, n);
                    emit(&input[pos..actual_end]);
                    pos = actual_end;
                }
                None => {
                    pos += 1;
                }
            }
        }
    }
}
