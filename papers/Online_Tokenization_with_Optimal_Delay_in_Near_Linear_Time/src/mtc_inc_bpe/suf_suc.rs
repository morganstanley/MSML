use std::collections::VecDeque;

use derive_more::Deref;

use crate::{
    SkipLen,
    aho_corasick::{AC_NODE_ROOT, ACAutomaton, ACNodeId},
    normalize::ATOMIC_TOKEN_PRIORITY,
    successor::{FOREST_VIRTUAL_ROOT, ForestNodeId, SucForest},
    typed_vec::TypedVec,
};

#[derive(Clone, Debug)]
pub(crate) struct SufSucNode<I = ForestNodeId> {
    pub repr_id: I,
    pub skip_len: SkipLen,
    pub suc_skip_len: SkipLen,
    pub valid_range: (ForestNodeId, ForestNodeId),
}

const _: () = {
    assert!(std::mem::size_of::<SufSucNode>() == 16);
    assert!(std::mem::size_of::<SufSucNode<u32>>() == 16);
};

impl<I> SufSucNode<I> {
    #[inline(always)]
    pub fn verify<F: FnOnce(usize) -> ForestNodeId>(&self, f: F) -> bool {
        self.verify_skipped(f(self.suc_skip_len as usize))
    }

    #[inline(always)]
    pub fn verify_skipped(&self, w: ForestNodeId) -> bool {
        let (u, v) = self.valid_range;
        u <= w && w < v
    }
}

#[derive(Debug, Deref)]
pub(crate) struct SufSucNodeSet {
    #[deref]
    nodes: TypedVec<ForestNodeId, SufSucNode>,
    pub suffix_parent: TypedVec<ForestNodeId, ForestNodeId>,
    pub longest_token_node: TypedVec<ACNodeId, ForestNodeId>,
}

impl SufSucNodeSet {
    pub fn new(forest: &SucForest, automaton: &ACAutomaton) -> Self {
        let automaton_size = automaton.num_of_nodes();
        let forest_size = forest.len();

        let mut longest_token_node = TypedVec::new_with(FOREST_VIRTUAL_ROOT, automaton_size);
        for (token_id, ac_node_id) in automaton.token_to_node.enumerate_copied() {
            let forest_node_id = forest.token_to_node_id[token_id];
            longest_token_node[ac_node_id] = forest_node_id;
        }

        let mut suffix_parent = TypedVec::new_with(FOREST_VIRTUAL_ROOT, forest_size);

        let mut queue = VecDeque::with_capacity(automaton_size.as_usize());
        queue.push_back(AC_NODE_ROOT);
        while let Some(node) = queue.pop_front() {
            let cur_longest = longest_token_node[node];
            for child in automaton.suffix.children(node) {
                if longest_token_node[child] == FOREST_VIRTUAL_ROOT {
                    longest_token_node[child] = cur_longest;
                } else {
                    suffix_parent[longest_token_node[child]] = cur_longest;
                }
                queue.push_back(child);
            }
        }

        let calc_valid_pre_node_id_range = |node_id: ForestNodeId| {
            let node = &forest[node_id];
            if node.skip_len <= 1 {
                debug_assert!(
                    node_id == FOREST_VIRTUAL_ROOT && node.skip_len == 0
                        || node.parent == FOREST_VIRTUAL_ROOT && node.skip_len == 1
                );
                (ForestNodeId::ZERO, ForestNodeId::MAX)
            } else {
                debug_assert!(
                    node.pre_id != FOREST_VIRTUAL_ROOT && node.priority < ATOMIC_TOKEN_PRIORITY
                );
                let pre = &forest[node.pre_id];
                let last = if pre
                    .children
                    .first()
                    .is_none_or(|&c| node.priority >= forest[c].priority)
                {
                    node.pre_id.next()
                } else if pre
                    .children
                    .last()
                    .is_some_and(|&c| node.priority < forest[c].priority)
                {
                    pre.subtree_last_node.next()
                } else {
                    match pre
                        .children
                        .binary_search_by_key(&!node.priority, |&i| !forest[i].priority)
                    {
                        Ok(idx) => pre.children[idx],
                        Err(idx) => pre.children[idx],
                    }
                };
                #[cfg(debug_assertions)]
                {
                    for &c in &pre.children {
                        debug_assert!((node.priority < forest[c].priority) ^ (c >= last));
                        debug_assert!(c > node.pre_id);
                    }
                }
                (node.pre_id, last)
            }
        };

        let nodes: TypedVec<ForestNodeId, _> = forest
            .enumerate()
            .map(|(i, node)| SufSucNode {
                repr_id: i,
                skip_len: node.skip_len,
                suc_skip_len: forest[node.parent].skip_len,
                valid_range: calc_valid_pre_node_id_range(i),
            })
            .collect();

        #[cfg(debug_assertions)]
        {
            for (i, node) in nodes.enumerate() {
                let parent = forest[i].parent;
                if parent == i {
                    debug_assert_eq!(i, FOREST_VIRTUAL_ROOT);
                }
                debug_assert_eq!(node.repr_id, i);
                debug_assert_eq!(node.skip_len, forest[i].skip_len);
                debug_assert_eq!(node.suc_skip_len, forest[parent].skip_len);
            }

            for node_id in nodes.keys() {
                if node_id == FOREST_VIRTUAL_ROOT {
                    continue;
                }
                let node = &forest[node_id];
                let mut ranges = Vec::with_capacity(node.children.len());
                for &child_id in &node.children {
                    let child = &nodes[child_id];
                    ranges.push(child.valid_range);
                }
                ranges.sort();
                for slice in ranges.windows(2) {
                    let (l, r) = slice[0];
                    let (u, v) = slice[1];
                    debug_assert!(l.max(u) >= r.min(v));
                }
            }
        }

        Self {
            nodes,
            suffix_parent,
            longest_token_node,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, NormalizedDict, Vocab,
        aho_corasick::{AC_NODE_ROOT, ACAutomaton},
        successor::{FOREST_VIRTUAL_ROOT, SucForest},
        suf_suc::SufSucNodeSet,
    };

    fn node_set_case(vocab: &[&str], rules: &[(&str, &str)]) {
        let vocab = Vocab::new(vocab.iter().map(|s| s.as_bytes().to_vec())).unwrap();

        let dict = Dictionary::new_from_token_pair(vocab, rules.iter().copied()).unwrap();
        let dict = NormalizedDict::new_in_bytes(dict.clone()).unwrap();
        let automaton = ACAutomaton::new(dict.iter_canonical_or_empty_tokens());
        let forest = SucForest::new(&dict);

        let node_set = SufSucNodeSet::new(&forest, &automaton);

        for (node_id, node) in forest.enumerate() {
            let s = if node_id.inner() == 0 {
                "(epsilon)"
            } else {
                std::str::from_utf8(&dict[node.token_id]).unwrap()
            };
            let suf_suc_node = &node_set[node_id];
            println!("{s:12} {node_id:2}: {node:?} {suf_suc_node:?}");
        }

        let mut stack = vec![(AC_NODE_ROOT, automaton.trie.children(AC_NODE_ROOT))];
        let mut cur_string = Vec::with_capacity(dict.tokens.iter().map(|t| t.len()).max().unwrap());
        println!("{:?}", automaton.token_to_node);
        while let Some((ac_node_id, child_iter)) = stack.last_mut() {
            let ac_node_id = *ac_node_id;
            let Some((child, byte)) = child_iter.next() else {
                stack.pop();
                if ac_node_id != AC_NODE_ROOT {
                    cur_string.pop();
                }
                continue;
            };
            stack.push((child, automaton.trie.children(child)));
            cur_string.push(byte);
            let longest = dict
                .tokens
                .keys()
                .filter(|&i| dict.is_canonical(i) && cur_string.ends_with(&dict[i]))
                .max_by_key(|&i| dict[i].len())
                .map(|i| forest.token_to_node_id[i])
                .unwrap_or(FOREST_VIRTUAL_ROOT);
            println!("{child}: {:?} {longest}", str::from_utf8(&cur_string));
            assert_eq!(node_set.longest_token_node[child], longest);
        }

        for (token_id, ac_node_id) in automaton.token_to_node.enumerate_copied() {
            let node_id = forest.token_to_node_id[token_id];
            if node_id == FOREST_VIRTUAL_ROOT {
                continue;
            }
            assert_eq!(node_set.longest_token_node[ac_node_id], node_id);
            let token = dict.get_token(token_id).unwrap();
            let suf_parent_id = dict
                .tokens
                .keys()
                .filter(|&i| {
                    dict.is_canonical(i) && token.ends_with(&dict[i]) && dict[i].len() < token.len()
                })
                .max_by_key(|&i| dict[i].len())
                .map(|i| forest.token_to_node_id[i])
                .unwrap_or(FOREST_VIRTUAL_ROOT);
            println!("{node_id}: {}", node_set.suffix_parent[node_id]);
            assert_eq!(node_set.suffix_parent[node_id], suf_parent_id);
        }
    }

    #[test]
    fn test_node_set() {
        let vocab = [
            "", "a", "abc", "abcde", "abcdef", "b", "ba", "bc", "bcdef", "c", "cd", "cde", "cdefg",
            "d", "de", "def", "e", "ef", "efg", "f", "g",
        ];
        node_set_case(
            &vocab,
            &[
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("b", "a"),
                ("a", "bc"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
        );
        node_set_case(
            &vocab,
            &[
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("a", "bc"),
                ("b", "a"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
        );
    }

    #[test]
    fn test_repeated_suf_suc() {
        let vocab: Vec<String> = ["".to_owned()]
            .into_iter()
            .chain((1..=16).map(|i| std::iter::repeat_n('a', i).collect()))
            .collect();
        let vocab_ref: Vec<_> = vocab.iter().map(|s| s.as_ref()).collect();
        node_set_case(
            &vocab_ref,
            &[
                ("a", "a"),
                ("aa", "a"),
                ("aa", "aa"),
                ("aaaa", "aaaa"),
                ("aaaa", "aa"),
                ("aa", "aaa"),
                ("aaaa", "aaa"),
                ("aaaaaaaa", "aaaaaaaa"),
            ],
        );
    }
}
