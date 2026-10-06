use derive_more::{Debug, Deref};

use crate::{
    TokenId,
    aho_corasick::{
        AC_NODE_ROOT, ACNodeId, ACSuffixLinkTree, ACTransTable, ACTrie,
        heavy_light::heavy_light_decomposition,
    },
    typed_vec::TypedVec,
};

#[derive(Debug, Deref)]
pub(crate) struct ACAutomaton {
    #[deref]
    pub trans_table: ACTransTable,
    #[cfg(test)]
    pub trie: ACTrie,
    pub suffix: ACSuffixLinkTree,
    pub token_to_node: TypedVec<TokenId, ACNodeId>,
}

impl ACAutomaton {
    pub fn new<T: AsRef<[u8]>, V: IntoIterator<Item = T>>(vocab: V) -> Self {
        let mut trie = ACTrie::default();

        let vocab = vocab.into_iter();
        let mut token_to_node = TypedVec::with_capacity(TokenId::from(vocab.size_hint().0));

        for token in vocab {
            let mut node = AC_NODE_ROOT;
            for &byte in token.as_ref() {
                node = trie.get_or_add(node, byte);
            }
            token_to_node.push(node);
        }

        let mut suffix = TypedVec::new_with(AC_NODE_ROOT, trie.len());
        for node in trie.bfs() {
            if node == AC_NODE_ROOT {
                continue;
            }
            for (child, byte) in trie.children(node) {
                let mut cursor = suffix[node];
                while cursor != AC_NODE_ROOT && trie.get(cursor, byte).is_none() {
                    cursor = suffix[cursor];
                }
                suffix[child] = trie.get(cursor, byte).unwrap_or(AC_NODE_ROOT);
            }
        }

        let relabeling = heavy_light_decomposition(&trie);

        let trie = trie.apply_relabeling(&relabeling);
        relabeling.apply_to_iter_mut(&mut token_to_node);

        relabeling.apply_to_iter_mut(&mut suffix);
        let suffix = ACSuffixLinkTree::new(relabeling.apply_to_typed_vec(suffix));

        let trans_table = ACTransTable::new(&trie, &suffix);

        Self {
            trans_table,
            #[cfg(test)]
            trie,
            suffix,
            token_to_node,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Vocab,
        aho_corasick::{AC_NODE_ROOT, ACAutomaton},
    };

    #[test]
    fn test_ac_automaton() {
        let vocab = Vocab::new([
            b"a" as &[u8],
            b"abc",
            b"abcde",
            b"abcdef",
            b"b",
            b"ba",
            b"bc",
            b"bcdef",
            b"c",
            b"cd",
            b"cde",
            b"cdefg",
            b"d",
            b"de",
            b"def",
            b"e",
            b"ef",
            b"efg",
            b"f",
            b"g",
        ])
        .unwrap();

        let automaton = ACAutomaton::new(vocab.tokens());

        for node in automaton.trie.keys() {
            let suffix = automaton.suffix[node];
            let children: Vec<_> = automaton.trie.children(node).collect();
            println!("{node:2} {suffix:2}: {children:?}");
        }
        for (id, token) in vocab.tokens.enumerate() {
            let node = automaton.token_to_node[id];
            let suffix = automaton.suffix[node];
            println!("{node:2} {suffix:2}: {}", str::from_utf8(token).unwrap());
        }

        let search = |s: &str| {
            let mut node = AC_NODE_ROOT;
            for &b in s.as_bytes() {
                if let Some(next) = automaton.trie.get(node, b) {
                    node = next;
                } else {
                    return None;
                }
            }
            Some(node)
        };

        let id_b = search("b").unwrap();
        let id_ba = search("ba").unwrap();
        assert!(search("babcd").is_none());
        let id_abcd = search("abcd").unwrap();
        let id_abcdef = search("abcdef").unwrap();
        assert!(search("abcdefg").is_none());
        assert!(search("bcdefg").is_none());
        let id_cdefg = search("cdefg").unwrap();

        let feed = |sequences: &[&str]| -> Vec<_> {
            let mut node = AC_NODE_ROOT;
            sequences
                .iter()
                .map(|&s| {
                    node = automaton.feed(node, s);
                    node
                })
                .collect()
        };

        let output = feed(&["b", "a", "bcd", "ef", "g"]);
        assert_eq!(output, vec![id_b, id_ba, id_abcd, id_abcdef, id_cdefg]);
    }
}
