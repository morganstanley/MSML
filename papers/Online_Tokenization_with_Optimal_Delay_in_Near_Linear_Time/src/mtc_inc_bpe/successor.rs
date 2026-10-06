use derive_more::Deref;
use tinyvec::TinyVec;

use crate::{
    RuleId, TokenId,
    normalize::{ATOMIC_TOKEN_PRIORITY, NormalizedView},
    typed_vec::{TypedVec, typed_vec_index},
    vocab::TokenIdVec,
};

pub type SkipLen = u16;

typed_vec_index!(pub(crate) ForestNodeId, u32);

pub(crate) const FOREST_VIRTUAL_ROOT: ForestNodeId = ForestNodeId::ZERO;

pub(crate) type ForestNodeIdVec = TinyVec<[ForestNodeId; 6]>;
const _: () = {
    assert!(std::mem::size_of::<ForestNodeIdVec>() == 32);
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SucNode {
    pub token_id: TokenId,
    pub priority: RuleId,
    pub skip_len: SkipLen,
    pub pre_id: ForestNodeId,
    pub parent: ForestNodeId,
    pub subtree_last_node: ForestNodeId,
    pub children: ForestNodeIdVec,
}

#[derive(Debug, Deref)]
pub(crate) struct SucForest {
    #[deref]
    nodes: TypedVec<ForestNodeId, SucNode>,
    pub(crate) token_to_node_id: TypedVec<TokenId, ForestNodeId>,
}

impl SucForest {
    pub fn new<D: NormalizedView + ?Sized>(dict: &D) -> Self {
        let num_tokens = dict.num_of_tokens();

        let mut roots = Vec::with_capacity(num_tokens.as_usize());
        let mut children = TypedVec::new_with(TokenIdVec::new(), num_tokens);
        for position in 0..num_tokens.as_usize() {
            let token_id = TokenId::from(position);
            let rule_id = dict.priority(token_id);
            if dict.is_atomic(token_id) {
                roots.push(token_id);
            } else if dict.is_canonical(token_id) {
                children[dict.rule(rule_id).suc].push(token_id);
            }
        }

        roots.sort();
        for vec in &mut children {
            vec.sort_by_key(|&i| !dict.priority(i));
        }

        let mut token_to_node_id = TypedVec::new_with(FOREST_VIRTUAL_ROOT, num_tokens);
        let virtual_root = SucNode {
            token_id: TokenId::MAX,
            priority: RuleId::MAX,
            skip_len: 0,
            pre_id: FOREST_VIRTUAL_ROOT,
            parent: FOREST_VIRTUAL_ROOT,
            subtree_last_node: FOREST_VIRTUAL_ROOT,
            children: Default::default(),
        };
        let mut nodes = TypedVec::with_capacity(ForestNodeId::new(dict.num_of_tokens().inner()));
        nodes.push(virtual_root);

        let mut alloc = {
            |token_id: TokenId, parent: ForestNodeId| {
                let node_id = nodes.len();
                nodes.push(SucNode {
                    token_id,
                    priority: dict.priority(token_id),
                    skip_len: 1,
                    parent,
                    pre_id: FOREST_VIRTUAL_ROOT,
                    subtree_last_node: node_id,
                    children: Default::default(),
                });
                node_id
            }
        };

        let mut stack = vec![(None::<TokenId>, 0usize)];

        while let Some((token_id, child_id)) = stack.last_mut() {
            if let Some(token_id) = *token_id {
                if *child_id >= children[token_id].len() {
                    stack.pop();
                    continue;
                }
                let child = children[token_id][*child_id];
                *child_id += 1;
                let parent = token_to_node_id[token_id];
                token_to_node_id[child] = alloc(child, parent);
                stack.push((Some(child), 0usize));
            } else {
                if *child_id >= roots.len() {
                    stack.pop();
                    continue;
                }
                let child = roots[*child_id];
                *child_id += 1;
                token_to_node_id[child] = alloc(child, FOREST_VIRTUAL_ROOT);
                stack.push((Some(child), 0usize));
            }
        }
        drop(children);

        for node_id in nodes.keys().rev() {
            nodes[node_id].children.reverse();

            #[cfg(debug_assertions)]
            {
                for &child in &nodes[node_id].children {
                    debug_assert!(child > node_id && nodes[child].parent == node_id);
                    debug_assert!(
                        (node_id == FOREST_VIRTUAL_ROOT)
                            ^ (nodes[child].priority < ATOMIC_TOKEN_PRIORITY)
                    );
                }
            }

            if node_id == FOREST_VIRTUAL_ROOT {
                continue;
            }

            #[cfg(debug_assertions)]
            {
                let node = &nodes[node_id];
                debug_assert!(node.children.is_sorted_by_key(|&c| !nodes[c].priority));
                for slice in node.children.windows(2) {
                    let u = slice[0];
                    let v = slice[1];
                    debug_assert!(u < v && nodes[u].priority > nodes[v].priority);
                }
            }

            let parent = nodes[node_id].parent;
            debug_assert!(parent < node_id);
            nodes[parent].subtree_last_node = nodes[parent]
                .subtree_last_node
                .max(nodes[node_id].subtree_last_node);
            nodes[parent].children.push(node_id);

            let rule_id = nodes[node_id].priority;
            if rule_id < ATOMIC_TOKEN_PRIORITY {
                debug_assert!(rule_id < dict.num_of_rules());
                let pre_id = token_to_node_id[dict.rule(rule_id).pre];
                nodes[node_id].pre_id = pre_id;
            }
        }

        for token_id in {
            let mut order: Vec<_> = (0..num_tokens.as_usize()).map(TokenId::from).collect();
            order.sort_by_key(|&i| dict.token(i).len());
            order
        } {
            let node_id = token_to_node_id[token_id];
            if !dict.is_canonical(token_id) {
                debug_assert_eq!(node_id, FOREST_VIRTUAL_ROOT);
                continue;
            }
            debug_assert_ne!(node_id, FOREST_VIRTUAL_ROOT);
            let node = &nodes[node_id];
            let (pre_id, parent) = (node.pre_id, node.parent);
            if node.parent == FOREST_VIRTUAL_ROOT {
                continue;
            }
            debug_assert_ne!(node.pre_id, FOREST_VIRTUAL_ROOT);
            nodes[node_id].skip_len = nodes[pre_id].skip_len + nodes[parent].skip_len;
        }

        Self {
            token_to_node_id,
            nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, NormalizedDict, Vocab,
        successor::{FOREST_VIRTUAL_ROOT, SucForest},
    };

    #[test]
    fn test_suc_forest() {
        let vocab = Vocab::new([
            b"" as &[_],
            b"a",
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

        let validate = |rules: &[(&str, &str)]| {
            let dict =
                Dictionary::new_from_token_pair(vocab.clone(), rules.iter().copied()).unwrap();
            let normalized = NormalizedDict::new_in_bytes(dict.clone()).unwrap();
            let forest = SucForest::new(&normalized);
            for (node_id, node) in forest.enumerate() {
                if node_id != FOREST_VIRTUAL_ROOT && node.token_id.inner() > 0 {
                    assert_eq!(vocab[node.token_id].len(), node.skip_len as usize);
                }
                let s = if node_id.0 == 0 {
                    "(epsilon)"
                } else {
                    std::str::from_utf8(&vocab[node.token_id]).unwrap()
                };
                println!("{s:12} {node_id:2}: {node:?}");
            }
            let normalized = NormalizedDict::new_in_utf8(dict.clone()).unwrap();
            let forest_b = SucForest::new(&normalized);

            assert_eq!(forest.token_to_node_id, forest_b.token_to_node_id);
            assert_eq!(forest.as_slice(), forest_b.as_slice());
        };

        validate(&[
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
        ]);

        validate(&[
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
        ]);

        let dict = Dictionary::new_from_token_pair(
            vocab.clone(),
            [("b", "c"), ("e", "f"), ("abc", "def")],
        )
        .unwrap();
        let dict = NormalizedDict::new_in_bytes(dict).unwrap();
        SucForest::new(&dict);
    }
}
