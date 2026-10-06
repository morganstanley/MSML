use std::{
    collections::{BTreeMap, LinkedList, VecDeque},
    ops::{Index, Range},
};

use derive_more::Deref;
use tinyvec::TinyVec;

use crate::{
    successor::{FOREST_VIRTUAL_ROOT, ForestNodeId, ForestNodeIdVec, SucForest, SucNode},
    suf_suc::{SufSucNode, SufSucNodeSet},
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub(crate) CentroidId, u16);
typed_vec_index!(SubTreeNodeId, u16);
typed_vec_index!(NodePoolId, u32);

type IntervalVec = TinyVec<[(ForestNodeId, ForestNodeId); 4]>;
type CentroidChildVec = TinyVec<[CentroidId; 7]>;
type SubTreeChildVec = TinyVec<[SubTreeNodeId; 7]>;

const _: () = {
    assert!(std::mem::size_of::<IntervalVec>() == 40);
    assert!(std::mem::size_of::<CentroidChildVec>() == 24);
    assert!(std::mem::size_of::<SubTreeChildVec>() == 24);
};

#[derive(Debug, Deref)]
pub(crate) struct CentroidNode {
    #[deref]
    node: SufSucNode,
    subtree_root: SubTreeNodeId,
    intervals: IntervalVec,
    children: CentroidChildVec,
}

const _: () = {
    assert!(std::mem::size_of::<CentroidNode>() == std::mem::size_of::<SufSucNode>() + 24 + 40 + 8);
    assert!(std::mem::size_of::<[CentroidNode; 2]>() == std::mem::size_of::<CentroidNode>() * 2);
};

#[derive(Debug, Deref)]
pub(crate) struct SufSucCentroidTree {
    nodes: TypedVec<CentroidId, CentroidNode>,
}

#[derive(Debug)]
pub(crate) struct SufSucCentroidTreeView<'n> {
    nodes: &'n [CentroidNode],
}

#[derive(Debug)]
pub(crate) struct SufSucCentroidTrees {
    nodes: TypedVec<NodePoolId, CentroidNode>,
    trees: TypedVec<ForestNodeId, Range<NodePoolId>>,
}

impl CentroidNode {
    fn new(node: SubTreeNodeRef, subtree_root: SubTreeNodeId) -> Self {
        Self {
            node: node.suf_suc_node.clone(),
            subtree_root,
            intervals: Default::default(),
            children: Default::default(),
        }
    }
}

impl SufSucCentroidTrees {
    pub fn new(node_set: &SufSucNodeSet, forest: &SucForest) -> Self {
        let chain_len = {
            let mut chain_len = TypedVec::new_with(0u16, forest.len());
            let mut children = TypedVec::new_with(ForestNodeIdVec::new(), forest.len());
            for node_id in forest.keys() {
                if node_id == FOREST_VIRTUAL_ROOT {
                    continue;
                }
                children[node_set.suffix_parent[node_id]].push(node_id);
            }
            let mut queue = VecDeque::with_capacity(forest.len().as_usize());
            queue.push_back(FOREST_VIRTUAL_ROOT);
            while let Some(node_id) = queue.pop_front() {
                if node_id != FOREST_VIRTUAL_ROOT {
                    chain_len[node_id] = chain_len[node_set.suffix_parent[node_id]] + 1;
                }
                queue.extend(children[node_id].iter().copied());
            }
            chain_len
        };
        let num_of_nodes = chain_len.iter().copied().map(|v| v as u32).sum();

        let mut nodes = TypedVec::with_capacity(NodePoolId::new(num_of_nodes));
        let mut trees = TypedVec::with_capacity(forest.len());
        for forest_id in forest.keys() {
            let tree = SufSucCentroidTree::new(forest_id, node_set, forest);
            debug_assert_eq!(tree.len().inner(), chain_len[forest_id]);
            let start = nodes.len();
            nodes.extend(tree.nodes);
            let end = nodes.len();
            trees.push(start..end);
        }
        Self { nodes, trees }
    }

    #[inline(always)]
    pub fn get(&self, forest_id: ForestNodeId) -> SufSucCentroidTreeView<'_> {
        let range = &self.trees[forest_id];
        SufSucCentroidTreeView {
            nodes: &self.nodes.as_slice()[range.start.as_usize()..range.end.as_usize()],
        }
    }
}

#[derive(Clone, Copy, Debug, Deref)]
struct SubTreeNodeRef<'a> {
    #[deref]
    forest_node: &'a SucNode,
    suf_suc_node: &'a SufSucNode,
}

#[derive(Debug, Deref)]
struct SubTreeNode<'a> {
    #[deref]
    node: SubTreeNodeRef<'a>,
    parent: Option<SubTreeNodeId>,
    children: SubTreeChildVec,
    size: u16,
}

impl SufSucCentroidTree {
    pub fn new(start: ForestNodeId, node_set: &SufSucNodeSet, forest: &SucForest) -> Self {
        if start == FOREST_VIRTUAL_ROOT {
            return Self {
                nodes: TypedVec::with_capacity(CentroidId::ZERO),
            };
        }

        let mut subtree = {
            let mut chain = LinkedList::new();
            let mut cursor = start;
            while cursor != FOREST_VIRTUAL_ROOT {
                let forest_node = &forest[cursor];
                let suf_suc_node = &node_set[cursor];
                chain.push_back(SubTreeNodeRef {
                    forest_node,
                    suf_suc_node,
                });
                cursor = node_set.suffix_parent[cursor];
            }
            debug_assert!(!chain.is_empty());
            debug_assert_eq!(chain.back().unwrap().parent, FOREST_VIRTUAL_ROOT);

            let mut forest_to_node_id = BTreeMap::new();

            let mut nodes = TypedVec::with_capacity(SubTreeNodeId::from(chain.len()));
            for node in chain.into_iter().rev() {
                let forest_id = node.suf_suc_node.repr_id;
                if node.parent == FOREST_VIRTUAL_ROOT {
                    let id = nodes.push(SubTreeNode {
                        node,
                        parent: None,
                        children: Default::default(),
                        size: 1,
                    });
                    forest_to_node_id.insert(forest_id, id);
                } else {
                    let parent = forest_to_node_id[&node.parent];
                    let id = nodes.push(SubTreeNode {
                        node,
                        parent: Some(parent),
                        children: Default::default(),
                        size: 1,
                    });
                    forest_to_node_id.insert(forest_id, id);
                    nodes[parent].children.push(id);
                }
            }

            for id in nodes.keys().rev() {
                let node = &nodes[id];
                if let Some(parent) = node.parent {
                    nodes[parent].size += node.size;
                    debug_assert!(parent < id);
                } else {
                    debug_assert_eq!(id, SubTreeNodeId::ZERO);
                }
            }

            nodes
        };

        let mut roots = vec![(SubTreeNodeId::ZERO, None::<CentroidId>)];
        let mut centroids = TypedVec::with_capacity(CentroidId::from(subtree.len().inner()));

        while let Some((root_id, parent_centroid)) = roots.pop() {
            let half_size = subtree[root_id].size / 2;
            let next_large_subtree = |id| -> Option<(usize, SubTreeNodeId)> {
                subtree[id]
                    .children
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, c)| subtree[*c].size > half_size)
            };

            let centroid = if let Some(child) = next_large_subtree(root_id) {
                let mut large_child = (root_id, child.0, child.1);
                while let Some(child) = next_large_subtree(large_child.2) {
                    large_child = (large_child.2, child.0, child.1);
                }

                let (parent, child_idx, centroid) = large_child;
                subtree[parent].children.swap_remove(child_idx);
                subtree[centroid].parent = None;

                let centroid_size = subtree[centroid].size;
                let mut parent = Some(parent);
                while let Some(parent_id) = parent {
                    subtree[parent_id].size -= centroid_size;
                    parent = subtree[parent_id].parent;
                }

                centroid
            } else {
                root_id
            };

            debug_assert!(
                subtree[centroid]
                    .children
                    .iter()
                    .all(|&i| subtree[i].size <= half_size)
            );

            let id = centroids.push(CentroidNode::new(*subtree[centroid], root_id));
            if let Some(parent) = parent_centroid {
                let parent_node = &mut centroids[parent];
                parent_node
                    .intervals
                    .push(subtree[root_id].suf_suc_node.valid_range);
                parent_node.children.push(id);
            }
            for c in std::mem::take(&mut subtree[centroid].children) {
                let child = &mut subtree[c];
                child.parent = None;
                subtree[centroid].size -= child.size;
                roots.push((c, Some(id)));
            }
            if centroid != root_id {
                roots.push((root_id, None));
                debug_assert!(subtree[root_id].size <= half_size);
            }
        }

        #[cfg(debug_assertions)]
        {
            for node in subtree {
                debug_assert!(node.size == 1 && node.parent.is_none() && node.children.is_empty());
            }
        }

        for id in centroids.keys() {
            let mut order = Vec::from_iter(0..centroids[id].children.len());
            order.sort_by_key(|&i| centroids[id].intervals[i].0);
            let children = order
                .iter()
                .copied()
                .map(|i| centroids[id].children[i])
                .collect();
            centroids[id].children = children;
            let intervals = order
                .iter()
                .copied()
                .map(|i| centroids[id].intervals[i])
                .collect();
            centroids[id].intervals = intervals;
        }

        Self { nodes: centroids }
    }
}

impl<'n> Index<CentroidId> for SufSucCentroidTreeView<'n> {
    type Output = CentroidNode;

    #[inline(always)]
    fn index(&self, index: CentroidId) -> &Self::Output {
        &self.nodes[index.as_usize()]
    }
}

impl<'n> SufSucCentroidTreeView<'n> {
    #[inline(always)]
    pub fn len(&self) -> CentroidId {
        CentroidId::from(self.nodes.len())
    }

    #[inline(always)]
    pub fn search<F: Fn(usize) -> ForestNodeId>(&self, skip_to: F) -> ForestNodeId {
        let len = self.len();
        let to_parent = |node: CentroidId| {
            Some(node.next()).filter(|&parent| {
                parent < len && self[parent].subtree_root == self[node].subtree_root
            })
        };

        let next_subtree = |node_id: CentroidId| {
            let node = &self[node_id];
            if node.children.is_empty() {
                return None;
            }
            let val = skip_to(node.skip_len as _);
            match node.intervals.binary_search_by_key(&val, |&(l, _)| l) {
                Ok(i) => Some(node.children[i]),
                Err(i) => {
                    if i == 0 {
                        return None;
                    }
                    let (_, r) = node.intervals[i - 1];
                    if val >= r {
                        return None;
                    }
                    Some(node.children[i - 1])
                }
            }
        };

        let mut current = CentroidId::ZERO;

        loop {
            if !self[current].verify(&skip_to) {
                if let Some(parent) = to_parent(current) {
                    current = parent;
                    continue;
                } else {
                    debug_assert!(false, "{self:?}");
                    break;
                }
            }
            if let Some(child) = next_subtree(current) {
                current = child;
            } else {
                break;
            }
        }
        self[current].repr_id
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, NormalizedDict, Vocab,
        aho_corasick::ACAutomaton,
        centroid::{CentroidId, SufSucCentroidTrees},
        successor::{FOREST_VIRTUAL_ROOT, SucForest},
        suf_suc::SufSucNodeSet,
    };

    fn centroid_case(rules: &[(&str, &str)]) {
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

        let dict = Dictionary::new_from_token_pair(vocab, rules.iter().copied()).unwrap();
        let dict = NormalizedDict::new_in_bytes(dict).unwrap();
        let automaton = ACAutomaton::new(dict.iter_canonical_or_empty_tokens());
        let forest = SucForest::new(&dict);
        let node_set = SufSucNodeSet::new(&forest, &automaton);
        let trees = SufSucCentroidTrees::new(&node_set, &forest);

        for (id, tree) in forest.keys().map(|i| (i, trees.get(i))) {
            if id == FOREST_VIRTUAL_ROOT {
                continue;
            }
            let token = &dict[forest[id].token_id];
            let num_valid_tokens = dict
                .tokens
                .iter()
                .filter(|t| !t.is_empty() && token.ends_with(t))
                .count();
            assert_eq!(num_valid_tokens, tree.len().as_usize());
            for u in (0..tree.len().as_usize()).map(CentroidId::from) {
                let v = u.next();
                if v >= tree.len() {
                    continue;
                }
                assert_ne!(tree[u].repr_id, tree[v].repr_id);
                let is_parent = {
                    let mut w = forest[tree[u].repr_id].parent;
                    while w != FOREST_VIRTUAL_ROOT && w != tree[v].repr_id {
                        w = forest[w].parent;
                    }
                    w == tree[v].repr_id
                };
                assert!(is_parent ^ (tree[v].subtree_root != tree[u].subtree_root));
            }
        }
    }

    #[test]
    fn test_centroid() {
        centroid_case(&[
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
        centroid_case(&[
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
    }
}
