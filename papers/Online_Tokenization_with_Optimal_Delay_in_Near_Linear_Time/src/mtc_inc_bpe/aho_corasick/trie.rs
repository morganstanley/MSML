use std::{collections::BTreeMap, iter::FusedIterator};

use derive_more::{Debug, Deref};

use crate::{
    aho_corasick::{AC_NODE_ROOT, ACNodeId, relabeling::Relabeling},
    typed_vec::{TypedVec, vec_with_head},
};

#[derive(Debug, Deref)]
pub(crate) struct ACTrie {
    nodes: TypedVec<ACNodeId, BTreeMap<u8, ACNodeId>>,
}

#[derive(Debug)]
pub(crate) struct BfsIter<'t> {
    trie: &'t ACTrie,
    queue: Vec<ACNodeId>,
    current: usize,
}

impl Default for ACTrie {
    fn default() -> Self {
        Self {
            nodes: [Default::default()].into_iter().collect(),
        }
    }
}

impl ACTrie {
    #[inline(always)]
    pub fn children(
        &self,
        id: ACNodeId,
    ) -> impl DoubleEndedIterator<Item = (ACNodeId, u8)> + ExactSizeIterator + FusedIterator {
        self[id].iter().map(|(&u, &v)| (v, u))
    }

    #[inline(always)]
    pub fn children_nodes(
        &self,
        id: ACNodeId,
    ) -> impl DoubleEndedIterator<Item = ACNodeId> + ExactSizeIterator + FusedIterator {
        self[id].values().copied()
    }

    #[inline(always)]
    pub fn get(&self, node_id: ACNodeId, byte: u8) -> Option<ACNodeId> {
        self.nodes[node_id].get(&byte).copied()
    }

    pub fn get_or_add(&mut self, node_id: ACNodeId, byte: u8) -> ACNodeId {
        if let Some(&child) = self.nodes[node_id].get(&byte) {
            child
        } else {
            let child_id = self.nodes.push(Default::default());
            self.nodes[node_id].insert(byte, child_id);
            child_id
        }
    }

    pub(super) fn apply_relabeling(mut self, relabeling: &Relabeling<ACNodeId>) -> ACTrie {
        self.nodes = relabeling.apply_to_typed_vec(self.nodes);
        for node in &mut self.nodes {
            relabeling.apply_to_iter_mut(node.values_mut());
        }
        self
    }

    pub fn bfs(&self) -> BfsIter<'_> {
        BfsIter {
            trie: self,
            queue: vec_with_head(AC_NODE_ROOT, self.len()),
            current: 0,
        }
    }
}

impl<'t> Iterator for BfsIter<'t> {
    type Item = ACNodeId;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&node_id) = self.queue.get(self.current) {
            self.queue.extend(self.trie.children_nodes(node_id));
            self.current += 1;
            Some(node_id)
        } else {
            None
        }
    }
}

impl<'t> FusedIterator for BfsIter<'t> {}
