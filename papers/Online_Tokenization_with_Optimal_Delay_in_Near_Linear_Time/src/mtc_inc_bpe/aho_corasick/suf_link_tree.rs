use std::iter::FusedIterator;

use derive_more::Deref;

use crate::{
    aho_corasick::{AC_NODE_ROOT, ACNodeId, ACNodeIdInlineVec},
    typed_vec::TypedVec,
};

#[derive(Debug, Deref)]
pub(crate) struct ACSuffixLinkTree {
    #[deref]
    suffix: TypedVec<ACNodeId, ACNodeId>,
    children: TypedVec<ACNodeId, ACNodeIdInlineVec>,
}

impl ACSuffixLinkTree {
    pub fn new(suffix: TypedVec<ACNodeId, ACNodeId>) -> Self {
        let len = suffix.len();

        let mut children = TypedVec::new_with(ACNodeIdInlineVec::new(), len);
        for (cur, suf) in suffix.enumerate_copied() {
            if cur == suf {
                debug_assert_eq!(cur, AC_NODE_ROOT);
                continue;
            }
            children[suf].push(cur);
        }

        Self { suffix, children }
    }

    #[inline(always)]
    pub fn children(
        &self,
        id: ACNodeId,
    ) -> impl DoubleEndedIterator<Item = ACNodeId> + ExactSizeIterator + FusedIterator {
        self.children[id].iter().copied()
    }
}
