use tinyvec::TinyVec;

use crate::typed_vec::typed_vec_index;

typed_vec_index!(pub(crate) ACNodeId, u32);

pub(crate) const AC_NODE_ROOT: ACNodeId = ACNodeId::ZERO;

pub(crate) type ACNodeIdInlineVec = TinyVec<[ACNodeId; 6]>;

const _: () = {
    assert!(std::mem::size_of::<ACNodeIdInlineVec>() == 32);
};
