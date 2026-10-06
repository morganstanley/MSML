use crate::typed_vec::{TypedVec, TypedVecIndex};

#[derive(Debug)]
pub(super) struct Relabeling<I> {
    order: TypedVec<I, I>,
    rank: TypedVec<I, I>,
}

impl<I: TypedVecIndex + Copy> Relabeling<I> {
    pub fn new(order: TypedVec<I, I>) -> Self {
        let mut rank = TypedVec::new_with(I::from_usize(0), order.len());
        for (new_id, old_id) in order.enumerate_copied() {
            rank[old_id] = new_id;
        }
        Self { order, rank }
    }

    pub fn apply_to_typed_vec<T: Default>(&self, mut seq: TypedVec<I, T>) -> TypedVec<I, T> {
        seq.keys()
            .map(|i| std::mem::take(&mut seq[self.order[i]]))
            .collect()
    }

    pub fn apply_to_iter_mut<'a>(&self, iter: impl IntoIterator<Item = &'a mut I>) {
        for slot in iter {
            *slot = self.rank[*slot];
        }
    }
}
