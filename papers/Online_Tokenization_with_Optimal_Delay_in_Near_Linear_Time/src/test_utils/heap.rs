use crate::typed_vec::{TypedVec, TypedVecIndex, typed_vec_index};

typed_vec_index!(NodeId, u32);

const ROOT: NodeId = NodeId::ZERO.next();

#[derive(Debug)]
pub(super) struct AdjustableHeap<Pos, Key> {
    nodes: TypedVec<NodeId, (Key, Pos)>,
    pos_to_node_id: TypedVec<Pos, NodeId>,
}

impl NodeId {
    #[inline]
    pub fn is_nil(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn parent(self) -> Self {
        Self(self.0 >> 1)
    }

    #[inline]
    pub fn left(self) -> Self {
        Self(self.0 << 1)
    }

    #[inline]
    pub fn right(self) -> Self {
        Self((self.0 << 1) | 1)
    }
}

impl<Pos, Key> AdjustableHeap<Pos, Key>
where
    Key: Copy + Default + Ord,
    Pos: Copy + Default + Ord + TypedVecIndex,
{
    #[inline]
    fn swap(&mut self, a: NodeId, b: NodeId) {
        let u = self.nodes[a].1;
        let v = self.nodes[b].1;
        self.nodes.swap(a, b);
        self.pos_to_node_id.swap(u, v);
    }

    #[inline]
    fn move_up(&mut self, mut cur: NodeId) {
        let mut parent = cur.parent();
        while !parent.is_nil() && self.nodes[cur] < self.nodes[parent] {
            self.swap(cur, parent);
            (cur, parent) = (parent, parent.parent());
        }
    }

    #[inline]
    fn move_down(&mut self, mut cur: NodeId) {
        let (mut left, mut right) = (cur.left(), cur.right());
        while right < self.nodes.len() {
            let min = if self.nodes[left] < self.nodes[right] {
                left
            } else {
                right
            };
            if self.nodes[min] >= self.nodes[cur] {
                return;
            }
            self.swap(min, cur);
            cur = min;
            (left, right) = (cur.left(), cur.right());
        }
        if left < self.nodes.len() && self.nodes[left] < self.nodes[cur] {
            self.swap(left, cur);
        }
    }

    fn adjust(&mut self, pos: Pos, key: Key) -> Key {
        let node_id = self.pos_to_node_id[pos];
        let old_key = self.nodes[node_id].0;
        self.nodes[node_id].0 = key;
        match key.cmp(&old_key) {
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Less => {
                self.move_up(node_id);
            }
            std::cmp::Ordering::Greater => {
                self.move_down(node_id);
            }
        }
        old_key
    }

    pub fn remove(&mut self, pos: Pos) -> Option<Key> {
        if pos >= self.pos_to_node_id.len() {
            return None;
        }
        let node_id = self.pos_to_node_id[pos];
        if node_id.is_nil() {
            return None;
        }
        let old_val = self.nodes[node_id];
        let last = self.nodes.len().prev();
        if last != node_id {
            self.swap(last, node_id);
        }
        self.nodes.pop();
        self.pos_to_node_id[pos] = NodeId::ZERO;
        if node_id < self.nodes.len() {
            let new_val = self.nodes[node_id];
            match new_val.cmp(&old_val) {
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Less => {
                    self.move_up(node_id);
                }
                std::cmp::Ordering::Greater => {
                    self.move_down(node_id);
                }
            }
        }
        Some(old_val.0)
    }

    pub fn pop(&mut self) -> Option<(Pos, Key)> {
        if self.nodes.len() <= ROOT {
            return None;
        }
        let last = self.nodes.len().prev();
        if last > ROOT {
            self.swap(last, ROOT);
        }
        let (key, pos) = self.nodes.pop().unwrap();
        self.pos_to_node_id[pos] = NodeId::ZERO;
        self.move_down(ROOT);
        Some((pos, key))
    }

    pub fn push(&mut self, pos: Pos, key: Key) {
        let node_id = self.nodes.len();
        self.nodes.push((key, pos));
        self.pos_to_node_id[pos] = node_id;
        self.move_up(node_id);
    }

    pub fn set(&mut self, pos: Pos, key: Key) -> Option<Key> {
        if self.pos_to_node_id[pos].is_nil() {
            self.push(pos, key);
            None
        } else {
            Some(self.adjust(pos, key))
        }
    }

    pub fn new<T: IntoIterator<Item = (Pos, Key)>>(pos_size: Pos, iter: T) -> Self {
        let mut nodes = TypedVec::<NodeId, _>::from_iter(
            [Default::default()]
                .into_iter()
                .chain(iter.into_iter().map(|(u, v)| (v, u))),
        );

        let mut cur = nodes.len().prev();
        while let parent = cur.parent()
            && !parent.is_nil()
        {
            if nodes[cur] < nodes[parent] {
                nodes.swap(parent, cur);
            }
            cur = cur.prev();
        }

        let mut pos_to_node_id = TypedVec::new_with(NodeId::ZERO, pos_size);
        for (node_id, (_, pos)) in nodes.enumerate().skip(1) {
            pos_to_node_id[*pos] = node_id;
        }

        Self {
            nodes,
            pos_to_node_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        test_utils::heap::AdjustableHeap,
        typed_vec::{TypedVec, typed_vec_index},
    };

    typed_vec_index!(KeyId, u32);
    typed_vec_index!(PosId, u32);
    type Heap = AdjustableHeap<PosId, KeyId>;

    fn validate_heap(heap: &Heap) {
        let nodes = &heap.nodes;
        let pos_to_node_id = &heap.pos_to_node_id;
        for (pos, node_id) in pos_to_node_id.enumerate_copied() {
            if node_id.is_nil() {
                continue;
            }
            assert_eq!(nodes[node_id].1, pos);
        }
        for (node_id, (rule_id, pos)) in nodes.enumerate_copied() {
            if node_id.is_nil() {
                continue;
            }
            let parent = node_id.parent();
            if !parent.is_nil() {
                assert!(nodes[parent].0 <= rule_id);
            }
            let left = node_id.left();
            if left < nodes.len() {
                assert!(rule_id <= nodes[left].0);
            }
            let right = node_id.right();
            if right < nodes.len() {
                assert!(rule_id <= nodes[right].0);
            }
            assert_eq!(pos_to_node_id[pos], node_id);
        }
    }

    fn rand_test(pos_size: u32, key_size: u32, num_ops: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut cur_keys = TypedVec::new_with(None::<KeyId>, PosId::new(pos_size));
        let mut heap = Heap::new(PosId(pos_size), []);
        validate_heap(&heap);
        for _ in 0..num_ops {
            match rng.random_range(0..3) {
                0 => {
                    let pos = PosId(rng.random_range(1..pos_size) - 1);
                    let new_rule_id = KeyId(rng.random_range(0..key_size));
                    let old_value = cur_keys[pos];
                    cur_keys[pos] = Some(new_rule_id);
                    assert_eq!(heap.set(pos, new_rule_id), old_value);
                }
                1 => {
                    let pos = PosId(rng.random_range(1..pos_size) - 1);
                    let old_value = cur_keys[pos];
                    cur_keys[pos] = None;
                    assert_eq!(heap.remove(pos), old_value);
                }
                2 => {
                    let min = cur_keys
                        .enumerate_copied()
                        .filter_map(|(i, v)| v.map(|r| (r, i)))
                        .min()
                        .map(|(r, i)| (i, r));
                    assert_eq!(heap.pop(), min);
                    if let Some((pos, _)) = min {
                        cur_keys[pos] = None;
                    }
                }
                _ => {
                    unreachable!();
                }
            }
            validate_heap(&heap);
        }
    }

    #[test]
    fn test_heap() {
        let mut heap = Heap::new(PosId::from(7usize), []);
        for (pos, rule_id) in [3, 9, 1, 10, 9, 6].into_iter().enumerate() {
            validate_heap(&heap);
            heap.push(PosId::from(pos + 1), KeyId(rule_id));
        }
        validate_heap(&heap);

        for seed in [391, 1096, 716] {
            rand_test(2, 100, 10000, seed);
            rand_test(100, 1, 10000, seed);
            rand_test(20, 100, 10000, seed);
            rand_test(1000, 1, 1000, seed);
            rand_test(1000, 1000, 1000, seed);
            rand_test(10000, 10000, 20, seed);
        }
    }
}
