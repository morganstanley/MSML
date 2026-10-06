use crate::{
    aho_corasick::{AC_NODE_ROOT, ACNodeId, ACTrie, relabeling::Relabeling},
    typed_vec::{TypedVec, vec_with_head},
};

pub(super) fn heavy_light_decomposition(trie: &ACTrie) -> Relabeling<ACNodeId> {
    let len = trie.len();

    let mut size = TypedVec::new_with(1u32, len);
    let mut largest = TypedVec::new_with(AC_NODE_ROOT, len);

    let mut stack = vec_with_head(
        (AC_NODE_ROOT, None::<ACNodeId>, trie.children(AC_NODE_ROOT)),
        len,
    );
    while let Some((node_id, parent_id, child_iter)) = stack.last_mut() {
        let node_id = *node_id;

        if let Some((child_id, _)) = child_iter.next() {
            stack.push((child_id, Some(node_id), trie.children(child_id)));
            continue;
        }

        if let Some(parent_id) = *parent_id {
            let cur_size = size[node_id];
            size[parent_id] += cur_size;
            if largest[parent_id] == AC_NODE_ROOT || size[largest[parent_id]] < cur_size {
                largest[parent_id] = node_id;
            }
        }

        stack.pop();
    }

    let mut stack = vec_with_head(AC_NODE_ROOT, len);
    let mut order = TypedVec::with_capacity(len);
    while let Some(mut head) = stack.pop() {
        loop {
            order.push(head);
            let next_head = largest[head];
            if next_head == AC_NODE_ROOT {
                break;
            }
            for (child_id, _) in trie.children(head).rev() {
                if child_id != next_head {
                    stack.push(child_id);
                }
            }
            head = next_head;
        }
    }
    debug_assert_eq!(order[AC_NODE_ROOT], AC_NODE_ROOT);
    debug_assert_eq!(order.len(), len);
    Relabeling::new(order)
}

#[cfg(test)]
mod tests {
    use crate::aho_corasick::{
        AC_NODE_ROOT, ACNodeId, ACTrie, heavy_light::heavy_light_decomposition,
    };

    #[test]
    fn test_ac_trie_heavy_light_decomposition() {
        let mut trie = ACTrie::default();
        for token in ["abcd", "aaa", "acd", "rose", "rad", "rand", "rb"] {
            let mut node = AC_NODE_ROOT;
            for &byte in token.as_bytes() {
                node = trie.get_or_add(node, byte);
            }
        }
        let relabeling = heavy_light_decomposition(&trie);
        let order = relabeling.apply_to_typed_vec(trie.keys().collect());
        let order: Vec<_> = order.into_iter().map(|i| i.inner()).collect();
        let expected = [0, 9, 13, 15, 16, 14, 17, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(order, expected);

        let trie = trie.apply_relabeling(&relabeling);

        for (node_id, expected) in [
            (0, &[(1u32, b'r'), (10, b'a')] as &[_]),
            (1, &[(2, b'a'), (6, b'b'), (7, b'o')]),
            (2, &[(3, b'n'), (5, b'd')]),
            (10, &[(11, b'b'), (14, b'a'), (16, b'c')]),
        ] {
            let mut children: Vec<_> = trie
                .children(ACNodeId::new(node_id))
                .map(|(u, v)| (u.inner(), v))
                .collect();
            children.sort();
            let expected = expected.to_vec();
            assert_eq!(children, expected);
        }
    }
}
