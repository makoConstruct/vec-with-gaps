<https://crates.io/crates/vec-with-gaps>

`VecWithGaps` is a data structure that behaves like a vec of vecs, but where the subvecs are kept in one contiguous section of memory, which improves cache performance for some workloads.

Mechanically, it is a vec with gaps, sections of uninitialized values interspersed between sections of initialized values, providing free space to add more elements to the subvecs in constant time.

Good for situations where the contents of subvecs change only a little bit over time and where they're often read in order. Not good if data changes a lot (unless changes are always concentrated at the very end).

On mako's computer, sequential read performance benefits start to become substantial once the VecWithGaps is storing around million words. If you are storing only, say, 20_000 words, and unless there's something mulching your the adjacency of the allocations while you're generating the data structure, the cache benefits of using a `VecWithGaps` instead of a `Vec<Vec<V>>` are pretty negligible.

Construction of `VecWithGaps` can be quite slow if you're inserting into the middle a lot, if you are, it might be a good idea to use a `Vec<Vec<V>>` and then convert it via `VecWithGaps::from_vec_vec` once generation is complete. If you're only pushing to the end, though, `VecWithGaps` will perform well.