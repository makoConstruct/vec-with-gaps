A data structure that behaves like a vec of vecs, but where the subvecs are kept contiguous in memory, improving cache performance for some workloads.

Good for situation where the contents of subvecs change only a little bit over time and where they're often read in order. Not good if data changes a lot (unless changes are always concentrated at the very end).

On mako's computer, sequential read performance benefits start to become substantial at about the point where there are 2_000_000 words. If you are storing only, say, 20_000 words, and unless there's something mulching your cache coherency as you generate the data structure, the cache benefits of using a `VecWithGaps` instead of a `Vec<Vec<V>>` are pretty negligible.

Construction of `VecWithGaps` can be quite slow if you're inserting into the middle a lot. If you're just pushing to the end, though, it's good.