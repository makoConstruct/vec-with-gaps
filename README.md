A data structure that behaves like a vec of vecs, but where the subvecs are kept contiguous in memory, potentially improving cache performance for some workloads.

Or, at least, it's supposed to improve cache performance for some workloads. Benchmarks indicate that somehow it fails to improve upon a vec of vecs for sequential reads. I don't understand it. I'm not sure where to go from here.

Construction of a `VecWithGaps` takes about twice as long as creating a vec of vecs, this is mostly expected, as expanding a subvec of a `VecWithGaps` usually entails moving every subvec after it over a bit, though it will often avoid dealing with the global allocator.

More paradoxical is it that reading performance is only very slightly improved, sequentially reading the entire structure is only 1.5% quicker, despite its perfect adjacency.

I also don't understand why `VWGUglyIter` is being marginally but noticeably outperformed by a high level combination of standard iterators (the `lego` iterator).