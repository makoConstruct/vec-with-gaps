#![feature(
    generic_associated_types,
    allocator_api,
    exact_size_is_empty,
    type_ascription
)]
use std::{
    alloc::{Allocator, Global, Layout},
    cmp::{
        max, Ordering,
        Ordering::{Equal, Greater, Less},
    },
    iter::{ExactSizeIterator, FromIterator, Iterator},
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr,
    ptr::{write, NonNull},
    slice,
    vec::Vec,
};

pub struct VWGIter<'a, V> {
    sections_start: *const VWGSection,
    sections_end: *const VWGSection,
    within_section_start: *const V,
    within_section_end: *const V,
    mem: *const V,
    _phanto: PhantomData<&'a V>,
}
impl<'a, V: 'a> Iterator for VWGIter<'a, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<&'a V> {
        let Self {
            mem,
            ref mut sections_start,
            sections_end,
            ref mut within_section_start,
            ref mut within_section_end,
            ..
        } = *self;
        unsafe {
            loop {
                if within_section_start != within_section_end {
                    let ret = Some(&**within_section_start);
                    *within_section_start = within_section_start.wrapping_add(1);
                    return ret;
                } else {
                    loop {
                        if *sections_start != sections_end {
                            let VWGSection {
                                start: ss,
                                length: sl,
                            } = **sections_start;
                            *within_section_start = mem.add(ss);
                            *within_section_end = mem.wrapping_add(ss + sl);
                            *sections_start = sections_start.wrapping_add(1);
                            if *within_section_start != *within_section_end {
                                break;
                            }
                        } else {
                            return None;
                        }
                    }
                }
            }
        }
    }
}

pub struct VWGSectionIter<'a, V> {
    sections_start: *const VWGSection,
    sections_end: *const VWGSection,
    mem: *const V,
    _phanto: PhantomData<&'a V>,
}
impl<'a, V: 'a> Iterator for VWGSectionIter<'a, V> {
    type Item = &'a [V];
    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            mem,
            ref mut sections_start,
            sections_end,
            ..
        } = *self;
        if *sections_start != sections_end {
            unsafe {
                loop {
                    let VWGSection { start, length } = **sections_start;
                    let ret = Some(slice::from_raw_parts(mem.add(start), length));
                    *sections_start = sections_start.add(1);
                    return ret;
                }
            }
        } else {
            return None;
        }
    }
}
impl<'a, V: 'a> ExactSizeIterator for VWGSectionIter<'a, V> {
    fn len(&self) -> usize {
        unsafe { self.sections_end.offset_from(self.sections_start) as usize }
    }
    fn is_empty(&self) -> bool {
        self.sections_end == self.sections_start
    }
}

// the following three things are used by batched sorted inserts
/// assumes that the inputs are sorted. Doesn't actually enact the insertions, reports them to the callbacks
fn insertions_for_merge<'b, B: 'b, V>(
    source: &'b [B],
    dest_slice: &mut [V],
    comparator: impl Fn(&B, &V) -> Ordering,
    mut on_collision: impl FnMut(&'b B, &mut V) -> (),
    mut on_insertion: impl FnMut(&'b B, usize) -> (),
    conf: &impl VecWithGapsConfig,
) {
    //iterating is faster than binary-searching if the sizing ratios of the arrays are similar, so check for that
    if source.len() * conf.merge_sparseness_limit() < dest_slice.len() {
        for sv in source.iter() {
            let sr = dest_slice.binary_search_by(|c| comparator(sv, c).reverse());
            match sr {
                Ok(s) => on_collision(sv, &mut dest_slice[s]),
                Err(s) => on_insertion(sv, s),
            }
        }
    } else {
        let mut si = source.iter().peekable();
        for (de, dv) in dest_slice.iter_mut().enumerate() {
            loop {
                if let Some(sip) = si.peek() {
                    match comparator(sip, dv) {
                        Less => {
                            on_insertion(sip, de);
                        }
                        Equal => {
                            on_collision(sip, dv);
                        }
                        Greater => {
                            break;
                        }
                    }
                    si.next();
                } else {
                    return;
                }
            }
        }
        //push the rest into the back
        for sv in si {
            on_insertion(sv, dest_slice.len());
        }
    }
}
struct Pry {
    section_start: usize,
    pry_by: usize,
}
#[derive(Copy, Clone)]
struct InsertHeader {
    into_section: usize,
    number_of_elements: usize,
}

//TODO: an insert instruction that just says "the dst is empty, memcpy this slice in"
//DOTO: consider trying not storing any insert instructions, and instead going through and doing all of insert calculations again
// wait did we really need to do the insert computations in advance? If we just assumed there weren't going to be any sets, that they were indeed all inserts, it would be much much faster and simpler, mixing sets and inserts....... is insane
//for posterity, we did indeed need to do the insert computations eventually *in full*, so it made sense to do them in advance and save whatever memory might have been saved, saving the instructions, *because you need to save the instructions anyway* even if you're just doing the insertions one at a time (in order to do them backwards, for exactly the same reason you have to do the pries backwards)

struct InsertElement<'a, B> {
    dst_index: usize,
    inserting_value: &'a B,
}
impl<'a, B> Clone for InsertElement<'a, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, B> Copy for InsertElement<'a, B> {}

union Insert<'a, B> {
    header: InsertHeader,
    element: InsertElement<'a, B>,
}

#[derive(Copy, Clone)]
pub struct VWGSection {
    pub start: usize,
    pub length: usize,
}
/// A data structure that behaves like a vec of vecs, but where the subvecs are kept, in order, in one contiguous section of memory, which improves cache performance for some workloads. Automates the logic of expansion so that each section essentially behaves like a `Vec`.
/// (The members are pub because rust's restrictions on private members are so strict that I do not believe that private members should be used, and because if you're using this, you know how it works and you care enough about performance to want to be able to mess with it.)
//TODO: `HeaderVec` with `sections` as the vec. It's not obvious to me that this would save any cache misses, as HeaderVec moves `mem` behind a pointer to a place that may be far enough from the `section` pointer that this actually constitutes adding a third dereference to what would have otherwise only required two dereferences (sections and mem, the rest of the non-HeaderVec VecWithGaps is already on the stack)
pub struct VecWithGaps<V, A: Allocator = Global, Conf: VecWithGapsConfig = DefaultConf> {
    pub sections: Vec<VWGSection>,
    pub total_capacity: usize,
    pub mem: NonNull<V>,
    pub total: usize,
    pub allocator: A,
    pub conf: Conf,
}
#[derive(Clone, Default)]
pub struct DefaultConf();
pub trait VecWithGapsConfig: Clone {
    fn initial_capacity(&self) -> usize;
    /// when loading from iterators, the gap that's inserted between sections
    fn initial_default_gaps(&self) -> usize;
    /// the ratio of inserted elements to incumbent elements that must be present to justify using a binary search instead of a linear interleaving when merging two sorted lists
    fn merge_sparseness_limit(&self) -> usize;
    /// the proportion by which the total backing vec increases when it is outgrown
    fn increase_total_proportion(&self) -> f64;
    /// when being initialized from sized iterators, the extra capacity allocated after actual content is this times the length of the actual content
    fn initial_extra_total_proportion(&self) -> f64;
    /// the proportion by which a section increases its capacity when outgrown
    fn section_growth_multiple(&self) -> f64;
    /// after a section has begun to outgrow its bounds, the smallest capacity that will be cleared for it
    fn min_nonzero_section_capacity(&self) -> usize;
    // /// the maximum number of segments it will try to nudge before just extending the whole
    // fn max_nudge_size()-> usize;
    fn compute_next_total_capacity_encompassing(&self, encompassing: usize) -> usize {
        let mut ret = self.initial_capacity();
        while ret <= encompassing {
            let mr = ret as f64 * self.increase_total_proportion();
            if mr > usize::MAX as f64 {
                panic!("tried to grow total capacity to a size that overflowed usize");
            }
            ret = mr as usize;
        }
        ret
    }
    fn compute_next_section_capacity_encompassing(&self, encompassing: usize) -> usize {
        let mut ret = self.min_nonzero_section_capacity();
        while ret <= encompassing {
            let mr = ret as f64 * self.section_growth_multiple();
            if mr > usize::MAX as f64 {
                panic!("tried to grow a section capacity to a size that overflowed usize");
            }
            ret = mr as usize;
        }
        ret
    }
}
impl VecWithGapsConfig for DefaultConf {
    fn section_growth_multiple(&self) -> f64 {
        1.5f64
    }
    fn merge_sparseness_limit(&self) -> usize {
        36
    }
    fn initial_extra_total_proportion(&self) -> f64 {
        1.3f64
    }
    fn initial_default_gaps(&self) -> usize {
        8
    }
    fn initial_capacity(&self) -> usize {
        4
    }
    fn min_nonzero_section_capacity(&self) -> usize {
        8
    }
    fn increase_total_proportion(&self) -> f64 {
        1.4
    }
}

impl<'a, V: Clone, A: Allocator + Clone + Default, Conf: VecWithGapsConfig + Default>
    FromIterator<&'a [V]> for VecWithGaps<V, A, Conf>
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a [V]>,
    {
        Self::from_iters_detailed(
            iter.into_iter().map(|sl| sl.iter()),
            A::default(),
            Conf::default(),
        )
    }
}

impl<V, A: Allocator + Clone, Conf: VecWithGapsConfig> Clone for VecWithGaps<V, A, Conf> {
    fn clone(&self) -> Self {
        let new_mem = if let Some(bs) = self.sections.last() {
            let nm = self
                .allocator
                .allocate(Self::layout(self.total_capacity))
                .unwrap()
                .cast();
            unsafe {
                ptr::copy_nonoverlapping(
                    self.mem.as_ptr(),
                    nm.cast().as_ptr(),
                    bs.start + bs.length,
                );
            }
            nm
        } else {
            self.allocator
                .allocate(Self::layout(self.conf.initial_capacity()))
                .unwrap()
                .cast()
        };

        Self {
            sections: self.sections.clone(),
            total_capacity: self.total_capacity,
            mem: new_mem,
            total: self.total,
            allocator: self.allocator.clone(),
            conf: self.conf.clone(),
        }
    }
}

impl<V> VecWithGaps<V, Global, DefaultConf> {
    pub fn empty() -> Self {
        Self::empty_detailed(Global::default(), DefaultConf())
    }
}

impl<V, A: Allocator + Clone, Conf: VecWithGapsConfig> VecWithGaps<V, A, Conf> {
    pub fn empty_detailed(allocator: A, conf: Conf) -> Self {
        Self {
            sections: vec![],
            total_capacity: conf.initial_capacity(),
            total: 0,
            mem: allocator
                .allocate(
                    Layout::from_size_align(
                        size_of::<V>() * conf.initial_capacity(),
                        align_of::<V>(),
                    )
                    .unwrap(),
                )
                .unwrap()
                .cast(),
            allocator: allocator,
            conf: conf,
        }
    }
}

impl<'a, V: 'a + Clone, A: Allocator + Clone, Conf: VecWithGapsConfig> VecWithGaps<V, A, Conf> {
    pub fn from_slice_slice_detailed(i: &[&[V]], allocator: A, conf: Conf) -> Self {
        Self::from_iters_detailed(i.iter().map(|ii| ii.iter()), allocator, conf)
    }
    pub fn from_iters_detailed<I: Iterator<Item = BI>, BI: Iterator<Item = &'a V>>(
        mut i: I,
        allocator: A,
        conf: Conf,
    ) -> Self {
        let mut ret = Self::empty_detailed(allocator, conf);
        let process_section = |ret: &mut Self, bi: BI| {
            bi.for_each(move |bin| ret.push(bin.clone()));
        };
        if let Some(si) = i.next() {
            ret.push_section();
            process_section(&mut ret, si);
            for si in i {
                ret.push_section_after_gap(ret.conf.initial_default_gaps());
                process_section(&mut ret, si);
            }
        }
        ret
    }
    pub fn from_vec_vec_detailed(v: &Vec<Vec<V>>, allocator: A, conf: Conf) -> Self {
        Self::from_sized_iters_detailed(v.iter().map(|vs| vs.iter().cloned()), allocator, conf)
    }
    /// where sizing information is available, faster than from_iters
    pub fn from_sized_iters_detailed<
        I: ExactSizeIterator<Item = BI> + Clone,
        BI: ExactSizeIterator<Item = V>,
    >(
        i: I,
        allocator: A,
        conf: Conf,
    ) -> Self
    where
        V: Clone,
    {
        let content_len_total = i.clone().fold(0, |a, b| a + b.len());
        let actual_content_len = content_len_total + i.len() * conf.initial_default_gaps();
        let total_capacity = max(
            conf.initial_capacity(),
            (actual_content_len as f64 * conf.initial_extra_total_proportion()) as usize,
        );
        let mem: NonNull<V> = allocator
            .allocate(Self::layout(total_capacity))
            .unwrap()
            .cast();
        let mut start = 0;
        let sections = i
            .map(|ss| {
                let length = ss.len();
                let ret = VWGSection { start, length };
                let mut si = start;
                for sv in ss {
                    unsafe { ptr::write(mem.as_ptr().add(si), sv) };
                    si += 1;
                }
                start += length + conf.initial_default_gaps();
                ret
            })
            .collect();
        Self {
            sections,
            total: content_len_total,
            allocator,
            mem: mem,
            conf,
            total_capacity,
        }
    }
}

impl<'a, V: 'a + Clone> VecWithGaps<V, Global, DefaultConf> {
    pub fn from_slice_slice(i: &[&[V]]) -> Self {
        Self::from_slice_slice_detailed(i, Global::default(), DefaultConf())
    }
    pub fn from_iters<I: Iterator<Item = BI>, BI: Iterator<Item = &'a V>>(i: I) -> Self {
        Self::from_iters_detailed(i, Global::default(), DefaultConf())
    }
    pub fn from_vec_vec(v: &Vec<Vec<V>>) -> Self {
        Self::from_vec_vec_detailed(v, Global::default(), DefaultConf())
    }
    /// where sizing information is available, faster than from_iters
    pub fn from_sized_iters<
        I: ExactSizeIterator<Item = BI> + Clone,
        BI: ExactSizeIterator<Item = V>,
    >(
        i: I,
    ) -> Self
    where
        V: Clone,
    {
        Self::from_sized_iters_detailed(i, Global::default(), DefaultConf())
    }
}

fn section_not_found_panic(section_count: usize, section: usize) -> ! {
    panic!(
        "index out of bounds: there are only {} sections, but the section index is {}",
        section_count, section
    )
}
fn section_bounds_panic(item_count: usize, at: usize) -> ! {
    panic!(
        "index out of bounds: there are only {} items, but the index was {}",
        item_count, at
    );
}

impl<V, A: Allocator, Conf: VecWithGapsConfig> VecWithGaps<V, A, Conf> {
    /// returns the index of the new section
    pub fn push_section_after_gap(&mut self, gap_width: usize) -> usize {
        let next_section_index = self.sections.len();
        let start = if let Some(ref bs) = self.sections.last() {
            bs.start + bs.length + gap_width
        } else {
            0
        };
        self.sections.push(VWGSection {
            start: start,
            length: 0,
        });
        self.increase_capacity_to_fit(self.end_of_used_volume() + gap_width);
        next_section_index
    }
    /// cuts out the gaps to increase adjacency, increasing iteration speed and potentially conserving memory for push-heavy insertions, but if many things are being inserted into earlier sections, the gaps will reemerge and it will be expensive.
    pub fn trim_gaps(&mut self) {
        let Self {
            ref mut sections,
            mem,
            ..
        } = *self;
        let mut sepi = sections.iter_mut();
        if let Some(prev_sec) = sepi.next() {
            let mut last_end = prev_sec.start + prev_sec.length;
            for cur_sec in sepi {
                unsafe {
                    ptr::copy(
                        mem.as_ptr().add(cur_sec.start),
                        mem.as_ptr().add(last_end),
                        cur_sec.length,
                    );
                }
                cur_sec.start = last_end;
                last_end = cur_sec.start + cur_sec.length;
            }
        }
    }
    fn end_of_used_volume(&self) -> usize {
        if let Some(s) = self.sections.last() {
            s.start + s.length
        } else {
            0
        }
    }
    fn increase_capacity_to_fit(&mut self, increasing_to: usize) {
        if increasing_to < self.total_capacity {
            return;
        }
        // find the nearest next power growth
        let new_total_capacity = self
            .conf
            .compute_next_total_capacity_encompassing(increasing_to);
        let new_layout = Self::layout(new_total_capacity);
        self.mem = unsafe {
            self.allocator.grow(
                self.mem.cast(),
                // NonNull::new_unchecked(mem.as_ptr() as *mut u8),
                Self::layout(self.total_capacity),
                new_layout,
            )
        }
        .unwrap_or_else(|_| std::alloc::handle_alloc_error(new_layout))
        .cast();
        self.total_capacity = new_total_capacity;
    }
    pub fn push_section(&mut self) -> usize {
        self.push_section_after_gap(0)
    }
    pub fn get(&self, section: usize, at: usize) -> Option<&V> {
        let Self {
            mem, ref sections, ..
        } = *self;
        sections.get(section).and_then(|se| {
            if at >= se.length {
                None
            } else {
                Some(unsafe { &*(mem.as_ptr().add(se.start + at)) })
            }
        })
    }
    pub fn get_mut(&mut self, section: usize, at: usize) -> Option<&mut V> {
        let Self {
            mem,
            ref mut sections,
            ..
        } = *self;
        sections.get_mut(section).and_then(|se| {
            if at >= se.length {
                None
            } else {
                Some(unsafe { &mut *(mem.as_ptr().add(se.start + at)) })
            }
        })
    }
    pub fn len(&self) -> usize {
        self.sections.len()
    }
    /// the number of elements in the entire VecWithGaps, every section summed.
    pub fn total(&self) -> usize {
        self.total
    }

    pub fn section_slice(&self, section: usize) -> &[V] {
        let sl = self.sections.len();
        let s = match self.sections.get(section) {
            Some(a) => a,
            None => section_not_found_panic(sl, section),
        };
        unsafe { slice::from_raw_parts(self.mem.as_ptr().add(s.start), s.length) }
    }

    pub fn iter_lego_ugly_hybrid<'a>(&'a self) -> impl Iterator<Item = &'a V> {
        self.section_iter().flat_map(|s| s.iter())
    }

    fn layout(size: usize) -> Layout {
        // from_size_align_unchecked is fine for these parameters
        unsafe { Layout::from_size_align_unchecked(size * size_of::<V>(), align_of::<V>()) }
    }
    #[cold]
    fn expand(
        &mut self,
        section: usize,
        section_length: usize,
        next_section_start_if_next_section_exists: Option<usize>,
    ) {
        let Self {
            ref allocator,
            ref mut sections,
            total_capacity,
            mem,
            ref conf,
            ..
        } = *self;
        let end_of_defined_volume = sections.last().map(|s| s.start + s.length).unwrap(); //we know from ↑↑ that there is at least one section
        let free_space_at_end = total_capacity - end_of_defined_volume;
        let new_section_capacity =
            conf.compute_next_section_capacity_encompassing(section_length + 1);
        let section_capacity_increase = new_section_capacity - section_length;

        if free_space_at_end < section_capacity_increase {
            //attempt to grow
            let new_total_capacity = self.conf.compute_next_total_capacity_encompassing(
                total_capacity + section_capacity_increase,
            );
            //attempt to just enlarge. If it can't be enlarged, allocate a new section of memory, copy the front as well as the back, and use more efficient nonoverlapping copying functions
            let new_layout = Self::layout(new_total_capacity);
            //TODO, when allocators are stabilized, there should be a grow_in_place call. This will allow you to detect when growing fails and copy the data over more efficiently with a pair of copy_unaligneds instead of letting grow copy everything and then copying overlapping the tail sections on top of that.
            //This is the code that you will use for that
            // //didn't grow
            // let nmem = allocator.allocate(Self::layout(new_total_capacity)).unwrap();
            // unsafe { ptr::copy_nonoverlapping(mem, nmem, section_start + at); }
            // if let Some(next_section_start) = next_section_start_if_next_section_exists {
            //   unsafe { ptr::copy_nonoverlapping(
            //     mem + next_start,
            //     nmem + next_start + section_capacity_increase,
            //     end_of_defined_volume - next_start
            //   ); }
            // }
            // if at != section_length {
            //   //copy elements in the section after the insert point
            //   unsafe { ptr::copy_nonoverlapping(
            //     mem + section_start + at,
            //     nmem + section_start + at + 1,
            //     section_length - at,
            //   ); }
            //   *mem = nmem
            // }
            self.mem = unsafe {
                allocator.grow(
                    mem.cast(),
                    // NonNull::new_unchecked(mem.as_ptr() as *mut u8),
                    Self::layout(self.total_capacity),
                    new_layout,
                )
            }
            .unwrap_or_else(|_| std::alloc::handle_alloc_error(new_layout))
            .cast();
            self.total_capacity = new_total_capacity;
        }
        //later sections
        if let Some(next_section_start) = next_section_start_if_next_section_exists {
            let size_of_moving_volume = end_of_defined_volume - next_section_start;
            unsafe {
                ptr::copy(
                    self.mem.as_ptr().add(next_section_start),
                    self.mem
                        .as_ptr()
                        .add(next_section_start + section_capacity_increase),
                    size_of_moving_volume,
                );
            }
        }

        //update all of the section.starts
        for s in &mut self.sections[section + 1..] {
            s.start += section_capacity_increase;
        }
    }
    pub fn push_into_section(&mut self, section: usize, v: V) {
        let at = self.sections[section].length;
        self.insert_into_section(section, at, v);
    }
    /// pushes to the backmost section, creating a section if none exist
    pub fn push(&mut self, v: V) {
        let si = if self.sections.len() == 0 {
            self.push_section();
            0
        } else {
            self.sections.len() - 1
        };
        self.push_into_section(si, v)
    }
    /// true iff insertion happened
    fn insert_into_section_detailed<F>(
        &mut self,
        section: usize,
        v: V,
        specified_insert: Result<usize, F>,
    )-> bool where
        F: FnMut(&V, &V) -> Ordering,
    {
        let Self {
            ref sections,
            ref mut total_capacity,
            mem,
            ..
        } = *self;
        let sn = sections.len();
        let mut si = sections.iter().skip(section);
        let &VWGSection {
            start: section_start,
            length: section_length,
        } = si.next().unwrap_or_else(|| {
            section_not_found_panic(sn, section);
        });
        let next_section_start_if_next_section_exists: Option<usize> = si.next().map(|s| s.start);

        let at = match specified_insert {
            Ok(at) => {
                if at > section_length {
                    section_bounds_panic(section_length, at);
                }
                at
            }
            Err(mut comparator) => {
                let s = unsafe {
                    slice::from_raw_parts_mut(mem.as_ptr().add(section_start), section_length)
                };
                match s.binary_search_by(|b| comparator(&v, b)) {
                    Ok(_) => {
                        return false;
                    }
                    Err(i) => i,
                }
            }
        };

        //if there's more content, see if it needs to be moved along and move if so
        let bound = next_section_start_if_next_section_exists.unwrap_or_else(|| *total_capacity);
        // if match next_section_start { Some(next_start)=> section_end == next_start, None=>false } {
        if bound == section_start + section_length {
            //needs to expand the bound
            self.expand(
                section,
                section_length,
                next_section_start_if_next_section_exists,
            );
        }
        if at != section_length {
            //within section
            unsafe {
                ptr::copy(
                    self.mem.as_ptr().add(section_start + at),
                    self.mem.as_ptr().add(section_start + at + 1),
                    section_length - at,
                );
            }
        }
        unsafe {
            ptr::write(self.mem.as_ptr().add(section_start + at), v);
        }
        self.sections[section].length += 1;
        self.total += 1;
        true
    }
    // if we're going to have batch_sorted_merge_insert we might as well have this?
    /// true iff insertion happened
    pub fn insert_into_sorted_section(
        &mut self,
        section: usize,
        v: V,
        comparator: impl FnMut(&V, &V) -> Ordering,
    )-> bool {
        self.insert_into_section_detailed(section, v, Err(comparator))
    }
    pub fn insert_into_section(&mut self, section: usize, at: usize, v: V) {
        //needs to be given a type parameter, otherwise it "can't infer" F and freaks out
        self.insert_into_section_detailed::<fn(&V, &V) -> Ordering>(section, v, Ok(at));
    }
    pub fn take_from_section(&mut self, section: usize, at: usize) -> V {
        let se = match self.sections.get_mut(section) {
            Some(a) => a,
            None => section_not_found_panic(self.sections.len(), section),
        };
        if se.length == 0 {
            panic!("index out of bounds: tried to remove item at index {}, but the section contains no elements", at);
        }
        //doing a frankly excessive thing where sometimes we move the front forward instead of moving the back backwards
        let ret = unsafe { ptr::read(self.mem.as_ptr().add(se.start + at)) };
        if section > 0 && at < se.length / 2 {
            if at != 0 {
                unsafe {
                    ptr::copy(
                        self.mem.as_ptr().add(se.start),
                        self.mem.as_ptr().add(se.start + 1),
                        at,
                    )
                };
            }
            se.start += 1;
        } else {
            if at < se.length - 1 {
                unsafe {
                    ptr::copy(
                        self.mem.as_ptr().add(se.start + at + 1),
                        self.mem.as_ptr().add(se.start + at),
                        se.length - 1 - at,
                    )
                };
            }
        }
        se.length -= 1;
        self.total -= 1;
        ret
    }
    pub fn remove_from_section(&mut self, section: usize, at: usize) {
        self.take_from_section(section, at);
    }
    pub fn remove_section(&mut self, section: usize) {
        let sl = self.sections.len();
        let se = self
            .sections
            .get_mut(section)
            .unwrap_or_else(|| section_not_found_panic(sl, section));
        for i in 0..se.length {
            unsafe {
                ptr::drop_in_place(&mut *self.mem.as_ptr().add(se.start + i));
            }
        }
        self.sections.remove(section);
    }
    pub fn section_iter<'a>(&'a self) -> impl Iterator<Item = &'a [V]> {
        let Self {
            ref sections, mem, ..
        } = *self;
        sections
            .iter()
            .map(move |s| unsafe { slice::from_raw_parts(mem.as_ptr().add(s.start), s.length) })
    }
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a V> {
        self.section_iter().flat_map(|s| s.iter())
    }

    pub fn section_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut [V]> {
        let Self {
            ref mut sections,
            mem,
            ..
        } = *self;
        sections
            .iter_mut()
            .map(move |s| unsafe { slice::from_raw_parts_mut(mem.as_ptr().add(s.start), s.length) })
    }
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut V> {
        self.section_iter_mut().flat_map(|s| s.iter_mut())
    }
    pub fn ugly_ptr_iter<'a>(&'a self) -> VWGIter<'a, V> {
        let secp = self.sections.as_ptr();
        let sl = self.sections.len();
        if sl != 0 {
            let &VWGSection {
                start: tss,
                length: tsl,
            } = unsafe { &*secp };
            VWGIter {
                mem: self.mem.as_ptr(),
                sections_start: secp.wrapping_add(1),
                sections_end: secp.wrapping_add(sl),
                within_section_start: self.mem.as_ptr().wrapping_add(tss),
                within_section_end: self.mem.as_ptr().wrapping_add(tss + tsl),
                _phanto: Default::default(),
            }
        } else {
            VWGIter {
                mem: self.mem.as_ptr(),
                sections_start: ptr::null(),
                sections_end: ptr::null(),
                within_section_start: ptr::null(),
                within_section_end: ptr::null(),
                _phanto: Default::default(),
            }
        }
    }

    /// to be safe, requires that the inputs are sorted, and that each section is sorted. A usize is likely to overflow if they aren't, and I can't rule out access to uninitialized memory. (if you need to batch insert unsorted, please write the *drastically simpler* version optimized for unsorted workloads, and submit a pull request)
    /// It's also not supposed to insert duplicate elements (in that case, overwrite is called). The non-duplication invariants can also be broken if `src_insertions` contains duplicate elements, but I'm fairly sure that wouldn't cause safety violations.
    /// if you're wondering why this is so strangely abstracted (and why I went to the extreme efforts of writing a batched insertion function at all), it was needed for CSR++
    /// in `src_insertions`, the first element of the tuple is the index of the section, second is the things to be inserted into that section
    pub unsafe fn batch_sorted_merge_insert_detailed<'a, B>(
        &mut self,
        src_insertions: impl Iterator<Item = (usize, &'a [B])>,
        comparator: impl Fn(&B, &V) -> Ordering + Clone,
        overwrite: impl Fn(&B, &mut V) + Clone,
        inserting: impl Fn(&B) -> V,
    ) where
        B: Clone + 'a,
        V: 'a,
    {
        //overview:
        //  for each segment
        //  goes through seeing which vertices will be inserted or not (and overwriting the values of the collisions), which allows it to figure out which spaces need to be expanded, which it then encodes in prying_commands
        //  executes those commands on the VecWithGaps in the reverse direction

        // insertion format:
        //   read backwards
        //     section: number
        //     number_of_elements: number
        //     each element:
        //       dst_index: number
        //       element: value

        let Self {
            ref mut sections,
            ref mut total_capacity,
            ref mut total,
            ref mut mem,
            ref conf,
            ref allocator,
        } = *self;
        let mut prying_commands: Vec<Pry> = Vec::with_capacity(16);
        let mut insertion_commands: Vec<Insert<'a, B>> = Vec::with_capacity(32);
        let mut pry_running_total = 0; //represents the amount that the memory has been pried bigger at any given point
        let mut max_volume_needed = 0; //first tracks the max extents needed by the growing sections, then may be overwritten by the max extent created by pries if that is greater, representing the total volume that will be needed

        //split again by origin vertex
        for (sectioni, for_vertex) in src_insertions {
            let VWGSection { start, length } = sections[sectioni];
            let section_slice = { slice::from_raw_parts_mut(mem.as_ptr().add(start), length) };
            let mut number_of_additions = 0;
            //insertion
            insertions_for_merge(
                for_vertex,
                section_slice,
                comparator.clone(),
                overwrite.clone(),
                |inserting_value, dst_index| {
                    insertion_commands.push(Insert {
                        element: InsertElement {
                            inserting_value,
                            dst_index,
                        },
                    });
                    number_of_additions += 1;
                },
                conf,
            );
            insertion_commands.push(Insert {
                header: InsertHeader {
                    number_of_elements: number_of_additions,
                    into_section: sectioni,
                },
            });
            *total += number_of_additions;
            let required_length = length + number_of_additions;
            max_volume_needed = pry_running_total + start + required_length; //has to register this because the only way max_volume_needed can exceed the figure computed below is if the length of the final section expands, and if it did, that would be captured here (and if it didn't, that wouldn't be captured here)
                                                                             //decide whether to pry
            if let Some(ns) = sections.get(sectioni + 1) {
                let section_capacity_end = ns.start;
                if start + required_length >= section_capacity_end {
                    let new_section_capacity =
                        conf.compute_next_section_capacity_encompassing(required_length);
                    let pry_distance = (start + new_section_capacity) - section_capacity_end;
                    let tpd = pry_running_total + pry_distance;
                    prying_commands.push(Pry {
                        section_start: sectioni + 1,
                        pry_by: tpd,
                    });
                    pry_running_total = tpd;
                }
            }
        }
        let whole_volume_end = if let Some(VWGSection { start, length }) = sections.last() {
            max_volume_needed = max(max_volume_needed, start + pry_running_total + length);
            start + length
        } else {
            0
        };

        //TODO when grow_in_place makes it into the Allocator API, activate some of this branching
        if max_volume_needed > *total_capacity {
            let new_capacity = conf.compute_next_total_capacity_encompassing(max_volume_needed);
            let newmem = {
                allocator
                    .grow(
                        mem.cast(),
                        Self::layout(*total_capacity),
                        Self::layout(new_capacity),
                    )
                    .unwrap()
                    .cast()
            };
            *total_capacity = new_capacity;
            *mem = newmem;
            {
                self.execute_pries(
                    ptr::copy,
                    inserting,
                    newmem.as_ptr(),
                    newmem.as_ptr(),
                    whole_volume_end,
                    &insertion_commands,
                    &prying_commands,
                );
            }
            // if unsuccessful grow_in_place, allocate new memory, and use copy_nonoverlapping, this will be faster :
            // execute_pries(ptr::copy_nonoverlapping, inserting, mem.as_ptr(), newmem, &insertion_commands, &prying_commands);
        } else {
            let mc = mem.as_ptr();
            {
                self.execute_pries(
                    ptr::copy,
                    inserting,
                    mc,
                    mc,
                    whole_volume_end,
                    &insertion_commands,
                    &prying_commands,
                );
            }
        }
    }

    unsafe fn execute_pries<'a, B>(
        &mut self,
        copy_fn: unsafe fn(*const V, *mut V, usize),
        inserting: impl Fn(&B) -> V,
        src: *const V,
        dst: *mut V,
        end_of_whole_volume: usize,
        insertion_commands: &Vec<Insert<'a, B>>,
        pry_commands: &Vec<Pry>,
    ) {
        // it would probably be possible to do the pries and insertions in one sweep, but more complicated, maybe try it later
        // now execute the prying commands
        let Self {
            ref mut sections,
            total_capacity,
            ..
        } = *self;
        let mut prev_sections_start = sections.len() - 1;
        let mut prev_element_start = end_of_whole_volume;
        for Pry {
            section_start,
            pry_by,
        } in pry_commands.iter().rev()
        {
            let pry_element_start = sections[*section_start].start;
            // let pse = sections[section_end];
            // let pry_element_end = pse.start + pse.length;
            if pry_element_start + pry_by + (prev_element_start - pry_element_start)
                > total_capacity
            {
                panic!("can't allocate there");
            }
            let volume_end = prev_element_start;
            copy_fn(
                src.add(pry_element_start),
                dst.add(pry_element_start + pry_by),
                volume_end - pry_element_start, // DOTO: This copies slightly more than it needs to, the end of a section is usually a bit before the start of the following section, but this is not noticeably slower than computing the end
            );
            for s in sections[*section_start..prev_sections_start].iter_mut() {
                s.start += pry_by;
            }
            prev_sections_start = *section_start;
            prev_element_start = pry_element_start;
        }

        let mut inserts = insertion_commands.iter().rev();
        while let Some(Insert {
            header:
                InsertHeader {
                    into_section,
                    number_of_elements,
                },
        }) = inserts.next()
        {
            let VWGSection {
                start,
                ref mut length,
            } = sections[*into_section];
            let mut prev_insertion_end = *length;
            for i in 0..*number_of_elements {
                let Insert {
                    element:
                        InsertElement {
                            dst_index,
                            inserting_value,
                        },
                } = inserts.next().unwrap();
                let pry = number_of_elements - i;
                copy_fn(
                    src.add(start + dst_index),
                    dst.add(start + dst_index + pry),
                    prev_insertion_end - dst_index,
                );
                write(
                    dst.add(start + dst_index + pry - 1),
                    inserting(inserting_value),
                );
                prev_insertion_end = *dst_index;
            }
            *length += number_of_elements;
        }
    }

    /// see `batch_sorted_merge_insert_detailed` for full documentation
    pub unsafe fn batch_sorted_merge_insert<'a>(
        &mut self,
        src_insertions: impl Iterator<Item = (usize, &'a [V])>,
    ) where
        V: Ord + Clone + 'a,
    {
        self.batch_sorted_merge_insert_detailed(
            src_insertions,
            |b, v| b.cmp(v),
            |b, v| *v = b.clone(),
            |b| b.clone(),
        );
    }
}
impl<V, A: Allocator, C: VecWithGapsConfig> Drop for VecWithGaps<V, A, C> {
    fn drop(&mut self) {
        let Self {
            ref allocator,
            ref sections,
            ref mut mem,
            total_capacity,
            ..
        } = *self;
        for s in sections {
            let ss = unsafe { slice::from_raw_parts_mut(mem.as_ptr().add(s.start), s.length) };
            for vv in ss {
                unsafe {
                    ptr::drop_in_place(&mut *vv);
                }
            }
        }
        unsafe { allocator.deallocate(mem.cast(), Self::layout(total_capacity)) }
    }
}

#[cfg(test)]
mod tests {
    extern crate itertools;
    extern crate rand;
    use super::*;
    use itertools::assert_equal;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_create_drop() {
        let _v: VecWithGaps<usize> = VecWithGaps::empty();
    }

    #[test]
    fn test_push_once() {
        let mut v: VecWithGaps<usize> = VecWithGaps::empty();
        v.push(2);
    }

    #[test]
    fn test_many_push() {
        let testo: &[usize] = &[1, 2, 3, 4, 5, 6];
        let mut v: VecWithGaps<usize> = VecWithGaps::empty();
        for u in testo {
            v.push(*u);
        }
        assert_equal(v.iter(), testo.iter());
    }

    #[test]
    fn test_segmented_push() {
        let testo: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
        const SEGMENT_LENGTH: usize = 3;
        let mut segi = 0;
        let mut v: VecWithGaps<usize> = VecWithGaps::empty();
        for neo in testo.iter() {
            v.push(*neo);
            segi += 1;
            if segi == SEGMENT_LENGTH {
                v.push_section_after_gap(4);
                segi = 0;
            }
        }
        assert_equal(v.iter(), testo.iter());
    }

    #[test]
    fn big_attack() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut v: VecWithGaps<usize> = VecWithGaps::empty();
        const DAYS: usize = 500;
        let mut pushed = 0;
        for _dayi in 0..DAYS {
            // new user comes
            v.push_section();
            // each user has a 1/10 chance of engaging
            for i in 0..v.len() {
                if rng.gen::<f64>() < 0.1 {
                    v.push_into_section(i, 7);
                    pushed += 1;
                }
            }
        }
        assert_equal(v.iter(), std::iter::repeat(&7).take(pushed));
    }

    #[test]
    fn construct_from_iters() {
        let src: Vec<Vec<usize>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7], vec![8, 9, 10]];
        let v = VecWithGaps::from_iters(src.iter().map(|s| s.iter()));
        assert_equal(src.iter().flat_map(|s| s.iter()), v.ugly_ptr_iter());
    }

    // fn print<V: std::fmt::Debug>(v: &VecWithGaps<V>) {
    //     println!("-");
    //     for s in v.section_iter() {
    //         println!("  -");
    //         for ss in s.iter() {
    //             println!("    {:?}", *ss);
    //         }
    //     }
    // }

    #[test]
    fn removals() {
        let src: &[&[usize]] = &[&[1, 2, 3, 4], &[5, 6, 7, 8, 9, 10], &[11, 12, 13]];
        //  [1,2,3,4,5,6,7,8,9,10,11,12,13];
        let mut v = VecWithGaps::from_iters(src.iter().map(|s| s.iter()));
        assert_eq!(6, v.take_from_section(1, 1));
        assert_equal(v.iter(), [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13].iter());
        assert_eq!(9, v.take_from_section(1, 3));
        assert_equal(v.iter(), [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13].iter());
        v.remove_section(1);
        assert_equal(v.iter(), [1, 2, 3, 4, 11, 12, 13].iter());
    }

    #[test]
    fn drops() {
        use std::rc::Rc;
        let o = Rc::new(2usize);
        {
            let mut v = {
                let src = vec![
                    vec![o.clone(), o.clone(), o.clone()],
                    vec![o.clone(), o.clone()],
                    vec![o.clone()],
                    vec![o.clone()],
                    vec![o.clone()],
                ];
                assert_eq!(Rc::strong_count(&o), 9);
                VecWithGaps::from_vec_vec(&src)
            };
            assert_eq!(Rc::strong_count(&o), 9);
            v.remove_from_section(1, 0);
            assert_eq!(Rc::strong_count(&o), 8);
            v.remove_from_section(4, 0);
            assert_eq!(Rc::strong_count(&o), 7);
            v.remove_section(4);
            assert_eq!(Rc::strong_count(&o), 7);
            v.remove_section(0);
            assert_eq!(Rc::strong_count(&o), 4);
        }
        assert_eq!(Rc::strong_count(&o), 1);
    }

    #[test]
    fn test_batch_insert_sorted() {
        let ss: &[&[usize]] = &[&[1, 2, 3, 4], &[5, 6, 7, 8, 9, 10], &[11, 12, 13]];
        let mut v = VecWithGaps::from_slice_slice(ss);
        let insert: &[(usize, &[usize])] = &[(0, &[5])];
        unsafe {
            v.batch_sorted_merge_insert(insert.iter().cloned());
        }
        assert_equal(
            v.iter(),
            [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13].iter(),
        );
    }

    #[test]
    fn test_batch_insert_sorted_multi() {
        let ss: &[&[usize]] = &[&[1, 2, 3, 4], &[5, 6, 7, 8, 9, 10], &[11, 12, 13]];
        let mut v = VecWithGaps::from_slice_slice(ss);
        let insert: &[(usize, &[usize])] = &[(0, &[5]), (1, &[2]), (2, &[1, 12, 19])];
        unsafe {
            v.batch_sorted_merge_insert(insert.iter().cloned());
        }
        let rs = &[1, 2, 3, 4, 5, 2, 5, 6, 7, 8, 9, 10, 1, 11, 12, 13, 19];
        assert_equal(v.iter(), rs.iter());
        assert_eq!(rs.len(), v.total());
    }

    #[test]
    fn test_batch_insert_sorted_big_insert() {
        let ss: &[&[usize]] = &[&[1, 2, 3, 4], &[5, 6, 7, 8, 9, 10], &[11, 12, 13]];
        let mut v = VecWithGaps::from_slice_slice(ss);

        let amount_to_add: usize = 237;
        let rv: Vec<usize> = std::iter::repeat(9).take(amount_to_add).collect();
        unsafe {
            v.batch_sorted_merge_insert([(0, rv.as_slice())].iter().cloned());
        }
        let ought: usize = amount_to_add + ss.iter().map(|i| i.len()).sum(): usize;
        assert_eq!(ought, v.total());
    }

    //testing the batch insertion comparison benchmark
    fn fast_rng(seed: u64) -> impl Rng + Clone {
        rand::rngs::StdRng::seed_from_u64(seed)
    }
    use crate::tests::rand::prelude::SliceRandom;
    fn create_vwg(size: usize) -> VecWithGaps<usize> {
        // (this benchmark is trying to model users actively maintaining small sets of things, some users are active and some are less active)
        let mut rng = fast_rng(1);

        let mut active_sections: Vec<usize> = Vec::new();
        let mut less_active_sections: Vec<usize> = Vec::new();
        let mut ret = Vec::<Vec<usize>>::new();
        let mut total = 0;
        loop {
            let eventr: f64 = rng.gen();
            if eventr < 0.02 {
                //create a new section
                let si = ret.len();
                ret.push(Vec::new());
                // start it out with something in it

                if rng.gen::<f64>() < 0.8 {
                    less_active_sections.push(si);
                } else {
                    active_sections.push(si);
                }
            } else if eventr < 0.93 {
                //add to an active one
                if active_sections.len() != 0 {
                    ret[*active_sections.choose(&mut rng).unwrap()].push(rng.gen_range(0..size));
                    total += 1;
                }
            } else {
                //add to an old section
                if less_active_sections.len() != 0 {
                    ret[*less_active_sections.choose(&mut rng).unwrap()]
                        .push(rng.gen_range(0..size));
                    total += 1;
                }
            }
            if total >= size {
                return VecWithGaps::from_vec_vec(&ret);
            }
        }
    }

    fn binary_insert_if_not_present<V: Ord>(vs: &mut Vec<V>, p: V) {
        match vs.binary_search(&p) {
            Ok(_) => {}
            Err(i) => {
                vs.insert(i, p);
            }
        }
    }

    #[test]
    fn batch_insertion_benchmark() {
        let vwg_size = 200 * 13;
        let n_inserting = 50;
        let vwg = create_vwg(vwg_size);
        let addition_size = vwg_size + 90; //biases additions slightly towards the end, to reflect an increasing ID space
        let mut rng = fast_rng(60);

        let mut v = vwg.clone();
        let vl = v.len();
        let mut inserts: Vec<Vec<usize>> = (0..vl).map(|_| Vec::new()).collect();
        for _ in 0..n_inserting {
            binary_insert_if_not_present(
                &mut inserts[rng.gen_range(0..vl)],
                rng.gen_range(0..addition_size),
            );
        }
        unsafe {
            v.batch_sorted_merge_insert(
                inserts
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(e, i)| (e, i.as_slice())),
            );
        }
    }
    
    #[test]
    fn insertions_for_merge_test(){
      let inserting:&[usize] = &[2, 4, 8, 12];
      let receiving:&mut [usize] = &mut [1, 2, 3, 5, 6, 7, 8, 10, 14];
      let mut collisions = Vec::new();
      let mut insertions = Vec::new();
      insertions_for_merge(
        inserting,
        receiving,
        |a,b| a.cmp(b),
        |si, _rm| collisions.push(si),
        |_si, ri| insertions.push(ri),
        &DefaultConf()
      );
      itertools::assert_equal(collisions.iter(), [&2usize,&8usize].iter());
      itertools::assert_equal(insertions.iter(), [3usize,8usize].iter());
    }
}

// /// A version of VecWithGaps that has "quick delete" for sections, which, instead of removing a section from the vec, marks it invalid by setting its length to -1. This can also speed up inserts if they're happening at a similar rate as deletes (and aren't happening at the opposite end of the sections vec)
// struct VVQuickDelete<V> {
//     core: VecWithGaps<V>,
//     deletions: [usize;10], //remembers the last 10 deletions (sorted) so that section inserts wont have to copy over as many elements.
//     real_count: usize,
// }
