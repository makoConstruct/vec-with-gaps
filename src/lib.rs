#![feature(generic_associated_types, allocator_api, exact_size_is_empty)]
use std::{
    alloc::{Allocator, Global, Layout},
    cmp::max,
    iter::{ExactSizeIterator, Iterator, Peekable},
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr,
    ptr::NonNull,
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
    fn is_empty(&self) -> bool { self.sections_end == self.sections_start }
}

pub struct VWGMutIter<'a, V> {
    mem: *mut V,
    sections: Peekable<slice::Iter<'a, VWGSection>>,
    si: usize,
}
impl<'a, V: 'a> Iterator for VWGMutIter<'a, V> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<&'a mut V> {
        let Self {
            mem,
            ref mut sections,
            ref mut si,
        } = *self;
        while let Some(se) = sections.peek() {
            if *si < se.length {
                return Some(unsafe { &mut *(mem.offset((se.start + *si) as isize)) });
            }
            sections.next();
            *si = 0;
        }
        None
    }
}

#[derive(Clone)]
pub struct VWGSection {
    pub start: usize,
    pub length: usize,
}
/// A vec with empty spaces between sections. Useful when you will have lots of short series that you want to be cache coherent, but with gaps between them ready in case something needs to be inserted. Automates the logic of expansion so that each section essentially behaves like a `Vec`.
pub struct VecWithGaps<V, A: Allocator = Global, Conf: VecWithGapsConfig = DefaultConf> {
    pub sections: Vec<VWGSection>,
    pub total_capacity: usize,
    pub mem: NonNull<V>,
    pub total: usize,
    pub allocator: A,
    pub conf: Conf,
}
#[derive(Clone)]
pub struct DefaultConf();
pub trait VecWithGapsConfig: Clone {
    fn initial_capacity(&self) -> usize;
    /// when loading from iterators, the gap that's inserted between sections
    fn initial_default_gaps(&self) -> usize;
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
}
impl VecWithGapsConfig for DefaultConf {
    fn section_growth_multiple(&self) -> f64 {
        2f64
    }
    fn initial_extra_total_proportion(&self) -> f64 {
        1.3f64
    }
    fn initial_default_gaps(&self) -> usize {
        2
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
        let conf = DefaultConf();
        let allocator = Global::default();
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

impl<'a, V: 'a + Clone> VecWithGaps<V, Global, DefaultConf> {
    pub fn from_iters<I: Iterator<Item = BI> + Clone, BI: Iterator<Item = &'a V>>(
        mut i: I,
    ) -> Self {
        let mut ret = Self::empty();
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
    pub fn from_vec_vec(v: &Vec<Vec<V>>) -> Self {
        Self::from_sized_iters(v.iter().map(|vs| vs.iter()))
    }
    /// where sizing information is available, faster than from_iters
    pub fn from_sized_iters<
        I: ExactSizeIterator<Item = BI> + Clone,
        BI: ExactSizeIterator<Item = &'a V>,
    >(
        i: I,
    ) -> Self
    where
        V: Clone,
    {
        let content_len_total = i.clone().fold(0, |a, b| a + b.len());
        let conf = DefaultConf();
        let actual_content_len = content_len_total + i.len() * conf.initial_default_gaps();
        let total_capacity = max(
            conf.initial_capacity(),
            (actual_content_len as f64 * conf.initial_extra_total_proportion()) as usize,
        );
        let allocator = Global::default();
        let mem = allocator.allocate(Self::layout(total_capacity)).unwrap();
        let mut start = 0;
        let sections = i
            .map(|ss| {
                let length = ss.len();
                let ret = VWGSection { start, length };
                start += length + conf.initial_default_gaps();
                let mut si = start;
                for sv in ss {
                    unsafe { ptr::write(mem.cast::<V>().as_ptr().add(si), sv.clone()) };
                    si += 1;
                }
                ret
            })
            .collect();
        Self {
            sections,
            total: content_len_total,
            allocator,
            mem: mem.cast(),
            conf: conf,
            total_capacity,
        }
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
    fn compute_next_total_capacity_encompassing(&self, encompassing: usize) -> usize {
        let mut ret = max(self.conf.initial_capacity(), self.total_capacity);
        while ret <= encompassing {
            let (nret, overflowed) = ret.overflowing_mul(2);
            if overflowed {
                panic!("tried to grow to a size that overflowed usize");
            }
            ret = nret;
        }
        ret
    }
    fn increase_capacity_to_fit(&mut self, increasing_to: usize) {
        if increasing_to < self.total_capacity {
            return;
        }
        // find the nearest next power growth
        let new_total_capacity = self.compute_next_total_capacity_encompassing(increasing_to);
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

    pub fn section_iter<'a>(&'a self) -> VWGSectionIter<'a, V> {
        VWGSectionIter {
            sections_start: self.sections.as_ptr(),
            sections_end: self.sections.as_ptr().wrapping_add(self.sections.len()),
            mem: self.mem.as_ptr(),
            _phanto: Default::default(),
        }
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
        let new_section_capacity = max(section_length * 2, conf.min_nonzero_section_capacity());
        let section_capacity_increase = new_section_capacity - section_length;

        if free_space_at_end < section_capacity_increase {
            //attempt to grow
            let new_total_capacity = self.compute_next_total_capacity_encompassing(
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
    pub fn insert_into_section(&mut self, section: usize, at: usize, v: V) {
        let Self {
            ref sections,
            ref mut total_capacity,
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
        if at > section_length {
            section_bounds_panic(section_length, at);
        }
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
    }
    pub fn take_from_section(&mut self, section: usize, at: usize) -> V {
        let sl = self.sections.len();
        let se = self
            .sections
            .get_mut(section)
            .unwrap_or_else(|| section_not_found_panic(sl, section));
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
    pub fn remove_section(&mut self, section:usize) {
        let sl = self.sections.len();
        let se = self
            .sections
            .get_mut(section)
            .unwrap_or_else(|| section_not_found_panic(sl, section));
        for i in 0..se.length {
            unsafe{ ptr::drop_in_place(&mut *self.mem.as_ptr().add(i)); }
        }
        self.sections.remove(section);
    }
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a V> {
        let Self {
            ref sections, mem, ..
        } = *self;
        sections.iter().flat_map(move |s| {
            unsafe { slice::from_raw_parts(mem.as_ptr().add(s.start), s.length) }.iter()
        })
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
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut V> {
        let Self {
            ref mut sections,
            mem,
            ..
        } = *self;
        sections.iter_mut().flat_map(move |s| {
            unsafe { slice::from_raw_parts_mut(mem.as_ptr().add(s.start), s.length) }.iter_mut()
        })
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
}
