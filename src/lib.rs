#![feature(generic_associated_types, allocator_api)]
// use alloc::{alloc, dealloc, Allocator, Global, Layout};
use std::{
    alloc::{Allocator, Global, Layout},
    cmp::max,
    iter::{Iterator, Peekable},
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr,
    ptr::NonNull,
    slice,
    vec::Vec,
};

pub struct VWGIter<'a, V> {
    mem: *mut V,
    sections: Peekable<slice::Iter<'a, VWGSection>>,
    si: usize,
}

impl<'a, V: 'a> Iterator for VWGIter<'a, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<&'a V> {
        let Self {
            mem,
            ref mut sections,
            ref mut si,
        } = *self;
        while let Some(se) = sections.peek() {
            if *si < se.length {
                let ret = Some(unsafe { &*mem.add(se.start + *si) });
                *si += 1;
                return ret;
            }
            sections.next();
            *si = 0;
        }
        None
    }
}

pub struct VWGUglyPtrIter<'a, V> {
    sections_start: *const VWGSection,
    sections_end: *const VWGSection,
    within_section_start: *const V,
    within_section_end: *const V,
    mem: *const V,
    _phanto: PhantomData<&'a V>,
}
impl<'a, V: 'a> Iterator for VWGUglyPtrIter<'a, V> {
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
    pub allocator: A,
    pub conf: Conf,
}
#[derive(Clone)]
pub struct DefaultConf();
pub trait VecWithGapsConfig: Clone {
    fn initial_capacity(&self) -> usize;
    /// the proportion by which the total backing vec increases when it is outgrown
    fn increase_total_proportion(&self) -> f64;
    /// the proportion by which a section increases its capacity when outgrown
    fn section_growth_multiple(&self) -> f64;
    /// the smallest section capacity cleared for a section that is beginning to grow
    fn min_nonzero_section_capacity(&self) -> usize;
    // /// the maximum number of segments it will try to nudge before just extending the whole
    // fn max_nudge_size()-> usize;
}
impl VecWithGapsConfig for DefaultConf {
    // fn initial_spare_space_per_vertex()-> usize { 2 }
    fn section_growth_multiple(&self) -> f64 {
        2f64
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
            allocator: self.allocator.clone(),
            conf: self.conf.clone(),
        }
    }
}

impl<V> VecWithGaps<V, Global, DefaultConf> {
    pub fn new() -> Self {
        let conf = DefaultConf();
        let allocator = Global::default();
        Self {
            sections: vec![VWGSection {
                start: 0,
                length: 0,
            }],
            total_capacity: conf.initial_capacity(),
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

impl<V, A: Allocator, Conf: VecWithGapsConfig> VecWithGaps<V, A, Conf> {
    // fn new()-> Self {
    //   let conf = DefaultConf();
    //   Self{
    //     sections: vec![VWGSection{start:0, length:0}],
    //     total_capacity: conf.initial_capacity(),
    //     mem: NonNull::new(alloc(Layout::from_size_align(size_of::<V>()*conf.initial_capacity(), align_of::<V>()).unwrap()) as *mut V).unwrap(),
    //     allocator: Global,
    //     conf: conf,
    //   }
    // }
    // fn push(&mut self, v:V) {
    //   let Self { ref mut allocator, ref mut sections, ref mut mem } = *self;
    //   if let &[.., mut ref a, mut ref b] = sections.as_mut_slice() {
    //     if a.start + a.length == b.start {
    //       b.start += 1;
    //       //expand mem
    //       let new_capacity = if a.length != 0 {
    //         4
    //       }else{
    //         a.length*2
    //       };
    //       allocator.realloc(mem, new_capacity - a.length);
    //     }
    //     unsafe{ ptr::write(mem + a.length, v) };
    //     a.length += 1;
    //   }else{
    //     unsafe{ seriously_unreachable(); } //there are always two sections
    //   }
    // }
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
        // panic!("index out of bounds: there are {} sections, but the section index is {}", sections.len(), section);
        sections.get(section).and_then(|se| {
            if at >= se.length {
                None
                // panic!("index out of bounds: the section len is {} but the index is {}", s.length, at);
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
        // panic!("index out of bounds: there are {} sections, but the section index is {}", sections.len(), section);
        sections.get_mut(section).and_then(|se| {
            if at >= se.length {
                None
                // panic!("index out of bounds: the section len is {} but the index is {}", s.length, at);
            } else {
                Some(unsafe { &mut *(mem.as_ptr().add(se.start + at)) })
            }
        })
    }
    pub fn section_count(&self) -> usize {
        self.sections.len()
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
        //TODO, think about overflows, and think about panics that could break the structure's drop invariants
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
            panic!(
                "index out of bounds: there are only {} sections, but the section index is {}",
                sn, section
            )
        });
        let next_section_start_if_next_section_exists: Option<usize> = si.next().map(|s| s.start);
        if at > section_length {
            panic!(
                "index out of bounds: there are only {} items, but the index was {}",
                section_length, at
            );
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
    }
    pub fn iter<'a>(&'a self) -> VWGIter<'a, V> {
        VWGIter {
            mem: self.mem.as_ptr(),
            sections: self.sections.iter().peekable(),
            si: 0,
        }
    }
    pub fn iter_lego<'a>(&'a self) -> impl Iterator<Item = &'a V> {
        let Self {
            ref sections, mem, ..
        } = *self;
        sections.iter().flat_map(move |s| {
            unsafe { slice::from_raw_parts(mem.as_ptr().add(s.start), s.length) }.iter()
        })
    }
    pub fn ugly_ptr_iter<'a>(&'a self) -> VWGUglyPtrIter<'a, V> {
        let secp = self.sections.as_ptr();
        let sl = self.sections.len();
        if sl != 0 {
            let &VWGSection {
                start: tss,
                length: tsl,
            } = unsafe { &*secp };
            VWGUglyPtrIter {
                mem: self.mem.as_ptr(),
                sections_start: secp.wrapping_add(1),
                sections_end: secp.wrapping_add(sl),
                within_section_start: self.mem.as_ptr().wrapping_add(tss),
                within_section_end: self.mem.as_ptr().wrapping_add(tss + tsl),
                _phanto: Default::default(),
            }
        } else {
            VWGUglyPtrIter {
                mem: self.mem.as_ptr(),
                sections_start: ptr::null(),
                sections_end: ptr::null(),
                within_section_start: ptr::null(),
                within_section_end: ptr::null(),
                _phanto: Default::default(),
            }
        }
    }
    pub fn iter_mut<'a>(&'a mut self) -> VWGMutIter<'a, V> {
        VWGMutIter {
            mem: self.mem.as_ptr(),
            sections: self.sections.iter().peekable(),
            si: 0,
        }
    }
    // fn section_iter<'a>(&'a self)-> VWGSectionIter<'a, V> {
    //   VWGSectionIter{ mem: self.mem, sections.self.sections.iter() }
    // }
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
        let _v: VecWithGaps<usize> = VecWithGaps::new();
    }

    #[test]
    fn test_push_once() {
        let mut v: VecWithGaps<usize> = VecWithGaps::new();
        v.push(2);
    }

    #[test]
    fn test_many_push() {
        let testo: &[usize] = &[1, 2, 3, 4, 5, 6];
        let mut v: VecWithGaps<usize> = VecWithGaps::new();
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
        let mut v: VecWithGaps<usize> = VecWithGaps::new();
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
        let mut v: VecWithGaps<usize> = VecWithGaps::new();
        const DAYS: usize = 500;
        let mut pushed = 0;
        for _dayi in 0..DAYS {
            // new user comes
            v.push_section();
            // each user has a 1/10 chance of engaging
            for i in 0..v.section_count() {
                if rng.gen::<f64>() < 0.1 {
                    v.push_into_section(i, 7);
                    pushed += 1;
                }
            }
        }
        assert_equal(v.iter(), std::iter::repeat(&7).take(pushed));
    }
}
