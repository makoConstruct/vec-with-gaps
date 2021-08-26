#![allow(dead_code, incomplete_features)]
#![feature(test)]
extern crate test;

use std::fmt::Debug;
use test::Bencher;
use vec_with_gaps::*;

trait VWG<V>: Clone {
    fn empty() -> Self;
    fn push_section(&mut self) -> usize;
    fn push(&mut self, section: usize, v: V);
    fn fold<F: Fn(V, &V) -> V>(&self, first: V, f: F) -> V;
    fn print(&self);
    fn print_sizes(&self);
}

#[derive(Clone)]
struct DummyVwg<V> {
    sections: Vec<Vec<V>>,
}

impl<V: Clone + Debug> VWG<V> for DummyVwg<V> {
    fn empty() -> Self {
        Self { sections: vec![] }
    }
    fn push(&mut self, section: usize, v: V) {
        self.sections[section].push(v);
    }
    fn push_section(&mut self) -> usize {
        let ret = self.sections.len();
        self.sections.push( Vec::new() );
        ret
    }
    fn fold<F: Fn(V, &V) -> V>(&self, first: V, f: F) -> V {
        self.sections
            .iter()
            .flat_map(|s| s.iter())
            .fold(first, f)
    }
    fn print(&self) {
        println!("-");
        for s in self.sections.iter() {
            println!("  -");
            for ss in s.iter() {
                println!("    {:?}", *ss);
            }
        }
    }
    fn print_sizes(&self) {
        for s in self.sections.iter() {
            print!(" {}", s.len());
        }
        println!("");
    }
}

impl<V: Debug> VWG<V> for VecWithGaps<V> {
    fn empty() -> Self {
        Self::new()
    }
    fn push_section(&mut self) -> usize {
        self.push_section_after_gap(0)
    }
    fn push(&mut self, section: usize, v: V) {
        self.push_into_section(section, v);
    }
    fn fold<F: Fn(V, &V) -> V>(&self, first: V, f: F) -> V {
        self.iter().fold(first, f)
    }
    fn print(&self) {
        println!("-");
        for s in self.section_iter() {
            println!("  -");
            for ss in s.iter() {
                println!("    {:?}", *ss);
            }
        }
    }
    fn print_sizes(&self) {
        for s in self.section_iter() {
            print!(" {}", s.len());
        }
        println!("");
    }
}

fn create_vwg<TV: VWG<V>, V: Clone>(to_push: &V, v: &mut TV) {
    const EVENT_COUNT: usize = 52000;
    let mut active_sections: Vec<usize> = Vec::with_capacity(EVENT_COUNT / 100 / 2);
    let mut less_active_sections: Vec<usize> = Vec::with_capacity(EVENT_COUNT / 100 / 2);
    for n in 0..EVENT_COUNT {
        let odds = n % 100;
        if odds < 2 {
            //create a new section
            let si = v.push_section();
            // start it out with something in it

            if odds < 80 {
                less_active_sections.push(si);
            } else {
                active_sections.push(si);
            }
        } else if odds < 93 {
            //add to an active one
            let len = active_sections.len();
            if len != 0 {
                let i = n % len;
                v.push(active_sections[i], to_push.clone());
            }
        } else {
            //add to an old section
            let len = less_active_sections.len();
            if len != 0 {
                let i = n % len;
                v.push(less_active_sections[i], to_push.clone());
            }
        }
    }
}

fn bench_vwg<TV: VWG<[usize; 2]>>(b: &mut Bencher, vs: &TV) {
    let to_push = [1, 2];
    b.iter(|| {
        let mut ve = vs.clone();
        create_vwg(&to_push, &mut ve);

        // const READ_COUNT:usize = 30000;
        const READ_COUNT: usize = 200;
        let mut total = 0;
        for _ in 0..READ_COUNT {
            let r = ve.fold(to_push, |a, b| [a[0] + b[0], a[1] + b[1]]);
            total += r[0] * r[1];
        }
        total
    });
}

const READ_REPS: usize = 1;

fn bench_vwg_read<TV: VWG<[usize; 2]>>(b: &mut Bencher, ve: &TV) {
    b.iter(|| {
        let mut total = 0;
        for _ in 0..READ_REPS {
            let r = ve.fold([0, 0], |a, b| [a[0] + b[0] + b[1], a[1] + b[1]]);
            total += r[0] * r[1];
        }
        total
    });
}

fn bench_vwg_ugly_read(b: &mut Bencher, ve: &VecWithGaps<[usize; 2]>) {
    b.iter(|| {
        let mut total = 0;
        for _ in 0..READ_REPS {
            let r = ve
                .ugly_ptr_iter()
                .fold([0, 0], |a, b| [a[0] + b[0] + b[1], a[1] + b[1]]);
            total += r[0] * r[1];
        }
        total
    });
}

fn bench_vwg_regular_read(b: &mut Bencher, ve: &VecWithGaps<[usize; 2]>) {
    b.iter(|| {
        let mut total = 0;
        for _ in 0..READ_REPS {
            let r = ve
                .iter()
                .fold([0, 0], |a, b| [a[0] + b[0] + b[1], a[1] + b[1]]);
            total += r[0] * r[1];
        }
        total
    });
}

fn bench_vwg_lego_read(b: &mut Bencher, ve: &VecWithGaps<[usize; 2]>) {
    b.iter(|| {
        let mut total = 0;
        for _ in 0..READ_REPS {
            let r = ve
                .iter_lego()
                .fold([0, 0], |a, b| [a[0] + b[0] + b[1], a[1] + b[1]]);
            total += r[0] * r[1];
        }
        total
    });
}

// #[bench]
fn bench_main_vwg_create_and_read(b: &mut Bencher) {
    bench_vwg(b, &VecWithGaps::empty());
}

// #[bench]
fn bench_dummy_vwg_create_and_read(b: &mut Bencher) {
    bench_vwg(b, &DummyVwg::empty());
}

// #[bench]
fn bench_main_vwg_read(b: &mut Bencher) {
    let mut v = VecWithGaps::empty();
    create_vwg(&[5, 8], &mut v);
    bench_vwg_read(b, &v);
}

#[bench]
fn bench_main_vwg_create(b: &mut Bencher){
  b.iter(||{
    let mut v = VecWithGaps::empty();
    create_vwg(&[5, 8], &mut v);
    v
  });
}

#[bench]
fn bench_dummy_vwg_create(b: &mut Bencher){
  b.iter(||{
    let mut v = DummyVwg::empty();
    create_vwg(&[5, 8], &mut v);
    v
  });
}

#[bench]
fn bench_main_vwg_read_ugly(b: &mut Bencher) {
    let mut v = VecWithGaps::empty();
    create_vwg(&[5, 8], &mut v);
    bench_vwg_ugly_read(b, &v);
}

#[bench]
fn bench_main_vwg_regular_read(b: &mut Bencher) {
    let mut v = VecWithGaps::empty();
    create_vwg(&[5, 8], &mut v);
    bench_vwg_regular_read(b, &v);
}


#[bench]
fn bench_main_vwg_read_ugly_trimmed(b: &mut Bencher) {
    let mut v = VecWithGaps::empty();
    create_vwg(&[5, 8], &mut v);
    v.trim_gaps();
    v.print_sizes();
    bench_vwg_ugly_read(b, &v);
}

#[bench]
fn bench_dummy_vwg_read(b: &mut Bencher) {
    let mut v = DummyVwg::empty();
    create_vwg(&[5, 8], &mut v);
    v.print_sizes();
    bench_vwg_read(b, &v);
}

/// uses the lego iterator, an iterator made with fairly standard high level iterator compositions instead of custom pointer twiddling. For some reason, this iterator performs 30% faster.
#[bench]
fn bench_main_vwg_read_lego(b: &mut Bencher) {
    let mut v = VecWithGaps::empty();
    create_vwg(&[5, 8], &mut v);
    bench_vwg_lego_read(b, &v);
}

#[bench]
fn bench_main_vwg_read_lego_trimmed(b: &mut Bencher) {
    let mut v = VecWithGaps::empty();
    create_vwg(&[5, 8], &mut v);
    v.trim_gaps();
    bench_vwg_lego_read(b, &v);
}
