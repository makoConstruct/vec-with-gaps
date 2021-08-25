#![feature(test)]
extern crate test;
extern crate rand;

use vec_with_gaps::*;
use test::Bencher;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};


trait VWG<V>: Clone {
  fn empty()-> Self;
  fn push_section(&mut self)-> usize ;
    fn push(&mut self, section: usize, v: V);
    fn fold<F: Fn(V, &V) -> V>(&self, first: V, f: F) -> V;
}

#[derive(Clone)]
struct DummySection<V> {
    vec: Vec<V>,
}
#[derive(Clone)]
struct DummyVwg<V> {
    sections: Vec<DummySection<V>>,
}


impl<V:Clone> VWG<V> for DummyVwg<V> {
  fn empty()-> Self { Self{sections: vec!()} }
    fn push(&mut self, section: usize, v: V) {
        self.sections[section].vec.push(v);
    }
    fn push_section(&mut self)-> usize {
      let ret = self.sections.len();
      self.sections.push(DummySection{vec:Vec::new()});
      ret
    }
    fn fold<F: Fn(V, &V) -> V>(&self, first: V, f: F) -> V {
        self.sections
            .iter()
            .flat_map(|s| s.vec.iter())
            .fold(first, f)
    }
}


impl<V> VWG<V> for VecWithGaps<V> {
  fn empty()-> Self { Self::new() }
  fn push_section(&mut self)-> usize {
    self.push_section_after_gap(0)
  }
    fn push(&mut self, section: usize, v: V) {
        self.push_into_section(section, v);
    }
    fn fold<F: Fn(V, &V) -> V>(&self, first: V, f: F) -> V {
        self.iter().fold(first, f)
    }
}



fn create_vwg<TV: VWG<V>, V:Clone, RNG: Rng>(rng:&mut RNG, to_push:&V, src:&TV)-> TV {
  let mut v = src.clone();
  let mut active_sections:Vec<usize> = Vec::new();
  let mut less_active_sections:Vec<usize> = Vec::new();
  const EVENT_COUNT:usize = 49000;
  let eventr:f64 = rng.gen();
  for _ in 0..EVENT_COUNT {
    if eventr < 0.008 { //create a new section
      let si = v.push_section();
      // start it out with something in it
      
      if rng.gen::<f64>() < 0.7 {
        less_active_sections.push(si);
      }else{
        active_sections.push(si);
      }
    }else if eventr < 0.83 { //add to an active one
      if active_sections.len() != 0 {
        v.push(*active_sections.choose(rng).unwrap(), to_push.clone());
      }
    }else { //add to an old section
      if less_active_sections.len() != 0 {
        v.push(*less_active_sections.choose(rng).unwrap(), to_push.clone());
      }
    }
  }
  v
}

fn bench_vwg<TV: VWG<[usize;2]>, RNG:Rng + Clone>(b: &mut Bencher, rng:&mut RNG, ve:&TV) {
  let to_push = [1,2];
  b.iter(|| {
    let mut r = rng.clone();
    let v: TV = create_vwg(&mut r, &to_push, ve);
    
    const READ_COUNT:usize = 210;
    let mut total = 0;
    for _ in 0..READ_COUNT {
      total += v.fold(to_push, |a, b| [a[0] + b[0], a[1] + b[1]])[0];
    }
    total
  });
}

#[bench]
fn bench_main_vwg(b: &mut test::Bencher) {
  let mut r = StdRng::seed_from_u64(780);
  bench_vwg(b, &mut r, &VecWithGaps::empty());
}

#[bench]
fn bench_dummy_vwg(b: &mut test::Bencher) {
  let mut r = StdRng::seed_from_u64(780);
  bench_vwg(b, &mut r, &DummyVwg::empty());
}



fn straw_bench<TV: VWG<[usize;2]>, RNG:Rng + Clone>(b: &mut Bencher, rng:&mut RNG, ve:&TV) {
  
}