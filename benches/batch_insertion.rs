#![allow(dead_code)]
#![feature(test)]
extern crate rand;
extern crate test;

use rand::{seq::SliceRandom, Rng, SeedableRng};
use test::Bencher;
use vec_with_gaps::*;

fn fast_rng(seed: u64) -> impl Rng + Clone {
    rand::rngs::StdRng::seed_from_u64(seed)
}

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
                ret[*less_active_sections.choose(&mut rng).unwrap()].push(rng.gen_range(0..size));
                total += 1;
            }
        }
        if total >= size {
            return VecWithGaps::from_vec_vec(&ret);
        }
    }
}

fn single_insertion(b: &mut Bencher, vwg_size: usize, n_inserting: usize) {
    let vwg = create_vwg(vwg_size);
    let addition_size = vwg_size + 90; //biases additions slightly towards the end, to reflect an increasing ID space
    let mut rng = fast_rng(3);
    b.iter(move || {
        let mut v = vwg.clone();
        let vl = v.len();
        let mut inserts: Vec<Vec<usize>> = (0..vl).map(|_| Vec::new()).collect();
        for _ in 0..n_inserting {
            binary_insert_if_not_present(
                &mut inserts[rng.gen_range(0..vl)],
                rng.gen_range(0..addition_size),
            );
        }
        for (i, vs) in inserts.iter().enumerate() {
            for vsv in vs.iter() {
                v.insert_into_sorted_section_by(i, *vsv, false, |a, b| a.cmp(b)).unwrap();
            }
        }
        v
    });
}

fn binary_insert_if_not_present<V: Ord>(vs: &mut Vec<V>, p: V) {
    match vs.binary_search(&p) {
        Ok(_) => {}
        Err(i) => {
            vs.insert(i, p);
        }
    }
}

fn batch_insertion(b: &mut Bencher, vwg_size: usize, n_inserting: usize) {
    let vwg = create_vwg(vwg_size);
    let addition_size = vwg_size + 90; //biases additions slightly towards the end, to reflect an increasing ID space
    let mut rng = fast_rng(3);

    b.iter(move || {
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
            drop(v.batch_sorted_merge_insert(
                inserts
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(e, i)| (e, i.as_slice())),
            ));
        }
        v
    });
}

const NUMBER_OF_USERS: usize = 200;
const EDGES_PER_USER: usize = 13;
const DATA_TOTAL: usize = NUMBER_OF_USERS * EDGES_PER_USER;
#[bench]
fn bench_batch_insertion_one(b: &mut Bencher) {
    batch_insertion(b, DATA_TOTAL, 1);
}

#[bench]
fn bench_single_insertion_one(b: &mut Bencher) {
    single_insertion(b, DATA_TOTAL, 1);
}

#[bench]
fn bench_batch_insertion_10(b: &mut Bencher) {
    batch_insertion(b, DATA_TOTAL, 10);
}

#[bench]
fn bench_single_insertion_10(b: &mut Bencher) {
    single_insertion(b, DATA_TOTAL, 10);
}

#[bench]
fn bench_batch_insertion_50(b: &mut Bencher) {
    batch_insertion(b, DATA_TOTAL, 50);
}

#[bench]
fn bench_single_insertion_50(b: &mut Bencher) {
    single_insertion(b, DATA_TOTAL, 50);
}

#[bench]
fn bench_batch_insertion_100(b: &mut Bencher) {
    batch_insertion(b, DATA_TOTAL, 100);
}

#[bench]
fn bench_single_insertion_100(b: &mut Bencher) {
    single_insertion(b, DATA_TOTAL, 100);
}

#[bench]
fn bench_batch_insertion_200(b: &mut Bencher) {
    batch_insertion(b, DATA_TOTAL, 200);
}

#[bench]
fn bench_single_insertion_200(b: &mut Bencher) {
    single_insertion(b, DATA_TOTAL, 200);
}

#[bench]
fn bench_batch_insertion_1000(b: &mut Bencher) {
    batch_insertion(b, DATA_TOTAL, 1000);
}

#[bench]
fn bench_single_insertion_1000(b: &mut Bencher) {
    single_insertion(b, DATA_TOTAL, 1000);
}

//finding, batch insertion doesn't start to be faster until about 60 elements
