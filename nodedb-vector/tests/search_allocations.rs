// SPDX-License-Identifier: Apache-2.0

//! Heap allocations per HNSW search, counted by a global allocator.
//!
//! This file is its own test binary because it installs a
//! `#[global_allocator]`. It uses only the public API. Run it with
//! `--nocapture` to see the `ALLOC` lines.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use nodedb_types::vector_dtype::VectorStorageDtype;
use nodedb_vector::navix::{NavixSearchOptions, navix_search};
use nodedb_vector::{DistanceMetric, FilterThresholds, HnswIndex, HnswParams, adaptive_search};
use roaring::RoaringBitmap;

/// Counts `alloc` and `realloc` calls, then delegates to [`System`].
struct CountingAlloc;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: every method delegates to `System` with the caller's arguments.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller upholds the `GlobalAlloc::alloc` contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds the `GlobalAlloc::dealloc` contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller upholds the `GlobalAlloc::realloc` contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

const COUNT: usize = 2000;
const DIM: usize = 64;
const SEARCHES: u64 = 100;
const K: usize = 10;
const EF: usize = 64;

/// Deterministic vectors in `[0, 1)` from an xorshift64 stream.
fn seeded_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % 10_000) as f32 / 10_000.0
                })
                .collect()
        })
        .collect()
}

fn build(vectors: &[Vec<f32>], dtype: VectorStorageDtype) -> HnswIndex {
    let params = HnswParams {
        m: 16,
        m0: 32,
        ef_construction: 100,
        metric: DistanceMetric::L2,
        dtype,
    };
    let mut index = HnswIndex::with_seed(DIM, params, 42);
    for v in vectors {
        index.insert(v.clone()).expect("vector has the index dim");
    }
    index
}

/// Mean allocations per call of `search` over `SEARCHES` calls, after one
/// warm-up call. Each call takes the query index `i`.
fn per_search(mut search: impl FnMut(usize)) -> u64 {
    search(0);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for i in 0..SEARCHES as usize {
        search(i);
    }
    (ALLOCATIONS.load(Ordering::Relaxed) - before) / SEARCHES
}

/// The single test: one test in the binary keeps other threads out of the count.
#[test]
fn search_allocations_do_not_scale_with_distance_count() {
    let vectors = seeded_vectors(COUNT, DIM, 0x9E37_79B9_7F4A_7C15);
    let queries = seeded_vectors(SEARCHES as usize, DIM, 0xD1B5_4A32_D192_ED03);
    let f32_index = build(&vectors, VectorStorageDtype::F32);
    let f16_index = build(&vectors, VectorStorageDtype::F16);

    // 1% of ids: the adaptive search brute-forces, NaviX walks the graph.
    let sparse: RoaringBitmap = (0..COUNT as u32).step_by(100).collect();
    let half: RoaringBitmap = (0..COUNT as u32).step_by(2).collect();
    let thresholds = FilterThresholds::default();
    let navix_options = NavixSearchOptions {
        k: K,
        ef_search: EF,
        allowed: half.clone(),
        brute_force_threshold: 0.0,
    };

    for (name, index) in [("f32", &f32_index), ("f16", &f16_index)] {
        let search = per_search(|i| {
            index.search(&queries[i], K, EF).expect("search");
        });
        let filtered = per_search(|i| {
            index
                .search_filtered(&queries[i], K, EF, &half)
                .expect("filtered search");
        });
        let brute = per_search(|i| {
            adaptive_search(index, &queries[i], K, EF, &sparse, &thresholds)
                .expect("brute-force search");
        });
        let navix = per_search(|i| {
            navix_search(index, &queries[i], &navix_options, DistanceMetric::L2)
                .expect("navix search");
        });
        println!("ALLOC {name} search: {search} per search");
        println!("ALLOC {name} filtered search: {filtered} per search");
        println!("ALLOC {name} brute-force search: {brute} per search");
        println!("ALLOC {name} navix search: {navix} per search");
    }

    // Bound: an F32 search allocates a fixed set of buffers that grow by
    // doubling, so its count grows at most with log(ef). Distances allocate
    // nothing. From ef=16 to ef=256 visited nodes grow several times, but
    // allocations stay under ef=256 itself and under 4x the ef=16 count + 16.
    let narrow = per_search(|i| {
        f32_index.search(&queries[i], K, 16).expect("search");
    });
    let wide = per_search(|i| {
        f32_index.search(&queries[i], K, 256).expect("search");
    });
    println!("ALLOC f32 search ef=16: {narrow} per search");
    println!("ALLOC f32 search ef=256: {wide} per search");
    assert!(
        wide < 256,
        "ef=256 search allocates {wide} times, at least once per beam slot"
    );
    assert!(
        wide <= narrow * 4 + 16,
        "ef=256 search allocates {wide} times vs {narrow} at ef=16: count tracks visited nodes"
    );
}
