// SPDX-License-Identifier: Apache-2.0

//! ACORN-1 static 2-hop filtered traversal — feature-flagged baseline.
//!
//! Retained **only** for benchmarking against NaviX.  When a 1-hop neighbor
//! fails the filter, ACORN-1 always expands to that node's 2-hop neighbors and
//! scores those against the filter.  There is no per-hop adaptive switch;
//! 2-hop expansion is unconditional.
//!
//! Enable with `--features acorn-baseline`.

#[cfg(feature = "acorn-baseline")]
mod inner {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashSet};

    use roaring::RoaringBitmap;

    use crate::distance::distance;
    use crate::error::{VectorError, check_dim};
    use crate::hnsw::graph::{Candidate, HnswIndex};
    use crate::navix::traversal::SearchResult;

    /// Options for an ACORN-1 search (same shape as `NavixSearchOptions`).
    pub struct AcornSearchOptions {
        pub k: usize,
        pub ef_search: usize,
        pub allowed: RoaringBitmap,
        /// Brute-force fallback threshold (same semantics as NaviX).
        pub brute_force_threshold: f64,
    }

    impl Default for AcornSearchOptions {
        fn default() -> Self {
            Self {
                k: 10,
                ef_search: 64,
                allowed: RoaringBitmap::new(),
                brute_force_threshold: 0.001,
            }
        }
    }

    /// ACORN-1 filtered search.
    ///
    /// Uses static 2-hop expansion: when a 1-hop neighbor is not in `allowed`,
    /// expand to its 2-hop neighbors unconditionally. A query without the
    /// index dimension fails with [`VectorError::DimensionMismatch`].
    pub fn acorn_search(
        index: &HnswIndex,
        query: &[f32],
        options: &AcornSearchOptions,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> Result<Vec<SearchResult>, VectorError> {
        check_dim(index.dim(), query.len())?;
        if index.is_empty() || options.allowed.is_empty() || options.k == 0 {
            return Ok(Vec::new());
        }

        let total = index.len();
        let global_sel = options.allowed.len() as f64 / total as f64;

        if global_sel < options.brute_force_threshold {
            return Ok(brute_force_on_allowed(
                index,
                query,
                options.k,
                &options.allowed,
                metric,
            ));
        }

        let Some(ep) = index.entry_point() else {
            return Ok(Vec::new());
        };

        // Phase 1: greedy descent (unfiltered) to find best layer-0 entry.
        let mut current_ep = ep;
        for layer in (1..=index.max_layer()).rev() {
            let results = greedy_descent_layer(index, query, current_ep, layer, metric);
            if let Some(nearest) = results.first() {
                current_ep = nearest.id;
            }
        }

        // Phase 2: ACORN-1 filtered beam search at layer 0.
        let ef = options.ef_search.max(options.k);
        let results = acorn_search_layer_0(index, query, current_ep, ef, &options.allowed, metric);

        Ok(results
            .into_iter()
            .take(options.k)
            .map(|c| SearchResult {
                id: c.id,
                distance: c.dist,
            })
            .collect())
    }

    /// Minimal greedy single-layer descent used for Phase-1 layer navigation.
    fn greedy_descent_layer(
        index: &HnswIndex,
        query: &[f32],
        entry_point: u32,
        layer: usize,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> Vec<Candidate> {
        let mut current = entry_point;
        let mut current_dist = node_dist(index, query, current, metric);

        loop {
            let mut improved = false;
            for &nb in index.neighbors_at(current, layer) {
                if index.is_deleted(nb) {
                    continue;
                }
                let d = node_dist(index, query, nb, metric);
                if d < current_dist {
                    current_dist = d;
                    current = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        vec![Candidate {
            dist: current_dist,
            id: current,
        }]
    }

    /// ACORN-1 layer-0 beam search.
    ///
    /// When a 1-hop neighbor does not pass the filter, expand its 2-hop
    /// neighbors and score those (the "static 2-hop" rule).
    fn acorn_search_layer_0(
        index: &HnswIndex,
        query: &[f32],
        entry_point: u32,
        ef: usize,
        allowed: &RoaringBitmap,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(entry_point);

        let ep_dist = node_dist(index, query, entry_point, metric);
        let ep_cand = Candidate {
            dist: ep_dist,
            id: entry_point,
        };

        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        candidates.push(Reverse(ep_cand));

        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        if !index.is_deleted(entry_point) && allowed.contains(entry_point) {
            results.push(ep_cand);
        }

        while let Some(Reverse(current)) = candidates.pop() {
            if let Some(worst) = results.peek()
                && current.dist > worst.dist
                && results.len() >= ef
            {
                break;
            }

            for &nb in index.neighbors_at(current.id, 0) {
                if !visited.insert(nb) {
                    continue;
                }

                let d = node_dist(index, query, nb, metric);
                let nb_cand = Candidate { dist: d, id: nb };
                let worst_dist = results.peek().map_or(f32::INFINITY, |w| w.dist);

                if d < worst_dist || results.len() < ef {
                    candidates.push(Reverse(nb_cand));
                }

                if !index.is_deleted(nb) && allowed.contains(nb) {
                    // 1-hop neighbor passes filter — add to results.
                    results.push(nb_cand);
                    if results.len() > ef {
                        results.pop();
                    }
                } else {
                    // 1-hop neighbor does not pass filter — static 2-hop expansion.
                    for &nb2 in index.neighbors_at(nb, 0) {
                        if !visited.insert(nb2) {
                            continue;
                        }
                        let d2 = node_dist(index, query, nb2, metric);
                        let nb2_cand = Candidate { dist: d2, id: nb2 };
                        let worst_dist2 = results.peek().map_or(f32::INFINITY, |w| w.dist);
                        if d2 < worst_dist2 || results.len() < ef {
                            candidates.push(Reverse(nb2_cand));
                        }
                        if !index.is_deleted(nb2) && allowed.contains(nb2) {
                            results.push(nb2_cand);
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut v: Vec<Candidate> = results.into_vec();
        v.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        v
    }

    fn brute_force_on_allowed(
        index: &HnswIndex,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = allowed
            .iter()
            .filter_map(|id| {
                if index.is_deleted(id) {
                    return None;
                }
                let v = index.get_vector(id)?;
                Some(SearchResult {
                    id,
                    distance: distance(query, v, metric),
                })
            })
            .collect();

        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
        }
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    #[inline]
    fn node_dist(
        index: &HnswIndex,
        query: &[f32],
        node_id: u32,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> f32 {
        match index.get_vector(node_id) {
            Some(v) => distance(query, v, metric),
            None => f32::INFINITY,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::distance::DistanceMetric;
        use crate::hnsw::{HnswIndex, HnswParams};

        fn build_index(n: usize) -> HnswIndex {
            let mut idx = HnswIndex::with_seed(
                3,
                HnswParams {
                    m: 8,
                    m0: 16,
                    ef_construction: 50,
                    metric: DistanceMetric::L2,
                    dtype: nodedb_types::vector_dtype::VectorStorageDtype::F32,
                },
                42,
            );
            for i in 0..n {
                idx.insert(vec![i as f32, 0.0, 0.0]).unwrap();
            }
            idx
        }

        /// Mid-selectivity (~50%) — all results must be in the allowed set.
        #[test]
        fn acorn_results_subset_of_allowed() {
            let idx = build_index(20);
            let query = [10.0f32, 0.0, 0.0];

            let mut allowed = RoaringBitmap::new();
            for i in (0..20u32).step_by(2) {
                allowed.insert(i);
            }

            let opts = AcornSearchOptions {
                k: 5,
                ef_search: 64,
                allowed: allowed.clone(),
                brute_force_threshold: 0.001,
            };

            let res = acorn_search(&idx, &query, &opts, DistanceMetric::L2).unwrap();
            assert!(!res.is_empty());
            for r in &res {
                assert!(
                    allowed.contains(r.id),
                    "ACORN returned disallowed id {}",
                    r.id
                );
            }
        }

        #[test]
        fn acorn_wrong_dimension_query_is_a_typed_error() {
            let idx = build_index(20);
            let opts = AcornSearchOptions {
                k: 3,
                ef_search: 64,
                allowed: (0..20u32).collect(),
                brute_force_threshold: 0.001,
            };
            assert!(matches!(
                acorn_search(&idx, &[1.0], &opts, DistanceMetric::L2),
                Err(VectorError::DimensionMismatch {
                    expected: 3,
                    got: 1
                })
            ));
        }

        /// Very low selectivity (1 ID out of 20) — result must be that single ID.
        #[test]
        fn acorn_single_allowed_id() {
            let idx = build_index(20);
            let query = [3.0f32, 0.0, 0.0];

            let mut allowed = RoaringBitmap::new();
            allowed.insert(7);

            let opts = AcornSearchOptions {
                k: 5,
                ef_search: 64,
                allowed,
                brute_force_threshold: 0.001,
            };

            let res = acorn_search(&idx, &query, &opts, DistanceMetric::L2).unwrap();
            assert!(res.len() <= 1);
            if let Some(r) = res.first() {
                assert_eq!(r.id, 7);
            }
        }
    }
}

#[cfg(feature = "acorn-baseline")]
pub use inner::{AcornSearchOptions, acorn_search};
