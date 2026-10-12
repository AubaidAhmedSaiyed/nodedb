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
    use std::collections::HashSet;

    use roaring::RoaringBitmap;

    use crate::error::{VectorError, check_dim};
    use crate::hnsw::graph::{Candidate, HnswIndex, Probe};
    use crate::navix::beam::Beam;
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
    /// index dimension fails with [`VectorError::DimensionMismatch`]. A node
    /// without a vector is never returned.
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
        let probe = index.probe(query);

        let total = index.len();
        let global_sel = options.allowed.len() as f64 / total as f64;

        if global_sel < options.brute_force_threshold {
            return Ok(brute_force_on_allowed(
                index,
                &probe,
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
            current_ep = greedy_descent_layer(index, &probe, current_ep, layer, metric).id;
        }

        // Phase 2: ACORN-1 filtered beam search at layer 0.
        let ef = options.ef_search.max(options.k);
        let results = acorn_search_layer_0(index, &probe, current_ep, ef, &options.allowed, metric);

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
    ///
    /// A node without a vector is never chosen over the entry point.
    fn greedy_descent_layer(
        index: &HnswIndex,
        probe: &Probe<'_>,
        entry_point: u32,
        layer: usize,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> Candidate {
        let mut current = entry_point;
        let mut current_dist = index
            .dist_to_node(probe, current, metric)
            .unwrap_or(f32::INFINITY);

        loop {
            let mut improved = false;
            for &nb in index.neighbors_at(current, layer) {
                if index.is_deleted(nb) {
                    continue;
                }
                if let Some(d) = index.dist_to_node(probe, nb, metric)
                    && d < current_dist
                {
                    current_dist = d;
                    current = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        Candidate {
            dist: current_dist,
            id: current,
        }
    }

    /// ACORN-1 layer-0 beam search.
    ///
    /// When a 1-hop neighbor does not pass the filter, expand its 2-hop
    /// neighbors and score those (the "static 2-hop" rule).
    fn acorn_search_layer_0(
        index: &HnswIndex,
        probe: &Probe<'_>,
        entry_point: u32,
        ef: usize,
        allowed: &RoaringBitmap,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> Vec<Candidate> {
        let admitted = |id: u32| !index.is_deleted(id) && allowed.contains(id);
        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(entry_point);

        let mut beam = Beam::new(ef);
        beam.offer(
            entry_point,
            index.dist_to_node(probe, entry_point, metric),
            admitted(entry_point),
        );

        while let Some(current) = beam.pop_frontier() {
            for &nb in index.neighbors_at(current.id, 0) {
                if !visited.insert(nb) {
                    continue;
                }
                let admit = admitted(nb);
                beam.offer(nb, index.dist_to_node(probe, nb, metric), admit);
                if admit {
                    continue;
                }
                // 1-hop neighbor does not pass filter — static 2-hop expansion.
                for &nb2 in index.neighbors_at(nb, 0) {
                    if !visited.insert(nb2) {
                        continue;
                    }
                    beam.offer(nb2, index.dist_to_node(probe, nb2, metric), admitted(nb2));
                }
            }
        }

        beam.into_sorted()
    }

    fn brute_force_on_allowed(
        index: &HnswIndex,
        probe: &Probe<'_>,
        k: usize,
        allowed: &RoaringBitmap,
        metric: nodedb_types::vector_distance::DistanceMetric,
    ) -> Vec<SearchResult> {
        // A node without a vector has no distance and is not a hit.
        let mut results: Vec<SearchResult> = allowed
            .iter()
            .filter(|&id| !index.is_deleted(id))
            .filter_map(|id| {
                let distance = index.dist_to_node(probe, id, metric)?;
                Some(SearchResult { id, distance })
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

        const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

        /// Options that brute-force (`true`) or walk the graph (`false`).
        fn options(allowed: RoaringBitmap, brute: bool) -> AcornSearchOptions {
            AcornSearchOptions {
                k: 5,
                ef_search: 64,
                allowed,
                brute_force_threshold: if brute { 2.0 } else { 0.0 },
            }
        }

        fn pairs(results: &[SearchResult]) -> Vec<(u32, f32)> {
            results.iter().map(|r| (r.id, r.distance)).collect()
        }

        /// After a graph-only restore with a backing, both ACORN paths rank
        /// exactly like the source index, under a caller metric too.
        #[cfg(not(target_arch = "wasm32"))]
        #[test]
        fn acorn_after_graph_only_restore_matches_the_source_index() {
            use crate::test_support::{
                graph_only_restore_with_backing, seeded_index, seeded_vectors,
            };
            use nodedb_types::vector_dtype::VectorStorageDtype;

            let vectors = seeded_vectors(300, 8, SEED);
            let src = seeded_index(&vectors, DistanceMetric::L2, VectorStorageDtype::F32);
            let restored = graph_only_restore_with_backing(&src, &vectors);
            let allowed: RoaringBitmap = (0..300u32).step_by(3).collect();
            for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
                for brute in [true, false] {
                    let opts = options(allowed.clone(), brute);
                    let expected = acorn_search(&src, &vectors[7], &opts, metric).unwrap();
                    let got = acorn_search(&restored, &vectors[7], &opts, metric).unwrap();
                    assert_eq!(expected.len(), 5, "{metric:?} brute={brute}");
                    assert_eq!(pairs(&got), pairs(&expected), "{metric:?} brute={brute}");
                }
            }
        }

        /// A graph-only restore searched before `with_backing` has no vectors.
        /// Neither ACORN path may return its nodes.
        #[test]
        fn acorn_without_backing_returns_no_hits() {
            use crate::test_support::{graph_only_restore, seeded_index, seeded_vectors};
            use nodedb_types::vector_dtype::VectorStorageDtype;

            let vectors = seeded_vectors(300, 8, SEED);
            let src = seeded_index(&vectors, DistanceMetric::L2, VectorStorageDtype::F32);
            let idx = graph_only_restore(&src);
            let allowed: RoaringBitmap = (0..300u32).step_by(3).collect();
            for brute in [true, false] {
                let opts = options(allowed.clone(), brute);
                let res = acorn_search(&idx, &vectors[7], &opts, DistanceMetric::L2).unwrap();
                assert!(res.is_empty(), "brute={brute}: got {res:?}");
            }
        }
    }
}

#[cfg(feature = "acorn-baseline")]
pub use inner::{AcornSearchOptions, acorn_search};
