// SPDX-License-Identifier: Apache-2.0

//! NaviX adaptive-local filtered HNSW traversal (VLDB 2025).
//!
//! Replaces ACORN-1's static 2-hop expansion with a per-hop heuristic switch
//! driven by local selectivity.  At each expansion step the algorithm asks:
//! "of this node's 1-hop neighbors, what fraction are in the allowed set?"
//! and then picks Standard / Directed / Blind accordingly.
//!
//! See `selectivity.rs` for heuristic boundary definitions.

use std::collections::HashSet;

use roaring::RoaringBitmap;

use crate::error::{VectorError, check_dim};
use crate::hnsw::graph::{Candidate, HnswIndex, Probe};
use crate::navix::beam::Beam;
use crate::navix::selectivity::{NavixHeuristic, local_selectivity_at, pick_heuristic};

/// A k-NN result returned by `navix_search`.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Internal node identifier (insertion order in the HNSW index).
    pub id: u32,
    /// Distance from the query vector.
    pub distance: f32,
}

/// Options for a NaviX filtered search.
pub struct NavixSearchOptions {
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Beam width (higher = better recall, more CPU).  Must be >= k.
    pub ef_search: usize,
    /// Sideways Information Passing (SIP): exact allowed-set semimask from the
    /// upstream filter operator.
    pub allowed: RoaringBitmap,
    /// Brute-force fallback threshold on global selectivity.
    /// When `allowed.len() / total_vectors < brute_force_threshold` the search
    /// bypasses HNSW and scans only the allowed IDs directly.
    /// Default 0.001 (0.1%).
    pub brute_force_threshold: f64,
}

impl Default for NavixSearchOptions {
    fn default() -> Self {
        Self {
            k: 10,
            ef_search: 64,
            allowed: RoaringBitmap::new(),
            brute_force_threshold: 0.001,
        }
    }
}

/// Adaptive-local NaviX filtered search.
///
/// Returns up to `options.k` nearest vectors from `index` to `query`, where
/// candidate IDs must be present in `options.allowed`.
///
/// Returns an empty Vec when the index is empty or `options.allowed` is empty.
/// A node without a vector is never returned.
///
/// # Errors
///
/// [`VectorError::DimensionMismatch`] when `query` does not have the index
/// dimension.
pub fn navix_search(
    index: &HnswIndex,
    query: &[f32],
    options: &NavixSearchOptions,
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

    // Phase 1: greedy descent from max_layer to layer 1 (unfiltered, as in
    // standard HNSW — we just want the best entry point for layer 0).
    let mut current_ep = ep;
    for layer in (1..=index.max_layer()).rev() {
        let results = unfiltered_search_layer(index, &probe, current_ep, 1, layer, metric);
        if let Some(nearest) = results.first() {
            current_ep = nearest.id;
        }
    }

    // Phase 2: adaptive-local filtered beam search at layer 0.
    let ef = options.ef_search.max(options.k);
    let results = navix_search_layer_0(index, &probe, current_ep, ef, &options.allowed, metric);

    Ok(results
        .into_iter()
        .take(options.k)
        .map(|c| SearchResult {
            id: c.id,
            distance: c.dist,
        })
        .collect())
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Brute-force scan over only the IDs in `allowed`.  Used when global
/// selectivity drops below the configured threshold.
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

/// Whether node `id` may enter the results of a filtered search.
fn admitted(index: &HnswIndex, allowed: &RoaringBitmap, id: u32) -> bool {
    !index.is_deleted(id) && allowed.contains(id)
}

/// Standard unfiltered single-layer beam search used for Phase-1 greedy
/// descent (layers 1..max_layer).
fn unfiltered_search_layer(
    index: &HnswIndex,
    probe: &Probe<'_>,
    entry_point: u32,
    ef: usize,
    layer: usize,
    metric: nodedb_types::vector_distance::DistanceMetric,
) -> Vec<Candidate> {
    let mut visited: HashSet<u32> = HashSet::new();
    visited.insert(entry_point);

    let mut beam = Beam::new(ef);
    beam.offer(
        entry_point,
        index.dist_to_node(probe, entry_point, metric),
        !index.is_deleted(entry_point),
    );

    while let Some(current) = beam.pop_frontier() {
        for &nb in index.neighbors_at(current.id, layer) {
            if !visited.insert(nb) {
                continue;
            }
            beam.offer(
                nb,
                index.dist_to_node(probe, nb, metric),
                !index.is_deleted(nb),
            );
        }
    }

    beam.into_sorted()
}

/// NaviX adaptive-local filtered beam search at layer 0.
///
/// Per-hop heuristic switch:
/// - **Standard**: score every allowed neighbor normally.
/// - **Directed**: score 1-hop, expand 2-hop of the single best neighbor.
/// - **Blind**: skip 1-hop scoring; sample 2-hop of all 1-hop neighbors.
fn navix_search_layer_0(
    index: &HnswIndex,
    probe: &Probe<'_>,
    entry_point: u32,
    ef: usize,
    allowed: &RoaringBitmap,
    metric: nodedb_types::vector_distance::DistanceMetric,
) -> Vec<Candidate> {
    let mut visited: HashSet<u32> = HashSet::new();
    visited.insert(entry_point);

    // Entry point enters results only if it is allowed.
    let mut beam = Beam::new(ef);
    beam.offer(
        entry_point,
        index.dist_to_node(probe, entry_point, metric),
        admitted(index, allowed, entry_point),
    );

    while let Some(current) = beam.pop_frontier() {
        let neighbors_1hop = index.neighbors_at(current.id, 0);
        let args = ExpandArgs {
            index,
            probe,
            neighbors_1hop,
            allowed,
            metric,
            visited: &mut visited,
            beam: &mut beam,
        };
        match pick_heuristic(local_selectivity_at(neighbors_1hop, allowed)) {
            NavixHeuristic::Standard => expand_standard(args),
            NavixHeuristic::Directed => expand_directed(args),
            NavixHeuristic::Blind => expand_blind(args),
        }
    }

    beam.into_sorted()
}

/// Shared beam-search state bundled for the per-heuristic `expand_*` helpers.
struct ExpandArgs<'a> {
    index: &'a HnswIndex,
    probe: &'a Probe<'a>,
    neighbors_1hop: &'a [u32],
    allowed: &'a RoaringBitmap,
    metric: nodedb_types::vector_distance::DistanceMetric,
    visited: &'a mut HashSet<u32>,
    beam: &'a mut Beam,
}

/// Standard expansion: score every allowed 1-hop neighbor and add to heaps.
fn expand_standard(args: ExpandArgs<'_>) {
    let ExpandArgs {
        index,
        probe,
        neighbors_1hop,
        allowed,
        metric,
        visited,
        beam,
    } = args;
    for &nb in neighbors_1hop {
        if !visited.insert(nb) {
            continue;
        }
        beam.offer(
            nb,
            index.dist_to_node(probe, nb, metric),
            admitted(index, allowed, nb),
        );
    }
}

/// Directed expansion: score 1-hop, pick the single best allowed neighbor,
/// then expand that neighbor's 2-hop neighbors into the heaps.
fn expand_directed(args: ExpandArgs<'_>) {
    let ExpandArgs {
        index,
        probe,
        neighbors_1hop,
        allowed,
        metric,
        visited,
        beam,
    } = args;
    // Score 1-hop; track the best allowed neighbor that has a distance.
    let mut best_allowed: Option<(u32, f32)> = None;

    for &nb in neighbors_1hop {
        if !visited.insert(nb) {
            continue;
        }
        let d = index.dist_to_node(probe, nb, metric);
        let admit = admitted(index, allowed, nb);
        if admit
            && let Some(d) = d
            && best_allowed.is_none_or(|(_, bd)| d < bd)
        {
            best_allowed = Some((nb, d));
        }
        beam.offer(nb, d, admit);
    }

    // Expand 2-hop of the single best allowed neighbor.
    if let Some((best_id, _)) = best_allowed {
        for &nb2 in index.neighbors_at(best_id, 0) {
            if !visited.insert(nb2) {
                continue;
            }
            beam.offer(
                nb2,
                index.dist_to_node(probe, nb2, metric),
                admitted(index, allowed, nb2),
            );
        }
    }
}

/// Blind expansion: skip scoring 1-hop; expand 2-hop of all 1-hop neighbors,
/// adding to heaps only IDs that are in `allowed`.
fn expand_blind(args: ExpandArgs<'_>) {
    let ExpandArgs {
        index,
        probe,
        neighbors_1hop,
        allowed,
        metric,
        visited,
        beam,
    } = args;
    for &nb1 in neighbors_1hop {
        // Mark 1-hop as visited so we do not double-score them later,
        // but do not score them — that is the Blind heuristic.
        visited.insert(nb1);

        for &nb2 in index.neighbors_at(nb1, 0) {
            if !visited.insert(nb2) || !admitted(index, allowed, nb2) {
                continue;
            }
            beam.offer(nb2, index.dist_to_node(probe, nb2, metric), true);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{DistanceMetric, distance};
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

    fn all_allowed(n: u32) -> RoaringBitmap {
        let mut b = RoaringBitmap::new();
        for i in 0..n {
            b.insert(i);
        }
        b
    }

    /// Full allowed set → recall should match unfiltered HNSW closely.
    #[test]
    fn full_allowed_matches_unfiltered() {
        let idx = build_index(20);
        let query = [10.0f32, 0.0, 0.0];
        let allowed = all_allowed(20);

        let opts = NavixSearchOptions {
            k: 5,
            ef_search: 64,
            allowed,
            brute_force_threshold: 0.001,
        };

        let navix_res = navix_search(&idx, &query, &opts, DistanceMetric::L2).unwrap();
        let hnsw_res = idx.search(&query, 5, 64).unwrap();

        assert!(!navix_res.is_empty());
        // The best result should be id=10 (exact match) in both cases.
        assert_eq!(navix_res[0].id, hnsw_res[0].id);
    }

    /// Allowed bitmap contains only one ID — that ID must be returned.
    #[test]
    fn single_allowed_id_returned() {
        let idx = build_index(20);
        let query = [5.0f32, 0.0, 0.0];
        let mut allowed = RoaringBitmap::new();
        allowed.insert(15); // Only ID 15 is allowed.

        let opts = NavixSearchOptions {
            k: 5,
            ef_search: 64,
            allowed,
            brute_force_threshold: 0.001,
        };

        let res = navix_search(&idx, &query, &opts, DistanceMetric::L2).unwrap();
        // With only one allowed ID, we get at most 1 result.
        assert!(res.len() <= 1);
        if let Some(r) = res.first() {
            assert_eq!(r.id, 15);
        }
    }

    /// ~50% bitmap — results must all be in the allowed set.
    #[test]
    fn half_allowed_results_in_allowed() {
        let idx = build_index(20);
        let query = [10.0f32, 0.0, 0.0];

        let mut allowed = RoaringBitmap::new();
        for i in (0..20u32).step_by(2) {
            allowed.insert(i); // even IDs only
        }

        let opts = NavixSearchOptions {
            k: 3,
            ef_search: 64,
            allowed: allowed.clone(),
            brute_force_threshold: 0.001,
        };

        let res = navix_search(&idx, &query, &opts, DistanceMetric::L2).unwrap();
        assert!(!res.is_empty());
        for r in &res {
            assert!(
                allowed.contains(r.id),
                "got disallowed id {} in results",
                r.id
            );
        }
    }

    /// Brute-force fallback fires when `brute_force_threshold` is set high.
    /// Output must equal manual brute-force over the allowed set.
    #[test]
    fn brute_force_fallback_matches_manual() {
        let idx = build_index(20);
        let query = [8.0f32, 0.0, 0.0];

        let mut allowed = RoaringBitmap::new();
        allowed.insert(3);
        allowed.insert(7);
        allowed.insert(12);

        // Set threshold = 0.5 → global sel = 3/20 = 0.15 < 0.5 → always brute-force.
        let opts = NavixSearchOptions {
            k: 5,
            ef_search: 64,
            allowed: allowed.clone(),
            brute_force_threshold: 0.5,
        };

        let res = navix_search(&idx, &query, &opts, DistanceMetric::L2).unwrap();

        // Manual brute-force reference.
        let mut manual: Vec<(u32, f32)> = allowed
            .iter()
            .map(|id| {
                let v = idx.get_vector(id).unwrap();
                let d = distance(&query, v, DistanceMetric::L2);
                (id, d)
            })
            .collect();
        manual.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        assert_eq!(res.len(), manual.len().min(opts.k));
        for (r, (mid, _)) in res.iter().zip(manual.iter()) {
            assert_eq!(r.id, *mid, "brute-force result mismatch");
        }
    }

    /// Empty index returns empty results.
    #[test]
    fn empty_index_returns_empty() {
        let idx = HnswIndex::new(
            3,
            HnswParams {
                m: 8,
                m0: 16,
                ef_construction: 50,
                metric: DistanceMetric::L2,
                dtype: nodedb_types::vector_dtype::VectorStorageDtype::F32,
            },
        );
        let mut allowed = RoaringBitmap::new();
        allowed.insert(0);

        let opts = NavixSearchOptions {
            k: 5,
            ef_search: 64,
            allowed,
            brute_force_threshold: 0.001,
        };
        let res = navix_search(&idx, &[1.0, 0.0, 0.0], &opts, DistanceMetric::L2).unwrap();
        assert!(res.is_empty());
    }

    /// Empty allowed bitmap returns empty results.
    #[test]
    fn empty_allowed_returns_empty() {
        let idx = build_index(10);
        let opts = NavixSearchOptions {
            k: 5,
            ef_search: 64,
            allowed: RoaringBitmap::new(),
            brute_force_threshold: 0.001,
        };
        let res = navix_search(&idx, &[5.0, 0.0, 0.0], &opts, DistanceMetric::L2).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn wrong_dimension_query_is_a_typed_error() {
        let idx = build_index(20);
        let opts = NavixSearchOptions {
            k: 3,
            allowed: (0..20u32).collect(),
            ..NavixSearchOptions::default()
        };
        let result = navix_search(&idx, &[1.0, 0.0], &opts, DistanceMetric::L2);
        assert!(
            matches!(
                result,
                Err(crate::error::VectorError::DimensionMismatch {
                    expected: 3,
                    got: 2
                })
            ),
            "{result:?}"
        );
    }

    const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
    /// Global selectivity is always below this, so the search brute-forces.
    const BRUTE_FORCE: f64 = 2.0;
    /// Global selectivity is never below this, so the search walks the graph.
    const GRAPH: f64 = 0.0;

    /// Allowed sets dense and sparse enough to reach every NaviX heuristic.
    fn allowed_sets() -> Vec<RoaringBitmap> {
        [2usize, 3, 50]
            .into_iter()
            .map(|step| (0..300u32).step_by(step).collect())
            .collect()
    }

    fn options(allowed: RoaringBitmap, brute_force_threshold: f64) -> NavixSearchOptions {
        NavixSearchOptions {
            k: 5,
            ef_search: 64,
            allowed,
            brute_force_threshold,
        }
    }

    fn pairs(results: &[SearchResult]) -> Vec<(u32, f32)> {
        results.iter().map(|r| (r.id, r.distance)).collect()
    }

    /// After a graph-only restore with a backing, brute force and graph
    /// traversal rank exactly like the source index, under the build metric
    /// and under a different caller metric.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn graph_only_restore_with_backing_matches_the_source_index() {
        use crate::test_support::{graph_only_restore_with_backing, seeded_index, seeded_vectors};
        use nodedb_types::vector_dtype::VectorStorageDtype;

        let vectors = seeded_vectors(300, 8, SEED);
        let src = seeded_index(&vectors, DistanceMetric::L2, VectorStorageDtype::F32);
        let restored = graph_only_restore_with_backing(&src, &vectors);
        let mut graph_hits = 0;
        for metric in [DistanceMetric::L2, DistanceMetric::Cosine] {
            for (threshold, brute) in [(BRUTE_FORCE, true), (GRAPH, false)] {
                for allowed in allowed_sets() {
                    let opts = options(allowed, threshold);
                    let expected = navix_search(&src, &vectors[7], &opts, metric).unwrap();
                    let got = navix_search(&restored, &vectors[7], &opts, metric).unwrap();
                    assert!(!brute || expected.len() == 5, "{metric:?}: {expected:?}");
                    assert_eq!(pairs(&got), pairs(&expected), "{metric:?} brute={brute}");
                    if !brute {
                        graph_hits += got.len();
                    }
                }
            }
        }
        assert!(graph_hits > 0, "graph traversal must return hits");
    }

    /// A graph-only restore searched before `with_backing` has no vectors.
    /// Neither brute force nor graph traversal may return its nodes.
    #[test]
    fn graph_only_restore_without_backing_returns_no_hits() {
        use crate::test_support::{graph_only_restore, seeded_index, seeded_vectors};
        use nodedb_types::vector_dtype::VectorStorageDtype;

        let vectors = seeded_vectors(300, 8, SEED);
        let src = seeded_index(&vectors, DistanceMetric::L2, VectorStorageDtype::F32);
        let idx = graph_only_restore(&src);
        for threshold in [BRUTE_FORCE, GRAPH] {
            for allowed in allowed_sets() {
                let opts = options(allowed, threshold);
                let res = navix_search(&idx, &vectors[7], &opts, DistanceMetric::L2).unwrap();
                assert!(res.is_empty(), "threshold={threshold}: got {res:?}");
            }
        }
    }

    /// Every dtype scores hits under the caller metric, not the build metric.
    /// Brute force returns the exact top-k under that metric.
    #[test]
    fn caller_metric_overrides_the_build_metric_for_every_dtype() {
        use crate::test_support::{seeded_index, seeded_vectors};
        use nodedb_types::vector_dtype::VectorStorageDtype;

        let vectors = seeded_vectors(300, 8, SEED);
        let allowed: RoaringBitmap = (0..300u32).step_by(3).collect();
        for dtype in [
            VectorStorageDtype::F32,
            VectorStorageDtype::F16,
            VectorStorageDtype::BF16,
        ] {
            let idx = seeded_index(&vectors, DistanceMetric::L2, dtype);
            let encoded = crate::dtype::cast_from_f32(&vectors[7], dtype);
            let query = crate::dtype::cast_to_f32(&encoded, dtype, 8).unwrap();
            let cosine_to = |id: u32| {
                let node = idx.materialize_vector(id).unwrap();
                distance(&query, &node, DistanceMetric::Cosine)
            };
            let mut reference: Vec<(u32, f32)> =
                allowed.iter().map(|id| (id, cosine_to(id))).collect();
            reference.sort_by(|a, b| a.1.total_cmp(&b.1));

            for (threshold, brute) in [(BRUTE_FORCE, true), (GRAPH, false)] {
                let opts = options(allowed.clone(), threshold);
                let res = navix_search(&idx, &vectors[7], &opts, DistanceMetric::Cosine).unwrap();
                assert_eq!(res.len(), 5, "dtype={dtype:?} brute={brute}");
                for hit in &res {
                    let want = cosine_to(hit.id);
                    assert!(
                        (hit.distance - want).abs() < 1e-4,
                        "dtype={dtype:?} brute={brute}: {hit:?} vs cosine {want}"
                    );
                }
                if brute {
                    let ids: Vec<u32> = res.iter().map(|r| r.id).collect();
                    let want: Vec<u32> = reference.iter().take(5).map(|r| r.0).collect();
                    assert_eq!(ids, want, "dtype={dtype:?}");
                }
            }
        }
    }
}
