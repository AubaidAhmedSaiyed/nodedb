// SPDX-License-Identifier: Apache-2.0

//! Adaptive filtered search strategy selection.
//!
//! Selects the optimal search strategy based on filter selectivity:
//! - **Low selectivity (<50% filtered out):** pre-filter during HNSW traversal.
//! - **High selectivity (50-95% filtered out):** post-filter with over-fetch.
//! - **Extreme selectivity (>95% filtered out):** brute-force on matching IDs.

use roaring::RoaringBitmap;

use crate::distance::distance;
use crate::hnsw::{HnswIndex, SearchResult};

/// Filter strategy thresholds.
pub struct FilterThresholds {
    /// Below this selectivity, use pre-filter (default strategy).
    pub high_selectivity: f64,
    /// Above this selectivity, use brute-force on matching IDs.
    pub extreme_selectivity: f64,
}

impl Default for FilterThresholds {
    fn default() -> Self {
        Self {
            high_selectivity: 0.50,
            extreme_selectivity: 0.95,
        }
    }
}

/// Selected search strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum FilterStrategy {
    /// HNSW traversal with bitmap pre-filter (standard).
    PreFilter,
    /// HNSW unfiltered with k×10 over-fetch, then post-filter.
    PostFilter { over_fetch_factor: usize },
    /// Skip HNSW. Brute-force distance on matching IDs only.
    BruteForceMatching,
}

/// Estimate filter selectivity from a Roaring bitmap.
///
/// Returns the fraction of vectors that are filtered OUT (not matching).
pub fn estimate_selectivity(bitmap: &RoaringBitmap, total_vectors: usize) -> f64 {
    if total_vectors == 0 {
        return 0.0;
    }
    let matching = bitmap.len() as usize;
    1.0 - (matching as f64 / total_vectors as f64)
}

/// Select the optimal search strategy based on filter selectivity.
pub fn select_strategy(selectivity: f64, thresholds: &FilterThresholds) -> FilterStrategy {
    if selectivity >= thresholds.extreme_selectivity {
        FilterStrategy::BruteForceMatching
    } else if selectivity >= thresholds.high_selectivity {
        FilterStrategy::PostFilter {
            over_fetch_factor: 10,
        }
    } else {
        FilterStrategy::PreFilter
    }
}

/// Execute adaptive filtered search on an HNSW index.
///
/// A query without the index dimension fails with
/// [`VectorError::DimensionMismatch`](crate::error::VectorError::DimensionMismatch)
/// under every strategy.
pub fn adaptive_search(
    index: &HnswIndex,
    query: &[f32],
    top_k: usize,
    ef: usize,
    bitmap: &RoaringBitmap,
    thresholds: &FilterThresholds,
) -> Result<Vec<SearchResult>, crate::error::VectorError> {
    crate::error::check_dim(index.dim(), query.len())?;
    let total = index.len();
    let selectivity = estimate_selectivity(bitmap, total);
    let strategy = select_strategy(selectivity, thresholds);

    match strategy {
        FilterStrategy::PreFilter => index.search_filtered(query, top_k, ef, bitmap),
        FilterStrategy::PostFilter { over_fetch_factor } => {
            let fetch_k = top_k * over_fetch_factor;
            let results = index.search(query, fetch_k, ef.max(fetch_k))?;
            let mut filtered: Vec<SearchResult> = results
                .into_iter()
                .filter(|r| bitmap.contains(r.id))
                .collect();
            filtered.truncate(top_k);
            Ok(filtered)
        }
        FilterStrategy::BruteForceMatching => {
            let metric = index.params().metric;
            let mut results: Vec<SearchResult> = bitmap
                .iter()
                .filter_map(|id| {
                    let v = index.get_vector(id)?;
                    if index.is_deleted(id) {
                        return None;
                    }
                    Some(SearchResult {
                        id,
                        distance: distance(query, v, metric),
                    })
                })
                .collect();

            if results.len() > top_k {
                results.select_nth_unstable_by(top_k, |a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(top_k);
            }
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(results)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use crate::hnsw::{HnswIndex, HnswParams};

    fn build_test_index() -> HnswIndex {
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
        for i in 0..1000 {
            idx.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }
        idx
    }

    #[test]
    fn low_selectivity_uses_prefilter() {
        let thresholds = FilterThresholds::default();
        let strategy = select_strategy(0.2, &thresholds);
        assert_eq!(strategy, FilterStrategy::PreFilter);
    }

    #[test]
    fn high_selectivity_uses_postfilter() {
        let thresholds = FilterThresholds::default();
        let strategy = select_strategy(0.8, &thresholds);
        assert!(matches!(strategy, FilterStrategy::PostFilter { .. }));
    }

    #[test]
    fn extreme_selectivity_uses_bruteforce() {
        let thresholds = FilterThresholds::default();
        let strategy = select_strategy(0.99, &thresholds);
        assert_eq!(strategy, FilterStrategy::BruteForceMatching);
    }

    #[test]
    fn adaptive_search_extreme_filter() {
        let idx = build_test_index();
        let thresholds = FilterThresholds::default();

        let mut bitmap = RoaringBitmap::new();
        for i in 500..510 {
            bitmap.insert(i);
        }

        let results =
            adaptive_search(&idx, &[505.0, 0.0, 0.0], 3, 64, &bitmap, &thresholds).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(bitmap.contains(r.id), "got filtered-out id {}", r.id);
        }
        assert_eq!(results[0].id, 505);
    }

    #[test]
    fn adaptive_search_low_filter() {
        let idx = build_test_index();
        let thresholds = FilterThresholds::default();

        let mut bitmap = RoaringBitmap::new();
        for i in 0..800 {
            bitmap.insert(i);
        }

        let results =
            adaptive_search(&idx, &[100.0, 0.0, 0.0], 5, 64, &bitmap, &thresholds).unwrap();
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(bitmap.contains(r.id));
        }
    }

    #[test]
    fn selectivity_estimation() {
        let mut bitmap = RoaringBitmap::new();
        for i in 0..100 {
            bitmap.insert(i);
        }
        let sel = estimate_selectivity(&bitmap, 1000);
        assert!((sel - 0.9).abs() < 0.01);
    }

    /// Every strategy refuses a query of the wrong dimension, including the
    /// brute-force path that never touches the graph.
    #[test]
    fn wrong_dimension_query_is_a_typed_error() {
        let idx = build_test_index();
        let thresholds = FilterThresholds::default();
        for n in [5u32, 500, 1000] {
            let bitmap: RoaringBitmap = (0..n).collect();
            let result = adaptive_search(&idx, &[1.0, 0.0], 3, 64, &bitmap, &thresholds);
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
    }
}
