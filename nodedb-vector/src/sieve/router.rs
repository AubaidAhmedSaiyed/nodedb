// SPDX-License-Identifier: Apache-2.0

//! SieveRouter — routes a filtered ANN query to a specialized subindex when the
//! predicate signature matches, or falls back to NaviX on the global index.

use roaring::RoaringBitmap;

use super::collection::{PredicateSignature, SieveCollection};
use crate::hnsw::graph::{HnswIndex, SearchResult};
use crate::navix::traversal::{NavixSearchOptions, navix_search};
use nodedb_types::vector_distance::DistanceMetric;

/// Routes a filtered ANN query to the right index.
///
/// - If `predicate_signature` is `Some(sig)` and a subindex exists for `sig`,
///   the query is executed directly on that subindex (no bitmap needed).
/// - Otherwise the query falls back to `navix_search` on the global fallback
///   index using `allowed` as the sideways-information bitmap.
pub struct SieveRouter<'a> {
    /// Collection of specialized subindices.
    pub collection: &'a SieveCollection,
    /// Global index used when no subindex matches.
    pub fallback: &'a HnswIndex,
}

impl<'a> SieveRouter<'a> {
    /// Execute a filtered k-NN query.
    ///
    /// # Parameters
    ///
    /// - `query`               — query vector.
    /// - `predicate_signature` — optional stable predicate; if present and a
    ///   subindex exists for it, that subindex is used directly.
    /// - `allowed`             — bitmap of allowed IDs for the NaviX fallback
    ///   path.  Ignored when a subindex is matched.
    /// - `k`                   — number of nearest neighbours to return.
    /// - `ef_search`           — beam width for HNSW/NaviX traversal.
    /// - `metric`              — distance metric.
    ///
    /// # Returns
    ///
    /// Up to `k` nearest-neighbour results, sorted by ascending distance, or
    /// [`VectorError::DimensionMismatch`](crate::error::VectorError::DimensionMismatch)
    /// when `query` does not have the index dimension.
    pub fn route(
        &self,
        query: &[f32],
        predicate_signature: Option<&PredicateSignature>,
        allowed: RoaringBitmap,
        k: usize,
        ef_search: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>, crate::error::VectorError> {
        // Fast path: subindex hit.
        if let Some(sig) = predicate_signature
            && let Some(subindex) = self.collection.get(sig)
        {
            return subindex.search(query, k, ef_search);
        }

        // Slow path: NaviX adaptive-local filtered search on the global index.
        let opts = NavixSearchOptions {
            k,
            ef_search,
            allowed,
            brute_force_threshold: 0.001,
        };
        Ok(navix_search(self.fallback, query, &opts, metric)?
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{HnswIndex, HnswParams};
    use crate::sieve::collection::SieveCollection;
    use nodedb_types::vector_distance::DistanceMetric;

    fn build_fallback(n: usize) -> HnswIndex {
        let mut idx = HnswIndex::with_seed(
            3,
            HnswParams {
                m: 8,
                m0: 16,
                ef_construction: 50,
                metric: DistanceMetric::L2,
                dtype: nodedb_types::vector_dtype::VectorStorageDtype::F32,
            },
            99,
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

    /// When the predicate matches an existing subindex, results come from that
    /// subindex (not the fallback).  The subindex contains only 5 vectors, so
    /// the result IDs are all < 5.
    #[test]
    fn route_hits_subindex() {
        // Build a subindex with 5 vectors [0,0,0]..[4,0,0].
        let mut coll = SieveCollection::new(8);
        let sub_vecs: Vec<(u32, Vec<f32>)> =
            (0u32..5).map(|i| (i, vec![i as f32, 0.0, 0.0])).collect();
        coll.build_subindex("T".to_string(), &sub_vecs, 3, DistanceMetric::L2)
            .expect("build subindex");

        let fallback = build_fallback(20);
        let router = SieveRouter {
            collection: &coll,
            fallback: &fallback,
        };

        let results = router
            .route(
                &[2.0, 0.0, 0.0],
                Some(&"T".to_string()),
                all_allowed(20), // bitmap irrelevant on subindex path
                3,
                32,
                DistanceMetric::L2,
            )
            .unwrap();

        assert!(!results.is_empty());
        // All result IDs must be within the subindex range [0..5).
        for r in &results {
            assert!(r.id < 5, "expected subindex id < 5, got {}", r.id);
        }
    }

    /// When the predicate does not match any subindex, NaviX on the fallback
    /// index is used.  With full allowed set, the nearest vector must be found.
    #[test]
    fn route_falls_back_to_navix() {
        let coll = SieveCollection::new(8); // empty — no subindices
        let fallback = build_fallback(20);
        let router = SieveRouter {
            collection: &coll,
            fallback: &fallback,
        };

        let allowed = all_allowed(20);
        let results = router
            .route(
                &[10.0, 0.0, 0.0],
                Some(&"unknown_sig".to_string()),
                allowed,
                3,
                64,
                DistanceMetric::L2,
            )
            .unwrap();

        assert!(!results.is_empty());
        // The nearest vector to [10,0,0] in [0..20] is id=10.
        assert_eq!(results[0].id, 10);
    }

    /// With no predicate signature, NaviX fallback is always used.
    #[test]
    fn route_no_signature_uses_navix() {
        let mut coll = SieveCollection::new(8);
        let sub_vecs: Vec<(u32, Vec<f32>)> =
            (0u32..5).map(|i| (i, vec![i as f32, 0.0, 0.0])).collect();
        coll.build_subindex("T".to_string(), &sub_vecs, 3, DistanceMetric::L2)
            .expect("build subindex");

        let fallback = build_fallback(20);
        let router = SieveRouter {
            collection: &coll,
            fallback: &fallback,
        };

        let allowed = all_allowed(20);
        let results = router
            .route(&[5.0, 0.0, 0.0], None, allowed, 3, 64, DistanceMetric::L2)
            .unwrap();

        assert!(!results.is_empty());
        // Must include id=5 since fallback has all 20 vectors.
        assert_eq!(results[0].id, 5);
    }
}
