// SPDX-License-Identifier: Apache-2.0

//! SieveCollection — a keyed map of specialized HNSW subindices, one per
//! stable predicate signature (e.g. "tenant_id=42", "lang=en").

use std::collections::HashMap;

use crate::error::VectorError;
use crate::hnsw::graph::HnswIndex;
use nodedb_types::hnsw::HnswParams;
use nodedb_types::vector_distance::DistanceMetric;

/// A stable predicate signature string, e.g. `"tenant_id=42"` or `"lang=en"`.
pub type PredicateSignature = String;

/// A collection of specialized HNSW subindices, one per stable predicate.
///
/// Each subindex is built over the subset of vectors that match its predicate.
/// Smaller datasets mean a lower `sub_m` is sufficient, keeping memory overhead
/// proportional to the per-predicate population rather than the global index.
pub struct SieveCollection {
    /// Map from predicate signature → specialized HNSW subindex.
    subindices: HashMap<PredicateSignature, HnswIndex>,
    /// `M` parameter used when creating subindices.
    /// Smaller than the global M because each subindex covers fewer vectors.
    sub_m: usize,
}

impl SieveCollection {
    /// Create an empty collection.  `sub_m` is the HNSW `M` parameter used for
    /// every subindex built by this collection.
    pub fn new(sub_m: usize) -> Self {
        Self {
            subindices: HashMap::new(),
            sub_m,
        }
    }

    /// Build (or rebuild) a specialized subindex for `signature` from the
    /// provided `(id, vector)` pairs.
    ///
    /// All vectors must have length `dim`.  The subindex uses `metric` for
    /// distance computation.
    ///
    /// # Errors
    ///
    /// Returns `VectorError` if any insertion fails (e.g. dimension mismatch).
    pub fn build_subindex(
        &mut self,
        signature: PredicateSignature,
        vectors: &[(u32, Vec<f32>)],
        dim: usize,
        metric: DistanceMetric,
    ) -> Result<(), VectorError> {
        let params = HnswParams {
            m: self.sub_m,
            m0: self.sub_m * 2,
            ef_construction: 200,
            metric,
            dtype: nodedb_types::vector_dtype::VectorStorageDtype::F32,
        };
        let mut index = HnswIndex::new(dim, params);
        for (_, vec) in vectors {
            index.insert(vec.clone())?;
        }
        self.subindices.insert(signature, index);
        Ok(())
    }

    /// Returns `true` if a subindex exists for the given signature.
    pub fn has(&self, signature: &PredicateSignature) -> bool {
        self.subindices.contains_key(signature)
    }

    /// Returns a shared reference to the subindex for `signature`, or `None`.
    pub fn get(&self, signature: &PredicateSignature) -> Option<&HnswIndex> {
        self.subindices.get(signature)
    }

    /// Remove the subindex for `signature`, freeing its memory.
    pub fn drop(&mut self, signature: &PredicateSignature) {
        self.subindices.remove(signature);
    }

    /// All predicate signatures currently held in this collection.
    pub fn signatures(&self) -> Vec<&PredicateSignature> {
        self.subindices.keys().collect()
    }
}

// Expose SearchResult at crate level through this module for callers that
// import from `sieve::collection`.
pub use crate::hnsw::graph::SearchResult as SubindexSearchResult;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vectors(n: usize, dim: usize) -> Vec<(u32, Vec<f32>)> {
        (0..n).map(|i| (i as u32, vec![i as f32; dim])).collect()
    }

    #[test]
    fn build_subindex_has_and_get() {
        let mut coll = SieveCollection::new(8);
        let vecs = sample_vectors(5, 3);
        coll.build_subindex("tenant_id=42".to_string(), &vecs, 3, DistanceMetric::L2)
            .expect("build should succeed");

        assert!(coll.has(&"tenant_id=42".to_string()));
        let idx = coll.get(&"tenant_id=42".to_string());
        assert!(idx.is_some());
        assert_eq!(idx.unwrap().len(), 5);
    }

    #[test]
    fn drop_removes_subindex() {
        let mut coll = SieveCollection::new(8);
        let vecs = sample_vectors(5, 3);
        coll.build_subindex("lang=en".to_string(), &vecs, 3, DistanceMetric::Cosine)
            .expect("build should succeed");
        assert!(coll.has(&"lang=en".to_string()));

        coll.drop(&"lang=en".to_string());
        assert!(!coll.has(&"lang=en".to_string()));
        assert!(coll.get(&"lang=en".to_string()).is_none());
    }

    #[test]
    fn signatures_lists_all_keys() {
        let mut coll = SieveCollection::new(8);
        let vecs = sample_vectors(3, 2);
        coll.build_subindex("a".to_string(), &vecs, 2, DistanceMetric::L2)
            .expect("build a");
        coll.build_subindex("b".to_string(), &vecs, 2, DistanceMetric::L2)
            .expect("build b");

        let mut sigs: Vec<String> = coll.signatures().into_iter().cloned().collect();
        sigs.sort();
        assert_eq!(sigs, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn search_on_subindex() {
        let mut coll = SieveCollection::new(8);
        let vecs = sample_vectors(5, 3);
        coll.build_subindex("tenant_id=1".to_string(), &vecs, 3, DistanceMetric::L2)
            .expect("build");

        let idx = coll.get(&"tenant_id=1".to_string()).unwrap();
        let results = idx.search(&[2.0, 2.0, 2.0], 2, 32).unwrap();
        assert!(!results.is_empty());
    }
}
