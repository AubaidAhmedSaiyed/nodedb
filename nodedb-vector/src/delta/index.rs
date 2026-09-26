// SPDX-License-Identifier: Apache-2.0

//! In-memory delta index for SPFresh streaming inserts.
//!
//! Fresh vectors accumulate here until `DeltaIndex::is_full()` signals that
//! compaction should flush them into the main HNSW.  Tombstones are lazily
//! tracked here and filtered at query time.

use std::collections::HashSet;

use crate::distance::distance;
use crate::error::{VectorError, check_dim};
use nodedb_types::vector_distance::DistanceMetric;

/// Secondary in-memory index that absorbs fresh inserts before they are
/// patched into the main HNSW graph.
pub struct DeltaIndex {
    pub dim: usize,
    /// Newly inserted vectors awaiting merge: `(id, vector)`.
    fresh: Vec<(u32, Vec<f32>)>,
    /// IDs marked tombstoned (lazy physical removal).
    tombstones: HashSet<u32>,
    /// Soft cap; callers should flush when `is_full()` returns `true`.
    max_fresh: usize,
}

impl DeltaIndex {
    /// Create a new delta index for vectors of `dim` dimensions.
    ///
    /// `max_fresh` is the capacity threshold above which `is_full()` returns
    /// `true`, signalling that a compaction flush should be triggered.
    pub fn new(dim: usize, max_fresh: usize) -> Self {
        Self {
            dim,
            fresh: Vec::new(),
            tombstones: HashSet::new(),
            max_fresh,
        }
    }

    /// Stage a fresh insert.  Does not deduplicate — callers must ensure IDs
    /// are unique across the delta and the main HNSW. A vector without the
    /// index dimension fails with [`VectorError::DimensionMismatch`].
    pub fn insert(&mut self, id: u32, vector: Vec<f32>) -> Result<(), VectorError> {
        check_dim(self.dim, vector.len())?;
        self.fresh.push((id, vector));
        Ok(())
    }

    /// Mark `id` as tombstoned.  It will be excluded from `search` results
    /// and from the patch applied to the main HNSW.
    pub fn tombstone(&mut self, id: u32) {
        self.tombstones.insert(id);
    }

    /// Returns `true` when the number of fresh vectors has reached `max_fresh`.
    pub fn is_full(&self) -> bool {
        self.fresh.len() >= self.max_fresh
    }

    /// Number of un-drained fresh vectors.
    pub fn fresh_len(&self) -> usize {
        self.fresh.len()
    }

    /// Brute-force scan over fresh vectors (excluding tombstones), returning
    /// the top-`k` results sorted ascending by distance. A query without
    /// the index dimension fails with [`VectorError::DimensionMismatch`].
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        check_dim(self.dim, query.len())?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(u32, f32)> = self
            .fresh
            .iter()
            .filter(|(id, _)| !self.tombstones.contains(id))
            .map(|(id, vec)| (*id, distance(query, vec, metric)))
            .collect();

        // Partial sort: cheapest path to top-k.
        if k < scored.len() {
            scored.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
        }

        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored)
    }

    /// Drain all staged fresh vectors for patching into the main HNSW.
    /// After this call `fresh_len()` returns 0.
    pub fn drain_fresh(&mut self) -> Vec<(u32, Vec<f32>)> {
        std::mem::take(&mut self.fresh)
    }

    /// Drain the current tombstone set.
    /// After this call no IDs are considered tombstoned in this delta.
    pub fn drain_tombstones(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.tombstones).into_iter().collect()
    }

    /// Returns `true` if `id` is currently tombstoned in this delta.
    pub fn is_tombstoned(&self, id: u32) -> bool {
        self.tombstones.contains(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodedb_types::vector_distance::DistanceMetric;

    fn make_delta() -> DeltaIndex {
        let mut d = DeltaIndex::new(3, 16);
        for i in 0u32..10 {
            let v = vec![i as f32, 0.0, 0.0];
            d.insert(i, v).unwrap();
        }
        d
    }

    #[test]
    fn top_k_returns_nearest() {
        let d = make_delta();
        let query = [0.0f32, 0.0, 0.0];
        let results = d.search(&query, 3, DistanceMetric::L2).unwrap();
        assert_eq!(results.len(), 3);
        // Nearest to [0,0,0] with L2^2 are ids 0,1,2
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
        assert_eq!(results[2].0, 2);
    }

    #[test]
    fn tombstone_excluded_from_search() {
        let mut d = make_delta();
        d.tombstone(0);
        let query = [0.0f32, 0.0, 0.0];
        let results = d.search(&query, 3, DistanceMetric::L2).unwrap();
        assert!(results.iter().all(|(id, _)| *id != 0));
    }

    #[test]
    fn is_full_triggers_at_threshold() {
        let mut d = DeltaIndex::new(3, 3);
        assert!(!d.is_full());
        d.insert(0, vec![0.0, 0.0, 0.0]).unwrap();
        d.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        assert!(!d.is_full());
        d.insert(2, vec![2.0, 0.0, 0.0]).unwrap();
        assert!(d.is_full());
    }

    #[test]
    fn drain_fresh_empties_buffer() {
        let mut d = make_delta();
        let drained = d.drain_fresh();
        assert_eq!(drained.len(), 10);
        assert_eq!(d.fresh_len(), 0);
    }

    #[test]
    fn drain_tombstones_empties_set() {
        let mut d = make_delta();
        d.tombstone(3);
        d.tombstone(7);
        let ts = d.drain_tombstones();
        assert_eq!(ts.len(), 2);
        assert!(!d.is_tombstoned(3));
        assert!(!d.is_tombstoned(7));
    }

    #[test]
    fn wrong_dimension_is_a_typed_error() {
        let mut d = DeltaIndex::new(3, 8);
        assert!(matches!(
            d.insert(0, vec![0.0; 2]),
            Err(VectorError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
        assert!(matches!(
            d.search(&[0.0; 4], 1, DistanceMetric::L2),
            Err(VectorError::DimensionMismatch {
                expected: 3,
                got: 4
            })
        ));
    }
}
