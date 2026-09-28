// SPDX-License-Identifier: Apache-2.0

//! SPFresh LIRE-style topology-aware local patching.
//!
//! Rather than rebuilding the entire HNSW on every update, `LirePatcher`
//! stitches fresh delta vectors into the main graph one node at a time using
//! the graph's own `insert` routine (which runs heuristic neighbor selection
//! and bidirectional edge maintenance internally).  Tombstones are forwarded
//! to the HNSW as soft-deletes so search skips them immediately; physical
//! removal is deferred to a later background compaction sweep via
//! `HnswIndex::compact`.
//!
//! ## Partition-quality drift estimate
//!
//! After inserting each fresh node we inspect its assigned neighbors.  For
//! each neighbor `n`, we count how many of *n*'s own neighbors were already
//! neighbors of any other newly-inserted node in this batch.  A high overlap
//! means the delta nodes clustered into an already-dense region — minimal
//! drift.  A low overlap means the new node landed in a sparse region that
//! may require broader re-wiring.
//!
//! This is an O(patch_size × M) approximation of LIRE's full local-rebuild
//! signal (SOSP 2023 §4.2).  When the average overlap fraction across all
//! patched nodes falls below `drift_threshold`, we record that subgraph in
//! `PatchStats::drift_subgraphs` so the caller can schedule a deeper
//! re-pruning pass at lower priority.

use crate::delta::index::DeltaIndex;
use crate::error::VectorError;
use crate::hnsw::HnswIndex;

/// SPFresh LIRE-style topology-aware local patcher.
///
/// Holds mutable references to both the main HNSW and the delta buffer so
/// both can be updated atomically within a single flush.
pub struct LirePatcher<'a> {
    /// Main HNSW graph that fresh vectors will be patched into.
    pub main: &'a mut HnswIndex,
    /// Delta buffer whose fresh vectors (and tombstones) will be drained.
    pub delta: &'a mut DeltaIndex,
    /// Drift threshold in `[0.0, 1.0]`.  When the average neighbor-overlap
    /// fraction for a batch falls below this value the subgraph is flagged
    /// for deeper re-pruning.  Default: `0.3`.
    pub drift_threshold: f32,
}

/// Statistics returned by a single `LirePatcher::patch` call.
#[derive(Debug, Default, Clone)]
pub struct PatchStats {
    /// Number of fresh vectors successfully patched into the main HNSW.
    pub patched: usize,
    /// Number of vectors tombstoned in the main HNSW during this flush.
    pub tombstoned_marked: usize,
    /// Number of subgraphs flagged for deeper re-pruning due to drift.
    pub drift_subgraphs: usize,
}

impl<'a> LirePatcher<'a> {
    /// Create a patcher with the default drift threshold (`0.3`).
    pub fn new(main: &'a mut HnswIndex, delta: &'a mut DeltaIndex) -> Self {
        Self {
            main,
            delta,
            drift_threshold: 0.3,
        }
    }

    /// Flush the delta buffer into the main HNSW.
    ///
    /// ## Steps
    ///
    /// 1. Drain tombstones → call `HnswIndex::delete` on each.
    /// 2. Drain fresh vectors → call `HnswIndex::insert` on each.
    /// 3. After each insert, estimate local topology drift for the newly
    ///    assigned node and accumulate the overlap fraction.
    /// 4. If average overlap fraction < `drift_threshold`, increment
    ///    `drift_subgraphs`.
    ///
    /// `k_neighbors` and `ef_construction` are accepted for API completeness
    /// and forward-compatibility with future Vamana-style patchers; the
    /// current HNSW implementation derives its own neighbor count from
    /// `HnswParams` stored on the index, so these values are informational.
    pub fn patch(
        &mut self,
        _k_neighbors: usize,
        _ef_construction: usize,
    ) -> Result<PatchStats, VectorError> {
        let mut stats = PatchStats::default();

        // --- Step 1: Forward tombstones to the main HNSW ---
        let tombstone_ids = self.delta.drain_tombstones();
        for id in tombstone_ids {
            if self.main.delete(id) {
                stats.tombstoned_marked += 1;
            }
        }

        // --- Step 2 + 3: Insert fresh vectors and estimate drift ---
        let fresh = self.delta.drain_fresh();

        // Collect the node IDs that will be assigned to freshly inserted nodes
        // so we can measure neighborhood overlap after each insert.
        // The HNSW appends nodes sequentially, so the new id = len() before insert.
        let mut overlap_fractions: Vec<f32> = Vec::with_capacity(fresh.len());
        // Track the set of recently-patched node ids for overlap estimation.
        let mut patched_ids: std::collections::HashSet<u32> =
            std::collections::HashSet::with_capacity(fresh.len());

        for (user_id, vector) in fresh {
            // Skip tombstoned fresh inserts — they were deleted before we
            // could patch them.
            if self.delta.is_tombstoned(user_id) {
                continue;
            }

            // The HNSW uses its own internal monotonic IDs (insertion order).
            // We record what the next id will be before the insert.
            let new_internal_id = self.main.len() as u32;

            self.main.insert(vector)?;
            stats.patched += 1;

            // --- Drift estimation (LIRE approximation) ---
            // Inspect neighbors assigned to the new node at layer 0.
            let neighbors_l0 = self.main.hnsw_neighbors_layer0(new_internal_id);

            let overlap_fraction = if neighbors_l0.is_empty() {
                // First node or isolated — perfect connectivity by definition.
                1.0f32
            } else {
                // Count how many neighbors are themselves in the current
                // patched-ids set (i.e., recently inserted into this batch).
                let overlap = neighbors_l0
                    .iter()
                    .filter(|&&nid| patched_ids.contains(&nid))
                    .count();
                overlap as f32 / neighbors_l0.len() as f32
            };

            overlap_fractions.push(overlap_fraction);
            patched_ids.insert(new_internal_id);
        }

        // --- Step 4: Flag drift subgraphs ---
        if !overlap_fractions.is_empty() {
            let avg_overlap =
                overlap_fractions.iter().sum::<f32>() / overlap_fractions.len() as f32;
            if avg_overlap < self.drift_threshold {
                stats.drift_subgraphs += 1;
            }
        }

        Ok(stats)
    }
}

// ---------------------------------------------------------------------------
// Additive accessor on HnswIndex required by LirePatcher.
//
// `mark_deleted` does not exist on `HnswIndex`; the equivalent is `delete`.
// We add a thin helper that exposes layer-0 neighbors so the drift estimator
// can read them without re-exposing internal fields.
// ---------------------------------------------------------------------------
impl HnswIndex {
    /// Return the layer-0 neighbor list of `node_id`, or an empty slice if
    /// the node does not exist or has no layer-0 neighbors.
    ///
    /// Used by `LirePatcher` for local topology-drift estimation.
    pub fn hnsw_neighbors_layer0(&self, node_id: u32) -> Vec<u32> {
        self.neighbors_at(node_id, 0).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswIndex;
    use nodedb_types::hnsw::HnswParams;

    fn small_params() -> HnswParams {
        HnswParams {
            m: 4,
            m0: 8,
            ef_construction: 20,
            ..HnswParams::default()
        }
    }

    #[test]
    fn patch_grows_hnsw_and_drains_delta() {
        let mut main = HnswIndex::with_seed(3, small_params(), 1);
        // Pre-populate main with 10 vectors so fresh nodes have neighbors.
        for i in 0u32..10 {
            let v = vec![i as f32, 0.0, 0.0];
            main.insert(v).expect("pre-populate insert failed");
        }
        assert_eq!(main.len(), 10);

        let mut delta = DeltaIndex::new(3, 32);
        for i in 10u32..15 {
            let v = vec![i as f32, 1.0, 0.0];
            delta.insert(i, v).unwrap();
        }
        assert_eq!(delta.fresh_len(), 5);

        let mut patcher = LirePatcher::new(&mut main, &mut delta);
        let stats = patcher.patch(8, 20).expect("patch failed");

        assert_eq!(stats.patched, 5);
        assert_eq!(delta.fresh_len(), 0);
        assert_eq!(main.len(), 15);
    }

    #[test]
    fn tombstone_forwarded_to_hnsw() {
        let mut main = HnswIndex::with_seed(3, small_params(), 2);
        for i in 0u32..5 {
            let v = vec![i as f32, 0.0, 0.0];
            main.insert(v).expect("insert failed");
        }
        assert!(!main.is_deleted(2));

        let mut delta = DeltaIndex::new(3, 16);
        delta.tombstone(2);

        let mut patcher = LirePatcher::new(&mut main, &mut delta);
        let stats = patcher.patch(4, 20).expect("patch failed");

        assert_eq!(stats.tombstoned_marked, 1);
        assert!(main.is_deleted(2));
    }

    #[test]
    fn patch_empty_delta_is_noop() {
        let mut main = HnswIndex::with_seed(3, small_params(), 3);
        for i in 0u32..3 {
            main.insert(vec![i as f32, 0.0, 0.0])
                .expect("insert failed");
        }
        let initial_len = main.len();

        let mut delta = DeltaIndex::new(3, 16);
        let mut patcher = LirePatcher::new(&mut main, &mut delta);
        let stats = patcher.patch(4, 20).expect("patch failed");

        assert_eq!(stats.patched, 0);
        assert_eq!(stats.tombstoned_marked, 0);
        assert_eq!(main.len(), initial_len);
    }
}
