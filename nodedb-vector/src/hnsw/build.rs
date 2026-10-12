// SPDX-License-Identifier: Apache-2.0

//! HNSW insert algorithm (Malkov & Yashunin, Algorithm 1).

use crate::dtype::cast_from_f32;
use crate::error::VectorError;
use crate::hnsw::graph::{Candidate, HnswIndex, Node, NodeStorage};
use crate::hnsw::search::search_layer;
use nodedb_types::vector_dtype::VectorStorageDtype;

impl HnswIndex {
    /// Insert a vector into the index.
    ///
    /// 1. Assign a random layer using the exponential distribution
    /// 2. Greedily descend from the entry point to the new node's layer + 1
    /// 3. At each layer from the node's layer down to 0, search for nearest
    ///    neighbors, select via the diversity heuristic, and add bidirectional edges
    /// 4. Prune over-connected nodes to maintain the M/M0 invariant
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<(), VectorError> {
        // Materialize flat neighbor storage on first mutation.
        self.ensure_mutable_neighbors();

        if vector.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let new_id = self.nodes.len() as u32;
        let new_layer = self.random_layer();

        let storage = match self.params.dtype {
            VectorStorageDtype::F32 => NodeStorage::F32(vector.clone()),
            dtype => NodeStorage::Bytes {
                dtype,
                bytes: cast_from_f32(&vector, dtype),
            },
        };
        let node = Node {
            storage,
            neighbors: (0..=new_layer).map(|_| Vec::new()).collect(),
            deleted: false,
        };
        self.nodes.push(node);

        let Some(ep) = self.entry_point else {
            self.entry_point = Some(new_id);
            self.max_layer = new_layer;
            return Ok(());
        };

        // Encode query once to the index dtype for all dist_to_node calls below.
        let query = vector;
        let query_bytes = cast_from_f32(&query, self.params.dtype);

        let mut current_ep = ep;

        // Phase 1: Greedy descent from top layer to new_layer + 1.
        if self.max_layer > new_layer {
            for layer in (new_layer + 1..=self.max_layer).rev() {
                let results = search_layer(self, &query_bytes, current_ep, 1, layer, None, 0);
                if let Some(nearest) = results.first() {
                    current_ep = nearest.id;
                }
            }
        }

        // Phase 2: Insert at each layer from min(new_layer, max_layer) down to 0.
        let insert_top = new_layer.min(self.max_layer);
        for layer in (0..=insert_top).rev() {
            let ef = self.params.ef_construction;
            let candidates = search_layer(self, &query_bytes, current_ep, ef, layer, None, 0);

            let m = self.max_neighbors(layer);
            let selected = select_neighbors_heuristic(self, &candidates, m);

            self.nodes[new_id as usize].neighbors[layer] = selected.iter().map(|c| c.id).collect();

            for neighbor in &selected {
                let nid = neighbor.id as usize;
                self.nodes[nid].neighbors[layer].push(new_id);

                if self.nodes[nid].neighbors[layer].len() > m {
                    let node_bytes = self.node_bytes(neighbor.id).into_owned();
                    self.prune_neighbors(nid, layer, &node_bytes, m);
                }
            }

            if let Some(nearest) = candidates.first() {
                current_ep = nearest.id;
            }
        }

        if new_layer > self.max_layer {
            self.entry_point = Some(new_id);
            self.max_layer = new_layer;
        }

        Ok(())
    }

    /// Prune a node's neighbor list using the diversity heuristic.
    fn prune_neighbors(&mut self, node_idx: usize, layer: usize, node_bytes: &[u8], m: usize) {
        let neighbor_ids: Vec<u32> = self.nodes[node_idx].neighbors[layer].clone();

        let mut candidates: Vec<Candidate> = neighbor_ids
            .iter()
            .map(|&nid| Candidate {
                id: nid,
                dist: self.dist_to_node(node_bytes, nid),
            })
            .collect();
        candidates.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));

        let selected = select_neighbors_heuristic(self, &candidates, m);
        self.nodes[node_idx].neighbors[layer] = selected.iter().map(|c| c.id).collect();
    }
}

/// Heuristic neighbor selection (Malkov & Yashunin, Algorithm 4).
fn select_neighbors_heuristic(
    index: &HnswIndex,
    candidates: &[Candidate],
    m: usize,
) -> Vec<Candidate> {
    let mut selected: Vec<Candidate> = Vec::with_capacity(m);

    for candidate in candidates {
        if selected.len() >= m {
            break;
        }

        let candidate_bytes = index.node_bytes(candidate.id);
        let is_diverse = selected.iter().all(|s| {
            let dist_to_selected = crate::distance::dispatch::distance_typed(
                index.params.metric,
                index.params.dtype,
                &candidate_bytes,
                &index.node_bytes(s.id),
                index.dim,
            )
            .unwrap_or(f32::MAX);
            candidate.dist <= dist_to_selected
        });

        if is_diverse {
            selected.push(*candidate);
        }
    }

    // Backfill if heuristic was too aggressive.
    if selected.len() < m {
        let selected_ids: std::collections::HashSet<u32> = selected.iter().map(|c| c.id).collect();
        for candidate in candidates {
            if selected.len() >= m {
                break;
            }
            if !selected_ids.contains(&candidate.id) {
                selected.push(*candidate);
            }
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use crate::distance::DistanceMetric;
    use crate::hnsw::{HnswIndex, HnswParams};

    fn make_index() -> HnswIndex {
        HnswIndex::with_seed(
            3,
            HnswParams {
                m: 4,
                m0: 8,
                ef_construction: 32,
                metric: DistanceMetric::L2,
                dtype: nodedb_types::vector_dtype::VectorStorageDtype::F32,
            },
            12345,
        )
    }

    #[test]
    fn insert_single() {
        let mut idx = make_index();
        idx.insert(vec![1.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.entry_point(), Some(0));
    }

    #[test]
    fn insert_many_maintains_invariants() {
        let mut idx = make_index();
        for i in 0..100 {
            let v = vec![(i as f32) * 0.1, (i as f32) * 0.2, (i as f32) * 0.3];
            idx.insert(v).unwrap();
        }
        assert_eq!(idx.len(), 100);
        assert!(idx.entry_point().is_some());

        for node in &idx.nodes {
            assert!(node.neighbors[0].len() <= idx.params.m0);
        }
        for node in &idx.nodes {
            for (layer, neighbors) in node.neighbors.iter().enumerate().skip(1) {
                assert!(
                    neighbors.len() <= idx.params.m,
                    "layer {layer} neighbor count {} exceeds m={}",
                    neighbors.len(),
                    idx.params.m
                );
            }
        }
    }

    #[test]
    fn all_nodes_reachable_from_entry() {
        let mut idx = make_index();
        for i in 0..20 {
            idx.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }

        for target in 0..20u32 {
            let query = idx.get_vector(target).unwrap().to_vec();
            let results = idx.search(&query, 1, 32).unwrap();
            assert_eq!(results[0].id, target, "node {target} not reachable");
        }
    }

    /// Deterministic vectors in `[0, 1)` from an xorshift64 stream. `seed` must
    /// be non-zero.
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

    /// A graph-only restore with an attached backing must insert exactly like a
    /// full restore. Restored nodes hold no local vector, so neighbor pruning
    /// and the diversity heuristic read them through the backing. A new node
    /// must gain incoming edges, or search can never reach it.
    ///
    /// `segment_backing` is gated off on wasm32, so the test is too.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn inserts_after_graph_only_restore_match_a_full_restore() {
        use std::sync::Arc;

        use crate::segment_backing::VectorSegmentBacking;

        /// Serves an in-memory copy of the base vectors.
        struct VecBacking {
            dim: usize,
            vectors: Vec<Vec<f32>>,
        }
        impl VectorSegmentBacking for VecBacking {
            fn len(&self) -> usize {
                self.vectors.len()
            }
            fn dim(&self) -> usize {
                self.dim
            }
            fn get_vector(&self, id: u32) -> Option<&[f32]> {
                self.vectors.get(id as usize).map(Vec::as_slice)
            }
            fn get_surrogate(&self, _id: u32) -> Option<u64> {
                None
            }
        }

        const DIM: usize = 8;
        let params = HnswParams {
            m: 4,
            m0: 8,
            ef_construction: 32,
            metric: DistanceMetric::L2,
            dtype: nodedb_types::vector_dtype::VectorStorageDtype::F32,
        };
        let base = seeded_vectors(200, DIM, 0x9E37_79B9_7F4A_7C15);
        let extra = seeded_vectors(5, DIM, 0xD1B5_4A32_D192_ED03);

        let mut src = HnswIndex::with_seed(DIM, params, 12345);
        for v in &base {
            src.insert(v.clone()).unwrap();
        }

        let mut full = HnswIndex::from_checkpoint(&src.checkpoint_to_bytes().unwrap())
            .unwrap()
            .expect("full checkpoint must be recognized");
        let mut graph_only = HnswIndex::from_checkpoint(&src.graph_checkpoint_to_bytes().unwrap())
            .unwrap()
            .expect("graph checkpoint must be recognized");
        graph_only
            .with_backing(Arc::new(VecBacking {
                dim: DIM,
                vectors: base.clone(),
            }))
            .expect("a complete backing must attach");

        for v in &extra {
            full.insert(v.clone()).unwrap();
            graph_only.insert(v.clone()).unwrap();
        }

        let neighbors = graph_only.export_neighbors();
        assert_eq!(
            full.export_neighbors(),
            neighbors,
            "a graph-only restore must build the same graph as a full restore"
        );

        for (offset, v) in extra.iter().enumerate() {
            let new_id = (base.len() + offset) as u32;
            let hits = graph_only.search(v, 1, 64).unwrap();
            assert_eq!(
                hits.first().map(|h| h.id),
                Some(new_id),
                "exact query for inserted node {new_id} returns it, got {hits:?}"
            );
            assert!(
                hits[0].distance < 1e-6,
                "exact query for node {new_id} scores ~0, got {}",
                hits[0].distance
            );
            assert!(
                neighbors
                    .iter()
                    .any(|layers| layers.first().is_some_and(|l0| l0.contains(&new_id))),
                "node {new_id} has no incoming layer-0 edge"
            );
        }

        let m = graph_only.params().m;
        let m0 = graph_only.params().m0;
        for (id, layers) in neighbors.iter().enumerate() {
            for (layer, list) in layers.iter().enumerate() {
                let cap = if layer == 0 { m0 } else { m };
                assert!(
                    list.len() <= cap,
                    "node {id} layer {layer} has {} neighbors, cap {cap}",
                    list.len()
                );
            }
        }
    }

    #[test]
    fn compact_removes_tombstones() {
        let mut idx = make_index();
        for i in 0..20u32 {
            idx.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }

        for i in (0..20u32).step_by(2) {
            assert!(idx.delete(i));
        }
        assert_eq!(idx.compact(), 10);
        assert_eq!(idx.len(), 10);

        for target_old_id in (1..20u32).step_by(2) {
            let query = vec![target_old_id as f32, 0.0, 0.0];
            let results = idx.search(&query, 1, 32).unwrap();
            assert!(!results.is_empty());
            let found_vec = idx.get_vector(results[0].id).unwrap();
            assert_eq!(found_vec[0], target_old_id as f32);
        }
    }
}
