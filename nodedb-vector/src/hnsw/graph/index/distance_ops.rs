// SPDX-License-Identifier: Apache-2.0

//! Layer assignment and distance computation for [`HnswIndex`].

use std::borrow::Cow;

use nodedb_types::vector_dtype::VectorStorageDtype;

use super::HnswIndex;
use crate::distance::dispatch::distance_typed;
use crate::distance::{DistanceMetric, distance};
use crate::dtype::cast_from_f32;
use crate::hnsw::graph::{MAX_LAYER_CAP, NodeStorage, Probe};

impl HnswIndex {
    /// Assign a random layer using the exponential distribution.
    ///
    /// Capped at `MAX_LAYER_CAP` to prevent pathological RNG draws from
    /// promoting the index's `max_layer` to hundreds or thousands, which
    /// would make every search's Phase-1 greedy descent O(max_layer).
    pub(crate) fn random_layer(&mut self) -> usize {
        let ml = 1.0 / (self.params.m as f64).ln();
        let r = self.rng.next_f64().max(f64::MIN_POSITIVE);
        let layer = (-r.ln() * ml).floor() as usize;
        layer.min(MAX_LAYER_CAP)
    }

    /// Prepares `query` for scoring against this index's nodes.
    ///
    /// An F32 index borrows the query. F16/BF16 indexes encode it once here,
    /// so callers build one probe per search or insert.
    pub(crate) fn probe<'a>(&self, query: &'a [f32]) -> Probe<'a> {
        match self.params.dtype {
            VectorStorageDtype::F32 => Probe::F32(query),
            dtype => Probe::Encoded(Cow::Owned(cast_from_f32(query, dtype))),
        }
    }

    /// Node `node_id`'s vector as a probe, borrowed without copying.
    ///
    /// Returns `None` when the node has no vector to compare.
    pub(crate) fn node_probe(&self, node_id: u32) -> Option<Probe<'_>> {
        match self.params.dtype {
            VectorStorageDtype::F32 => self.node_f32(node_id).map(Probe::F32),
            _ => self
                .node_encoded(node_id)
                .map(|bytes| Probe::Encoded(Cow::Borrowed(bytes))),
        }
    }

    /// Distance from `probe` to node `node_id` under `metric`.
    ///
    /// `metric` is caller-supplied, so filtered traversals can rank under a
    /// metric other than the build metric.
    ///
    /// Returns `None` when the node has no vector, for example after a
    /// graph-only restore with no backing attached. Such a node is never a hit.
    pub(crate) fn dist_to_node(
        &self,
        probe: &Probe<'_>,
        node_id: u32,
        metric: DistanceMetric,
    ) -> Option<f32> {
        match probe {
            Probe::F32(query) => self
                .node_f32(node_id)
                .map(|node| distance(query, node, metric)),
            Probe::Encoded(query) => {
                let node = self.node_encoded(node_id)?;
                distance_typed(metric, self.params.dtype, query, node, self.dim).ok()
            }
        }
    }

    /// Node `node_id`'s f32 vector: local storage first, then the backing.
    ///
    /// The backing serves f32 vectors. Only F32 indexes hold backed nodes,
    /// because a checkpoint restores an F32 index.
    fn node_f32(&self, node_id: u32) -> Option<&[f32]> {
        match &self.nodes.get(node_id as usize)?.storage {
            NodeStorage::F32(v) if !v.is_empty() => Some(v.as_slice()),
            NodeStorage::F32(_) => self.backing_f32(node_id),
            NodeStorage::Bytes { .. } => None,
        }
    }

    /// Node `node_id`'s bytes in the index dtype, for F16/BF16 indexes.
    fn node_encoded(&self, node_id: u32) -> Option<&[u8]> {
        match &self.nodes.get(node_id as usize)?.storage {
            NodeStorage::Bytes { bytes, .. } if !bytes.is_empty() => Some(bytes.as_slice()),
            NodeStorage::Bytes { .. } | NodeStorage::F32(_) => None,
        }
    }

    /// Node `node_id`'s vector in the attached segment backing.
    #[cfg(not(target_arch = "wasm32"))]
    fn backing_f32(&self, node_id: u32) -> Option<&[f32]> {
        self.backing.as_ref()?.get_vector(node_id)
    }

    /// WASM targets have no segment backing (it requires mmap).
    #[cfg(target_arch = "wasm32")]
    fn backing_f32(&self, _node_id: u32) -> Option<&[f32]> {
        None
    }

    /// Max neighbors allowed at a given layer.
    pub(crate) fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m0
        } else {
            self.params.m
        }
    }
}
