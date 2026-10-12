// SPDX-License-Identifier: Apache-2.0

//! Layer assignment and distance computation for [`HnswIndex`].

use std::borrow::Cow;

use nodedb_types::vector_dtype::VectorStorageDtype;

use super::HnswIndex;
use crate::distance::dispatch::distance_typed;
use crate::distance::{DistanceMetric, distance};
use crate::dtype::cast_from_f32;
use crate::hnsw::graph::MAX_LAYER_CAP;

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

    /// Compute distance between a pre-encoded query and a stored node.
    ///
    /// `query_bytes` must already be encoded in `self.params.dtype`; callers
    /// encode once at the top of search/insert and pass the same buffer to
    /// every `dist_to_node` call within that operation.
    ///
    /// When the node's local vector storage is empty (graph-checkpoint-only
    /// restore) and a `backing` is attached, the vector bytes are fetched from
    /// the backing.  This is the Lite cold-load path; Origin never hits it.
    ///
    /// A node whose vector cannot be resolved scores [`f32::INFINITY`] — search
    /// then never selects it — rather than panicking. `with_backing` rejects a
    /// backing that cannot serve the index, so reaching that branch means a node
    /// has no vector source at all; aborting a search worker (and poisoning the
    /// locks it held) is strictly worse than ranking one node last.
    pub(crate) fn dist_to_node(&self, query_bytes: &[u8], node_id: u32) -> f32 {
        distance_typed(
            self.params.metric,
            self.params.dtype,
            query_bytes,
            &self.node_bytes(node_id),
            self.dim,
        )
        .unwrap_or(f32::INFINITY)
    }

    /// Node `node_id`'s vector bytes, encoded in `self.params.dtype`.
    ///
    /// Every distance computation on the insert, prune, and search paths reads
    /// node vectors through this accessor. After a graph-checkpoint-only
    /// restore, local storage is empty and the vector lives in the attached
    /// backing. Reading `node.storage` directly there yields an empty slice,
    /// and every distance against it errors.
    ///
    /// The backing holds F32 vectors. An F32 index borrows them as bytes. Any
    /// other dtype encodes them into an owned buffer.
    ///
    /// Returns an empty slice when `node_id` is out of range, or the node has
    /// no local vector and no backing serves one. Distance against an empty
    /// slice errors, so callers score that node [`f32::INFINITY`].
    pub(crate) fn node_bytes(&self, node_id: u32) -> Cow<'_, [u8]> {
        let local = self
            .nodes
            .get(node_id as usize)
            .map_or(&[][..], |n| n.storage.as_bytes());
        #[cfg(not(target_arch = "wasm32"))]
        {
            if local.is_empty()
                && let Some(v) = self.backing.as_ref().and_then(|b| b.get_vector(node_id))
            {
                return match self.params.dtype {
                    VectorStorageDtype::F32 => Cow::Borrowed(bytemuck::cast_slice(v)),
                    dtype => Cow::Owned(cast_from_f32(v, dtype)),
                };
            }
        }
        Cow::Borrowed(local)
    }

    /// Compute distance between an f32 query and a stored node under `metric`.
    ///
    /// `metric` is caller-supplied, so filtered traversals that rank under a
    /// metric other than the build metric use this accessor too. The node
    /// vector comes from [`Self::node_bytes`], so restored nodes resolve
    /// through the attached backing.
    ///
    /// F32 indexes compare the query to the node vector without copying.
    /// F16/BF16 indexes encode the query on each call, which allocates.
    ///
    /// A node whose vector cannot be resolved, or a query of the wrong length,
    /// scores [`f32::INFINITY`].
    pub(crate) fn dist_to_node_f32(
        &self,
        query: &[f32],
        node_id: u32,
        metric: DistanceMetric,
    ) -> f32 {
        let node_bytes = self.node_bytes(node_id);
        match self.params.dtype {
            VectorStorageDtype::F32 => match bytemuck::try_cast_slice::<u8, f32>(&node_bytes) {
                Ok(node) if node.len() == query.len() => distance(query, node, metric),
                // Empty node or wrong query length.
                Ok(_) => f32::INFINITY,
                // Bytes not aligned to f32: `distance_typed` copies them out.
                Err(_) => distance_typed(
                    metric,
                    VectorStorageDtype::F32,
                    bytemuck::cast_slice(query),
                    &node_bytes,
                    self.dim,
                )
                .unwrap_or(f32::INFINITY),
            },
            dtype => {
                let query_bytes = cast_from_f32(query, dtype);
                distance_typed(metric, dtype, &query_bytes, &node_bytes, self.dim)
                    .unwrap_or(f32::INFINITY)
            }
        }
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
