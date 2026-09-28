// SPDX-License-Identifier: Apache-2.0

//! IVF-PQ index parameters.

use crate::distance::DistanceMetric;

/// IVF-PQ index configuration.
#[derive(Debug, Clone, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub struct IvfPqParams {
    /// Number of Voronoi cells (partitions). Typical: sqrt(N).
    pub n_cells: usize,
    /// Number of PQ subvectors. Must divide dimension evenly.
    pub pq_m: usize,
    /// Centroids per PQ subvector (at most 256 for u8 codes).
    pub pq_k: usize,
    /// Number of cells to probe at query time. Higher = better recall.
    pub nprobe: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
}

impl IvfPqParams {
    /// Vectors an index needs before it can train: one per coarse cell and
    /// one per PQ centroid, since both k-means runs need at least `k` points.
    pub fn training_threshold(&self) -> usize {
        self.n_cells.max(self.pq_k).max(1)
    }
}

impl Default for IvfPqParams {
    fn default() -> Self {
        Self {
            n_cells: 256,
            pq_m: 8,
            pq_k: 256,
            nprobe: 16,
            metric: DistanceMetric::L2,
        }
    }
}
