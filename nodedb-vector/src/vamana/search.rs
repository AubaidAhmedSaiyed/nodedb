// SPDX-License-Identifier: Apache-2.0

//! Beam-search for the Vamana graph.
//!
//! Single-layer greedy traversal (Vamana has no upper layers unlike HNSW).
//! The hot path calls `VectorCodec::exact_asymmetric_distance` against
//! quantized vectors.  Full-precision rerank via `NodeFetcher::fetch_fp32`
//! is available when the caller explicitly reranks after search.
//!
//! ## Pre-fetch contract
//!
//! Before computing distances for a candidate's neighbors, `beam_search`
//! calls `NodeFetcher::prefetch_batch` with the neighbor index set.  On the
//! in-memory path this is a no-op.  On the `IoUringNodeFetcher` path it
//! submits `IORING_OP_READ` SQEs for all neighbor vectors in parallel so
//! the TPC reactor is never blocked waiting for a major page fault.
//!
//! Reference: Algorithm 1 in "DiskANN: Fast Accurate Billion-point Nearest
//! Neighbor Search on a Single Node", NeurIPS 2019.

use std::collections::{BinaryHeap, HashSet};

use nodedb_codec::vector_quant::codec::VectorCodec;

use crate::distance::scalar::l2_squared;
use crate::error::{VectorError, check_dim};
use crate::vamana::graph::VamanaGraph;
use crate::vamana::node_fetcher::NodeFetcher;

/// A single result from a beam-search query.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamSearchResult {
    /// External identifier of the matched node.
    pub id: u64,
    /// Approximate distance from the query (computed by the codec).
    pub distance: f32,
}

/// Ordered wrapper for the priority queues inside beam-search.
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    dist: f32,
    idx: u32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap by distance; break ties by index for determinism.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.idx.cmp(&self.idx))
    }
}

/// Greedy beam-search over a `VamanaGraph`.
///
/// Uses `codec.exact_asymmetric_distance` for all distance evaluations.
/// `l_search` is the beam width (candidate list size); larger values improve
/// recall at the cost of more distance evaluations.
///
/// # Arguments
///
/// * `graph` — the Vamana adjacency index.
/// * `query` — prepared query produced by `codec.prepare_query`.
/// * `codec` — quantization codec used during index construction.
/// * `quantized` — per-node quantized vectors in insertion order.
/// * `fetcher` — provides full-precision FP32 vectors; `prefetch_batch` is
///   called for each frontier expansion so io_uring reads land before
///   `rerank` touches the vectors.
/// * `k` — number of nearest neighbors to return.
/// * `l_search` — beam width (must be ≥ `k`).
///
/// # Returns
///
/// Up to `k` results sorted by ascending distance.
///
/// The query arrives already prepared by the codec, so its dimension is
/// checked where the caller prepares it; [`rerank`] checks the FP32 query.
///
/// # Errors
///
/// [`VectorError::InvalidInput`] when `quantized` does not hold one vector
/// per graph node.
pub fn beam_search<C, F>(
    graph: &VamanaGraph,
    query: &C::Query,
    codec: &C,
    quantized: &[C::Quantized],
    fetcher: &mut F,
    k: usize,
    l_search: usize,
) -> Result<Vec<BeamSearchResult>, VectorError>
where
    C: VectorCodec,
    F: NodeFetcher,
{
    if graph.is_empty() || quantized.is_empty() {
        return Ok(Vec::new());
    }
    if quantized.len() != graph.len() {
        return Err(VectorError::InvalidInput {
            detail: format!(
                "Vamana search got {} quantized vectors for {} graph nodes",
                quantized.len(),
                graph.len()
            ),
        });
    }

    let l = l_search.max(k);

    // Visited set — node indices that have been expanded.
    let mut visited: HashSet<u32> = HashSet::new();

    // Candidate min-heap (closest at top).
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();

    // Result set — we maintain up to `l` best results.
    // We use a max-heap so we can efficiently evict the farthest element.
    let mut result: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();

    // Seed from the entry point.
    let entry_idx = graph.entry as u32;
    let entry_dist = codec.exact_asymmetric_distance(query, &quantized[graph.entry]);
    candidates.push(Candidate {
        dist: entry_dist,
        idx: entry_idx,
    });
    visited.insert(entry_idx);

    // Scratch buffer reused every iteration to avoid per-hop allocation.
    let mut frontier_indices: Vec<u32> = Vec::new();

    // Greedy expansion.
    while let Some(current) = candidates.pop() {
        // Prune: if the current candidate is worse than our l-th result, stop.
        if result.len() >= l
            && let Some(worst) = result.peek()
            && current.dist > worst.0.dist
        {
            break;
        }

        // Add to result set.
        result.push(std::cmp::Reverse(current));
        if result.len() > l {
            result.pop(); // drop farthest
        }

        // Collect unvisited neighbors into frontier.
        frontier_indices.clear();
        for &neighbor_idx in graph.neighbors(current.idx as usize) {
            if !visited.contains(&neighbor_idx) && (neighbor_idx as usize) < quantized.len() {
                frontier_indices.push(neighbor_idx);
            }
        }

        // Pre-fetch FP32 vectors for the frontier so io_uring reads complete
        // asynchronously while we evaluate distances.  On the in-memory path
        // this is a no-op; on the io_uring path it submits IORING_OP_READ SQEs
        // without blocking.
        fetcher.prefetch_batch(&frontier_indices);

        // Expand neighbors.
        for &neighbor_idx in &frontier_indices {
            visited.insert(neighbor_idx);
            let d = codec.exact_asymmetric_distance(query, &quantized[neighbor_idx as usize]);
            candidates.push(Candidate {
                dist: d,
                idx: neighbor_idx,
            });
        }
    }

    // Collect, sort ascending, take k.
    let mut out: Vec<Candidate> = result.into_iter().map(|r| r.0).collect();
    out.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(k);

    Ok(out
        .into_iter()
        .map(|c| BeamSearchResult {
            id: graph.external_id(c.idx as usize),
            distance: c.dist,
        })
        .collect())
}

/// Rerank a candidate list using full-precision FP32 vectors from `fetcher`.
///
/// Accepts `BeamSearchResult` items from `beam_search`, fetches their
/// full-precision vectors, recomputes exact L2-squared distance, and
/// returns the top-`k` results sorted ascending.
///
/// On the `IoUringNodeFetcher` path the pre-fetch issued during `beam_search`
/// expansion means completions are already queued; `fetch_fp32` collects
/// them without re-submitting.
///
/// This is the "SSD fetch + rerank" step described in the DiskANN paper.
///
/// # Errors
///
/// - [`VectorError::DimensionMismatch`] when `query_fp32` does not have the
///   graph dimension.
/// - [`VectorError::StoredDimensionMismatch`] when a fetched vector does not
///   have the graph dimension.
pub fn rerank<F: NodeFetcher>(
    candidates: Vec<BeamSearchResult>,
    query_fp32: &[f32],
    fetcher: &mut F,
    graph: &VamanaGraph,
    k: usize,
) -> Result<Vec<BeamSearchResult>, VectorError> {
    check_dim(graph.dim, query_fp32.len())?;
    // Build id → internal index map.
    let id_to_idx: std::collections::HashMap<u64, usize> =
        graph.iter().map(|(idx, node)| (node.id, idx)).collect();

    // Pre-fetch all candidate vectors in one batch before iterating.
    let candidate_indices: Vec<u32> = candidates
        .iter()
        .filter_map(|c| id_to_idx.get(&c.id).map(|&i| i as u32))
        .collect();
    fetcher.prefetch_batch(&candidate_indices);

    let mut reranked: Vec<BeamSearchResult> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let Some(&idx) = id_to_idx.get(&c.id) else {
            continue;
        };
        let Some(vec) = fetcher.fetch_fp32(idx as u32) else {
            continue;
        };
        if vec.len() != graph.dim {
            return Err(VectorError::StoredDimensionMismatch {
                expected: graph.dim,
                got: vec.len(),
            });
        }
        reranked.push(BeamSearchResult {
            id: c.id,
            distance: l2_squared(query_fp32, &vec),
        });
    }

    reranked.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reranked.truncate(k);
    Ok(reranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vamana::build::build_vamana;
    use crate::vamana::node_fetcher::InMemoryFetcher;

    /// Minimal stub codec that uses exact L2 as both distance functions.
    struct L2Codec;

    struct L2Quantized(Vec<f32>);

    impl AsRef<nodedb_codec::vector_quant::layout::UnifiedQuantizedVector> for L2Quantized {
        fn as_ref(&self) -> &nodedb_codec::vector_quant::layout::UnifiedQuantizedVector {
            panic!("stub: UnifiedQuantizedVector not needed for L2Codec tests")
        }
    }

    impl VectorCodec for L2Codec {
        type Quantized = L2Quantized;
        type Query = Vec<f32>;

        fn encode(&self, v: &[f32]) -> Self::Quantized {
            L2Quantized(v.to_vec())
        }

        fn prepare_query(&self, q: &[f32]) -> Self::Query {
            q.to_vec()
        }

        fn fast_symmetric_distance(&self, a: &Self::Quantized, b: &Self::Quantized) -> f32 {
            l2_squared(&a.0, &b.0)
        }

        fn exact_asymmetric_distance(&self, q: &Self::Query, v: &Self::Quantized) -> f32 {
            l2_squared(q, &v.0)
        }
    }

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed.max(1);
        let mut xorshift = move || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32) / (u64::MAX as f32)
        };
        (0..n)
            .map(|_| (0..dim).map(|_| xorshift()).collect())
            .collect()
    }

    #[test]
    fn beam_search_finds_self_as_nearest() {
        let dim = 8;
        let n = 50;
        let codec = L2Codec;

        let vecs = random_vecs(n, dim, 42);
        let ids: Vec<u64> = (0..n as u64).collect();
        let quantized: Vec<L2Quantized> = vecs.iter().map(|v| codec.encode(v)).collect();

        let graph = build_vamana(&vecs, &ids, &codec, &quantized, 8, 1.2, 20).unwrap();

        // Query with the vector at index 7; it should be the nearest result.
        let query_vec = vecs[7].clone();
        let query = codec.prepare_query(&query_vec);
        let mut fetcher = InMemoryFetcher::new(dim, vecs.clone());

        let results = beam_search(&graph, &query, &codec, &quantized, &mut fetcher, 5, 20).unwrap();

        assert!(
            !results.is_empty(),
            "beam_search must return at least one result"
        );
        assert_eq!(
            results[0].id, 7,
            "nearest result must be the query vector itself"
        );
        assert!(
            results[0].distance < 1e-6,
            "distance to self must be near zero"
        );
    }

    #[test]
    fn wrong_dimension_is_a_typed_error() {
        let dim = 4;
        let codec = L2Codec;
        let vecs = random_vecs(10, dim, 7);
        let ids: Vec<u64> = (0..10).collect();
        let quantized: Vec<L2Quantized> = vecs.iter().map(|v| codec.encode(v)).collect();
        let graph = build_vamana(&vecs, &ids, &codec, &quantized, 4, 1.2, 8).unwrap();
        let mut fetcher = InMemoryFetcher::new(dim, vecs.clone());
        let query = codec.prepare_query(&vecs[0]);
        let candidates =
            beam_search(&graph, &query, &codec, &quantized, &mut fetcher, 3, 8).unwrap();
        assert!(matches!(
            rerank(candidates, &[0.0; 3], &mut fetcher, &graph, 3),
            Err(VectorError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        ));

        let mut ragged = vecs.clone();
        ragged[3] = vec![0.0; 2];
        let ragged_q: Vec<L2Quantized> = ragged.iter().map(|v| codec.encode(v)).collect();
        assert!(matches!(
            build_vamana(&ragged, &ids, &codec, &ragged_q, 4, 1.2, 8),
            Err(VectorError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
        assert!(matches!(
            beam_search(&graph, &query, &codec, &quantized[..5], &mut fetcher, 3, 8),
            Err(VectorError::InvalidInput { .. })
        ));
    }
}
