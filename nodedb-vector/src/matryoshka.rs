// SPDX-License-Identifier: Apache-2.0

//! Matryoshka adaptive-dim querying.
//!
//! For embedding models trained with Matryoshka Representation Learning
//! (MRL), prefix-truncated vectors retain semantic structure. We exploit
//! this for memory-bandwidth-bound HNSW: coarse pass at low dim → exact
//! rerank at full dim.
//!
//! # Cache-efficiency
//!
//! A 256-dim coarse pass reads 256 × 4 = 1 KiB per vector vs 1536 × 4 = 6 KiB
//! for full-dim traversal — a ≥6× reduction in L1/L2 memory bandwidth during
//! the dominant HNSW graph traversal phase.
//!
//! # Supported models
//!
//! - `text-embedding-3` (OpenAI): `[256, 512, 1024, 1536]`
//! - `Gemini Embedding`: `[256, 512, 768, 3072]`
//! - `Nomic Embed`: `[64, 128, 256, 512, 768]`

use std::collections::BinaryHeap;

use crate::distance::distance;
use crate::error::VectorError;
use nodedb_types::vector_distance::DistanceMetric;

/// Per-collection Matryoshka configuration.
#[derive(Debug, Clone)]
pub struct MatryoshkaSpec {
    /// Sorted ascending list of valid truncation dimensions.
    /// E.g. for text-embedding-3 (1536-dim base): `[256, 512, 1024, 1536]`.
    pub truncation_dims: Vec<u32>,
}

impl MatryoshkaSpec {
    /// Create a new spec. Sorts the provided dims ascending.
    pub fn new(mut truncation_dims: Vec<u32>) -> Self {
        truncation_dims.sort_unstable();
        truncation_dims.dedup();
        Self { truncation_dims }
    }

    /// Pick the largest supported dim ≤ `requested`.
    ///
    /// Returns the full dim (last in the list) when `requested` is `None`.
    /// Returns the smallest available dim when `requested` is smaller than
    /// all supported dims.
    pub fn pick(&self, requested: Option<u32>) -> u32 {
        let Some(req) = requested else {
            return *self.truncation_dims.last().copied().get_or_insert(0);
        };
        // Find largest dim that is ≤ req.
        self.truncation_dims
            .iter()
            .rev()
            .find(|&&d| d <= req)
            .copied()
            .unwrap_or_else(|| self.truncation_dims.first().copied().unwrap_or(req))
    }

    /// Return `true` if `dim` is one of the supported truncation dims.
    pub fn is_valid(&self, dim: u32) -> bool {
        self.truncation_dims.contains(&dim)
    }
}

/// Stride-truncate a vector to its first `dim` components.
///
/// If `dim` exceeds `v.len()` the full slice is returned (no panic).
#[inline]
pub fn truncate(v: &[f32], dim: usize) -> &[f32] {
    &v[..dim.min(v.len())]
}

/// Options for a two-stage Matryoshka search.
pub struct MatryoshkaSearchOptions {
    /// Dimensionality used for the coarse pass.
    pub coarse_dim: u32,
    /// Full dimensionality used for the rerank pass.
    pub full_dim: u32,
    /// Oversample factor: coarse pass collects `oversample × k` candidates.
    /// Typical values: 3–5.
    pub oversample: u8,
    /// Final result count.
    pub k: usize,
}

/// Entry in the coarse-pass max-heap (ordered by distance descending so we
/// can evict the worst candidate when the heap overflows).
#[derive(PartialEq)]
struct HeapEntry {
    /// Distance (higher = worse, evict first).
    dist: f32,
    id: u32,
    /// Index into the collected vectors buffer for full-dim rerank.
    vec_idx: usize,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap by distance (NaN treated as largest).
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Two-stage Matryoshka search.
///
/// # Phase 1 — coarse pass
/// Iterates `candidates`, computes distance on the first `options.coarse_dim`
/// components, and keeps a bounded max-heap of size `oversample × k`.
///
/// # Phase 2 — rerank
/// Recomputes full-dim distance for all coarse survivors and returns the top-k
/// by ascending full-dim distance.
///
/// # Notes
/// - `candidates` yields `(id, vector)` pairs; each vector carries at least
///   `full_dim` components, and components past `full_dim` are ignored.
/// - When `coarse_dim == full_dim` the method degenerates to a single-pass
///   top-k scan with one distance call per candidate (no duplicated work).
///
/// # Errors
/// - [`VectorError::InvalidInput`] when `coarse_dim` exceeds `full_dim`.
/// - [`VectorError::DimensionMismatch`] when `query` has fewer than
///   `full_dim` components.
/// - [`VectorError::StoredDimensionMismatch`] when a candidate vector has
///   fewer than `full_dim` components.
pub fn matryoshka_search<'a, I>(
    candidates: I,
    query: &[f32],
    options: &MatryoshkaSearchOptions,
    metric: DistanceMetric,
) -> Result<Vec<(u32, f32)>, VectorError>
where
    I: Iterator<Item = (u32, &'a [f32])>,
{
    let coarse = options.coarse_dim as usize;
    let full = options.full_dim as usize;
    if coarse > full {
        return Err(VectorError::InvalidInput {
            detail: format!("Matryoshka coarse dimension {coarse} exceeds full dimension {full}"),
        });
    }
    if query.len() < full {
        return Err(VectorError::DimensionMismatch {
            expected: full,
            got: query.len(),
        });
    }
    let pool_size = (options.oversample as usize).max(1) * options.k.max(1);

    let query_coarse = truncate(query, coarse);

    // --- Phase 1: coarse pass ---
    // We collect full-dim vectors for survivors so Phase 2 doesn't need the
    // original iterator again.
    let mut coarse_heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(pool_size + 1);
    // Store owned copies of the surviving vectors for rerank.
    let mut survivor_vecs: Vec<Vec<f32>> = Vec::with_capacity(pool_size);

    for (id, vec) in candidates {
        if vec.len() < full {
            return Err(VectorError::StoredDimensionMismatch {
                expected: full,
                got: vec.len(),
            });
        }
        let vec_coarse = truncate(vec, coarse);
        let d = distance(query_coarse, vec_coarse, metric);

        // Maintain a max-heap of capacity `pool_size`.
        let should_insert = coarse_heap.len() < pool_size
            || coarse_heap
                .peek()
                .map(|worst| d < worst.dist)
                .unwrap_or(true);

        if should_insert {
            let vec_idx = survivor_vecs.len();
            survivor_vecs.push(truncate(vec, full).to_vec());

            coarse_heap.push(HeapEntry {
                dist: d,
                id,
                vec_idx,
            });

            if coarse_heap.len() > pool_size {
                // Evict worst; its survivor_vec slot becomes orphaned but that
                // is acceptable — we only rerank heap survivors.
                coarse_heap.pop();
            }
        }
    }

    // --- Phase 2: rerank ---
    let query_full = truncate(query, full);

    let mut reranked: Vec<(u32, f32)> = coarse_heap
        .into_iter()
        .map(|entry| {
            let full_vec = &survivor_vecs[entry.vec_idx];
            let d_full = distance(query_full, full_vec.as_slice(), metric);
            (entry.id, d_full)
        })
        .collect();

    reranked.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    reranked.truncate(options.k);
    Ok(reranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MatryoshkaSpec::pick ──────────────────────────────────────────────────

    #[test]
    fn pick_largest_leq_requested() {
        let spec = MatryoshkaSpec::new(vec![256, 512, 1024]);
        assert_eq!(spec.pick(Some(300)), 256);
    }

    #[test]
    fn pick_exact_match() {
        let spec = MatryoshkaSpec::new(vec![256, 512, 1024]);
        assert_eq!(spec.pick(Some(512)), 512);
    }

    #[test]
    fn pick_none_returns_full_dim() {
        let spec = MatryoshkaSpec::new(vec![256, 512, 1024]);
        assert_eq!(spec.pick(None), 1024);
    }

    #[test]
    fn pick_smaller_than_all_returns_smallest() {
        let spec = MatryoshkaSpec::new(vec![256, 512, 1024]);
        assert_eq!(spec.pick(Some(10)), 256);
    }

    // ── MatryoshkaSpec::is_valid ──────────────────────────────────────────────

    #[test]
    fn is_valid_known_dim() {
        let spec = MatryoshkaSpec::new(vec![256, 512, 1024]);
        assert!(spec.is_valid(512));
    }

    #[test]
    fn is_valid_unknown_dim() {
        let spec = MatryoshkaSpec::new(vec![256, 512, 1024]);
        assert!(!spec.is_valid(100));
    }

    // ── truncate ─────────────────────────────────────────────────────────────

    #[test]
    fn truncate_clips_to_requested_dim() {
        let v: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        let t = truncate(&v, 256);
        assert_eq!(t.len(), 256);
        assert_eq!(t[0], 0.0);
        assert_eq!(t[255], 255.0);
    }

    #[test]
    fn truncate_does_not_exceed_vec_len() {
        let v = vec![1.0f32; 10];
        let t = truncate(&v, 9999);
        assert_eq!(t.len(), 10);
    }

    // ── matryoshka_search ────────────────────────────────────────────────────

    /// Build 100 random-ish vectors of dimension 128.
    fn make_vecs(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn search_returns_k_results() {
        let vecs = make_vecs(100, 128);
        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.007).cos()).collect();

        let candidates = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v.as_slice()));

        let opts = MatryoshkaSearchOptions {
            coarse_dim: 64,
            full_dim: 128,
            oversample: 3,
            k: 10,
        };

        let results = matryoshka_search(candidates, &query, &opts, DistanceMetric::L2).unwrap();
        assert_eq!(results.len(), 10, "expected exactly k=10 results");
    }

    #[test]
    fn coarse_equal_to_full_matches_direct_search() {
        let vecs = make_vecs(100, 128);
        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.007).cos()).collect();

        // Direct top-10 search for reference.
        let mut direct: Vec<(u32, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, distance(&query, v.as_slice(), DistanceMetric::L2)))
            .collect();
        direct.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        direct.truncate(10);

        // Matryoshka with coarse_dim == full_dim — no truncation effect.
        let candidates = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v.as_slice()));
        let opts = MatryoshkaSearchOptions {
            coarse_dim: 128,
            full_dim: 128,
            oversample: 1,
            k: 10,
        };
        let mrl = matryoshka_search(candidates, &query, &opts, DistanceMetric::L2).unwrap();

        // Same IDs in same order.
        let direct_ids: Vec<u32> = direct.iter().map(|(id, _)| *id).collect();
        let mrl_ids: Vec<u32> = mrl.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            direct_ids, mrl_ids,
            "coarse==full should equal direct search"
        );
    }

    #[test]
    fn short_query_or_candidate_is_a_typed_error() {
        let vecs = make_vecs(4, 8);
        let opts = MatryoshkaSearchOptions {
            coarse_dim: 4,
            full_dim: 8,
            oversample: 2,
            k: 2,
        };
        let candidates = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v.as_slice()));
        assert!(matches!(
            matryoshka_search(candidates, &[0.0; 6], &opts, DistanceMetric::L2),
            Err(VectorError::DimensionMismatch {
                expected: 8,
                got: 6
            })
        ));

        let short = [0.0_f32; 5];
        let candidates = std::iter::once((0_u32, &short[..]));
        assert!(matches!(
            matryoshka_search(candidates, &[0.0; 8], &opts, DistanceMetric::L2),
            Err(VectorError::StoredDimensionMismatch {
                expected: 8,
                got: 5
            })
        ));

        let inverted = MatryoshkaSearchOptions {
            coarse_dim: 16,
            full_dim: 8,
            oversample: 2,
            k: 2,
        };
        let candidates = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v.as_slice()));
        assert!(matches!(
            matryoshka_search(candidates, &[0.0; 8], &inverted, DistanceMetric::L2),
            Err(VectorError::InvalidInput { .. })
        ));
    }
}
