// SPDX-License-Identifier: Apache-2.0

//! PLAID-style centroid-based candidate pruning for multi-vector search.
//!
//! Builds K-means centroids over all document vectors.  Each document is
//! encoded as a sorted bag of centroid IDs.  At query time the query's
//! centroid bag is computed and only documents whose centroid bag overlaps
//! the query bag are returned as candidates.
//!
//! Reference: Santhanam et al., "PLAID: An Efficient Engine for Late
//! Interaction Retrieval", CIKM 2022.

use std::collections::{HashMap, HashSet};

use crate::distance::scalar::scalar_distance;
use crate::error::{VectorError, check_dim};
use nodedb_types::vector_distance::DistanceMetric;

use super::storage::MultiVectorStore;

// ---------------------------------------------------------------------------
// Internal Lloyd's K-means (tiny, self-contained)
// ---------------------------------------------------------------------------

/// Assign each vector to its nearest centroid index.
fn assign(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    vectors
        .iter()
        .map(|v| {
            centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, scalar_distance(v, c, DistanceMetric::L2)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

/// Deterministic LCG step for reproducible randomness across train calls.
fn lcg_next(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

/// Distance from `v` to the nearest centroid in `centroids`.
fn min_dist_to_centroids(v: &[f32], centroids: &[Vec<f32>]) -> f32 {
    centroids
        .iter()
        .map(|c| scalar_distance(v, c, DistanceMetric::L2))
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(f32::INFINITY)
}

/// Recompute centroids as the mean of their assigned vectors. Empty clusters
/// are re-seeded to the input vector farthest from all live centroids — this
/// guarantees every centroid covers some part of the input space and prevents
/// k-means from collapsing into fewer-than-k effective clusters.
fn recompute(
    vectors: &[Vec<f32>],
    assignments: &[usize],
    num_centroids: usize,
    dim: usize,
    prev_centroids: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut sums = vec![vec![0.0f32; dim]; num_centroids];
    let mut counts = vec![0usize; num_centroids];

    for (v, &c) in vectors.iter().zip(assignments.iter()) {
        for (s, x) in sums[c].iter_mut().zip(v.iter()) {
            *s += x;
        }
        counts[c] += 1;
    }

    // First pass: average populated clusters in place.
    for (s, &n) in sums.iter_mut().zip(counts.iter()) {
        if n > 0 {
            s.iter_mut().for_each(|x| *x /= n as f32);
        }
    }

    // Second pass: re-seed empty clusters from vectors farthest from any live
    // centroid. Snapshot the live set first so each empty slot is filled
    // deterministically and subsequent re-seeds in the same call see the
    // updated pool.
    for c_idx in 0..num_centroids {
        if counts[c_idx] != 0 {
            continue;
        }
        let live: Vec<Vec<f32>> = counts
            .iter()
            .enumerate()
            .filter(|(i, cnt)| *i != c_idx && **cnt > 0)
            .map(|(i, _)| sums[i].clone())
            .collect();
        let seed_pool: &[Vec<f32>] = if live.is_empty() {
            prev_centroids
        } else {
            &live
        };
        let farthest = vectors
            .iter()
            .map(|v| (v, min_dist_to_centroids(v, seed_pool)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(v, _)| v.clone());
        if let Some(v) = farthest {
            sums[c_idx] = v;
            counts[c_idx] = 1; // mark live so later empty slots see it as a seed.
        } else if c_idx < prev_centroids.len() {
            sums[c_idx] = prev_centroids[c_idx].clone();
            counts[c_idx] = 1;
        }
    }

    sums
}

/// k-means++ initialisation: first centroid uniform random, subsequent
/// centroids drawn with probability proportional to squared distance from the
/// nearest already-chosen centroid. Deterministic given `seed`.
fn kmeans_plus_plus_init(vectors: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed.wrapping_add(1);
    let first = (lcg_next(&mut state) as usize) % vectors.len();
    let mut centroids: Vec<Vec<f32>> = vec![vectors[first].clone()];

    while centroids.len() < k {
        let dists: Vec<f32> = vectors
            .iter()
            .map(|v| {
                let d = min_dist_to_centroids(v, &centroids);
                d * d
            })
            .collect();
        let total: f64 = dists.iter().map(|&d| d as f64).sum();
        if total <= 0.0 {
            // All remaining vectors coincide with existing centroids; just
            // pick any unique-by-index vector to fill k.
            let idx = (lcg_next(&mut state) as usize) % vectors.len();
            centroids.push(vectors[idx].clone());
            continue;
        }
        // Deterministic weighted draw from the LCG.
        let r = (lcg_next(&mut state) as f64) / (u64::MAX as f64) * total;
        let mut acc = 0.0f64;
        let mut pick = vectors.len() - 1;
        for (i, &d) in dists.iter().enumerate() {
            acc += d as f64;
            if acc >= r {
                pick = i;
                break;
            }
        }
        centroids.push(vectors[pick].clone());
    }

    centroids
}

/// Run Lloyd's K-means with k-means++ initialisation. Empty clusters are
/// re-seeded each iteration so the result always has exactly `k` distinct
/// centroids covering the input space.
fn kmeans(
    vectors: &[Vec<f32>],
    num_centroids: usize,
    iters: usize,
    seed: u64,
    dim: usize,
) -> Vec<Vec<f32>> {
    if vectors.is_empty() || num_centroids == 0 {
        return Vec::new();
    }

    let k = num_centroids.min(vectors.len());
    let mut centroids = kmeans_plus_plus_init(vectors, k, seed);

    for _ in 0..iters {
        let assignments = assign(vectors, &centroids);
        let new_centroids = recompute(vectors, &assignments, k, dim, &centroids);
        centroids = new_centroids;
    }

    centroids
}

// ---------------------------------------------------------------------------
// PlaidPruner
// ---------------------------------------------------------------------------

/// PLAID centroid-based candidate pruner.
///
/// After `train`, call `candidates` at query time to get the set of document
/// IDs whose centroid bag overlaps the query's centroid bag.
pub struct PlaidPruner {
    pub centroids: Vec<Vec<f32>>,
    /// Sorted list of centroid IDs for each document.
    doc_centroids: HashMap<u32, Vec<u16>>,
}

impl PlaidPruner {
    /// Train the pruner from a `MultiVectorStore`.
    ///
    /// * `num_centroids` — number of K-means clusters.
    /// * `kmeans_iters` — Lloyd iterations.
    /// * `seed` — deterministic seed for centroid initialisation.
    pub fn train(
        store: &MultiVectorStore,
        num_centroids: u16,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        let dim = store.dim;
        let nc = num_centroids as usize;

        // Collect all document vectors for K-means training.
        let all_vectors: Vec<Vec<f32>> = store
            .iter()
            .flat_map(|doc| doc.vectors.iter().cloned())
            .collect();

        let centroids = kmeans(&all_vectors, nc, kmeans_iters, seed, dim);

        // Encode each document as a sorted, deduplicated bag of centroid IDs.
        let doc_centroids: HashMap<u32, Vec<u16>> = store
            .iter()
            .map(|doc| {
                let mut ids: Vec<u16> = doc
                    .vectors
                    .iter()
                    .map(|v| {
                        centroids
                            .iter()
                            .enumerate()
                            .map(|(i, c)| (i as u16, scalar_distance(v, c, DistanceMetric::L2)))
                            .min_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    })
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                (doc.doc_id, ids)
            })
            .collect();

        Self {
            centroids,
            doc_centroids,
        }
    }

    /// Return candidate doc IDs whose centroid bag overlaps the query's
    /// centroid bag.
    ///
    /// The query centroid bag is the set of nearest centroids for each query
    /// vector.
    ///
    /// A query vector without the centroid dimension fails with
    /// [`VectorError::DimensionMismatch`].
    pub fn candidates(&self, query: &[Vec<f32>]) -> Result<Vec<u32>, VectorError> {
        let Some(first) = self.centroids.first() else {
            return Ok(Vec::new());
        };
        for v in query {
            check_dim(first.len(), v.len())?;
        }
        if query.is_empty() {
            return Ok(Vec::new());
        }

        // Build query centroid bag.
        let query_bag: HashSet<u16> = query
            .iter()
            .filter_map(|v| {
                self.centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i as u16, scalar_distance(v, c, DistanceMetric::L2)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(id, _)| id)
            })
            .collect();

        // Collect docs that share at least one centroid with the query.
        Ok(self
            .doc_centroids
            .iter()
            .filter(|(_, doc_ids)| doc_ids.iter().any(|id| query_bag.contains(id)))
            .map(|(&doc_id, _)| doc_id)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multivec::storage::{MultiVecMode, MultiVectorDoc, MultiVectorStore};

    fn build_store() -> MultiVectorStore {
        let mut store = MultiVectorStore::new(2, MultiVecMode::PerToken);

        // Three well-separated clusters of documents.
        // Cluster A: docs 0–2 near (0,0).
        // Cluster B: docs 3–5 near (10,0).
        // Cluster C: docs 6–8 near (0,10).
        for i in 0u32..3 {
            store
                .insert(MultiVectorDoc {
                    doc_id: i,
                    vectors: vec![vec![i as f32 * 0.1, i as f32 * 0.1]],
                })
                .unwrap();
        }
        for i in 3u32..6 {
            store
                .insert(MultiVectorDoc {
                    doc_id: i,
                    vectors: vec![vec![10.0 + i as f32 * 0.1, 0.0]],
                })
                .unwrap();
        }
        for i in 6u32..9 {
            store
                .insert(MultiVectorDoc {
                    doc_id: i,
                    vectors: vec![vec![0.0, 10.0 + i as f32 * 0.1]],
                })
                .unwrap();
        }
        store
    }

    #[test]
    fn train_produces_correct_centroid_count() {
        let store = build_store();
        let pruner = PlaidPruner::train(&store, 3, 10, 42);
        assert_eq!(pruner.centroids.len(), 3);
    }

    #[test]
    fn centroids_have_correct_dim() {
        let store = build_store();
        let pruner = PlaidPruner::train(&store, 3, 10, 42);
        for c in &pruner.centroids {
            assert_eq!(c.len(), 2);
        }
    }

    #[test]
    fn candidates_non_empty_for_matching_query() {
        let store = build_store();
        let pruner = PlaidPruner::train(&store, 3, 10, 42);

        // A query near cluster A should return at least some candidates.
        let query = vec![vec![0.0f32, 0.0f32]];
        let cands = pruner.candidates(&query).unwrap();
        assert!(!cands.is_empty(), "expected at least one candidate");
    }

    #[test]
    fn candidates_empty_when_no_centroids() {
        // An empty store produces a pruner with no centroids.
        let store = MultiVectorStore::new(2, MultiVecMode::PerToken);
        let pruner = PlaidPruner::train(&store, 3, 5, 1);
        let query = vec![vec![0.0f32, 0.0f32]];
        assert!(pruner.candidates(&query).unwrap().is_empty());
    }

    #[test]
    fn candidates_cover_input_range() {
        // After training, the set of all-doc candidates (using multiple query
        // vectors spanning the whole space) should cover all 9 documents.
        let store = build_store();
        let pruner = PlaidPruner::train(&store, 3, 15, 7);
        let query = vec![
            vec![0.0f32, 0.0f32],
            vec![10.0f32, 0.0f32],
            vec![0.0f32, 10.0f32],
        ];
        let mut cands = pruner.candidates(&query).unwrap();
        cands.sort_unstable();
        cands.dedup();
        assert_eq!(cands.len(), 9, "all docs should be candidates: {:?}", cands);
    }
}
