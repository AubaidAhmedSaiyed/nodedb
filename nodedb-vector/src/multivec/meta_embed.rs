// SPDX-License-Identifier: Apache-2.0

//! MetaEmbed-specific search helpers.
//!
//! MetaEmbed (ICLR 2026) stores K pre-computed Meta Token vectors per document
//! instead of one vector per token.  At query time the Matryoshka ordering of
//! the query Meta Tokens allows a `budget` parameter to trade recall for
//! latency: `budget=k` is full accuracy; `budget=1` is fastest.
//!
//! `meta_embed_search` wires together:
//! 1. Optional PLAID candidate pruning.
//! 2. Budgeted MaxSim scoring over the remaining candidates.
//! 3. Top-k selection.

use nodedb_types::vector_distance::DistanceMetric;

use super::plaid::PlaidPruner;
use super::scoring::budgeted_maxsim;
use super::storage::MultiVectorStore;
use crate::error::{VectorError, check_dim};

/// Search a `MultiVectorStore` using budgeted MaxSim with optional PLAID
/// candidate pruning.
///
/// # Parameters
/// * `store`  — the document collection.
/// * `plaid`  — optional PLAID pruner (pass `None` to scan all docs).
/// * `query`  — query Meta Token vectors (Matryoshka ordering).
/// * `budget` — number of leading query tokens to use; 0 falls back to all.
/// * `k`      — number of top documents to return.
/// * `metric` — distance metric (Cosine recommended for MetaEmbed).
///
/// # Returns
/// A `Vec<(doc_id, score)>` sorted descending by score, length ≤ `k`, or
/// [`VectorError::DimensionMismatch`] when a query vector does not have the
/// store dimension.
pub fn meta_embed_search(
    store: &MultiVectorStore,
    plaid: Option<&PlaidPruner>,
    query: &[Vec<f32>],
    budget: u8,
    k: usize,
    metric: DistanceMetric,
) -> Result<Vec<(u32, f32)>, VectorError> {
    for v in query {
        check_dim(store.dim, v.len())?;
    }
    if k == 0 || query.is_empty() {
        return Ok(Vec::new());
    }

    // Effective budget: 0 means use all query vectors.
    let effective_budget = if budget == 0 {
        query.len() as u8
    } else {
        budget
    };

    // Determine candidate set.
    let candidate_ids: Vec<u32> = match plaid {
        Some(pruner) => pruner.candidates(query)?,
        None => store.iter().map(|doc| doc.doc_id).collect(),
    };

    // Score each candidate.
    let mut scored: Vec<(u32, f32)> = candidate_ids
        .into_iter()
        .filter_map(|doc_id| {
            store.get(doc_id).map(|doc| {
                let score = budgeted_maxsim(query, &doc.vectors, effective_budget, metric);
                (doc_id, score)
            })
        })
        .collect();

    // Sort descending by score.
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multivec::storage::{MultiVecMode, MultiVectorDoc, MultiVectorStore};

    /// Build a store with `n` documents, each with one unit-vector at
    /// dimension `i % dim`.
    fn build_store(n: u32, dim: usize, k: u8) -> MultiVectorStore {
        let mut store = MultiVectorStore::new(dim, MultiVecMode::MetaToken { k });
        for i in 0..n {
            let mut vecs: Vec<Vec<f32>> = Vec::new();
            for j in 0..k as usize {
                let mut v = vec![0.0f32; dim];
                // Each Meta Token of doc i points in direction (i + j) % dim.
                v[(i as usize + j) % dim] = 1.0;
                vecs.push(v);
            }
            store
                .insert(MultiVectorDoc {
                    doc_id: i,
                    vectors: vecs,
                })
                .unwrap();
        }
        store
    }

    #[test]
    fn search_returns_at_most_k_results() {
        let store = build_store(10, 4, 2);
        let query = vec![vec![1.0f32, 0.0, 0.0, 0.0]];
        let results =
            meta_embed_search(&store, None, &query, 2, 3, DistanceMetric::Cosine).unwrap();
        assert!(results.len() <= 3);
    }

    #[test]
    fn search_results_sorted_descending() {
        let store = build_store(8, 4, 2);
        let query = vec![vec![1.0f32, 0.0, 0.0, 0.0]];
        let results =
            meta_embed_search(&store, None, &query, 2, 8, DistanceMetric::Cosine).unwrap();
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "not sorted: {:?}", results);
        }
    }

    #[test]
    fn plaid_filtered_results_are_subset_of_unfiltered() {
        let store = build_store(9, 2, 2);

        // Train PLAID on the store.
        let pruner = PlaidPruner::train(&store, 3, 10, 99);

        let query = vec![vec![1.0f32, 0.0f32]];
        let unfiltered =
            meta_embed_search(&store, None, &query, 2, 9, DistanceMetric::Cosine).unwrap();
        let filtered =
            meta_embed_search(&store, Some(&pruner), &query, 2, 9, DistanceMetric::Cosine).unwrap();

        let unfiltered_ids: std::collections::HashSet<u32> =
            unfiltered.iter().map(|(id, _)| *id).collect();

        for (id, _) in &filtered {
            assert!(
                unfiltered_ids.contains(id),
                "filtered result {id} not in unfiltered set"
            );
        }
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let store = build_store(5, 4, 2);
        let results = meta_embed_search(&store, None, &[], 2, 5, DistanceMetric::Cosine).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let store = build_store(5, 4, 2);
        let query = vec![vec![1.0f32, 0.0, 0.0, 0.0]];
        let results =
            meta_embed_search(&store, None, &query, 2, 0, DistanceMetric::Cosine).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn top_result_is_best_matching_doc() {
        // Doc 0 has its first Meta Token at direction 0 (index 0).
        // Query is also in direction 0 — doc 0 should rank first.
        let store = build_store(4, 4, 1);
        let query = vec![vec![1.0f32, 0.0, 0.0, 0.0]];
        let results =
            meta_embed_search(&store, None, &query, 1, 1, DistanceMetric::Cosine).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0, "expected doc_id=0 to be top result");
    }

    #[test]
    fn wrong_dimension_query_is_a_typed_error() {
        let store = build_store(4, 4, 2);
        let pruner = PlaidPruner::train(&store, 2, 3, 7);
        let query = vec![vec![1.0f32, 0.0, 0.0]];
        for plaid in [None, Some(&pruner)] {
            let result = meta_embed_search(&store, plaid, &query, 1, 2, DistanceMetric::Cosine);
            assert!(
                matches!(
                    result,
                    Err(VectorError::DimensionMismatch {
                        expected: 4,
                        got: 3
                    })
                ),
                "{result:?}"
            );
        }
    }
}
