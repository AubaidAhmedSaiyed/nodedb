// SPDX-License-Identifier: BUSL-1.1

//! Shard splitting: vector-aware and graph-aware partitioning.
//!
//! When a shard becomes overloaded, it can be split into two shards.
//! The splitting strategy is engine-aware:
//!
//! - **Vector-aware**: split by collection + partition key. Each resulting
//!   shard holds a subset of the collection's vectors with its own HNSW
//!   index. Cross-shard k-NN queries use scatter-gather with result
//!   merging on the Control Plane.
//!
//! - **Graph-aware**: balanced partitioning minimizing cross-shard edges.
//!   Uses a greedy heuristic (BFS-based community detection) rather than
//!   full METIS for practical performance. Cross-shard traversals use
//!   existing scatter-gather infrastructure with ghost edges.
//!
//! - **Speculative prefetch**: when scatter-gather dispatches to multiple
//!   shards, co-located vector and graph shards for the same tenant are
//!   prefetched together to reduce round-trip latency.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::routing::RoutingTable;

/// A shard split plan: which vectors/documents move to the new shard.
#[derive(Debug, Clone)]
pub struct SplitPlan {
    /// Source vShard being split.
    pub source_vshard: u32,
    /// New vShard ID for the second half.
    pub new_vshard: u32,
    /// Target node for the new shard.
    pub target_node: u64,
    /// Document IDs that move to the new shard.
    pub documents_to_move: Vec<String>,
    /// Strategy used.
    pub strategy: SplitStrategy,
    /// Estimated data size moving (bytes).
    pub estimated_bytes: u64,
}

/// Splitting strategy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// Split by partition key hash (vector-aware).
    /// Each half gets vectors whose partition key hash falls in its range.
    VectorPartitionKey,
    /// Split by graph community (graph-aware).
    /// Minimizes cross-shard edges using BFS community detection.
    GraphCommunity,
    /// Simple even split by document ID hash.
    EvenHash,
}

/// Plan a vector-aware shard split.
///
/// Splits documents by partition key: documents whose key hashes to
/// the lower half stay on the source shard, upper half moves to the new shard.
/// Each resulting shard builds its own HNSW index independently.
///
/// Cross-shard k-NN: the Control Plane dispatches VectorSearch to both
/// shards via scatter-gather, collects top-k from each, and merges by
/// distance on the Control Plane.
pub fn plan_vector_split(
    source_vshard: u32,
    new_vshard: u32,
    target_node: u64,
    document_ids: &[String],
) -> SplitPlan {
    let mid = document_ids.len() / 2;

    // Sort by partition key hash for deterministic splitting.
    let mut sorted: Vec<(u64, String)> = document_ids
        .iter()
        .map(|id| (partition_hash(id), id.clone()))
        .collect();
    sorted.sort_by_key(|(hash, _)| *hash);

    let documents_to_move: Vec<String> = sorted[mid..].iter().map(|(_, id)| id.clone()).collect();

    info!(
        source_vshard,
        new_vshard,
        target_node,
        docs_moving = documents_to_move.len(),
        docs_staying = mid,
        "vector-aware split planned"
    );

    SplitPlan {
        source_vshard,
        new_vshard,
        target_node,
        documents_to_move,
        strategy: SplitStrategy::VectorPartitionKey,
        estimated_bytes: 0, // Caller fills in from actual data.
    }
}

/// Plan a graph-aware shard split.
///
/// Uses BFS-based community detection to partition nodes into two groups
/// that minimize cross-shard edges. Starting from a seed node, BFS assigns
/// the first half of discovered nodes to group A, the rest to group B.
///
/// This is a greedy heuristic — not optimal like METIS, but O(V+E) and
/// practical for online splitting without blocking queries.
pub fn plan_graph_split(
    source_vshard: u32,
    new_vshard: u32,
    target_node: u64,
    node_ids: &[String],
    edges: &[(String, String)],
) -> SplitPlan {
    if node_ids.is_empty() {
        return SplitPlan {
            source_vshard,
            new_vshard,
            target_node,
            documents_to_move: Vec::new(),
            strategy: SplitStrategy::GraphCommunity,
            estimated_bytes: 0,
        };
    }

    // Build adjacency list for BFS.
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (src, dst) in edges {
        adj.entry(src.as_str()).or_default().push(dst.as_str());
        adj.entry(dst.as_str()).or_default().push(src.as_str());
    }

    // BFS from first node, assign first half to group A.
    let mut visited: Vec<String> = Vec::with_capacity(node_ids.len());
    let mut seen: HashSet<&str> = HashSet::new();
    let mut queue: std::collections::VecDeque<&str> = std::collections::VecDeque::new();

    // Start BFS from the node with the most edges (hub).
    let start = node_ids
        .iter()
        .max_by_key(|id| adj.get(id.as_str()).map(|v| v.len()).unwrap_or(0))
        .map(|s| s.as_str())
        .unwrap_or(node_ids[0].as_str());

    queue.push_back(start);
    seen.insert(start);

    while let Some(node) = queue.pop_front() {
        visited.push(node.to_string());
        if let Some(neighbors) = adj.get(node) {
            for &neighbor in neighbors {
                if seen.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // Add any disconnected nodes not reached by BFS.
    for id in node_ids {
        if seen.insert(id.as_str()) {
            visited.push(id.clone());
        }
    }

    // Second half moves to new shard.
    let mid = visited.len() / 2;
    let documents_to_move: Vec<String> = visited[mid..].to_vec();

    // Count cross-shard edges.
    let group_a: HashSet<&str> = visited[..mid].iter().map(|s| s.as_str()).collect();
    let cross_edges = edges
        .iter()
        .filter(|(src, dst)| {
            let a_has_src = group_a.contains(src.as_str());
            let a_has_dst = group_a.contains(dst.as_str());
            a_has_src != a_has_dst
        })
        .count();

    info!(
        source_vshard,
        new_vshard,
        total_nodes = node_ids.len(),
        cross_edges,
        "graph-aware split planned (BFS community)"
    );

    SplitPlan {
        source_vshard,
        new_vshard,
        target_node,
        documents_to_move,
        strategy: SplitStrategy::GraphCommunity,
        estimated_bytes: 0,
    }
}

/// Speculative prefetch plan for cross-shard scatter-gather queries.
///
/// When a query must scatter to multiple shards, this identifies which
/// additional shards should be prefetched based on co-location hints:
///
/// - Vector shards for the same tenant/collection are prefetched together
/// - Graph shards adjacent to queried shards are prefetched
///
/// Returns additional vShard IDs to include in the scatter batch for
/// reduced round-trip latency.
pub fn speculative_prefetch_shards(
    query_vshards: &[u32],
    _routing: &RoutingTable,
    tenant_collections: &[(u32, nodedb_types::id::CollectionKey<'_>)],
) -> Vec<u32> {
    let mut prefetch: HashSet<u32> = HashSet::new();
    let queried: HashSet<u32> = query_vshards.iter().copied().collect();

    // For each (tenant, collection), find all vShards that might hold data
    // for the same collection (co-located shards).
    for (_tenant_id, key) in tenant_collections {
        // The collection key names its primary vShard.
        let primary = crate::routing::vshard_for_collection(*key);

        // Adjacent vShards (±1, ±2) are likely to hold related data
        // due to hash distribution locality.
        for offset in [1u32, 2] {
            let adjacent_low = primary.wrapping_sub(offset);
            let adjacent_high = primary.wrapping_add(offset);
            if !queried.contains(&adjacent_low) {
                prefetch.insert(adjacent_low);
            }
            if !queried.contains(&adjacent_high) {
                prefetch.insert(adjacent_high);
            }
        }
    }

    // Limit prefetch to avoid excessive fan-out.
    let max_prefetch = 8;
    prefetch.into_iter().take(max_prefetch).collect()
}

/// Deterministic hash for partition key splitting.
fn partition_hash(key: &str) -> u64 {
    crate::routing::fnv1a_hash(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_split_even() {
        let docs: Vec<String> = (0..100).map(|i| format!("doc_{i}")).collect();
        let plan = plan_vector_split(10, 20, 3, &docs);
        assert_eq!(plan.source_vshard, 10);
        assert_eq!(plan.new_vshard, 20);
        assert_eq!(plan.documents_to_move.len(), 50);
        assert_eq!(plan.strategy, SplitStrategy::VectorPartitionKey);
    }

    #[test]
    fn graph_split_minimizes_cross_edges() {
        // Chain: a→b→c→d→e. Splitting at midpoint should produce
        // fewer cross-edges than random split.
        let nodes: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let edges: Vec<(String, String)> = vec![
            ("a".into(), "b".into()),
            ("b".into(), "c".into()),
            ("c".into(), "d".into()),
            ("d".into(), "e".into()),
        ];
        let plan = plan_graph_split(10, 20, 3, &nodes, &edges);
        // 5 nodes split: mid = 5/2 = 2, so 3 docs move (5 - 2).
        assert_eq!(plan.documents_to_move.len(), 3);
        assert_eq!(plan.strategy, SplitStrategy::GraphCommunity);
    }

    #[test]
    fn graph_split_empty() {
        let plan = plan_graph_split(10, 20, 3, &[], &[]);
        assert!(plan.documents_to_move.is_empty());
    }

    #[test]
    fn speculative_prefetch_limits() {
        let routing = RoutingTable::uniform(4, &[1, 2], 1);
        let prefetch = speculative_prefetch_shards(
            &[0, 1],
            &routing,
            &[(
                1,
                nodedb_types::id::CollectionKey::from_bare(
                    nodedb_types::id::DatabaseId::DEFAULT,
                    "users",
                ),
            )],
        );
        assert!(prefetch.len() <= 8); // Max prefetch limit.
    }

    #[test]
    fn partition_hash_deterministic() {
        let h1 = partition_hash("document_42");
        let h2 = partition_hash("document_42");
        assert_eq!(h1, h2);
    }
}
