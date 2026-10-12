// SPDX-License-Identifier: Apache-2.0

//! Frontier and result heaps shared by the NaviX and ACORN traversals.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::hnsw::graph::Candidate;

/// Heaps for one single-layer beam search of width `ef`.
pub(crate) struct Beam {
    frontier: BinaryHeap<Reverse<Candidate>>,
    results: BinaryHeap<Candidate>,
    ef: usize,
}

impl Beam {
    pub(crate) fn new(ef: usize) -> Self {
        Self {
            frontier: BinaryHeap::new(),
            results: BinaryHeap::new(),
            ef,
        }
    }

    /// Offers node `id` at distance `dist`.
    ///
    /// The node joins the frontier while it can still improve the results.
    /// It joins the results only when `admit` holds and it has a distance.
    /// A node without a vector stays traversable at infinite distance.
    pub(crate) fn offer(&mut self, id: u32, dist: Option<f32>, admit: bool) {
        let candidate = Candidate {
            dist: dist.unwrap_or(f32::INFINITY),
            id,
        };
        let worst = self.results.peek().map_or(f32::INFINITY, |w| w.dist);
        if candidate.dist < worst || self.results.len() < self.ef {
            self.frontier.push(Reverse(candidate));
        }
        if admit && dist.is_some() {
            self.results.push(candidate);
            if self.results.len() > self.ef {
                self.results.pop();
            }
        }
    }

    /// Pops the nearest frontier node.
    ///
    /// Returns `None` when the frontier is empty, or when `ef` results exist
    /// and the nearest frontier node is farther than all of them.
    pub(crate) fn pop_frontier(&mut self) -> Option<Candidate> {
        let Reverse(current) = self.frontier.pop()?;
        if let Some(worst) = self.results.peek()
            && current.dist > worst.dist
            && self.results.len() >= self.ef
        {
            return None;
        }
        Some(current)
    }

    /// The results, nearest first.
    pub(crate) fn into_sorted(self) -> Vec<Candidate> {
        let mut results = self.results.into_vec();
        results.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_without_a_vector_is_traversed_but_never_a_result() {
        let mut beam = Beam::new(4);
        beam.offer(7, None, true);
        beam.offer(3, Some(1.0), true);
        let first = beam.pop_frontier().expect("scored node is on the frontier");
        assert_eq!(first.id, 3);
        let second = beam
            .pop_frontier()
            .expect("vectorless node stays traversable");
        assert_eq!(second.id, 7);
        let ids: Vec<u32> = beam.into_sorted().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![3]);
    }

    #[test]
    fn results_keep_the_ef_nearest_admitted_nodes() {
        let mut beam = Beam::new(2);
        beam.offer(1, Some(3.0), true);
        beam.offer(2, Some(1.0), false);
        beam.offer(3, Some(2.0), true);
        beam.offer(4, Some(0.5), true);
        let ids: Vec<u32> = beam.into_sorted().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![4, 3]);
    }
}
