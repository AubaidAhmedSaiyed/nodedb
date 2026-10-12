// SPDX-License-Identifier: BUSL-1.1

//! Apply-convergence barriers.
//!
//! A barrier waits until every node applied every entry its hosted groups
//! committed when the barrier looked. On timeout it panics with each lagging
//! group's commit target and every node's commit and applied index.
//!
//! The target of a group is the highest commit index any node reports for it.
//! A node's own applied index alone is no target: after a restart, every node
//! can hold the same applied index while committed entries above it are
//! unapplied everywhere.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use super::TestCluster;

/// One hosted group's positions on one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupPosition {
    pub node_id: u64,
    pub group_id: u64,
    /// The leader this node knows for the group, `0` when it knows none.
    pub leader_id: u64,
    pub commit: u64,
    /// This node's apply watermark for the group.
    pub applied: u64,
}

impl TestCluster {
    /// Every hosted group's positions on every node, keyed by group. A node
    /// that runs no Raft yet reports nothing.
    pub fn group_positions(&self) -> BTreeMap<u64, Vec<GroupPosition>> {
        let mut groups: BTreeMap<u64, Vec<GroupPosition>> = BTreeMap::new();
        for node in &self.nodes {
            let Some(status_fn) = node.shared.raft_status_fn.get() else {
                continue;
            };
            for status in status_fn() {
                groups
                    .entry(status.group_id)
                    .or_default()
                    .push(GroupPosition {
                        node_id: node.node_id,
                        group_id: status.group_id,
                        leader_id: status.leader_id,
                        commit: status.commit_index,
                        applied: node.shared.applied_index_watcher(status.group_id).current(),
                    });
            }
        }
        groups
    }

    /// [`Self::group_positions`] as text, one group per line, for a panic
    /// message.
    pub fn group_positions_report(&self) -> String {
        render(&self.group_positions(), &BTreeMap::new())
    }

    /// Block until every node applied every entry its hosted groups
    /// committed. Groups and commits are re-read on every poll, so a group
    /// that appears late is covered.
    ///
    /// # Panics
    ///
    /// At `timeout`, naming each lagging group with its commit target and
    /// every node's commit and applied index.
    pub async fn wait_for_full_apply_convergence(&self, timeout: Duration) {
        self.wait_for_groups_applied(timeout, None).await;
    }

    /// [`Self::wait_for_full_apply_convergence`] for the metadata group only.
    pub(crate) async fn wait_for_metadata_apply_convergence(&self, timeout: Duration) {
        self.wait_for_groups_applied(timeout, Some(nodedb_cluster::METADATA_GROUP_ID))
            .await;
    }

    async fn wait_for_groups_applied(&self, timeout: Duration, only: Option<u64>) {
        let deadline = Instant::now() + timeout;
        loop {
            // Commit targets first, apply positions after: an applied index
            // only grows, so a node at or past the target applied every
            // entry committed when the targets were read.
            let targets = commit_targets(&self.group_positions(), only);
            let mut positions = self.group_positions();
            positions.retain(|group_id, _| targets.contains_key(group_id));
            let lagging = lagging_groups(&positions, &targets);
            let silent = self.nodes_without_status();
            if lagging.is_empty() && silent.is_empty() {
                return;
            }
            if Instant::now() >= deadline {
                let lagging_positions: BTreeMap<u64, Vec<GroupPosition>> = positions
                    .into_iter()
                    .filter(|(group_id, _)| lagging.contains(group_id))
                    .collect();
                panic!(
                    "the cluster did not apply every committed entry within {timeout:?}.\n\
                     nodes reporting no group status: {silent:?}\n\
                     lagging groups:\n{}",
                    render(&lagging_positions, &targets)
                );
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Nodes whose Raft status is not installed yet.
    fn nodes_without_status(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .filter(|node| node.shared.raft_status_fn.get().is_none())
            .map(|node| node.node_id)
            .collect()
    }
}

/// The highest commit index any node reports, per group.
fn commit_targets(
    groups: &BTreeMap<u64, Vec<GroupPosition>>,
    only: Option<u64>,
) -> BTreeMap<u64, u64> {
    groups
        .iter()
        .filter(|(group_id, _)| only.is_none_or(|only| only == **group_id))
        .map(|(group_id, positions)| {
            let target = positions.iter().map(|p| p.commit).max().unwrap_or(0);
            (*group_id, target)
        })
        .collect()
}

/// Groups where some hosting node applied less than the group's target.
fn lagging_groups(
    groups: &BTreeMap<u64, Vec<GroupPosition>>,
    targets: &BTreeMap<u64, u64>,
) -> Vec<u64> {
    groups
        .iter()
        .filter(|(group_id, positions)| {
            let target = targets.get(group_id).copied().unwrap_or(0);
            positions.iter().any(|p| p.applied < target)
        })
        .map(|(group_id, _)| *group_id)
        .collect()
}

fn render(groups: &BTreeMap<u64, Vec<GroupPosition>>, targets: &BTreeMap<u64, u64>) -> String {
    let mut out = String::new();
    for (group_id, positions) in groups {
        let _ = write!(out, "  group {group_id}");
        if let Some(target) = targets.get(group_id) {
            let _ = write!(out, " (commit target {target})");
        }
        let _ = write!(out, ":");
        for p in positions {
            let _ = write!(
                out,
                " [node {} leader {} commit {} applied {}]",
                p.node_id, p.leader_id, p.commit, p.applied
            );
        }
        out.push('\n');
    }
    out
}
