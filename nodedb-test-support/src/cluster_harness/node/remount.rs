// SPDX-License-Identifier: BUSL-1.1

//! Unmount one data group on a [`TestClusterNode`] and mount it again from
//! its log on disk, as a node that left a group and rejoined it does.

use std::sync::Arc;

use nodedb::control::state::SharedState;
use nodedb_cluster::ClusterError;

use crate::cluster_harness::node::lifecycle::TestClusterNode;

/// Why a remount of a data group failed. Every case names the node and the
/// group.
#[derive(Debug, thiserror::Error)]
pub enum RemountError {
    #[error("node {node_id}: the remount task of group {group_id} did not finish: {source}")]
    Task {
        node_id: u64,
        group_id: u64,
        source: tokio::task::JoinError,
    },
    #[error("node {node_id}: its Raft groups are not started, so group {group_id} has none")]
    NotStarted { node_id: u64, group_id: u64 },
    #[error("node {node_id}: group {group_id} is not mounted")]
    NotMounted { node_id: u64, group_id: u64 },
    #[error("node {node_id}: dropping the redo streams of group {group_id}: {source}")]
    DropStreams {
        node_id: u64,
        group_id: u64,
        source: nodedb::Error,
    },
    #[error("node {node_id}: reopening group {group_id}: {source}")]
    Reopen {
        node_id: u64,
        group_id: u64,
        source: ClusterError,
    },
    #[error("node {node_id}: group {group_id} mounted again while its lock was held")]
    MountedTwice { node_id: u64, group_id: u64 },
}

impl TestClusterNode {
    /// Unmount `group_id` here, drop its redo streams as the membership pass
    /// drops the streams of a group this node no longer mounts, and mount the
    /// group again from its log on disk with the same voters and learners.
    ///
    /// The node's `MultiRaft` lock is held throughout, so no tick mounts the
    /// group in between.
    pub async fn remount_data_group(&self, group_id: u64) -> Result<(), RemountError> {
        let shared = Arc::clone(&self.shared);
        let node_id = self.node_id;
        tokio::task::spawn_blocking(move || remount(&shared, node_id, group_id))
            .await
            .map_err(|source| RemountError::Task {
                node_id,
                group_id,
                source,
            })?
    }
}

/// Run the remount on a blocking thread: closing and opening a group's log
/// blocks on disk.
fn remount(shared: &SharedState, node_id: u64, group_id: u64) -> Result<(), RemountError> {
    let multi_raft = shared
        .multi_raft
        .get()
        .ok_or(RemountError::NotStarted { node_id, group_id })?;
    let mut mr = multi_raft.lock().unwrap_or_else(|p| p.into_inner());
    let membership = mr
        .group_membership(group_id)
        .ok_or(RemountError::NotMounted { node_id, group_id })?;
    let starts_as_learner = membership.learners.contains(&node_id);
    let peers: Vec<u64> = membership
        .voters
        .into_iter()
        .filter(|&id| id != node_id)
        .collect();
    let learners: Vec<u64> = membership
        .learners
        .into_iter()
        .filter(|&id| id != node_id)
        .collect();
    // Dropping the replica closes its log file, which the mount reopens.
    drop(mr.unmount_group(group_id));
    shared
        .redo_chunks
        .drop_unhosted_groups(
            |group| mr.contains_group(group),
            shared.credentials.catalog(),
        )
        .map_err(|source| RemountError::DropStreams {
            node_id,
            group_id,
            source,
        })?;
    let opened = mr
        .mount_spec(group_id, peers, learners, starts_as_learner)
        .open()
        .map_err(|source| RemountError::Reopen {
            node_id,
            group_id,
            source,
        })?;
    if mr.insert_opened(opened).is_some() {
        return Err(RemountError::MountedTwice { node_id, group_id });
    }
    Ok(())
}
