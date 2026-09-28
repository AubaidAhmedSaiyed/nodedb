// SPDX-License-Identifier: BUSL-1.1

//! The handle DDL proposes metadata entries through.

use std::sync::{Arc, Weak};

use nodedb_cluster::ClusterError;
use nodedb_raft::RaftError;

use crate::error::Error;

/// Type-erased handle for proposing to the metadata raft group.
///
/// The apply watermark for the metadata group lives on
/// [`crate::control::state::SharedState::applied_index_watcher`] (keyed by
/// [`nodedb_cluster::METADATA_GROUP_ID`]); callers of [`Self::propose`]
/// look it up there rather than receiving it through this handle.
pub trait MetadataRaftHandle: Send + Sync {
    /// Propose a raw encoded `MetadataEntry` to the metadata group.
    /// Returns its assigned log index on success.
    fn propose(&self, bytes: Vec<u8>) -> Result<u64, Error>;
}

/// Concrete impl wrapping `nodedb_cluster::RaftLoop`.
///
/// Holds the loop weakly: this handle lives on `SharedState`, which is
/// itself kept alive transitively by the `RaftLoop`, so a strong
/// reference here would close a cycle that pins both forever and blocks
/// clean shutdown. The loop is kept alive by its own spawned tasks;
/// `upgrade` therefore succeeds throughout normal operation and only
/// fails once the loop has been dropped on shutdown.
pub struct RaftLoopProposerHandle {
    raft_loop: Weak<
        nodedb_cluster::RaftLoop<
            crate::control::cluster::SpscCommitApplier,
            crate::control::LocalPlanExecutor,
        >,
    >,
}

impl RaftLoopProposerHandle {
    pub fn new(
        raft_loop: Arc<
            nodedb_cluster::RaftLoop<
                crate::control::cluster::SpscCommitApplier,
                crate::control::LocalPlanExecutor,
            >,
        >,
    ) -> Self {
        Self {
            raft_loop: Arc::downgrade(&raft_loop),
        }
    }
}

impl MetadataRaftHandle for RaftLoopProposerHandle {
    fn propose(&self, bytes: Vec<u8>) -> Result<u64, Error> {
        // The cluster crate's `propose_to_metadata_group_via_leader`
        // is async because it may need to forward to the metadata
        // leader over QUIC. The trait method is sync because every
        // caller (catalog DDL handlers, lease grant/release helpers)
        // is itself sync but runs inside a tokio task. Wrap in
        // `block_in_place` + the current runtime's `block_on` so the
        // forwarding QUIC round-trip drives without starving the
        // raft tick that produces the leader_hint.
        // `upgrade` fails only once the raft loop has been dropped on
        // shutdown; a request racing shutdown then fails cleanly with a
        // typed error instead of panicking.
        let raft_loop = self.raft_loop.upgrade().ok_or_else(|| Error::Config {
            detail: "metadata propose: cluster not running".into(),
        })?;
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(raft_loop.propose_to_metadata_group_via_leader(bytes))
        })
        .map_err(metadata_propose_error)
    }
}

/// The error a metadata proposal returns for a cluster error.
fn metadata_propose_error(error: ClusterError) -> Error {
    match error {
        // An election in progress is transient, not a failure of this
        // proposal. Keep it typed rather than flattening it into a generic
        // config error, so callers can wait the election out instead of
        // failing the statement — a node that has just restarted answers
        // every metadata proposal this way for a moment.
        ClusterError::Raft(RaftError::NotLeader { leader_hint: None }) => {
            Error::MetadataLeaderUnavailable
        }
        // A typed verdict keeps its class.
        ClusterError::DataPlane { code } => Error::DataPlane(code.into()),
        ClusterError::ShardExecution { error, .. } | ClusterError::StreamTerminal { error, .. } => {
            Error::from(*error)
        }
        other @ (ClusterError::Raft(
            RaftError::NotLeader {
                leader_hint: Some(_),
            }
            | RaftError::LogCompacted { .. }
            | RaftError::CompactionAheadOfApplied { .. }
            | RaftError::ProposalRejected { .. }
            | RaftError::InvalidTransferTarget { .. }
            | RaftError::LeadershipTransferInProgress
            | RaftError::GroupNotFound { .. }
            | RaftError::Transport { .. }
            | RaftError::Storage { .. }
            | RaftError::Serialization { .. }
            | RaftError::SnapshotFormat { .. }
            | RaftError::Shutdown,
        )
        | ClusterError::VShardNotMapped { .. }
        | ClusterError::GroupNotFound { .. }
        | ClusterError::LearnerNotCaughtUp { .. }
        | ClusterError::MigrationInProgress { .. }
        | ClusterError::MigrationPauseBudgetExceeded { .. }
        | ClusterError::NodeUnreachable { .. }
        | ClusterError::GhostNotFound { .. }
        | ClusterError::Transport { .. }
        | ClusterError::ShardTimeout { .. }
        | ClusterError::Storage { .. }
        | ClusterError::Codec { .. }
        | ClusterError::UnsupportedWireVersion { .. }
        | ClusterError::CircuitOpen { .. }
        | ClusterError::JoinGroupDisappeared { .. }
        | ClusterError::JoinCommitTimeout { .. }
        | ClusterError::ReadIndexNotLeader { .. }
        | ClusterError::ReadIndexTimeout { .. }
        | ClusterError::Config { .. }
        | ClusterError::MigrationCheckpoint(_)
        | ClusterError::MigrationRecovery(_)
        | ClusterError::WrongOwner { .. }
        | ClusterError::Calvin(_)
        | ClusterError::SnapshotCrcMismatch { .. }
        | ClusterError::SnapshotOffsetRegression { .. }
        | ClusterError::PartialSnapshotCorrupt { .. }
        | ClusterError::PartialSnapshotCleanupFailed { .. }
        | ClusterError::SnapshotApplyFailed { .. }
        | ClusterError::Mirror(_)
        | ClusterError::BspBarrier(_)
        | ClusterError::VectorGather(_)
        | ClusterError::SpatialGather(_)
        | ClusterError::Bm25Gather(_)
        | ClusterError::TsGather(_)
        | ClusterError::RemoteUntyped { .. }) => Error::Config {
            detail: format!("metadata propose: {other}"),
        },
    }
}
