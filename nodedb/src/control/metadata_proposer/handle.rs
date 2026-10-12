// SPDX-License-Identifier: BUSL-1.1

//! The handle DDL proposes metadata entries through.

use std::sync::{Arc, Weak};

use nodedb_cluster::ClusterError;
use nodedb_raft::RaftError;

use crate::error::Error;
use crate::types::RequestId;

/// Type-erased handle for proposing to the metadata raft group.
///
/// The apply watermark for the metadata group lives on
/// [`crate::control::state::SharedState::applied_index_watcher`] (keyed by
/// [`nodedb_cluster::METADATA_GROUP_ID`]); callers of [`Self::propose`]
/// look it up there rather than receiving it through this handle.
pub trait MetadataRaftHandle: Send + Sync {
    /// Propose a raw encoded `MetadataEntry` to the metadata group. Resolves
    /// to its assigned log index. Runs on any runtime flavor.
    fn propose_async<'a>(&'a self, bytes: Vec<u8>) -> ProposeFuture<'a>;
}

/// The future [`MetadataRaftHandle::propose_async`] returns.
pub type ProposeFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64, Error>> + Send + 'a>>;

/// Concrete impl wrapping `nodedb_cluster::RaftLoop`.
///
/// Holds the loop weakly: this handle lives on `SharedState`, which is
/// itself kept alive transitively by the `RaftLoop`, so a strong
/// reference here closes a cycle that pins both forever and blocks
/// clean shutdown. The loop is kept alive by its own spawned tasks;
/// `upgrade` therefore succeeds throughout normal operation and only
/// fails once the loop has been dropped on shutdown.
///
/// Every entry it proposes carries the metadata leader's HLC stamp, taken as
/// the leader appends it. A restore keeps an entry only when that stamp is
/// below the restore point's watermark, the rule it applies to data writes.
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

impl RaftLoopProposerHandle {
    /// The live raft loop. `upgrade` fails only once the raft loop has been
    /// dropped on shutdown; a request racing shutdown then fails cleanly with
    /// a typed error instead of panicking.
    fn live_loop(
        &self,
    ) -> Result<
        Arc<
            nodedb_cluster::RaftLoop<
                crate::control::cluster::SpscCommitApplier,
                crate::control::LocalPlanExecutor,
            >,
        >,
        Error,
    > {
        self.raft_loop.upgrade().ok_or_else(|| Error::Config {
            detail: "metadata propose: cluster not running".into(),
        })
    }
}

impl MetadataRaftHandle for RaftLoopProposerHandle {
    fn propose_async<'a>(&'a self, bytes: Vec<u8>) -> ProposeFuture<'a> {
        Box::pin(async move {
            let raft_loop = self.live_loop()?;
            raft_loop
                .propose_stamped_to_metadata_group_via_leader(bytes)
                .await
                .map_err(metadata_propose_error)
        })
    }
}

/// The error a metadata proposal returns for a cluster error.
///
/// A group with no leader to take the proposal proposed nothing, so the
/// caller can re-propose: [`Error::MetadataLeaderUnavailable`]. A forward
/// with no answer can have been proposed, so its outcome is unknown:
/// [`Error::DeadlineExceeded`]. A typed verdict keeps its class.
fn metadata_propose_error(error: ClusterError) -> Error {
    match error {
        // An election, a transfer in flight, or a leader that stepped down.
        // Nothing was proposed.
        ClusterError::Raft(
            RaftError::NotLeader { .. } | RaftError::LeadershipTransferInProgress,
        ) => Error::MetadataLeaderUnavailable,
        // A forward that never reached the leader whole proposed nothing.
        error if error.is_unsent() => Error::MetadataLeaderUnavailable,
        // The forward is not resend-safe and got no answer. The leader can
        // have appended the entry, so the outcome is unknown.
        error if error.is_lost_answer() => {
            tracing::warn!(
                %error,
                "metadata proposal forward got no answer from the leader"
            );
            Error::DeadlineExceeded {
                request_id: RequestId::new(0),
            }
        }
        // A typed verdict keeps its class.
        ClusterError::DataPlane { code } => Error::DataPlane(code.into()),
        ClusterError::ShardExecution { error, .. } | ClusterError::StreamTerminal { error, .. } => {
            Error::from(*error)
        }
        other @ (ClusterError::Raft(
            RaftError::LogCompacted { .. }
            | RaftError::CompactionAheadOfApplied { .. }
            | RaftError::ProposalRejected { .. }
            | RaftError::InvalidTransferTarget { .. }
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
        | ClusterError::ShufflePush(_)
        | ClusterError::RemoteUntyped { .. }
        // The guards above take these three. They stay named for exhaustiveness.
        | ClusterError::Transport { .. }
        | ClusterError::ShardTimeout { .. }
        | ClusterError::Unanswered { .. }) => Error::Config {
            detail: format!("metadata propose: {other}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A forward with no answer can have appended the entry. It is an
    /// unknown outcome, never a config error and never retried.
    #[test]
    fn a_lost_metadata_answer_is_an_unknown_outcome() {
        let unanswered = metadata_propose_error(ClusterError::Unanswered {
            node_id: 2,
            detail: "connection lost".into(),
        });
        assert!(matches!(unanswered, Error::DeadlineExceeded { .. }));

        let timed_out = metadata_propose_error(ClusterError::ShardTimeout {
            vshard_id: 0,
            elapsed_ms: 50,
        });
        assert!(matches!(timed_out, Error::DeadlineExceeded { .. }));
    }

    /// A forward that never went out proposed nothing. The caller waits for
    /// a leader and re-proposes.
    #[test]
    fn an_unsent_metadata_forward_waits_for_a_leader() {
        let unsent = metadata_propose_error(ClusterError::Transport {
            detail: "connect to node 2 refused".into(),
        });
        assert!(matches!(unsent, Error::MetadataLeaderUnavailable));

        let stale = metadata_propose_error(ClusterError::Raft(RaftError::NotLeader {
            leader_hint: Some(2),
            term: 3,
        }));
        assert!(matches!(stale, Error::MetadataLeaderUnavailable));

        let moving =
            metadata_propose_error(ClusterError::Raft(RaftError::LeadershipTransferInProgress));
        assert!(matches!(moving, Error::MetadataLeaderUnavailable));
    }
}
