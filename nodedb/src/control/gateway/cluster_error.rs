// SPDX-License-Identifier: BUSL-1.1

//! Map a remote node's [`TypedClusterError`], and a failed send to it, to
//! the gateway's [`Error`].

use nodedb_cluster::ClusterError;
use nodedb_cluster::rpc_codec::TypedClusterError;

use crate::Error;
use crate::types::VShardId;

/// Map a [`TypedClusterError`] to an internal [`Error`].
///
/// `NotLeader` is mapped such that the gateway retry loop can extract the
/// hinted leader from `Error::NotLeader.leader_node` and update the routing
/// table before the next attempt.
pub(super) fn map_typed_cluster_error(err: TypedClusterError, vshard_id: u64) -> Error {
    match err {
        TypedClusterError::NotLeader {
            leader_node_id,
            leader_addr,
            term,
            ..
        } => Error::NotLeader {
            vshard_id: VShardId::new((vshard_id % VShardId::COUNT as u64) as u32),
            leader_node: leader_node_id.unwrap_or(0),
            leader_addr: leader_addr.unwrap_or_default(),
            leader_term: term,
        },
        TypedClusterError::DescriptorMismatch {
            collection,
            expected_version,
            actual_version,
        } => {
            // A repeating mismatch means the planner and leaseholder disagree
            // persistently — a bug, not the transient race the retry assumes.
            tracing::debug!(
                %collection,
                expected_version,
                actual_version,
                "gateway: descriptor version mismatch at leaseholder"
            );
            Error::RetryableSchemaChanged {
                descriptor: collection,
            }
        }
        TypedClusterError::DeadlineExceeded { elapsed_ms } => {
            tracing::warn!(
                elapsed_ms,
                "gateway: a remote node reported the request's deadline exceeded"
            );
            Error::DeadlineExceeded {
                request_id: crate::types::RequestId::new(0),
            }
        }
        // Remote Data-Plane verdict: keep the code so the client sees the
        // SQLSTATE local execution renders, not a generic internal error.
        TypedClusterError::DataPlane { code } => Error::DataPlane(code.into()),
        // Remote constraint refusal: keep the kind so the client sees 23502
        // vs 23505, exactly as a local refusal on this node renders.
        TypedClusterError::RejectedConstraint {
            collection,
            constraint,
            detail,
        } => Error::RejectedConstraint {
            collection,
            constraint,
            detail,
        },
        // A numeric class crosses as `Error::RemoteTyped`, so the client sees
        // the SQLSTATE the executing node gave it. Only a code of 0 (no class)
        // decodes as `Error::Internal`.
        internal @ TypedClusterError::Internal { .. } => Error::from(internal),
        // A Calvin abort keeps the error a local submit returns.
        aborted @ TypedClusterError::CalvinAborted { .. } => Error::from(aborted),
    }
}

/// The error of a failed `ExecuteRequest` or `ExecuteStreamRequest` send.
///
/// A request the target never received, and a lost answer to a read, retry as
/// `NotLeader` with `leader_node = 0`. The unreachable node is then not kept as
/// leader, and the next try routes locally to the real leader. A lost answer
/// to any other plan is an unknown outcome: the plan can have run, and a
/// second send would run it again.
pub(super) fn send_error(
    e: ClusterError,
    resend_safe: bool,
    node_id: u64,
    vshard_id: u64,
) -> Error {
    if e.is_lost_answer() && !resend_safe {
        tracing::warn!(
            node_id,
            vshard_id,
            error = %e,
            "gateway: a remote execute reached its deadline with no answer"
        );
        return Error::DeadlineExceeded {
            request_id: crate::types::RequestId::new(0),
        };
    }
    Error::NotLeader {
        vshard_id: VShardId::new((vshard_id % VShardId::COUNT as u64) as u32),
        leader_node: 0,
        leader_addr: format!("node-{node_id} (transport error: {e})"),
        leader_term: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unanswered() -> ClusterError {
        ClusterError::Unanswered {
            node_id: 2,
            detail: "stream reset".into(),
        }
    }

    /// A write whose answer was lost can have applied: it is never sent again.
    #[test]
    fn a_lost_answer_to_a_write_is_an_unknown_outcome() {
        assert!(matches!(
            send_error(unanswered(), false, 2, 7),
            Error::DeadlineExceeded { .. }
        ));
    }

    /// A read whose answer was lost retries through the leader lookup.
    #[test]
    fn a_lost_answer_to_a_read_retries() {
        assert!(matches!(
            send_error(unanswered(), true, 2, 7),
            Error::NotLeader { leader_node: 0, .. }
        ));
    }

    /// A send that timed out is a lost answer: a read retries, a write is an
    /// unknown outcome.
    #[test]
    fn a_timed_out_send_follows_the_lost_answer_rule() {
        let timed_out = || ClusterError::ShardTimeout {
            vshard_id: 7,
            elapsed_ms: 1_000,
        };
        assert!(matches!(
            send_error(timed_out(), true, 2, 7),
            Error::NotLeader { leader_node: 0, .. }
        ));
        assert!(matches!(
            send_error(timed_out(), false, 2, 7),
            Error::DeadlineExceeded { .. }
        ));
    }

    /// A request that never reached the target retries, read or write.
    #[test]
    fn an_unsent_request_retries() {
        let unsent = ClusterError::Transport {
            detail: "connect refused".into(),
        };
        assert!(matches!(
            send_error(unsent, false, 2, 7),
            Error::NotLeader { leader_node: 0, .. }
        ));
    }

    #[test]
    fn map_not_leader() {
        let err = TypedClusterError::NotLeader {
            group_id: 0,
            leader_node_id: Some(5),
            leader_addr: Some("10.0.0.5:9400".into()),
            term: 3,
        };
        match map_typed_cluster_error(err, 7) {
            Error::NotLeader {
                leader_node,
                leader_term,
                ..
            } => assert_eq!((leader_node, leader_term), (5, 3)),
            other => panic!("expected NotLeader, got {other:?}"),
        }
    }

    #[test]
    fn map_descriptor_mismatch() {
        let err = TypedClusterError::DescriptorMismatch {
            collection: "orders".into(),
            expected_version: 1,
            actual_version: 2,
        };
        match map_typed_cluster_error(err, 0) {
            Error::RetryableSchemaChanged { descriptor } => assert_eq!(descriptor, "orders"),
            other => panic!("expected RetryableSchemaChanged, got {other:?}"),
        }
    }

    /// A remote error with a numeric class keeps it, never `Internal`.
    #[test]
    fn map_internal_keeps_its_numeric_class() {
        let err = TypedClusterError::Internal {
            code: u32::from(nodedb_types::error::ErrorCode::AUTHORIZATION_DENIED.0),
            message: "permission denied on orders".into(),
        };
        match map_typed_cluster_error(err, 0) {
            Error::RemoteTyped { code, .. } => {
                assert_eq!(code, nodedb_types::error::ErrorCode::AUTHORIZATION_DENIED);
            }
            other => panic!("expected RemoteTyped, got {other:?}"),
        }
    }

    #[test]
    fn map_deadline_exceeded() {
        let err = TypedClusterError::DeadlineExceeded { elapsed_ms: 100 };
        assert!(matches!(
            map_typed_cluster_error(err, 0),
            Error::DeadlineExceeded { .. }
        ));
    }
}
