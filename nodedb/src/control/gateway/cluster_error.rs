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

/// One `ExecuteRequest` or `ExecuteStreamRequest` send, as its failure
/// mapping needs it.
#[derive(Debug, Clone, Copy)]
pub(super) struct RemoteSend {
    pub node_id: u64,
    pub vshard_id: u64,
    /// A second run of the plan is harmless: the plan only reads.
    pub resend_safe: bool,
    /// The statement's deadline for this send.
    pub deadline: tokio::time::Instant,
}

/// The error of a failed `ExecuteRequest` or `ExecuteStreamRequest` send.
///
/// A request the target never received, and a lost answer to a read, retry as
/// `NotLeader` with `leader_node = 0`. The unreachable node is then not kept as
/// leader, and the next try routes locally to the real leader.
///
/// These end the statement as `DeadlineExceeded` instead:
/// - a lost answer to any other plan: it can have run, and a second send
///   would run it again;
/// - a lost answer once the deadline passed: the wait used the whole budget,
///   and no retry has time left;
/// - a `ShardTimeout`: the wait used the whole budget.
pub(super) fn send_error(e: ClusterError, send: &RemoteSend) -> Error {
    if matches!(e, ClusterError::ShardTimeout { .. }) {
        return deadline_exceeded(send, &e);
    }
    if e.is_lost_answer() && (!send.resend_safe || tokio::time::Instant::now() >= send.deadline) {
        return deadline_exceeded(send, &e);
    }
    Error::NotLeader {
        vshard_id: VShardId::new((send.vshard_id % VShardId::COUNT as u64) as u32),
        leader_node: 0,
        leader_addr: format!("node-{} (transport error: {e})", send.node_id),
        leader_term: 0,
    }
}

/// The error of a remote execute whose reply wait expired.
///
/// The wait covers the statement's whole remaining budget plus the reply
/// margin, so no retry has time left: the statement ends as `DeadlineExceeded`.
pub(super) fn reply_wait_expired(send: &RemoteSend, reply_wait: std::time::Duration) -> Error {
    let timed_out = ClusterError::ShardTimeout {
        vshard_id: (send.vshard_id % VShardId::COUNT as u64) as u32,
        elapsed_ms: u64::try_from(reply_wait.as_millis()).unwrap_or(u64::MAX),
    };
    deadline_exceeded(send, &timed_out)
}

fn deadline_exceeded(send: &RemoteSend, cause: &ClusterError) -> Error {
    tracing::warn!(
        node_id = send.node_id,
        vshard_id = send.vshard_id,
        error = %cause,
        "gateway: a remote execute ended with no answer within the statement budget"
    );
    Error::DeadlineExceeded {
        request_id: crate::types::RequestId::new(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A send to node 2 for vShard 7 whose statement has time left.
    fn send(resend_safe: bool) -> RemoteSend {
        RemoteSend {
            node_id: 2,
            vshard_id: 7,
            resend_safe,
            deadline: tokio::time::Instant::now() + std::time::Duration::from_secs(30),
        }
    }

    /// The same send once its statement deadline passed.
    fn spent(resend_safe: bool) -> RemoteSend {
        RemoteSend {
            deadline: tokio::time::Instant::now(),
            ..send(resend_safe)
        }
    }

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
            send_error(unanswered(), &send(false)),
            Error::DeadlineExceeded { .. }
        ));
    }

    /// A read whose answer was lost retries through the leader lookup.
    #[test]
    fn a_lost_answer_to_a_read_retries() {
        assert!(matches!(
            send_error(unanswered(), &send(true)),
            Error::NotLeader { leader_node: 0, .. }
        ));
    }

    /// A send that timed out used the statement's whole budget: a read and a
    /// write both end as `DeadlineExceeded`, with no retry.
    #[test]
    fn a_timed_out_read_ends_the_statement() {
        let timed_out = || ClusterError::ShardTimeout {
            vshard_id: 7,
            elapsed_ms: 1_000,
        };
        assert!(matches!(
            send_error(timed_out(), &send(true)),
            Error::DeadlineExceeded { .. }
        ));
        assert!(matches!(
            send_error(timed_out(), &send(false)),
            Error::DeadlineExceeded { .. }
        ));
    }

    /// A live node holds a read past its budget: the transport's read timeout
    /// ends the wait as a lost answer once the deadline passed. No retry has
    /// time left, so the read ends as `DeadlineExceeded`, never `NotLeader`.
    #[test]
    fn a_lost_answer_to_a_read_past_its_deadline_ends_the_statement() {
        assert!(matches!(
            send_error(unanswered(), &spent(true)),
            Error::DeadlineExceeded { .. }
        ));
    }

    /// A request that never went out leaves nothing to wait for, so it
    /// retries even at the deadline. The retry loop decides whether time is left.
    #[test]
    fn an_unsent_read_past_its_deadline_stays_retryable() {
        let unsent = ClusterError::Transport {
            detail: "connect refused".into(),
        };
        assert!(matches!(
            send_error(unsent, &spent(true)),
            Error::NotLeader { leader_node: 0, .. }
        ));
    }

    /// An expired reply wait ends the statement, never a retry.
    #[test]
    fn an_expired_reply_wait_is_deadline_exceeded() {
        assert!(matches!(
            reply_wait_expired(&send(true), std::time::Duration::from_secs(14)),
            Error::DeadlineExceeded { .. }
        ));
    }

    /// A transport error is unsent: a read and a write both retry.
    #[test]
    fn an_unsent_read_retries() {
        let unsent = ClusterError::Transport {
            detail: "handshake timeout".into(),
        };
        assert!(matches!(
            send_error(unsent, &send(true)),
            Error::NotLeader { leader_node: 0, .. }
        ));
    }

    /// A request that never reached the target retries, read or write.
    #[test]
    fn an_unsent_request_retries() {
        let unsent = ClusterError::Transport {
            detail: "connect refused".into(),
        };
        assert!(matches!(
            send_error(unsent, &send(false)),
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
