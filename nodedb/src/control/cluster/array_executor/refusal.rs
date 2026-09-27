// SPDX-License-Identifier: BUSL-1.1

//! The cluster error an array shard answers with when its Data Plane refuses.
//!
//! A coded refusal crosses as `ClusterError::DataPlane`, so the coordinator
//! rebuilds `crate::Error::DataPlane(code)` and renders the SQLSTATE a
//! single-node execution renders. Only a refusal with no code is a storage
//! error. A local-execution error keeps its class where the cluster wire has
//! one.

use nodedb_cluster::error::ClusterError;
use nodedb_cluster::rpc_codec::DataPlaneErrorCode;

use crate::bridge::envelope::Response;

/// The cluster error for a Data-Plane response with an error status.
pub(super) fn refusal_error(context: &str, response: &Response) -> ClusterError {
    match response.error_code.as_deref() {
        Some(code) => ClusterError::DataPlane {
            code: code.clone().into(),
        },
        None => ClusterError::Storage {
            detail: format!("{context}: data plane returned an error status with no error code"),
        },
    }
}

/// The cluster error for a local-execution error.
///
/// - A Data-Plane verdict keeps its code.
/// - A deadline and a capacity refusal cross as their Data-Plane verdicts.
/// - A missing leader crosses as `WrongOwner`, so the coordinator re-reads
///   its routing and retries.
/// - Every other error is a storage error with `context` before its message.
pub(super) fn execution_error(context: &str, error: crate::Error) -> ClusterError {
    match error {
        crate::Error::DataPlane(code) => ClusterError::DataPlane { code: code.into() },
        crate::Error::DeadlineExceeded { .. } => ClusterError::DataPlane {
            code: DataPlaneErrorCode::DeadlineExceeded,
        },
        capacity @ crate::Error::DispatchCapacity { .. } => ClusterError::DataPlane {
            code: DataPlaneErrorCode::DispatchCapacity {
                reason: capacity.to_string(),
            },
        },
        crate::Error::NotLeader {
            vshard_id,
            leader_node,
            ..
        } => ClusterError::WrongOwner {
            vshard_id: vshard_id.as_u32(),
            expected_owner_node: (leader_node != 0).then_some(leader_node),
        },
        crate::Error::NoLeader { vshard_id } => ClusterError::WrongOwner {
            vshard_id: vshard_id.as_u32(),
            expected_owner_node: None,
        },
        other => ClusterError::Storage {
            detail: format!("{context}: {other}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::envelope::{ErrorCode, Payload, Status};
    use crate::types::{Lsn, RequestId};

    fn refusal(code: Option<ErrorCode>) -> Response {
        Response {
            request_id: RequestId::new(1),
            status: Status::Error,
            attempt: 1,
            partial: false,
            payload: Payload::empty(),
            watermark_lsn: Lsn::ZERO,
            error_code: code.map(Box::new),
            read_set_valid: None,
            read_version_lsn: Lsn::ZERO,
            write_set: Vec::new(),
        }
    }

    fn unsupported() -> ErrorCode {
        ErrorCode::Unsupported {
            detail: "not on this engine".into(),
        }
    }

    /// A coded refusal keeps its code through the cluster error and back to
    /// the coordinator's typed error.
    #[test]
    fn a_coded_refusal_keeps_its_code() {
        match refusal_error("array slice", &refusal(Some(unsupported()))) {
            ClusterError::DataPlane { code } => {
                assert_eq!(ErrorCode::from(code), unsupported());
            }
            other => panic!("expected the typed refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_refusal_with_no_code_is_a_storage_error() {
        match refusal_error("array slice", &refusal(None)) {
            ClusterError::Storage { detail } => assert!(detail.starts_with("array slice: ")),
            other => panic!("expected a storage error, got {other:?}"),
        }
    }

    #[test]
    fn a_local_deadline_crosses_as_the_deadline_verdict() {
        let error = crate::Error::DeadlineExceeded {
            request_id: RequestId::new(1),
        };
        assert!(matches!(
            execution_error("array put", error),
            ClusterError::DataPlane {
                code: DataPlaneErrorCode::DeadlineExceeded
            }
        ));
    }

    #[test]
    fn a_missing_leader_crosses_as_wrong_owner() {
        let error = crate::Error::NotLeader {
            vshard_id: crate::types::VShardId::new(9),
            leader_node: 4,
            leader_addr: "10.0.0.4:9000".into(),
        };
        assert!(matches!(
            execution_error("array put raft propose", error),
            ClusterError::WrongOwner {
                vshard_id: 9,
                expected_owner_node: Some(4)
            }
        ));
    }

    #[test]
    fn an_execution_verdict_keeps_its_code() {
        let error = crate::Error::DataPlane(unsupported());
        match execution_error("array put", error) {
            ClusterError::DataPlane { code } => {
                assert_eq!(ErrorCode::from(code), unsupported());
            }
            other => panic!("expected the typed refusal, got {other:?}"),
        }
    }
}
