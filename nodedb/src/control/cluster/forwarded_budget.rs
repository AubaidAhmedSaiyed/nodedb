// SPDX-License-Identifier: BUSL-1.1

//! The deadline a leader-side hook works to for a forwarded request.

use nodedb_cluster::TypedClusterError;
use nodedb_cluster::rpc_codec::forwarded_deadline;

use crate::bridge::envelope::ErrorCode;
use crate::control::cluster::execution_error_wire::execution_error_to_typed;

/// The deadline for a forwarded request that carries `deadline_remaining_ms`.
///
/// A spent budget refuses the request before it runs. The refusal is
/// `ExpiredBeforeExecution`, so the coordinator reports a definite refusal,
/// never an unknown outcome.
pub(crate) fn hook_deadline(
    deadline_remaining_ms: u64,
) -> Result<tokio::time::Instant, TypedClusterError> {
    match forwarded_deadline(deadline_remaining_ms) {
        Some(deadline) => Ok(deadline),
        None => Err(execution_error_to_typed(crate::Error::DataPlane(
            ErrorCode::ExpiredBeforeExecution,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::control::server::dispatch_utils::write_definitely_not_applied;

    /// A spent budget comes back to the coordinator as a definite refusal.
    #[test]
    fn a_spent_budget_is_a_definite_refusal() {
        let refusal = hook_deadline(0).expect_err("a spent budget is refused");
        let crate::Error::DataPlane(code) = crate::Error::from(refusal) else {
            panic!("expected a Data-Plane verdict");
        };
        assert_eq!(code, ErrorCode::ExpiredBeforeExecution);
        assert!(write_definitely_not_applied(&code));
    }

    /// A live budget is used as it arrived, never stretched or floored.
    #[tokio::test]
    async fn a_live_budget_ends_within_what_arrived() {
        let before = tokio::time::Instant::now();
        let deadline = hook_deadline(50).expect("a live budget");
        assert!(deadline >= before + Duration::from_millis(50));
        assert!(deadline <= tokio::time::Instant::now() + Duration::from_millis(50));
    }
}
