// SPDX-License-Identifier: BUSL-1.1

//! The two waits of a submitted Calvin transaction: its sequencer assignment
//! and its completion.
//!
//! Both end at the statement's deadline. A submission that reached the
//! sequencer inbox can still be sequenced after its wait ends, so an elapsed
//! wait is an unknown outcome, never a definite refusal.

use nodedb_cluster::calvin::{
    Assignment, AssignmentReceiver, CalvinCompletionRegistry, CompletionReport, TxnId,
};

use crate::Error;
use crate::types::RequestId;

/// The error of a wait that ended at the statement's deadline: the outcome
/// is unknown.
fn unknown_outcome() -> Error {
    Error::DeadlineExceeded {
        request_id: RequestId::new(0),
    }
}

/// Await the sequencer assignment of submission `inbox_seq` until `deadline`.
///
/// A closed channel means the sequencer rejected or discarded the submission
/// without sequencing it: a read/write cycle in its epoch, a leadership
/// change, or a halted sequencer. Nothing applied, so the caller gets a
/// retryable refusal. On the deadline the registration is dropped, so no
/// sender stays behind.
pub(crate) async fn await_assignment(
    registry: &CalvinCompletionRegistry,
    inbox_seq: u64,
    assignment_rx: AssignmentReceiver,
    deadline: tokio::time::Instant,
) -> crate::Result<Assignment> {
    match tokio::time::timeout_at(deadline, assignment_rx).await {
        Ok(Ok(assignment)) => Ok(assignment),
        Ok(Err(_)) => Err(Error::RetryableRefusal {
            reason: "the Calvin sequencer did not sequence the transaction; nothing was \
                     applied"
                .to_owned(),
        }),
        Err(_) => {
            registry.drop_assignment(inbox_seq);
            tracing::warn!(
                inbox_seq,
                "Calvin submission reached the statement deadline before its assignment"
            );
            Err(unknown_outcome())
        }
    }
}

/// Await the completion report of `txn_id` until `deadline`.
///
/// An elapsed wait files a `calvin_completion_timeout` report: it is the only
/// signal a never-completed Calvin write produces.
pub(crate) async fn await_completion(
    registry: &CalvinCompletionRegistry,
    txn_id: TxnId,
    participants: usize,
    deadline: tokio::time::Instant,
) -> crate::Result<CompletionReport> {
    let waiting_since = tokio::time::Instant::now();
    let completion_rx = registry.register_completion_report(txn_id, participants);
    match tokio::time::timeout_at(deadline, completion_rx).await {
        Ok(Ok(report)) => Ok(report),
        Ok(Err(_)) => Err(Error::Internal {
            detail: "Calvin completion channel closed".to_owned(),
        }),
        Err(_) => {
            let error = unknown_outcome();
            let waited_ms = u64::try_from(
                deadline
                    .saturating_duration_since(waiting_since)
                    .as_millis(),
            )
            .unwrap_or(u64::MAX);
            crate::diag::calvin_completion_timeout(
                &error,
                txn_id.epoch,
                txn_id.position,
                participants,
                waited_ms,
            );
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// The sequencer dropped the assignment of a submission it did not
    /// sequence. The caller fails at once with a retryable refusal, long
    /// before its deadline.
    #[tokio::test]
    async fn an_unsequenced_submission_is_a_retryable_refusal() {
        let registry = CalvinCompletionRegistry::new_detached();
        let rx = registry.register_submission(7);
        registry.drop_assignment(7);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3600);
        match await_assignment(&registry, 7, rx, deadline).await {
            Err(Error::RetryableRefusal { .. }) => {}
            other => panic!("expected a retryable refusal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_assigned_submission_returns_its_assignment() {
        let registry = CalvinCompletionRegistry::new_detached();
        let rx = registry.register_submission(3);
        registry.note_assigned(3, TxnId::new(5, 1), 2);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3600);
        let assignment = await_assignment(&registry, 3, rx, deadline)
            .await
            .expect("assigned");
        assert_eq!(assignment, (5, 1, 2));
    }

    /// A wait that reaches the deadline is an unknown outcome, never a
    /// refusal: the submission can still be sequenced.
    #[tokio::test]
    async fn a_timed_out_wait_is_not_a_refusal() {
        let registry = CalvinCompletionRegistry::new_detached();
        let rx = registry.register_submission(8);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(1);
        match await_assignment(&registry, 8, rx, deadline).await {
            Err(Error::DeadlineExceeded { .. }) => {}
            other => panic!("expected an unknown outcome, got {other:?}"),
        }
    }
}
