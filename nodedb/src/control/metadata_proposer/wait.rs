// SPDX-License-Identifier: BUSL-1.1

//! Waits on the metadata group's applied index.
//!
//! A metadata entry proposed for a statement waits for its apply here until
//! the statement's deadline. Past the deadline the entry is proposed but its
//! outcome here is unknown, so the wait ends with `Error::DeadlineExceeded`.
//! A slow apply is not a wedge: `/healthz` reports a wedged metadata apply.

use std::time::Instant;

use nodedb_cluster::{AppliedIndexWatcher, METADATA_GROUP_ID, WaitOutcome};

use crate::control::server::shared::session::StatementDeadline;
use crate::control::state::SharedState;
use crate::error::Error;
use crate::types::RequestId;

/// Wait until `watcher` reaches `log_index`, the index of a metadata entry
/// this node proposed, or `deadline` passes. Returns `log_index`.
pub(crate) async fn await_applied(
    watcher: &AppliedIndexWatcher,
    log_index: u64,
    deadline: Instant,
) -> Result<u64, Error> {
    let outcome = watcher.wait_for_async(log_index, deadline).await;
    applied_or_error(outcome, log_index, watcher.current())
}

/// Wait until this node applied the metadata entry it proposed at
/// `log_index`, until the statement deadline. Returns `log_index`.
pub(crate) async fn await_metadata_apply(
    shared: &SharedState,
    log_index: u64,
) -> Result<u64, Error> {
    let watcher = shared.applied_index_watcher(METADATA_GROUP_ID);
    let deadline = StatementDeadline::of_statement(shared).instant();
    await_applied(&watcher, log_index, deadline).await
}

/// The verdict on a proposed metadata entry from its apply wait. `applied`
/// is the watermark when the wait ended.
fn applied_or_error(outcome: WaitOutcome, log_index: u64, applied: u64) -> Result<u64, Error> {
    match outcome {
        WaitOutcome::Reached => Ok(log_index),
        WaitOutcome::TimedOut => {
            tracing::warn!(
                log_index,
                applied,
                "a proposed metadata entry did not apply here before the statement deadline; \
                 its outcome is unknown"
            );
            Err(Error::DeadlineExceeded {
                request_id: RequestId::new(0),
            })
        }
        WaitOutcome::GroupGone => Err(Error::MetadataGroupGone { log_index }),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A metadata group that left this node ends the wait with its own error.
    #[tokio::test]
    async fn a_gone_group_ends_the_wait_with_a_typed_error() {
        let watcher = AppliedIndexWatcher::new();
        watcher.close();
        let result = await_applied(&watcher, 5, Instant::now() + Duration::from_secs(5)).await;
        assert!(
            matches!(result, Err(Error::MetadataGroupGone { log_index: 5 })),
            "{result:?}"
        );
    }
}
