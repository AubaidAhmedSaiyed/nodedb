// SPDX-License-Identifier: BUSL-1.1

//! Capture sites for a committed entry the data-group apply loop holds.

use faultbox::{Capture, EventKind, error_chain_of};

use crate::diag::context;

/// Report a committed redo entry of `group_id` at `log_index` whose bytes do
/// not decode on this replica. Called from the apply loop, the one site that
/// holds the entry.
pub fn redo_entry_malformed(group_id: u64, log_index: u64, error: &crate::Error) {
    let ctx = context::RedoEntryMalformed {
        group_id,
        log_index,
        error: error.to_string(),
    };
    let _ = Capture::new(
        EventKind::InvariantViolation,
        "committed redo entry does not decode: the group's apply holds until a snapshot \
         installs",
    )
    .domain(&ctx)
    .error_chain(error_chain_of(error))
    .emit();
}

/// Report that the snapshot debt of `group_id`, whose apply holds the entry
/// at `log_index`, did not persist. Called from the apply loop, the one site
/// that records the debt of a held entry, once per failure run.
pub fn held_entry_debt_not_recorded(group_id: u64, log_index: u64, error: &crate::Error) {
    let ctx = context::HeldEntryDebtNotRecorded {
        group_id,
        log_index,
        error: error.to_string(),
    };
    let _ = Capture::new(
        EventKind::Error,
        "snapshot debt of a group whose apply holds an entry did not persist",
    )
    .domain(&ctx)
    .error_chain(error_chain_of(error))
    .emit();
}

/// Report that the enqueue or apply of the committed entry of `group_id` at
/// `log_index` panicked on this replica. Called from the apply loop, the one
/// site that collects the entry's apply.
pub fn apply_panicked(group_id: u64, log_index: u64, detail: &str) {
    let ctx = context::ApplyPanicked {
        group_id,
        log_index,
        detail: detail.to_string(),
    };
    let _ = Capture::new(
        EventKind::InvariantViolation,
        "a committed entry's apply panicked: the group's apply holds until a snapshot \
         installs",
    )
    .domain(&ctx)
    .emit();
}
