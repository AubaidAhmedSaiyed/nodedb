// SPDX-License-Identifier: BUSL-1.1

//! Forensic payloads for a committed entry a replica holds: its redo bytes
//! do not decode, its apply panicked, or its group's snapshot debt did not
//! persist.

use faultbox::DomainContext;
use faultbox::serde_json::{Value, json};

/// A committed redo entry whose own bytes do not decode as the redo they
/// name on this replica.
pub(in crate::diag) struct RedoEntryMalformed {
    pub group_id: u64,
    pub log_index: u64,
    /// The decode error.
    pub error: String,
}

impl DomainContext for RedoEntryMalformed {
    fn domain_kind(&self) -> &'static str {
        "nodedb.redo_entry_malformed"
    }

    fn grouping_key(&self) -> String {
        // One bug class: a committed redo that does not decode. Group and
        // index are the occurrence.
        "redo_entry_malformed".to_string()
    }

    fn to_json(&self) -> Value {
        json!({
            "group_id": self.group_id,
            "log_index": self.log_index,
            "error": self.error,
            "impact": "the group's apply on this replica holds at the entry. Its applied \
                       index stays below it and reads wait. The replica refuses log \
                       entries, steps down from leadership, and never campaigns until a \
                       snapshot of the group installs",
            "operator_action": "the group's leader sends a snapshot. When every replica \
                                 holds the same entry, no replica can lead the group: the \
                                 proposer encoded a redo no replica decodes, and the error \
                                 names the field",
        })
    }
}

/// The snapshot debt of a group whose apply holds an entry did not persist.
pub(in crate::diag) struct HeldEntryDebtNotRecorded {
    pub group_id: u64,
    pub log_index: u64,
    /// The catalog error.
    pub error: String,
}

impl DomainContext for HeldEntryDebtNotRecorded {
    fn domain_kind(&self) -> &'static str {
        "nodedb.held_entry_debt_not_recorded"
    }

    fn grouping_key(&self) -> String {
        // The apply loop retries on each Raft tick while the write fails: one
        // report per root cause.
        "held_entry_debt_not_recorded".to_string()
    }

    fn to_json(&self) -> Value {
        json!({
            "group_id": self.group_id,
            "log_index": self.log_index,
            "error": self.error,
            "impact": "the replica refuses log entries until a snapshot installs, but a \
                       sequencer snapshot install can lift that before the held entry is \
                       covered. The apply loop writes the debt again on each Raft tick \
                       and holds the requirement up until a write persists",
            "operator_action": "the error names why the system catalog refused the write",
        })
    }
}

/// A committed entry whose enqueue or apply panicked on this replica.
pub(in crate::diag) struct ApplyPanicked {
    pub group_id: u64,
    pub log_index: u64,
    /// The panic payload.
    pub detail: String,
}

impl DomainContext for ApplyPanicked {
    fn domain_kind(&self) -> &'static str {
        "nodedb.apply_panicked"
    }

    fn grouping_key(&self) -> String {
        // One bug class: an apply that panicked. Group and index are the
        // occurrence.
        "apply_panicked".to_string()
    }

    fn to_json(&self) -> Value {
        json!({
            "group_id": self.group_id,
            "log_index": self.log_index,
            "detail": self.detail,
            "impact": "the entry's effect on this replica is unknown. Its proposer \
                       receives an error. The group's apply holds at the entry, its \
                       applied index stays below it, and the replica refuses log entries \
                       until a snapshot of the group installs. Other groups keep applying",
            "operator_action": "the panic names the code that failed. The group's leader \
                                 sends a snapshot. When every replica panics on the entry, \
                                 no replica can lead the group",
        })
    }
}
