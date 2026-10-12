// SPDX-License-Identifier: BUSL-1.1

//! Forensic payloads for the data-group apply path.

use faultbox::DomainContext;
use faultbox::serde_json::{Value, json};

/// The apply loop reached a Raft log index it had already applied.
pub(in crate::diag) struct RaftEntryReapplied {
    pub group_id: u64,
    pub log_index: u64,
    /// The highest index of the group the apply loop applied before.
    pub highest_applied: u64,
}

impl DomainContext for RaftEntryReapplied {
    fn domain_kind(&self) -> &'static str {
        "nodedb.raft_entry_reapplied"
    }

    fn grouping_key(&self) -> String {
        // One bug class: a path handed an applied index to the apply loop
        // again. Group and index are the occurrence.
        "raft_entry_reapplied".to_string()
    }

    fn to_json(&self) -> Value {
        json!({
            "group_id": self.group_id,
            "log_index": self.log_index,
            "highest_applied": self.highest_applied,
            "why_fatal": "a committed entry applies once per replica. A second apply of an \
                          append-shaped write (timeseries, columnar, spatial, predicated \
                          update) stores its rows twice, mints a second redo record, fires \
                          AFTER triggers twice, and records the write's mark again",
            "operator_action": "the entry reached the apply loop twice past the applier's \
                                 delivery watermark. Find the path that re-queued it: \
                                 re-delivery after a refused hand-off, snapshot install, or \
                                 a second applier feeding the same channel",
        })
    }
}

/// The catalog refused a read or write of the stored dependent-read
/// barrier logs.
pub(in crate::diag) struct CalvinBarrierLogStoreFailed<'a> {
    /// The vShard. A persist of every vShard's rows names none.
    pub vshard_id: Option<u32>,
    /// The txn's `(epoch, position)`. An operation on every row of a vShard
    /// names none.
    pub txn: Option<(u64, u32)>,
    /// What the store did: `save`, `load`, `remove`, `forget` or `persist`.
    pub op: &'static str,
    pub error_class: &'a str,
}

impl DomainContext for CalvinBarrierLogStoreFailed<'_> {
    fn domain_kind(&self) -> &'static str {
        "nodedb.calvin_barrier_log_store_failed"
    }

    fn grouping_key(&self) -> String {
        // One report per vShard and operation: the txn changes per event,
        // and a retry loop repeats the same pair.
        match self.vshard_id {
            Some(vshard_id) => format!("calvin_barrier_log_store_failed:{}:{vshard_id}", self.op),
            None => format!("calvin_barrier_log_store_failed:{}:all", self.op),
        }
    }

    fn to_json(&self) -> Value {
        let effect = match self.op {
            "save" => {
                "the barrier event stays in memory, and every later event of the \
                       txn too. The entry does not count as durably applied, so a restart \
                       delivers it again from the log"
            }
            "load" => {
                "the scheduler keeps the txn's barrier open and reads the row \
                       again on its next pass"
            }
            "forget" => {
                "the rows of a vShard this node left stay stored. Every scheduler \
                       reconcile pass removes them again until one succeeds"
            }
            "persist" => {
                "the saved barrier rows are not durable yet. No applied floor moves \
                       this pass, so a restart delivers their entries again. The next \
                       settle pass persists again"
            }
            _ => {
                "the row of a finished txn stays stored. The scheduler's next \
                  stall-tick sweep removes it"
            }
        };
        json!({
            "vshard_id": self.vshard_id,
            "txn": self.txn,
            "op": self.op,
            "error_class": self.error_class,
            "effect": effect,
            "operator_action": "check the system catalog's disk: free space, I/O errors, \
                                and the redb file's permissions",
        })
    }
}

/// The catalog refused the forget of the stored Calvin state of vShards
/// this node left: their applied positions and bases.
pub(in crate::diag) struct CalvinLeftStateNotForgotten<'a> {
    pub vshards: &'a [u32],
    pub error_class: &'a str,
}

impl DomainContext for CalvinLeftStateNotForgotten<'_> {
    fn domain_kind(&self) -> &'static str {
        "nodedb.calvin_left_state_not_forgotten"
    }

    fn grouping_key(&self) -> String {
        // One root cause: the catalog refuses the write. The vShards are the
        // occurrence, and a retry loop repeats them.
        "calvin_left_state_not_forgotten".to_string()
    }

    fn to_json(&self) -> Value {
        json!({
            "vshards": self.vshards,
            "error_class": self.error_class,
            "effect": "the applied positions and base of a vShard this node left stay \
                       stored. Every scheduler reconcile pass forgets them again until one \
                       succeeds",
            "operator_action": "check the system catalog's disk: free space, I/O errors, \
                                and the redb file's permissions",
        })
    }
}
