// SPDX-License-Identifier: BUSL-1.1

//! Forensic payloads for inbound array-sync capture sites.

use faultbox::DomainContext;
use faultbox::serde_json::{Value, json};

/// A session's snapshot-assembly lock was poisoned: a thread panicked while
/// it held the lock, so no buffered assembly can be trusted.
pub(in crate::diag) struct ArraySnapshotBufferPoisoned<'a> {
    /// The array the frame that found the poison named.
    pub array: &'a str,
    pub tenant_id: u64,
    pub database_id: u64,
    /// Partial assemblies discarded with the lock.
    pub discarded: usize,
}

impl DomainContext for ArraySnapshotBufferPoisoned<'_> {
    fn domain_kind(&self) -> &'static str {
        "nodedb.array_snapshot_buffer_poisoned"
    }

    fn grouping_key(&self) -> String {
        // One root cause: a panic under the lock. The array, scope and
        // count are the occurrence.
        "array_snapshot_buffer_poisoned".to_string()
    }

    fn to_json(&self) -> Value {
        json!({
            "array": self.array,
            "tenant_id": self.tenant_id,
            "database_id": self.database_id,
            "discarded_assemblies": self.discarded,
            "effect": "every partial inbound snapshot of the session was discarded. The \
                       frame that found the poison drew a SnapshotBufferLost reject",
            "operator_action": "find the panic that poisoned the lock in this report's \
                                breadcrumbs and fix it. Clients re-send their snapshots",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_ignores_the_occurrence() {
        let first = ArraySnapshotBufferPoisoned {
            array: "a",
            tenant_id: 1,
            database_id: 0,
            discarded: 2,
        };
        let second = ArraySnapshotBufferPoisoned {
            array: "b",
            tenant_id: 9,
            database_id: 4,
            discarded: 7,
        };
        assert_eq!(first.grouping_key(), second.grouping_key());
    }
}
