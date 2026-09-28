// SPDX-License-Identifier: BUSL-1.1

//! Forensic payload for a REINDEX of a full-text or CSR index that did not
//! install.
//!
//! REINDEX CONCURRENTLY acknowledges once the rebuild starts. A rebuild the
//! core refuses at cutover, or one whose build fails, leaves the live index
//! in place with every write it took. Nothing but this report and a log
//! line shows that the requested rebuild never happened.

use faultbox::DomainContext;
use faultbox::serde_json::{Value, json};

/// One rebuild of one index that did not install.
pub(in crate::diag) struct IndexRebuildNotInstalled<'a> {
    /// `fts` or `csr`.
    pub index: &'static str,
    pub database_id: u64,
    pub tenant_id: u64,
    pub collection: &'a str,
    /// What stopped the install, without the per-occurrence detail.
    pub cause_class: &'a str,
}

impl DomainContext for IndexRebuildNotInstalled<'_> {
    fn domain_kind(&self) -> &'static str {
        "nodedb.index_rebuild_not_installed"
    }

    fn grouping_key(&self) -> String {
        // The index kind and the cause name the bug. The collection is the
        // occurrence, so repeated REINDEX runs that hit one cause file one
        // report.
        format!("index={};cause={}", self.index, self.cause_class)
    }

    fn to_json(&self) -> Value {
        json!({
            "index": self.index,
            "database_id": self.database_id,
            "tenant_id": self.tenant_id,
            "collection": self.collection,
            "cause_class": self.cause_class,
            "why_reported": "REINDEX acknowledged this rebuild when it started. The \
                             rebuilt index was discarded, so the live index stays as it \
                             was, with every write it took. Results stay correct; the \
                             compaction the rebuild was meant to install never happened",
            "operator_action": "read the cause: a journal overflow means more writes landed \
                                during the rebuild than its journal holds, so run REINDEX \
                                again at a quieter time. A purge or supersede means the \
                                collection was dropped or rebuilt again, and needs no action. \
                                Any other cause is a storage or snapshot error: check the \
                                core's log for the same collection",
        })
    }
}
