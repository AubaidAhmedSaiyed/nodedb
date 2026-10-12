// SPDX-License-Identifier: BUSL-1.1

//! Process-wide count of dispatches the bridge dispatcher refused at a
//! capacity limit, by scope.
//!
//! The dispatcher records each refusal at the one place it builds
//! [`crate::Error::DispatchCapacity`]. The label is the scope only, never a
//! tenant or database id, so the series count stays fixed. Every field is
//! atomic, so `/metrics` reads it without taking the dispatcher lock.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::DispatchCapacityScope;

/// The `scope` label of each counter, by index.
const SCOPE_LABELS: [&str; 3] = ["tenant_inflight", "database_suspended", "queue_full"];

/// The counter index of `scope`.
fn scope_index(scope: &DispatchCapacityScope) -> usize {
    match scope {
        DispatchCapacityScope::TenantInflight { .. } => 0,
        DispatchCapacityScope::DatabaseSuspended { .. } => 1,
        DispatchCapacityScope::QueueFull { .. } => 2,
    }
}

/// Capacity refusals since process start, one counter per scope.
#[derive(Debug, Default)]
pub struct DispatchCapacityMetrics {
    refusals: [AtomicU64; SCOPE_LABELS.len()],
}

impl DispatchCapacityMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count one refusal at `scope`.
    pub(super) fn record(&self, scope: &DispatchCapacityScope) {
        self.refusals[scope_index(scope)].fetch_add(1, Ordering::Relaxed);
    }

    /// Refusals counted at the scope of `scope`, whatever its ids.
    pub fn refusals(&self, scope: &DispatchCapacityScope) -> u64 {
        self.refusals[scope_index(scope)].load(Ordering::Relaxed)
    }

    /// Append the `nodedb_dispatch_capacity_busy_total` counter.
    pub fn write_prometheus(&self, out: &mut String) {
        use std::fmt::Write as _;

        let _ = out.write_str(
            "# HELP nodedb_dispatch_capacity_busy_total Dispatches refused at a capacity \
             limit. Nothing was enqueued, and the client retries.\n\
             # TYPE nodedb_dispatch_capacity_busy_total counter\n",
        );
        for (label, count) in SCOPE_LABELS.iter().zip(&self.refusals) {
            let _ = writeln!(
                out,
                r#"nodedb_dispatch_capacity_busy_total{{scope="{label}"}} {}"#,
                count.load(Ordering::Relaxed)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DatabaseId, TenantId};

    fn tenant_scope() -> DispatchCapacityScope {
        DispatchCapacityScope::TenantInflight {
            tenant_id: TenantId::new(7),
            inflight: 4,
            cap: 4,
        }
    }

    fn database_scope() -> DispatchCapacityScope {
        DispatchCapacityScope::DatabaseSuspended {
            database_id: DatabaseId::new(3),
            core_id: 1,
        }
    }

    fn queue_scope() -> DispatchCapacityScope {
        DispatchCapacityScope::QueueFull {
            core_id: 0,
            capacity: 8,
        }
    }

    #[test]
    fn each_scope_counts_apart() {
        let metrics = DispatchCapacityMetrics::new();
        metrics.record(&tenant_scope());
        metrics.record(&tenant_scope());
        metrics.record(&queue_scope());

        assert_eq!(metrics.refusals(&tenant_scope()), 2);
        assert_eq!(metrics.refusals(&database_scope()), 0);
        assert_eq!(metrics.refusals(&queue_scope()), 1);
    }

    #[test]
    fn the_label_is_the_scope_not_its_ids() {
        let metrics = DispatchCapacityMetrics::new();
        metrics.record(&tenant_scope());
        metrics.record(&DispatchCapacityScope::TenantInflight {
            tenant_id: TenantId::new(99),
            inflight: 1,
            cap: 1,
        });
        assert_eq!(metrics.refusals(&tenant_scope()), 2);
    }

    #[test]
    fn exposition_has_one_series_per_scope() {
        let metrics = DispatchCapacityMetrics::new();
        metrics.record(&database_scope());
        let mut out = String::new();
        metrics.write_prometheus(&mut out);

        assert!(
            out.contains("# TYPE nodedb_dispatch_capacity_busy_total counter"),
            "{out}"
        );
        assert!(
            out.contains(r#"nodedb_dispatch_capacity_busy_total{scope="tenant_inflight"} 0"#),
            "{out}"
        );
        assert!(
            out.contains(r#"nodedb_dispatch_capacity_busy_total{scope="database_suspended"} 1"#),
            "{out}"
        );
        assert!(
            out.contains(r#"nodedb_dispatch_capacity_busy_total{scope="queue_full"} 0"#),
            "{out}"
        );
        assert_eq!(out.lines().count(), 2 + SCOPE_LABELS.len(), "{out}");
    }
}
