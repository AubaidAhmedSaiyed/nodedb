// SPDX-License-Identifier: BUSL-1.1

//! One CRDT admission and the deadline it runs under.
//!
//! Every phase ends at one deadline: the slot wait, the catch-up, the
//! preview and the proposal. A deadline that passes with no proposal in
//! flight refuses the delta, and nothing applied. A deadline that passes
//! while a proposal is in flight leaves the outcome unknown.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::control::server::shared::session::StatementDeadline;
use crate::control::state::SharedState;
use crate::event::EventSource;
use crate::types::{DatabaseId, RequestId, TenantId, VShardId};

use super::policy::CrdtPostImagePolicy;

/// Frontier mismatches an admission retries before it gives up.
pub(super) const FRONTIER_RETRY_LIMIT: usize = 8;

pub(super) struct CrdtAdmissionWorkflow<'a> {
    pub(super) state: &'a SharedState,
    pub(super) tenant_id: TenantId,
    pub(super) database_id: DatabaseId,
    pub(super) vshard_id: VShardId,
    /// The plan's database-qualified collection name.
    pub(super) collection: &'a str,
    pub(super) deadline: StatementDeadline,
    pub(super) event_source: EventSource,
    pub(super) policy: &'a dyn CrdtPostImagePolicy,
    /// The budget left when the admission started, for the refusal message.
    budget_ms: u64,
    /// Set while a proposal is in flight.
    proposing: AtomicBool,
}

impl<'a> CrdtAdmissionWorkflow<'a> {
    pub(super) fn new(
        state: &'a SharedState,
        tenant_id: TenantId,
        database_id: DatabaseId,
        collection: &'a str,
        deadline: StatementDeadline,
        event_source: EventSource,
        policy: &'a dyn CrdtPostImagePolicy,
    ) -> crate::Result<Self> {
        let vshard_id =
            nodedb_types::CollectionKey::from_qualified_str(database_id, collection)?.vshard();
        Ok(Self {
            state,
            tenant_id,
            database_id,
            vshard_id,
            collection,
            deadline,
            event_source,
            policy,
            budget_ms: u64::try_from(deadline.remaining().as_millis()).unwrap_or(u64::MAX),
            proposing: AtomicBool::new(false),
        })
    }

    /// Run `operation` in the vShard's admission slot, once this node applied
    /// every write admitted before it. The whole run ends at the deadline.
    pub(super) async fn run<T, F, Fut>(&self, operation: F) -> crate::Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = crate::Result<T>>,
    {
        let admission = self.state.vshard_admission_sequencer.run_after_proposed(
            self.vshard_id,
            |last_proposed| async move {
                crate::control::vshard_admission::await_admitted_applies(
                    self.state,
                    self.vshard_id,
                    last_proposed,
                    self.deadline.instant(),
                )
                .await?;
                operation().await
            },
        );
        match tokio::time::timeout_at(self.deadline.tokio_instant(), admission).await {
            Ok(result) => result,
            Err(_) => Err(self.expired()),
        }
    }

    /// Await `proposal` with the in-flight mark set. A deadline that already
    /// passed refuses before anything is proposed.
    pub(super) async fn propose<T>(
        &self,
        proposal: impl Future<Output = crate::Result<T>>,
    ) -> crate::Result<T> {
        if self.deadline.remaining().is_zero() {
            return Err(self.expired());
        }
        self.proposing.store(true, Ordering::Release);
        let outcome = proposal.await;
        self.proposing.store(false, Ordering::Release);
        outcome
    }

    /// The error of a passed deadline. With no proposal in flight it is a
    /// definite refusal. With one in flight the outcome is unknown.
    pub(super) fn expired(&self) -> crate::Error {
        if self.proposing.load(Ordering::Acquire) {
            crate::Error::DeadlineExceeded {
                request_id: RequestId::new(0),
            }
        } else {
            crate::Error::CrdtAdmissionTimeout {
                vshard_id: self.vshard_id,
                timeout_ms: self.budget_ms,
            }
        }
    }

    /// `error` of a dispatch that runs before any proposal. A passed deadline
    /// there refuses the delta.
    pub(super) fn before_propose(&self, error: crate::Error) -> crate::Error {
        match error {
            crate::Error::DeadlineExceeded { .. } => self.expired(),
            other => other,
        }
    }
}
