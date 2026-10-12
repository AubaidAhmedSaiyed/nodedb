// SPDX-License-Identifier: BUSL-1.1

//! Advisory acknowledgement and catch-up handlers for array sync.
//!
//! A refused advisory request draws an `ArrayAdvisoryRefused` reply. It
//! names no local op, so it never touches the client's push window.

use std::sync::Arc;

use nodedb_array::sync::hlc::Hlc;
use nodedb_types::sync::wire::array::{
    ArrayAckMsg, ArrayAdvisoryKind, ArrayAdvisoryRefusedMsg, ArrayCatchupRequestMsg,
    ArrayRejectReason,
};
use tracing::warn;

use super::catchup::OriginCatchupServer;
use super::inbound::{InboundOutcome, OriginArrayInbound};
use super::reject::reason_for_engine_error;

impl OriginArrayInbound {
    /// Record a peer ack for GC frontier tracking.
    ///
    /// Forwards the ack into the `ArrayAckRegistry` on `SharedState` so the
    /// GC task can compute the min-ack frontier for each array.
    pub fn handle_ack(&self, msg: &ArrayAckMsg) -> Result<InboundOutcome, ArrayAdvisoryRefusedMsg> {
        let ack_hlc = Hlc::from_bytes(&msg.ack_hlc_bytes);
        let _authorization = self
            .authorize_array(
                &msg.array,
                ack_hlc,
                crate::control::security::identity::Permission::Write,
            )
            .map_err(|reject| {
                ArrayAdvisoryRefusedMsg::from_reject(ArrayAdvisoryKind::Ack, reject)
            })?;
        let replica_id = nodedb_array::sync::replica_id::ReplicaId::new(msg.replica_id);
        self.shared.array_ack_registry.record_in_database(
            self.database_id(),
            self.tenant_id().as_u64(),
            &msg.array,
            replica_id,
            ack_hlc,
        );
        tracing::debug!(
            array = %msg.array,
            replica_id = msg.replica_id,
            ack_hlc = ?ack_hlc,
            "array_inbound: peer ack recorded"
        );
        Ok(InboundOutcome::AckRecorded)
    }

    /// Handle a catch-up request from a Lite peer.
    ///
    /// Delegates to [`OriginCatchupServer`] which selects the op-stream or
    /// snapshot delivery path and enqueues outbound frames. A denial, an
    /// unknown array, and a serve error each draw a typed refusal.
    pub fn handle_catchup_request(
        &self,
        msg: &ArrayCatchupRequestMsg,
        session_id: &str,
    ) -> Result<InboundOutcome, ArrayAdvisoryRefusedMsg> {
        let refuse = |reason: ArrayRejectReason, detail: String| ArrayAdvisoryRefusedMsg {
            array: msg.array.clone(),
            request: ArrayAdvisoryKind::CatchupRequest,
            reason,
            detail,
        };
        let from_hlc = Hlc::from_bytes(&msg.from_hlc_bytes);
        let _authorization = self
            .authorize_array(
                &msg.array,
                from_hlc,
                crate::control::security::identity::Permission::Read,
            )
            .map_err(|reject| {
                ArrayAdvisoryRefusedMsg::from_reject(ArrayAdvisoryKind::CatchupRequest, reject)
            })?;
        if self
            .schemas
            .schema_hlc_in_database(self.database_id(), self.tenant_id().as_u64(), &msg.array)
            .is_none()
        {
            return Err(refuse(
                ArrayRejectReason::ArrayUnknown,
                format!("array '{}' not known to this replica", msg.array),
            ));
        }
        let server = OriginCatchupServer::new(
            crate::control::array_sync::ArrayServerScope::new(
                self.database_id(),
                self.tenant_id().as_u64(),
            ),
            Arc::clone(&self.shared.array_sync_op_log),
            Arc::clone(&self.schemas),
            Arc::clone(&self.shared.array_snapshot_store),
            Arc::clone(&self.shared.array_delivery),
            Arc::clone(&self.shared.array_subscriber_cursors),
            Arc::clone(&self.shared.array_ack_registry),
        );

        if let Err(error) = server.serve(msg, session_id) {
            warn!(
                session = %session_id,
                array = %msg.array,
                error = %error,
                "array_inbound: catchup server error"
            );
            return Err(refuse(
                reason_for_engine_error(&error),
                format!("catch-up serve: {error}"),
            ));
        }

        Ok(InboundOutcome::CatchupRequested)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use nodedb_types::sync::wire::ack_status::AckStatus;

    use super::*;
    use crate::bridge::dispatch::Dispatcher;
    use crate::control::array_sync::OriginApplyEngine;
    use crate::control::security::identity::{AuthMethod, AuthenticatedIdentity, Permission};
    use crate::control::state::SharedState;
    use crate::types::{DatabaseId, TenantId};
    use crate::wal::WalManager;

    const ARRAY: &str = "advisory-array";
    const TENANT: u64 = 7;

    fn inbound_with_opposite_permission(
        permission: Permission,
    ) -> (OriginArrayInbound, Arc<SharedState>, tempfile::TempDir) {
        let directory = tempfile::tempdir().expect("create advisory test directory");
        let wal = Arc::new(
            WalManager::open_for_testing(&directory.path().join("advisory.wal"))
                .expect("open advisory test WAL"),
        );
        let (dispatcher, _data_sides) = Dispatcher::new(1, 64);
        let shared = SharedState::new(dispatcher, wal).expect("construct advisory test state");
        let identity = AuthenticatedIdentity::new_regular(
            1,
            "advisory-user",
            TenantId::new(TENANT),
            AuthMethod::Trust,
            vec![],
            Some(DatabaseId::DEFAULT),
            AuthenticatedIdentity::default_database_set(false),
        );
        shared
            .permissions
            .grant(
                &format!("collection:0:{TENANT}:{ARRAY}"),
                "user:advisory-user",
                permission,
                "test",
                None,
            )
            .expect("grant opposite advisory permission");
        let engine = Arc::new(OriginApplyEngine::new(
            Arc::clone(&shared.array_sync_schemas),
            Arc::clone(&shared.array_sync_op_log),
        ));
        let inbound = OriginArrayInbound::new(
            engine,
            Arc::clone(&shared.array_sync_schemas),
            Arc::clone(&shared),
            identity,
        );
        (inbound, shared, directory)
    }

    #[tokio::test]
    async fn ack_with_only_read_permission_rejects_before_ack_registry_mutation() {
        let (inbound, shared, _directory) = inbound_with_opposite_permission(Permission::Read);
        let msg = ArrayAckMsg {
            array: ARRAY.to_owned(),
            replica_id: 42,
            ack_hlc_bytes: Hlc::ZERO.to_bytes(),
            applied_seq: 0,
            status: AckStatus::Applied,
        };

        let rejection = inbound
            .handle_ack(&msg)
            .expect_err("read permission must not authorize ack mutation");

        assert_eq!(
            rejection.reason,
            nodedb_types::sync::wire::array::ArrayRejectReason::PermissionDenied
        );
        assert_eq!(rejection.request, ArrayAdvisoryKind::Ack);
        assert_eq!(rejection.array, ARRAY);
        assert_eq!(
            shared
                .array_ack_registry
                .min_ack_hlc_in_database(DatabaseId::DEFAULT, TENANT, ARRAY,),
            None,
            "denied ack must not populate the persistent ack cache"
        );
    }

    #[tokio::test]
    async fn catchup_with_only_write_permission_rejects_before_cursor_or_delivery_changes() {
        let (inbound, shared, _directory) = inbound_with_opposite_permission(Permission::Write);
        let session_id = "denied-catchup";
        let mut delivery = shared.array_delivery.register(session_id.to_owned());
        let sessions_registered = shared
            .array_delivery
            .sessions_registered
            .load(Ordering::Relaxed);
        let frames_dropped = shared.array_delivery.frames_dropped.load(Ordering::Relaxed);
        let msg = ArrayCatchupRequestMsg {
            array: ARRAY.to_owned(),
            from_hlc_bytes: Hlc::ZERO.to_bytes(),
        };

        let rejection = inbound
            .handle_catchup_request(&msg, session_id)
            .expect_err("write permission must not authorize catchup reads");

        assert_eq!(
            rejection.reason,
            nodedb_types::sync::wire::array::ArrayRejectReason::PermissionDenied
        );
        assert_eq!(rejection.request, ArrayAdvisoryKind::CatchupRequest);
        assert_eq!(rejection.array, ARRAY);
        assert!(
            shared
                .array_subscriber_cursors
                .get_in_database(session_id, DatabaseId::DEFAULT, TENANT, ARRAY)
                .is_none(),
            "denied catchup must not register a subscriber cursor"
        );
        assert!(
            delivery.try_recv().is_err(),
            "denied catchup must not enqueue delivery"
        );
        assert_eq!(
            shared
                .array_delivery
                .sessions_registered
                .load(Ordering::Relaxed),
            sessions_registered,
            "denied catchup must not alter delivery registration"
        );
        assert_eq!(
            shared.array_delivery.frames_dropped.load(Ordering::Relaxed),
            frames_dropped,
            "denied catchup must not alter delivery backpressure state"
        );
    }
}
