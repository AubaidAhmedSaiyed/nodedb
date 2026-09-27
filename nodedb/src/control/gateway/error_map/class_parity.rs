// SPDX-License-Identifier: BUSL-1.1

//! Every Data-Plane `ErrorCode`, and each Control-Plane error a client acts
//! on, answers one class on native, pgwire and HTTP.
//!
//! pgwire renders a Data-Plane verdict as its SQLSTATE. A native client reads
//! the numeric `nodedb_types` code on the frame. The two agree when the
//! numeric code renders, through the numeric-code SQLSTATE table, in the same
//! SQLSTATE class (the first two characters) as the pgwire SQLSTATE. That same
//! table renders a code that crossed a node as a bare number, so agreement
//! also keeps a verdict's class across nodes.

use nodedb_types::error::sqlstate;
use nodedb_types::sync::violation::ViolationType;
use nodedb_types::sync::wire::SyncProvenance;

use crate::bridge::envelope::{CounterFault, ErrorCode, SyncHold};
use crate::control::server::native::dispatch::{error_code_to_native, native_error_fields};
use crate::control::server::pgwire::types::error_map::numeric_code_to_sqlstate;
use crate::control::server::pgwire::types::error_to_sqlstate;

use super::gateway_map::GatewayErrorMap;

/// The number of `ErrorCode` variants [`variant_index`] numbers.
const VARIANT_COUNT: usize = 41;

/// A dense index per variant. Exhaustive, so a new variant fails to compile
/// here until it gets an index, and [`every_variant_has_a_sample`] then fails
/// until [`samples`] carries it.
fn variant_index(code: &ErrorCode) -> usize {
    match code {
        ErrorCode::DeadlineExceeded => 0,
        ErrorCode::RejectedConstraint { .. } => 1,
        ErrorCode::RejectedPrevalidation { .. } => 2,
        ErrorCode::RetryableRefusal { .. } => 3,
        ErrorCode::SyncRejected { .. } => 4,
        ErrorCode::SyncNotApplied { .. } => 5,
        ErrorCode::NotFound => 6,
        ErrorCode::RejectedAuthz { .. } => 7,
        ErrorCode::ConflictRetry => 8,
        ErrorCode::CrdtFrontierMismatch { .. } => 9,
        ErrorCode::FanOutExceeded => 10,
        ErrorCode::ResourcesExhausted => 11,
        ErrorCode::RejectedDanglingEdge { .. } => 12,
        ErrorCode::DuplicateWrite => 13,
        ErrorCode::AppendOnlyViolation { .. } => 14,
        ErrorCode::BalanceViolation { .. } => 15,
        ErrorCode::PeriodLocked { .. } => 16,
        ErrorCode::PeriodLockMisconfigured { .. } => 17,
        ErrorCode::RetentionViolation { .. } => 18,
        ErrorCode::LegalHoldActive { .. } => 19,
        ErrorCode::StateTransitionViolation { .. } => 20,
        ErrorCode::TransitionCheckViolation { .. } => 21,
        ErrorCode::TypeGuardViolation { .. } => 22,
        ErrorCode::TypeMismatch { .. } => 23,
        ErrorCode::CounterFault { .. } => 24,
        ErrorCode::InsufficientBalance { .. } => 25,
        ErrorCode::RateExceeded { .. } => 26,
        ErrorCode::CollectionDraining { .. } => 27,
        ErrorCode::RecursionDepthExceeded { .. } => 28,
        ErrorCode::UndefinedColumn { .. } => 29,
        ErrorCode::Internal { .. } => 30,
        ErrorCode::Unsupported { .. } => 31,
        ErrorCode::RollbackFailed { .. } => 32,
        ErrorCode::OllpRetryRequired => 33,
        ErrorCode::TxnOverlayMemoryExceeded { .. } => 34,
        ErrorCode::DivisionByZero => 35,
        ErrorCode::UndefinedFunction { .. } => 36,
        ErrorCode::DataException { .. } => 37,
        ErrorCode::DispatchCapacity { .. } => 38,
        ErrorCode::ExpiredBeforeExecution => 39,
        ErrorCode::BadRequest { .. } => 40,
    }
}

fn provenance() -> SyncProvenance {
    SyncProvenance {
        producer_id: 1,
        epoch: 1,
        stream_id: 1,
        seq: 1,
    }
}

/// One sample per variant, plus one per value that picks a different
/// SQLSTATE: each constraint kind and each counter fault.
fn samples() -> Vec<ErrorCode> {
    let text = || "detail".to_owned();
    let collection = || "c".to_owned();
    let mut samples = vec![
        ErrorCode::DeadlineExceeded,
        ErrorCode::RejectedPrevalidation { reason: text() },
        ErrorCode::RetryableRefusal { reason: text() },
        ErrorCode::SyncRejected {
            violation: ViolationType::PermissionDenied,
            applied_seq: 1,
            provenance: provenance(),
        },
        ErrorCode::SyncRejected {
            violation: ViolationType::RateLimited,
            applied_seq: 1,
            provenance: provenance(),
        },
        ErrorCode::SyncNotApplied {
            hold: SyncHold::Gap { expected: 2 },
            applied_seq: 1,
        },
        ErrorCode::NotFound,
        ErrorCode::RejectedAuthz { resource: text() },
        ErrorCode::ConflictRetry,
        ErrorCode::CrdtFrontierMismatch {
            expected: [0; 32],
            actual: [1; 32],
        },
        ErrorCode::FanOutExceeded,
        ErrorCode::ResourcesExhausted,
        ErrorCode::RejectedDanglingEdge {
            missing_node: text(),
        },
        ErrorCode::DuplicateWrite,
        ErrorCode::AppendOnlyViolation {
            collection: collection(),
        },
        ErrorCode::BalanceViolation {
            collection: collection(),
            detail: text(),
        },
        ErrorCode::PeriodLocked {
            collection: collection(),
        },
        ErrorCode::PeriodLockMisconfigured {
            collection: collection(),
            ref_table: "periods".into(),
            status_column: "status".into(),
            row_identity: "p1".into(),
        },
        ErrorCode::RetentionViolation {
            collection: collection(),
        },
        ErrorCode::LegalHoldActive {
            collection: collection(),
        },
        ErrorCode::StateTransitionViolation {
            collection: collection(),
            detail: text(),
        },
        ErrorCode::TransitionCheckViolation {
            collection: collection(),
            detail: text(),
        },
        ErrorCode::TypeGuardViolation {
            collection: collection(),
            detail: text(),
        },
        ErrorCode::TypeMismatch {
            collection: collection(),
            detail: text(),
        },
        ErrorCode::InsufficientBalance {
            collection: collection(),
            detail: text(),
        },
        ErrorCode::RateExceeded {
            gate: "g".into(),
            retry_after_ms: 10,
        },
        ErrorCode::CollectionDraining {
            collection: collection(),
        },
        ErrorCode::RecursionDepthExceeded {
            cte_name: "walk".into(),
            max_depth: 100,
        },
        ErrorCode::UndefinedColumn { column: "x".into() },
        ErrorCode::Internal { detail: text() },
        ErrorCode::Unsupported { detail: text() },
        ErrorCode::RollbackFailed {
            entry_index: 0,
            detail: text(),
        },
        ErrorCode::OllpRetryRequired,
        ErrorCode::TxnOverlayMemoryExceeded { limit: 1 << 20 },
        ErrorCode::DivisionByZero,
        ErrorCode::UndefinedFunction { name: "f".into() },
        ErrorCode::DataException { detail: text() },
        ErrorCode::DispatchCapacity { reason: text() },
        ErrorCode::ExpiredBeforeExecution,
        ErrorCode::BadRequest { detail: text() },
    ];
    for constraint in [
        "not_null",
        "unique",
        "generated_always",
        "fk_missing",
        "rls_policy",
        "permission_denied",
        "check",
    ] {
        samples.push(ErrorCode::RejectedConstraint {
            constraint: constraint.into(),
            detail: text(),
        });
    }
    for fault in [
        CounterFault::NotAnInteger,
        CounterFault::NotAFloat,
        CounterFault::IntegerOverflow,
        CounterFault::NonFinite,
    ] {
        samples.push(ErrorCode::CounterFault {
            collection: collection(),
            fault,
        });
    }
    samples
}

fn class(state: &str) -> &str {
    state.get(..2).unwrap_or(state)
}

#[test]
fn every_variant_has_a_sample() {
    let mut seen = [false; VARIANT_COUNT];
    for code in samples() {
        seen[variant_index(&code)] = true;
    }
    let missing: Vec<usize> = (0..VARIANT_COUNT).filter(|i| !seen[*i]).collect();
    assert!(missing.is_empty(), "variants with no sample: {missing:?}");
}

/// The native frame carries pgwire's SQLSTATE, and its numeric code has the
/// same SQLSTATE class, on both native renderings: the typed `Err` and the
/// raw response frame.
#[test]
fn every_data_plane_code_has_one_class_on_native_and_pgwire() {
    for code in samples() {
        let err = crate::Error::DataPlane(code.clone());
        let (_, pg_state, _) = error_to_sqlstate(&err);

        let native = native_error_fields(&err);
        assert_eq!(native.sqlstate, pg_state, "native SQLSTATE for {code:?}");
        let native_state = numeric_code_to_sqlstate(native.code);
        assert_eq!(
            class(native_state),
            class(pg_state),
            "{code:?}: pgwire sends {pg_state}, native code {} renders {native_state}",
            native.code
        );

        let frame = error_code_to_native(1, Some(&code));
        let payload = frame.error.expect("error frames carry a payload");
        assert_eq!(
            payload.code, pg_state,
            "response-frame SQLSTATE for {code:?}"
        );
        assert_eq!(
            payload.ndb_code, native.code.0,
            "response-frame code for {code:?}"
        );
    }
}

/// A classified Data-Plane verdict never reads as a server fault over HTTP.
#[test]
fn classified_data_plane_codes_are_not_http_500() {
    for code in samples() {
        let err = crate::Error::DataPlane(code.clone());
        let (_, pg_state, _) = error_to_sqlstate(&err);
        let (status, _) = GatewayErrorMap::to_http(&err);
        if pg_state == sqlstate::INTERNAL_ERROR {
            assert_eq!(status, 500, "{code:?}");
        } else {
            assert_ne!(status, 500, "{code:?} is {pg_state} on pgwire");
        }
    }
}

/// The SQLSTATE status table agrees with the gateway status table for every
/// Data-Plane code. A DDL error and a query error of one class answer one
/// HTTP status.
#[test]
fn sqlstate_status_agrees_with_the_gateway_status() {
    for code in samples() {
        let err = crate::Error::DataPlane(code.clone());
        let (_, pg_state, _) = error_to_sqlstate(&err);
        let (status, _) = GatewayErrorMap::to_http(&err);
        assert_eq!(
            GatewayErrorMap::sqlstate_to_http(pg_state),
            status,
            "{code:?} is {pg_state} on pgwire"
        );
    }
}

/// `Unsupported` is feature-not-supported on every surface.
#[test]
fn unsupported_is_feature_not_supported_everywhere() {
    let err = crate::Error::DataPlane(ErrorCode::Unsupported {
        detail: "not on this engine".into(),
    });
    let (_, pg_state, _) = error_to_sqlstate(&err);
    assert_eq!(pg_state, sqlstate::FEATURE_NOT_SUPPORTED);
    let native = native_error_fields(&err);
    assert_eq!(native.code, nodedb_types::error::ErrorCode::SQL_NOT_ENABLED);
    assert_eq!(GatewayErrorMap::to_http(&err).0, 501);
}

/// Control-Plane errors a client acts on. Each has a class of its own.
fn control_plane_samples() -> Vec<crate::Error> {
    vec![
        crate::Error::RetryableSchemaChanged {
            descriptor: "orders".into(),
        },
        crate::Error::SessionTokenExpired,
    ]
}

/// A Control-Plane error has one class on native and pgwire, and its native
/// numeric code renders in that class.
#[test]
fn control_plane_errors_have_one_class_on_native_and_pgwire() {
    for err in control_plane_samples() {
        let (_, pg_state, _) = error_to_sqlstate(&err);
        assert_ne!(pg_state, sqlstate::INTERNAL_ERROR, "{err:?} has no class");

        let native = native_error_fields(&err);
        assert_eq!(native.sqlstate, pg_state, "native SQLSTATE for {err:?}");
        let native_state = numeric_code_to_sqlstate(native.code);
        assert_eq!(
            class(native_state),
            class(pg_state),
            "{err:?}: pgwire sends {pg_state}, native code {} renders {native_state}",
            native.code
        );
    }
}

/// A schema change the server could not absorb is the retryable
/// serialization class on every surface.
#[test]
fn schema_change_is_a_retryable_serialization_failure() {
    let err = crate::Error::RetryableSchemaChanged {
        descriptor: "orders".into(),
    };
    assert_eq!(error_to_sqlstate(&err).1, sqlstate::SERIALIZATION_FAILURE);
    let native = native_error_fields(&err);
    assert_eq!(native.sqlstate, sqlstate::SERIALIZATION_FAILURE);
    assert_eq!(native.code, nodedb_types::error::ErrorCode::WRITE_CONFLICT);
    assert!(crate::error_classify::classify(&err).is_retriable());
    let status = GatewayErrorMap::to_http(&err).0;
    assert_eq!(status, 409);
    assert_eq!(
        GatewayErrorMap::sqlstate_to_http(sqlstate::SERIALIZATION_FAILURE),
        status
    );
}

/// An expired session token is invalid authorization on every surface.
#[test]
fn expired_session_token_is_invalid_authorization_everywhere() {
    let err = crate::Error::SessionTokenExpired;
    assert_eq!(error_to_sqlstate(&err).1, sqlstate::INVALID_AUTHORIZATION);
    let native = native_error_fields(&err);
    assert_eq!(native.sqlstate, sqlstate::INVALID_AUTHORIZATION);
    assert_eq!(native.code, nodedb_types::error::ErrorCode::AUTH_EXPIRED);
    assert_eq!(
        numeric_code_to_sqlstate(native.code),
        sqlstate::INVALID_AUTHORIZATION
    );
    let status = GatewayErrorMap::to_http(&err).0;
    assert_eq!(status, 401);
    assert_eq!(
        GatewayErrorMap::sqlstate_to_http(sqlstate::INVALID_AUTHORIZATION),
        status
    );
}

/// The number of `crate::Error` variants [`error_variant_index`] numbers.
const ERROR_VARIANT_COUNT: usize = 108;

/// A dense index per `crate::Error` variant. Exhaustive, so a new variant
/// fails to compile here until it gets an index, and
/// [`every_error_variant_has_a_sample`] then fails until
/// [`error_samples`] carries it.
pub(crate) fn error_variant_index(err: &crate::Error) -> usize {
    use crate::Error as E;
    match err {
        E::RejectedConstraint { .. } => 0,
        E::TxnOverlayMemoryExceeded { .. } => 1,
        E::RejectedAuthz { .. } => 2,
        E::OffsetRegression { .. } => 3,
        E::DeadlineExceeded { .. } => 4,
        E::ConflictRetry { .. } => 5,
        E::CalvinSerializationConflict => 6,
        E::CalvinParticipantError => 7,
        E::RejectedPrevalidation { .. } => 8,
        E::RetryableRefusal { .. } => 9,
        E::AppendOnlyViolation { .. } => 10,
        E::BalanceViolation { .. } => 11,
        E::MaterializedSumTargetNotFound { .. } => 12,
        E::MaterializedSumResolutionMissing { .. } => 13,
        E::PeriodLocked { .. } => 14,
        E::PeriodLockMisconfigured { .. } => 15,
        E::RetentionViolation { .. } => 16,
        E::LegalHoldActive { .. } => 17,
        E::StateTransitionViolation { .. } => 18,
        E::TransitionCheckViolation { .. } => 19,
        E::TypeGuardViolation { .. } => 20,
        E::TypeMismatch { .. } => 21,
        E::InsufficientBalance { .. } => 22,
        E::RateExceeded { .. } => 23,
        E::CollectionNotFound { .. } => 24,
        E::DocumentNotFound { .. } => 25,
        E::CollectionDeactivated { .. } => 26,
        E::VShardAdmissionCapacityExceeded { .. } => 27,
        E::CrdtAdmissionRetriesExhausted { .. } => 28,
        E::CrdtAdmissionInvalidPlan { .. } => 29,
        E::CrdtAdmissionCallerFence => 30,
        E::CrdtApplyRequiresAdmission => 31,
        E::CrdtApplyForbiddenInTransaction => 32,
        E::NotInTransactionBlock { .. } => 33,
        E::CrdtAdmissionTimeout { .. } => 34,
        E::NoLeader { .. } => 35,
        E::NotLeader { .. } => 36,
        E::FanOutExceeded { .. } => 37,
        E::CrossCollectionNotColocated { .. } => 38,
        E::SourceFrozen { .. } => 39,
        E::CloneWriteRequiresMaterialize { .. } => 40,
        E::BadRequest { .. } => 41,
        E::BackupTenantMismatch { .. } => 42,
        E::BackupKeyMismatch => 43,
        E::QuotaOvercommit { .. } => 44,
        E::PlanError { .. } => 45,
        E::FeatureNotSupported { .. } => 46,
        E::UndefinedFunction { .. } => 47,
        E::UndefinedObject { .. } => 48,
        E::ObjectNotInPrerequisiteState { .. } => 49,
        E::UndefinedColumn { .. } => 50,
        E::AmbiguousColumn { .. } => 51,
        E::UnknownStrictField { .. } => 52,
        E::DivisionByZero => 53,
        E::DataException { .. } => 54,
        E::InvalidLimitValue { .. } => 55,
        E::RetryableSchemaChanged { .. } => 56,
        E::RetryableLeaderChange { .. } => 57,
        E::GroupQuorumUnavailable { .. } => 58,
        E::GroupMarksUnavailable { .. } => 59,
        E::MetadataLeaderUnavailable => 60,
        E::AuthorizationStateBehind { .. } => 61,
        E::ExecutionLimitExceeded { .. } => 62,
        E::LimitExceeded { .. } => 63,
        E::Wal(_) => 64,
        E::Dispatch { .. } => 65,
        E::DispatchCapacity { .. } => 66,
        E::Storage { .. } => 67,
        E::ColdStorage { .. } => 68,
        E::Serialization { .. } => 69,
        E::Codec { .. } => 70,
        E::SegmentCorrupted { .. } => 71,
        E::MemoryExhausted { .. } => 72,
        E::Backpressure { .. } => 73,
        E::Crdt(_) => 74,
        E::Io(_) => 75,
        E::Config { .. } => 76,
        E::Encryption { .. } => 77,
        E::Bridge { .. } => 78,
        E::VersionCompat { .. } => 79,
        E::Internal { .. } => 80,
        E::Shaping(_) => 81,
        E::RemoteTyped { .. } => 82,
        E::DescriptorVersionAnomaly { .. } => 83,
        E::CollectionPurgeRowMissing { .. } => 84,
        E::CatalogIntegrityViolation { .. } => 85,
        E::DataPlane(_) => 86,
        E::Promql(_) => 87,
        E::DependentObjectsExist { .. } => 88,
        E::CascadeCycle { .. } => 89,
        E::CrossShardInExplicitTransaction => 90,
        E::SequencerUnavailable => 91,
        E::SessionCapExceeded { .. } => 92,
        E::SessionIdleTimeout => 93,
        E::SessionTokenExpired => 94,
        E::SessionKilledByAdmin => 95,
        E::SessionUserDropped => 96,
        E::OidcProviderTenantUnbound => 97,
        E::OidcProviderTenantUnavailable { .. } => 98,
        E::ExternalRoleUndefined { .. } => 99,
        E::OidcNoDefaultDatabase { .. } => 100,
        E::TenantVectorDimExceeded { .. } => 101,
        E::TenantGraphDepthExceeded { .. } => 102,
        E::RoleInheritanceCycle { .. } => 103,
        E::RoleInheritanceDepthExceeded { .. } => 104,
        E::OllpExhausted { .. } => 105,
        E::MirrorReadOnly { .. } => 106,
        E::StaleReadNotLeader { .. } => 107,
    }
}

/// One sample per `crate::Error` variant.
pub(crate) fn error_samples() -> Vec<crate::Error> {
    use crate::Error as E;
    use crate::types::{DatabaseId, RequestId, TenantId, VShardId};

    let text = || "detail".to_owned();
    let collection = || "c".to_owned();
    vec![
        E::RejectedConstraint {
            collection: collection(),
            constraint: "unique".into(),
            detail: text(),
        },
        E::TxnOverlayMemoryExceeded { limit: 1 << 20 },
        E::RejectedAuthz {
            tenant_id: TenantId::new(1),
            resource: text(),
        },
        E::OffsetRegression {
            stream: "s".into(),
            group: "g".into(),
            partition_id: 0,
            current_lsn: 2,
            current_sequence: 2,
            attempted_lsn: 1,
            attempted_sequence: 1,
        },
        E::DeadlineExceeded {
            request_id: RequestId::new(1),
        },
        E::ConflictRetry {
            collection: collection(),
            document_id: "d".into(),
        },
        E::CalvinSerializationConflict,
        E::CalvinParticipantError,
        E::RejectedPrevalidation {
            constraint: "check".into(),
            reason: text(),
        },
        E::RetryableRefusal { reason: text() },
        E::AppendOnlyViolation {
            collection: collection(),
            detail: text(),
        },
        E::BalanceViolation {
            collection: collection(),
            detail: text(),
        },
        E::MaterializedSumTargetNotFound {
            target_collection: "t".into(),
            join_column: "k".into(),
            join_value: "1".into(),
        },
        E::MaterializedSumResolutionMissing {
            target_collection: "t".into(),
            join_column: "k".into(),
            join_value: "1".into(),
        },
        E::PeriodLocked {
            collection: collection(),
            detail: text(),
        },
        E::PeriodLockMisconfigured {
            collection: collection(),
            ref_table: "periods".into(),
            status_column: "status".into(),
            row_identity: "p1".into(),
        },
        E::RetentionViolation {
            collection: collection(),
            detail: text(),
        },
        E::LegalHoldActive {
            collection: collection(),
            detail: text(),
        },
        E::StateTransitionViolation {
            collection: collection(),
            detail: text(),
        },
        E::TransitionCheckViolation {
            collection: collection(),
            detail: text(),
        },
        E::TypeGuardViolation {
            collection: collection(),
            detail: text(),
        },
        E::TypeMismatch {
            collection: collection(),
            key: "k".into(),
            detail: text(),
        },
        E::InsufficientBalance {
            collection: collection(),
            key: "k".into(),
            detail: text(),
        },
        E::RateExceeded {
            gate: "g".into(),
            detail: text(),
            retry_after_ms: 10,
        },
        E::CollectionNotFound {
            tenant_id: TenantId::new(1),
            collection: collection(),
        },
        E::DocumentNotFound {
            collection: collection(),
            document_id: "d".into(),
        },
        E::CollectionDeactivated {
            tenant_id: TenantId::new(1),
            collection: collection(),
            retention_expires_at_ns: 1,
        },
        E::VShardAdmissionCapacityExceeded {
            vshard_id: VShardId::new(1),
            capacity: 4,
        },
        E::CrdtAdmissionRetriesExhausted {
            vshard_id: VShardId::new(1),
            attempts: 3,
        },
        E::CrdtAdmissionInvalidPlan { reason: "empty" },
        E::CrdtAdmissionCallerFence,
        E::CrdtApplyRequiresAdmission,
        E::CrdtApplyForbiddenInTransaction,
        E::NotInTransactionBlock {
            statement: "VACUUM".into(),
        },
        E::CrdtAdmissionTimeout {
            vshard_id: VShardId::new(1),
            timeout_ms: 10,
        },
        E::NoLeader {
            vshard_id: VShardId::new(1),
        },
        E::NotLeader {
            vshard_id: VShardId::new(1),
            leader_node: 2,
            leader_addr: "10.0.0.1:9000".into(),
        },
        E::FanOutExceeded {
            shards_touched: 9,
            limit: 8,
        },
        E::CrossCollectionNotColocated {
            op: "insert-select",
            source_collection: "a".into(),
            target_collection: "b".into(),
        },
        E::SourceFrozen {
            database_id: DatabaseId::new(7),
        },
        E::CloneWriteRequiresMaterialize {
            collection: collection(),
            engine: "kv".into(),
            database: "db".into(),
            reason: "shadowed",
        },
        E::BadRequest { detail: text() },
        E::BackupTenantMismatch {
            expected: 1,
            actual: 2,
        },
        E::BackupKeyMismatch,
        E::QuotaOvercommit {
            field: "max_storage".into(),
            detail: text(),
        },
        E::PlanError { detail: text() },
        E::FeatureNotSupported { detail: text() },
        E::UndefinedFunction { name: "f".into() },
        E::UndefinedObject {
            kind: "sequence",
            name: "s".into(),
        },
        E::ObjectNotInPrerequisiteState {
            object: "s".into(),
            detail: text(),
        },
        E::UndefinedColumn { column: "x".into() },
        E::AmbiguousColumn {
            column: "id".into(),
        },
        E::UnknownStrictField {
            collection: collection(),
            column: "x".into(),
        },
        E::DivisionByZero,
        E::DataException { detail: text() },
        E::InvalidLimitValue {
            clause: "LIMIT",
            value: "-1".into(),
        },
        E::RetryableSchemaChanged {
            descriptor: "orders".into(),
        },
        E::RetryableLeaderChange {
            group_id: 1,
            log_index: 2,
        },
        E::GroupQuorumUnavailable {
            group_id: 1,
            voters: vec![1, 2, 3],
            unreachable: vec![2, 3],
        },
        E::GroupMarksUnavailable {
            group_id: 1,
            refused_by: vec![2],
        },
        E::MetadataLeaderUnavailable,
        E::AuthorizationStateBehind { detail: text() },
        E::ExecutionLimitExceeded { detail: text() },
        E::LimitExceeded {
            limit_name: "max_rows",
            value: 10,
            max: 5,
        },
        E::Wal(nodedb_wal::WalError::Sealed),
        E::Dispatch { detail: text() },
        E::DispatchCapacity {
            scope: crate::DispatchCapacityScope::QueueFull {
                core_id: 0,
                capacity: 4,
            },
        },
        E::Storage {
            engine: "kv".into(),
            detail: text(),
        },
        E::ColdStorage { detail: text() },
        E::Serialization {
            format: "msgpack".into(),
            detail: text(),
        },
        E::Codec { detail: text() },
        E::SegmentCorrupted { detail: text() },
        E::MemoryExhausted {
            engine: "kv".into(),
        },
        E::Backpressure {
            engine: nodedb_mem::EngineId::Vector,
        },
        E::Crdt(nodedb_crdt::CrdtError::ConstraintViolation {
            constraint: "unique".into(),
            collection: collection(),
            detail: text(),
        }),
        E::Io(std::io::Error::other("disk")),
        E::Config { detail: text() },
        E::Encryption { detail: text() },
        E::Bridge { detail: text() },
        E::VersionCompat { detail: text() },
        E::Internal { detail: text() },
        E::Shaping(Box::new(nodedb_types::NodeDbError::bad_request(text()))),
        E::RemoteTyped {
            code: nodedb_types::error::ErrorCode::WRITE_CONFLICT,
            message: text(),
        },
        E::DescriptorVersionAnomaly {
            descriptor: "orders".into(),
            carried: 5,
            prior: 2,
        },
        E::CollectionPurgeRowMissing {
            database_id: 1,
            tenant_id: 1,
            name: collection(),
        },
        E::CatalogIntegrityViolation {
            entry_kind: "PutCollection".into(),
            detail: text(),
        },
        E::DataPlane(ErrorCode::NotFound),
        E::Promql(crate::control::promql::PromqlError::UnexpectedEof),
        E::DependentObjectsExist {
            tenant_id: 1,
            root_kind: "collection",
            root_name: collection(),
            dependent_count: 1,
            dependents: vec![("view".into(), "v".into())],
        },
        E::CascadeCycle {
            tenant_id: 1,
            root: collection(),
            depth: 64,
        },
        E::CrossShardInExplicitTransaction,
        E::SequencerUnavailable,
        E::SessionCapExceeded { cap: 8 },
        E::SessionIdleTimeout,
        E::SessionTokenExpired,
        E::SessionKilledByAdmin,
        E::SessionUserDropped,
        E::OidcProviderTenantUnbound,
        E::OidcProviderTenantUnavailable { tenant_id: 1 },
        E::ExternalRoleUndefined {
            subject: "alice".into(),
            role: "auditor".into(),
            tenant_id: 1,
        },
        E::OidcNoDefaultDatabase {
            sub: "alice".into(),
        },
        E::TenantVectorDimExceeded {
            dim: 4096,
            limit: 1024,
        },
        E::TenantGraphDepthExceeded {
            depth: 20,
            limit: 10,
        },
        E::RoleInheritanceCycle {
            child: "a".into(),
            parent: "b".into(),
        },
        E::RoleInheritanceDepthExceeded { depth: 9, limit: 8 },
        E::OllpExhausted {
            retries: 3,
            cause: crate::OllpExhaustedCause::PredicateDrift,
        },
        E::MirrorReadOnly {
            database: "db".into(),
        },
        E::StaleReadNotLeader {
            database: "db".into(),
            source_cluster: "src".into(),
            detail: text(),
        },
    ]
}

#[test]
fn every_error_variant_has_a_sample() {
    let mut seen = [false; ERROR_VARIANT_COUNT];
    for err in error_samples() {
        seen[error_variant_index(&err)] = true;
    }
    let missing: Vec<usize> = (0..ERROR_VARIANT_COUNT).filter(|i| !seen[*i]).collect();
    assert!(
        missing.is_empty(),
        "error variants with no sample: {missing:?}"
    );
}

/// Every `crate::Error` variant answers the HTTP status its pgwire SQLSTATE
/// class has. Only an internal or system error reads as a 500.
#[test]
fn every_error_variant_has_the_http_status_of_its_sqlstate() {
    for err in error_samples() {
        let (_, pg_state, _) = error_to_sqlstate(&err);
        let (status, _) = GatewayErrorMap::to_http(&err);
        assert_eq!(
            status,
            GatewayErrorMap::sqlstate_to_http(pg_state),
            "{err:?} is {pg_state} on pgwire"
        );
        let server_fault = matches!(class(pg_state), "XX" | "58");
        assert_eq!(
            status == 500,
            server_fault,
            "{err:?} is {pg_state} on pgwire but HTTP {status}"
        );
    }
}

/// Variants whose pgwire SQLSTATE class has no public numeric code: `25`
/// (`CrdtApplyForbiddenInTransaction`, `NotInTransactionBlock`,
/// `CrossShardInExplicitTransaction`), `2B` (`DependentObjectsExist`), and
/// `40000` (`CalvinParticipantError`, whose code is deliberately not a write
/// conflict). The numeric wire form cannot carry their class.
const NUMERIC_CLASS_GAPS: [usize; 5] = [7, 32, 33, 88, 90];

/// Every `crate::Error` variant renders the SQLSTATE class it renders locally
/// after it crosses a node hop, through both wire encoders and the decoder.
#[test]
fn every_error_variant_keeps_its_class_across_a_node_hop() {
    use nodedb_cluster::rpc_codec::TypedClusterError;

    use crate::control::cluster::data_plane_error_wire::execution_error_to_typed;

    let encoders: [(&str, fn(crate::Error) -> TypedClusterError); 2] = [
        ("execution_error_to_typed", execution_error_to_typed),
        ("From<Error>", TypedClusterError::from),
    ];
    for (name, encode) in encoders {
        for (err, twin) in error_samples().into_iter().zip(error_samples()) {
            if NUMERIC_CLASS_GAPS.contains(&error_variant_index(&err)) {
                continue;
            }
            let (_, local, _) = error_to_sqlstate(&err);
            let rebuilt = crate::Error::from(encode(twin));
            let (_, remote, _) = error_to_sqlstate(&rebuilt);
            assert_eq!(
                class(remote),
                class(local),
                "{name}: {err:?} is {local} locally but {remote} after the hop as {rebuilt:?}"
            );
        }
    }
}

/// The SQLSTATE each Control-Plane variant renders where it has a class of
/// its own, pinned by variant index.
fn classified_sqlstates() -> Vec<(usize, &'static str)> {
    vec![
        (3, sqlstate::INVALID_PARAMETER_VALUE),
        (27, sqlstate::TOO_MANY_CONNECTIONS),
        (28, sqlstate::SERIALIZATION_FAILURE),
        (29, sqlstate::SYNTAX_ERROR),
        (30, sqlstate::SYNTAX_ERROR),
        (31, sqlstate::SYNTAX_ERROR),
        (32, sqlstate::ACTIVE_SQL_TRANSACTION),
        (34, sqlstate::QUERY_CANCELED),
        (44, sqlstate::QUOTA_OVERCOMMIT),
        (62, sqlstate::SYNTAX_ERROR),
        (63, sqlstate::SYNTAX_ERROR),
        (87, sqlstate::SYNTAX_ERROR),
        (88, sqlstate::DEPENDENT_OBJECTS_STILL_EXIST),
        (90, sqlstate::ACTIVE_SQL_TRANSACTION),
        (91, sqlstate::SYNTAX_ERROR),
        (92, sqlstate::SYNTAX_ERROR),
        (93, sqlstate::SYNTAX_ERROR),
        (95, sqlstate::SYNTAX_ERROR),
        (96, sqlstate::SYNTAX_ERROR),
        (97, sqlstate::SYNTAX_ERROR),
        (98, sqlstate::SYNTAX_ERROR),
        (99, sqlstate::SYNTAX_ERROR),
        (100, sqlstate::SYNTAX_ERROR),
        (101, sqlstate::QUOTA_EXCEEDED),
        (102, sqlstate::QUOTA_EXCEEDED),
        (103, sqlstate::SYNTAX_ERROR),
        (104, sqlstate::SYNTAX_ERROR),
        (106, sqlstate::READ_ONLY_SQL_TRANSACTION),
        (107, sqlstate::STALE_READ_NOT_LEADER),
    ]
}

/// A client-facing Control-Plane variant renders its own SQLSTATE, never the
/// internal-error default.
#[test]
fn client_facing_variants_render_their_own_sqlstate() {
    let expected = classified_sqlstates();
    let mut seen = 0;
    for err in error_samples() {
        let index = error_variant_index(&err);
        if let Some((_, state)) = expected.iter().find(|(i, _)| *i == index) {
            assert_eq!(error_to_sqlstate(&err).1, *state, "{err:?}");
            seen += 1;
        }
    }
    assert_eq!(seen, expected.len(), "a pinned variant has no sample");
}

/// A variant with a dedicated public code renders that code's class on the
/// numeric table too, so native and remote renderings agree with pgwire.
#[test]
fn dedicated_codes_render_the_class_of_their_variant() {
    use nodedb_types::error::ErrorCode as Ec;

    assert_eq!(
        numeric_code_to_sqlstate(Ec::QUOTA_OVERCOMMIT),
        sqlstate::QUOTA_OVERCOMMIT
    );
    assert_eq!(
        numeric_code_to_sqlstate(Ec::TENANT_VECTOR_DIM_EXCEEDED),
        sqlstate::QUOTA_EXCEEDED
    );
    assert_eq!(
        numeric_code_to_sqlstate(Ec::TENANT_GRAPH_DEPTH_EXCEEDED),
        sqlstate::QUOTA_EXCEEDED
    );
    assert_eq!(
        numeric_code_to_sqlstate(Ec::MIRROR_READ_ONLY),
        sqlstate::READ_ONLY_SQL_TRANSACTION
    );
    assert_eq!(
        numeric_code_to_sqlstate(Ec::STALE_READ_NOT_LEADER),
        sqlstate::STALE_READ_NOT_LEADER
    );
}
