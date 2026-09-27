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
