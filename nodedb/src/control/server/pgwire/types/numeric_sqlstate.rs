// SPDX-License-Identifier: BUSL-1.1

//! Public numeric `ErrorCode` to PostgreSQL SQLSTATE mapping.

use nodedb_types::error::sqlstate;

/// Map a numeric `ErrorCode` received from a remote node back to a SQLSTATE.
/// Local errors map by variant identity in `error_to_sqlstate`. A remote
/// error arrives as a bare numeric code, so this recovers the class. Each
/// bucket mirrors the SQLSTATE the local variant arm chooses for the same
/// numeric code, so a constraint violation (say) maps to the same SQLSTATE
/// whether it happened locally or on a remote node. `ErrorCode` is an open
/// numeric newtype, so an unmapped or unknown code renders `INTERNAL_ERROR`.
pub(crate) fn numeric_code_to_sqlstate(code: nodedb_types::error::ErrorCode) -> &'static str {
    use nodedb_types::error::ErrorCode as Ec;
    match code {
        // Mirrors the `RejectedConstraint` arm.
        Ec::CONSTRAINT_VIOLATION => sqlstate::UNIQUE_VIOLATION,
        // Mirrors the `ConflictRetry` / `CalvinSerializationConflict` /
        // `SourceFrozen` / `RetryableSchemaChanged` arms, and `OllpExhausted`
        // when it exhausted on drift.
        Ec::WRITE_CONFLICT => sqlstate::SERIALIZATION_FAILURE,
        // Mirrors the `DeadlineExceeded` arm.
        Ec::DEADLINE_EXCEEDED => sqlstate::QUERY_CANCELED,
        // Mirrors the `CollectionNotFound` / `CollectionDeactivated` arms.
        Ec::COLLECTION_NOT_FOUND | Ec::COLLECTION_DEACTIVATED => sqlstate::UNDEFINED_TABLE,
        // Mirrors the `DocumentNotFound` arm.
        Ec::DOCUMENT_NOT_FOUND => sqlstate::NO_DATA,
        // Mirrors the `BadRequest` / `PlanError` arms.
        Ec::BAD_REQUEST | Ec::PLAN_ERROR => sqlstate::SYNTAX_ERROR,
        // Mirrors the `UndefinedFunction` arm.
        Ec::UNDEFINED_FUNCTION => sqlstate::UNDEFINED_FUNCTION,
        // Mirrors the `UndefinedObject` arm.
        Ec::UNDEFINED_OBJECT => sqlstate::UNDEFINED_OBJECT,
        // Mirrors the `ObjectNotInPrerequisiteState` arm.
        Ec::OBJECT_NOT_READY => sqlstate::OBJECT_NOT_IN_PREREQUISITE_STATE,
        // Mirrors the `UndefinedColumn` arm.
        Ec::UNDEFINED_COLUMN => sqlstate::UNDEFINED_COLUMN,
        // Mirrors the `AmbiguousColumn` arm.
        Ec::AMBIGUOUS_COLUMN => sqlstate::AMBIGUOUS_COLUMN,
        // Mirrors the `DivisionByZero` arm.
        Ec::DIVISION_BY_ZERO => sqlstate::DIVISION_BY_ZERO,
        // Mirrors the `DataException` arm.
        Ec::DATA_EXCEPTION => sqlstate::DATA_EXCEPTION,
        // Mirrors the `InvalidLimitValue` arm.
        Ec::INVALID_LIMIT_VALUE => sqlstate::INVALID_LIMIT_VALUE,
        // Mirrors the `FanOutExceeded` arm.
        Ec::FAN_OUT_EXCEEDED => sqlstate::STATEMENT_TOO_COMPLEX,
        // Mirrors the `RejectedAuthz` arm.
        Ec::AUTHORIZATION_DENIED => sqlstate::INSUFFICIENT_PRIVILEGE,
        // Mirrors the `SessionTokenExpired` arm.
        Ec::AUTH_EXPIRED => sqlstate::INVALID_AUTHORIZATION,
        // Mirrors the `RateExceeded` arm.
        Ec::RATE_EXCEEDED => sqlstate::TOO_MANY_CONNECTIONS,
        // Mirrors the `MemoryExhausted` / `Backpressure` arms.
        Ec::MEMORY_EXHAUSTED => sqlstate::OUT_OF_MEMORY,
        // Mirrors the `DispatchCapacity` arm.
        Ec::SERVER_OVERLOAD => sqlstate::SERVER_OVERLOAD,
        // Mirrors the `NoLeader` arm.
        Ec::NO_LEADER => sqlstate::LOCK_NOT_AVAILABLE,
        // Mirrors the `NotLeader` arm.
        Ec::NOT_LEADER => sqlstate::DATABASE_DROPPED,
        // Mirrors the `CloneWriteRequiresMaterialize` arm.
        Ec::CLONE_WRITE_REQUIRES_MATERIALIZE => sqlstate::CLONE_WRITE_REQUIRES_MATERIALIZE.0,
        // Mirrors the `BackupTenantMismatch` arm.
        Ec::BACKUP_TENANT_MISMATCH => sqlstate::BACKUP_TENANT_MISMATCH,
        // Mirrors the `BackupKeyMismatch` arm.
        Ec::BACKUP_KEY_MISMATCH => sqlstate::BACKUP_KEY_MISMATCH,
        // Mirrors the `QuotaOvercommit` arm.
        Ec::QUOTA_OVERCOMMIT => sqlstate::QUOTA_OVERCOMMIT,
        // Mirrors the `TenantVectorDimExceeded` / `TenantGraphDepthExceeded`
        // arms.
        Ec::TENANT_VECTOR_DIM_EXCEEDED | Ec::TENANT_GRAPH_DEPTH_EXCEEDED => {
            sqlstate::QUOTA_EXCEEDED
        }
        // Mirrors the `MirrorReadOnly` arm.
        Ec::MIRROR_READ_ONLY => sqlstate::READ_ONLY_SQL_TRANSACTION,
        // Mirrors the `StaleReadNotLeader` arm.
        Ec::STALE_READ_NOT_LEADER => sqlstate::STALE_READ_NOT_LEADER,
        // The codes below mirror the Data-Plane code table
        // (`error_code_to_sqlstate`) for the public code each Data-Plane code
        // classifies to, so a verdict that crossed a node as a numeric code
        // renders in the class it has locally.
        Ec::PREVALIDATION_REJECTED | Ec::INSUFFICIENT_BALANCE => sqlstate::CHECK_VIOLATION,
        Ec::APPEND_ONLY_VIOLATION => sqlstate::APPEND_ONLY_VIOLATION,
        Ec::BALANCE_VIOLATION => sqlstate::BALANCE_VIOLATION,
        Ec::PERIOD_LOCKED => sqlstate::PERIOD_LOCKED,
        Ec::PERIOD_LOCK_MISCONFIGURED => sqlstate::PERIOD_LOCK_MISCONFIGURED,
        Ec::STATE_TRANSITION_VIOLATION => sqlstate::STATE_TRANSITION_VIOLATION,
        Ec::TRANSITION_CHECK_VIOLATION => sqlstate::TRANSITION_CHECK_VIOLATION,
        Ec::RETENTION_VIOLATION => sqlstate::RETENTION_VIOLATION,
        Ec::LEGAL_HOLD_ACTIVE => sqlstate::LEGAL_HOLD_ACTIVE,
        Ec::TYPE_GUARD_VIOLATION => sqlstate::TYPE_GUARD_VIOLATION,
        Ec::TYPE_MISMATCH => sqlstate::CANNOT_COERCE,
        Ec::OVERFLOW => sqlstate::NUMERIC_VALUE_OUT_OF_RANGE,
        Ec::COLLECTION_DRAINING => sqlstate::CANNOT_CONNECT_NOW,
        Ec::SQL_NOT_ENABLED => sqlstate::FEATURE_NOT_SUPPORTED,
        Ec::PROGRAM_LIMIT_EXCEEDED => sqlstate::PROGRAM_LIMIT_EXCEEDED,
        _ => sqlstate::INTERNAL_ERROR,
    }
}
