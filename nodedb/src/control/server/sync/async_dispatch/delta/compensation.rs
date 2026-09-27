// SPDX-License-Identifier: BUSL-1.1

//! The compensation hint a rejected CRDT delta carries to the edge.

use crate::bridge::envelope::ErrorCode;

use super::super::super::wire::CompensationHint;

/// Classify a dispatch failure into the hint the edge compensates against.
///
/// A failure that never judged the write is answered with a gap ack before
/// this runs. It still maps here to [`CompensationHint::Retry`], the class it
/// has, so no path turns it into a compensation.
pub(super) fn compensation_hint_for_dispatch_error(e: &crate::Error) -> CompensationHint {
    match e {
        crate::Error::DataPlane(code) => compensation_hint_for_code(code),
        crate::Error::RejectedConstraint {
            constraint, detail, ..
        } => CompensationHint::Custom {
            constraint: constraint.clone(),
            detail: detail.clone(),
        },
        crate::Error::RejectedPrevalidation { constraint, reason } => CompensationHint::Custom {
            constraint: constraint.clone(),
            detail: reason.clone(),
        },
        crate::Error::RejectedAuthz { .. } => CompensationHint::PermissionDenied,
        crate::Error::RateExceeded { retry_after_ms, .. } => CompensationHint::RateLimited {
            retry_after_ms: *retry_after_ms,
        },
        crate::Error::OllpExhausted { cause, .. } => match cause {
            crate::OllpExhaustedCause::PreAdmission(inner) => {
                compensation_hint_for_dispatch_error(inner)
            }
            crate::OllpExhaustedCause::PredicateDrift
            | crate::OllpExhaustedCause::AdmissionRefused { .. } => {
                CompensationHint::Retry { retry_after_ms: 0 }
            }
        },
        // Never judged the write: re-push it.
        crate::Error::DeadlineExceeded { .. }
        | crate::Error::ConflictRetry { .. }
        | crate::Error::CalvinSerializationConflict
        | crate::Error::CalvinParticipantError
        | crate::Error::RetryableRefusal { .. }
        | crate::Error::VShardAdmissionCapacityExceeded { .. }
        | crate::Error::CrdtAdmissionRetriesExhausted { .. }
        | crate::Error::CrdtAdmissionTimeout { .. }
        | crate::Error::NoLeader { .. }
        | crate::Error::NotLeader { .. }
        | crate::Error::SourceFrozen { .. }
        | crate::Error::RetryableSchemaChanged { .. }
        | crate::Error::RetryableLeaderChange { .. }
        | crate::Error::GroupQuorumUnavailable { .. }
        | crate::Error::GroupMarksUnavailable { .. }
        | crate::Error::MetadataLeaderUnavailable
        | crate::Error::AuthorizationStateBehind { .. }
        | crate::Error::DispatchCapacity { .. }
        | crate::Error::MemoryExhausted { .. }
        | crate::Error::Backpressure { .. }
        | crate::Error::SequencerUnavailable
        | crate::Error::StaleReadNotLeader { .. } => CompensationHint::Retry { retry_after_ms: 0 },
        // Refused on its merits, or a fault the same bytes reproduce.
        other @ (crate::Error::TxnOverlayMemoryExceeded { .. }
        | crate::Error::OffsetRegression { .. }
        | crate::Error::AppendOnlyViolation { .. }
        | crate::Error::BalanceViolation { .. }
        | crate::Error::MaterializedSumTargetNotFound { .. }
        | crate::Error::MaterializedSumResolutionMissing { .. }
        | crate::Error::PeriodLocked { .. }
        | crate::Error::PeriodLockMisconfigured { .. }
        | crate::Error::RetentionViolation { .. }
        | crate::Error::LegalHoldActive { .. }
        | crate::Error::StateTransitionViolation { .. }
        | crate::Error::TransitionCheckViolation { .. }
        | crate::Error::TypeGuardViolation { .. }
        | crate::Error::TypeMismatch { .. }
        | crate::Error::InsufficientBalance { .. }
        | crate::Error::CollectionNotFound { .. }
        | crate::Error::DocumentNotFound { .. }
        | crate::Error::CollectionDeactivated { .. }
        | crate::Error::CrdtAdmissionInvalidPlan { .. }
        | crate::Error::CrdtAdmissionCallerFence
        | crate::Error::CrdtApplyRequiresAdmission
        | crate::Error::CrdtApplyForbiddenInTransaction
        | crate::Error::NotInTransactionBlock { .. }
        | crate::Error::FanOutExceeded { .. }
        | crate::Error::CrossCollectionNotColocated { .. }
        | crate::Error::CloneWriteRequiresMaterialize { .. }
        | crate::Error::BadRequest { .. }
        | crate::Error::BackupTenantMismatch { .. }
        | crate::Error::BackupKeyMismatch
        | crate::Error::QuotaOvercommit { .. }
        | crate::Error::PlanError { .. }
        | crate::Error::FeatureNotSupported { .. }
        | crate::Error::UndefinedFunction { .. }
        | crate::Error::UndefinedObject { .. }
        | crate::Error::ObjectNotInPrerequisiteState { .. }
        | crate::Error::UndefinedColumn { .. }
        | crate::Error::AmbiguousColumn { .. }
        | crate::Error::UnknownStrictField { .. }
        | crate::Error::DivisionByZero
        | crate::Error::DataException { .. }
        | crate::Error::InvalidLimitValue { .. }
        | crate::Error::ExecutionLimitExceeded { .. }
        | crate::Error::LimitExceeded { .. }
        | crate::Error::Wal(_)
        | crate::Error::Dispatch { .. }
        | crate::Error::Storage { .. }
        | crate::Error::ColdStorage { .. }
        | crate::Error::Serialization { .. }
        | crate::Error::Codec { .. }
        | crate::Error::SegmentCorrupted { .. }
        | crate::Error::Crdt(_)
        | crate::Error::Io(_)
        | crate::Error::Config { .. }
        | crate::Error::Encryption { .. }
        | crate::Error::Bridge { .. }
        | crate::Error::VersionCompat { .. }
        | crate::Error::Internal { .. }
        | crate::Error::Shaping(_)
        | crate::Error::RemoteTyped { .. }
        | crate::Error::DescriptorVersionAnomaly { .. }
        | crate::Error::CollectionPurgeRowMissing { .. }
        | crate::Error::CatalogIntegrityViolation { .. }
        | crate::Error::Promql(_)
        | crate::Error::DependentObjectsExist { .. }
        | crate::Error::RoleInUse { .. }
        | crate::Error::CascadeCycle { .. }
        | crate::Error::CrossShardInExplicitTransaction
        | crate::Error::SessionCapExceeded { .. }
        | crate::Error::SessionIdleTimeout
        | crate::Error::SessionTokenExpired
        | crate::Error::SessionKilledByAdmin
        | crate::Error::SessionUserDropped
        | crate::Error::OidcProviderTenantUnbound
        | crate::Error::OidcProviderTenantUnavailable { .. }
        | crate::Error::ExternalRoleUndefined { .. }
        | crate::Error::OidcNoDefaultDatabase { .. }
        | crate::Error::TenantVectorDimExceeded { .. }
        | crate::Error::TenantGraphDepthExceeded { .. }
        | crate::Error::RoleInheritanceCycle { .. }
        | crate::Error::RoleInheritanceDepthExceeded { .. }
        | crate::Error::MirrorReadOnly { .. }) => CompensationHint::Custom {
            constraint: "apply_failed".into(),
            detail: other.to_string(),
        },
    }
}

/// [`compensation_hint_for_dispatch_error`] for a Data-Plane verdict.
fn compensation_hint_for_code(code: &ErrorCode) -> CompensationHint {
    match code {
        ErrorCode::RejectedConstraint { constraint, detail } => CompensationHint::Custom {
            constraint: constraint.clone(),
            detail: detail.clone(),
        },
        ErrorCode::RejectedPrevalidation { reason } => CompensationHint::Custom {
            constraint: "prevalidation".into(),
            detail: reason.clone(),
        },
        ErrorCode::RejectedAuthz { .. } => CompensationHint::PermissionDenied,
        ErrorCode::RateExceeded { retry_after_ms, .. } => CompensationHint::RateLimited {
            retry_after_ms: *retry_after_ms,
        },
        // Never judged the write: re-push it.
        ErrorCode::DeadlineExceeded
        | ErrorCode::ExpiredBeforeExecution
        | ErrorCode::ResourcesExhausted
        | ErrorCode::DispatchCapacity { .. }
        | ErrorCode::ConflictRetry
        | ErrorCode::OllpRetryRequired
        | ErrorCode::CrdtFrontierMismatch { .. }
        | ErrorCode::CollectionDraining { .. }
        | ErrorCode::RetryableRefusal { .. } => CompensationHint::Retry { retry_after_ms: 0 },
        other @ (ErrorCode::SyncRejected { .. }
        | ErrorCode::SyncNotApplied { .. }
        | ErrorCode::NotFound
        | ErrorCode::FanOutExceeded
        | ErrorCode::RejectedDanglingEdge { .. }
        | ErrorCode::DuplicateWrite
        | ErrorCode::AppendOnlyViolation { .. }
        | ErrorCode::BalanceViolation { .. }
        | ErrorCode::PeriodLocked { .. }
        | ErrorCode::PeriodLockMisconfigured { .. }
        | ErrorCode::RetentionViolation { .. }
        | ErrorCode::LegalHoldActive { .. }
        | ErrorCode::StateTransitionViolation { .. }
        | ErrorCode::TransitionCheckViolation { .. }
        | ErrorCode::TypeGuardViolation { .. }
        | ErrorCode::TypeMismatch { .. }
        | ErrorCode::CounterFault { .. }
        | ErrorCode::InsufficientBalance { .. }
        | ErrorCode::RecursionDepthExceeded { .. }
        | ErrorCode::UndefinedColumn { .. }
        | ErrorCode::Internal { .. }
        | ErrorCode::Unsupported { .. }
        | ErrorCode::RollbackFailed { .. }
        | ErrorCode::TxnOverlayMemoryExceeded { .. }
        | ErrorCode::DivisionByZero
        | ErrorCode::UndefinedFunction { .. }
        | ErrorCode::DataException { .. }
        | ErrorCode::BadRequest { .. }) => CompensationHint::Custom {
            constraint: "apply_failed".into(),
            detail: format!("{other:?}"),
        },
    }
}
