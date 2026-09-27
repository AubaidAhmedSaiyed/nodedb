// SPDX-License-Identifier: BUSL-1.1

//! Numeric-`ErrorCode` fallbacks shared by the HTTP and RESP surfaces.
//!
//! A remote peer on a newer build can mint a code this build does not know,
//! so both helpers degrade to the generic shape rather than misclassifying.

/// Map a numeric `ErrorCode` to an HTTP status, mirroring the local variant
/// arms in `to_http` for the same condition (e.g. `CONSTRAINT_VIOLATION`
/// mirrors `RejectedConstraint`'s 409).
///
/// Serves a `RemoteTyped` error and every Data-Plane verdict. Each code a
/// Data-Plane code classifies to has its own status here, so a typed
/// refusal never reads as a server fault.
pub(super) fn remote_code_to_http_status(code: nodedb_types::error::ErrorCode) -> u16 {
    use nodedb_types::error::ErrorCode as Ec;
    match code {
        Ec::NOT_LEADER
        | Ec::NO_LEADER
        | Ec::SERVER_OVERLOAD
        | Ec::MEMORY_EXHAUSTED
        | Ec::COLLECTION_DRAINING => 503,
        Ec::DEADLINE_EXCEEDED => 504,
        Ec::COLLECTION_NOT_FOUND | Ec::DOCUMENT_NOT_FOUND => 404,
        Ec::AUTHORIZATION_DENIED => 403,
        Ec::AUTH_EXPIRED => 401,
        Ec::BAD_REQUEST
        | Ec::PLAN_ERROR
        | Ec::TYPE_MISMATCH
        | Ec::OVERFLOW
        | Ec::DATA_EXCEPTION
        | Ec::DIVISION_BY_ZERO
        | Ec::UNDEFINED_COLUMN
        | Ec::UNDEFINED_FUNCTION
        | Ec::FAN_OUT_EXCEEDED
        | Ec::PROGRAM_LIMIT_EXCEEDED => 400,
        Ec::CONSTRAINT_VIOLATION
        | Ec::WRITE_CONFLICT
        | Ec::PREVALIDATION_REJECTED
        | Ec::APPEND_ONLY_VIOLATION
        | Ec::BALANCE_VIOLATION
        | Ec::PERIOD_LOCKED
        | Ec::PERIOD_LOCK_MISCONFIGURED
        | Ec::STATE_TRANSITION_VIOLATION
        | Ec::TRANSITION_CHECK_VIOLATION
        | Ec::TYPE_GUARD_VIOLATION
        | Ec::RETENTION_VIOLATION
        | Ec::LEGAL_HOLD_ACTIVE
        | Ec::INSUFFICIENT_BALANCE => 409,
        Ec::RATE_EXCEEDED => 429,
        Ec::SQL_NOT_ENABLED => 501,
        _ => 500,
    }
}

/// Map a numeric `ErrorCode` from a `RemoteTyped` error to a RESP error
/// prefix, mirroring the local variant arms in `to_resp`.
pub(super) fn remote_code_to_resp_prefix(code: nodedb_types::error::ErrorCode) -> &'static str {
    use nodedb_types::error::ErrorCode as Ec;
    match code {
        Ec::DEADLINE_EXCEEDED => "TIMEOUT",
        Ec::COLLECTION_NOT_FOUND => "NOTFOUND",
        Ec::AUTHORIZATION_DENIED => "NOPERM",
        Ec::CONSTRAINT_VIOLATION => "CONSTRAINT",
        Ec::TYPE_MISMATCH => "WRONGTYPE",
        Ec::SERVER_OVERLOAD => "BUSY",
        _ => "ERR",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_http_status_maps_known_code() {
        use nodedb_types::error::ErrorCode;
        // Mirrors `RejectedConstraint`'s 409 in `to_http`.
        assert_eq!(
            remote_code_to_http_status(ErrorCode::CONSTRAINT_VIOLATION),
            409
        );
        assert_eq!(
            remote_code_to_http_status(ErrorCode::AUTHORIZATION_DENIED),
            403
        );
    }

    #[test]
    fn remote_http_status_unmapped_code_falls_back_to_500() {
        use nodedb_types::error::ErrorCode;
        // A code with no explicit arm (e.g. one a newer remote node minted
        // that this build doesn't recognize) must degrade to the generic
        // 500 fallback, not silently misreport a specific status.
        assert_eq!(remote_code_to_http_status(ErrorCode(65000)), 500);
    }

    #[test]
    fn remote_resp_prefix_maps_known_code() {
        use nodedb_types::error::ErrorCode;
        assert_eq!(
            remote_code_to_resp_prefix(ErrorCode::AUTHORIZATION_DENIED),
            "NOPERM"
        );
        assert_eq!(
            remote_code_to_resp_prefix(ErrorCode::CONSTRAINT_VIOLATION),
            "CONSTRAINT"
        );
        assert_eq!(
            remote_code_to_resp_prefix(ErrorCode::TYPE_MISMATCH),
            "WRONGTYPE"
        );
    }

    #[test]
    fn remote_resp_prefix_unmapped_code_falls_back_to_err() {
        use nodedb_types::error::ErrorCode;
        // Same degrade path as the HTTP fallback: an unrecognized remote code
        // must still surface as the generic `ERR` prefix.
        assert_eq!(remote_code_to_resp_prefix(ErrorCode(65000)), "ERR");
    }
}
