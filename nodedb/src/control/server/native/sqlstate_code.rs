// SPDX-License-Identifier: BUSL-1.1

//! Native error frames authored as a bare SQLSTATE.
//!
//! A native error frame carries both a SQLSTATE and the stable numeric NodeDB
//! code, and the client rebuilds its typed error from the number: a frame that
//! ships `ndb_code == 0` collapses on arrival into a generic internal failure,
//! so `is_not_found()`, `is_auth_denied()` and `is_retriable()` all answer
//! wrongly for it.
//!
//! Most frames get their number from [`crate::error_classify::classify`],
//! which is the one internal-`Error`-to-public mapping the crate owns. This
//! module serves the frames that never held an `Error` to classify: the
//! session and dispatch guards that reject a request with a literal SQLSTATE
//! and a static message. For those the SQLSTATE *is* the only classification
//! the server produced, so the number comes from the one SQLSTATE-to-code
//! table, [`code_for_sqlstate`], which DDL refusals read too. A bare `0A000`
//! therefore carries the same feature-not-supported class on native as on
//! pgwire.
//!
//! A SQLSTATE that more than one code shares (`0A000`, `55006`, `57014`,
//! `28000`, `XX000`, `02000` in their special meanings) has a typed constant
//! a `&str` parameter rejects, so a site that means one of those special
//! codes builds its frame from the code, not from this table.

use nodedb_types::protocol::NativeResponse;

use crate::control::server::shared::ddl::result::code_for_sqlstate;

/// Build a native error frame from a bare SQLSTATE, classifying it through
/// [`code_for_sqlstate`].
///
/// Use this wherever a site rejects a request with a literal SQLSTATE and no
/// `Error` value. A site that holds an `Error` must use
/// `error_to_native` / `error_to_native_with_sqlstate` instead: those read the
/// classification the error already carries rather than inferring one.
pub(crate) fn sqlstate_error(
    seq: u64,
    sqlstate_str: impl Into<String>,
    message: impl Into<String>,
) -> NativeResponse {
    let sqlstate_str = sqlstate_str.into();
    let ndb_code = code_for_sqlstate(&sqlstate_str).0;
    NativeResponse::error_with_code(seq, sqlstate_str, message, ndb_code)
}

#[cfg(test)]
mod tests {
    use nodedb_types::error::ErrorCode;

    use super::*;

    fn frame_code(sqlstate: &str) -> u16 {
        sqlstate_error(1, sqlstate, "refused")
            .error
            .expect("error frames carry a payload")
            .ndb_code
    }

    #[test]
    fn classified_sqlstates_carry_their_code() {
        assert_eq!(frame_code("42P01"), ErrorCode::COLLECTION_NOT_FOUND.0);
        assert_eq!(frame_code("42501"), ErrorCode::AUTHORIZATION_DENIED.0);
        assert_eq!(frame_code("42601"), ErrorCode::BAD_REQUEST.0);
        assert_eq!(frame_code("3D000"), ErrorCode::DATABASE_NOT_FOUND.0);
        assert_eq!(frame_code("XX000"), ErrorCode::INTERNAL.0);
    }

    /// A retry loop reads the numeric code, so the SQLSTATE the server sends
    /// precisely to get a transaction retried must not arrive unclassified.
    #[test]
    fn serialization_failure_stays_retriable() {
        let frame = sqlstate_error(1, "40001", "OCC abort");
        let payload = frame.error.expect("error frames carry a payload");
        assert_eq!(payload.ndb_code, ErrorCode::WRITE_CONFLICT.0);
        assert!(
            nodedb_types::NodeDbError::from_wire(ErrorCode(payload.ndb_code), payload.message)
                .is_retriable()
        );
    }

    /// A bare `0A000` guard ("opcode not supported") carries the same class a
    /// DDL `0A000` refusal does, not the internal class.
    #[test]
    fn feature_not_supported_matches_the_ddl_class() {
        assert_eq!(frame_code("0A000"), ErrorCode::SQL_NOT_ENABLED.0);
        assert_eq!(
            frame_code("0A000"),
            crate::control::server::shared::ddl::DdlError::new("0A000", "x")
                .code
                .0
        );
    }

    /// Credential failures stay undistinguished: every one gets the same
    /// code, so a caller cannot tell a wrong password from an unknown user.
    #[test]
    fn credential_failures_share_one_code() {
        assert_eq!(frame_code("28P01"), frame_code("28000"));
        assert!(
            !nodedb_types::NodeDbError::from_wire(ErrorCode(frame_code("28P01")), "x")
                .is_retriable()
        );
    }

    /// `57014` is sent both for a deadline and for a cancellation that is not
    /// one, so it must not classify as the retriable deadline class.
    #[test]
    fn query_canceled_is_not_retriable() {
        assert_ne!(frame_code("57014"), ErrorCode::DEADLINE_EXCEEDED.0);
    }

    /// The frame keeps the SQLSTATE and message the site chose.
    #[test]
    fn frame_keeps_sqlstate_and_message() {
        let frame = sqlstate_error(7, "42P07", "table 'repro_t' already exists");
        let payload = frame.error.expect("error frames carry a payload");
        assert_eq!(payload.code, "42P07");
        assert_eq!(payload.message, "table 'repro_t' already exists");
        assert_eq!(payload.ndb_code, ErrorCode::ALREADY_EXISTS.0);
    }
}
