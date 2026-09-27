// SPDX-License-Identifier: BUSL-1.1

//! HTTP error shape: `(status_code, message)`.

use super::gateway_map::GatewayErrorMap;
use super::sqlstate_status::sqlstate_to_http_status;
use crate::Error;

impl GatewayErrorMap {
    /// Map a SQLSTATE into an HTTP status, for an error that reaches HTTP as
    /// a SQLSTATE, such as a DDL error. [`Self::to_http`] reads the same
    /// table, so a DDL error and a gateway error of one class answer one
    /// status.
    pub fn sqlstate_to_http(sqlstate: &str) -> u16 {
        sqlstate_to_http_status(sqlstate)
    }

    /// Map a gateway error into `(http_status_code, message)` for HTTP.
    ///
    /// The status follows the SQLSTATE pgwire renders for the error, through
    /// the one SQLSTATE status table. One error answers one class on pgwire,
    /// native and HTTP. Only an `XX000` or `58` class error reads as a 500.
    /// A Data-Plane verdict answers with its public message.
    pub fn to_http(err: &Error) -> (u16, String) {
        let (_severity, state, message) =
            crate::control::server::pgwire::types::error_to_sqlstate(err);
        let message = if let Error::DataPlane(_) = err {
            crate::error_classify::classify(err).message().to_owned()
        } else {
            message
        };
        (sqlstate_to_http_status(state), message)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{authz, deadline, internal, not_found, not_leader};
    use super::*;

    #[test]
    fn http_not_leader() {
        let (status, _) = GatewayErrorMap::to_http(&not_leader());
        assert_eq!(status, 503);
    }

    #[test]
    fn http_deadline() {
        let (status, _) = GatewayErrorMap::to_http(&deadline());
        assert_eq!(status, 504);
    }

    #[test]
    fn http_not_found() {
        let (status, _) = GatewayErrorMap::to_http(&not_found());
        assert_eq!(status, 404);
    }

    #[test]
    fn http_authz() {
        let (status, _) = GatewayErrorMap::to_http(&authz());
        assert_eq!(status, 403);
    }

    #[test]
    fn http_internal() {
        let (status, _) = GatewayErrorMap::to_http(&internal());
        assert_eq!(status, 500);
    }

    #[test]
    fn http_data_plane_not_found() {
        let err = Error::DataPlane(crate::bridge::envelope::ErrorCode::NotFound);
        assert_eq!(GatewayErrorMap::to_http(&err).0, 404);
    }

    #[test]
    fn http_conflict_retry() {
        let err = Error::ConflictRetry {
            collection: "orders".into(),
            document_id: "o1".into(),
        };
        assert_eq!(GatewayErrorMap::to_http(&err).0, 409);
    }

    #[test]
    fn http_backup_key_mismatch_is_invalid_authorization() {
        assert_eq!(GatewayErrorMap::to_http(&Error::BackupKeyMismatch).0, 401);
    }

    /// A remote rendering of an error keeps the status of the local one.
    #[test]
    fn http_backup_key_mismatch_keeps_its_status_across_nodes() {
        use nodedb_types::error::ErrorCode;
        let remote = Error::RemoteTyped {
            code: ErrorCode::BACKUP_KEY_MISMATCH,
            message: "wrong backup KEK".into(),
        };
        assert_eq!(
            GatewayErrorMap::to_http(&remote).0,
            GatewayErrorMap::to_http(&Error::BackupKeyMismatch).0
        );
    }

    #[test]
    fn to_http_remote_typed_is_wired_to_helper() {
        use nodedb_types::error::ErrorCode;
        let err = Error::RemoteTyped {
            code: ErrorCode::AUTHORIZATION_DENIED,
            message: "remote denied write".into(),
        };
        let (status, msg) = GatewayErrorMap::to_http(&err);
        assert_eq!(status, 403);
        assert_eq!(msg, "remote denied write");
    }
}
