// SPDX-License-Identifier: BUSL-1.1

//! Native-protocol error shape: `(numeric code, message)`.

use nodedb_types::error::ErrorCode;

use super::gateway_map::GatewayErrorMap;
use crate::Error;

impl GatewayErrorMap {
    /// Map a gateway error into `(code, message)` for the native protocol.
    ///
    /// The code and message are the ones the native error frame carries,
    /// from the one native mapping the listener uses. The code is the stable
    /// `nodedb_types::error::ErrorCode`, so a native client switches on it
    /// without string matching.
    pub fn to_native(err: &Error) -> (ErrorCode, String) {
        let fields = crate::control::server::native::dispatch::native_error_fields(err);
        (fields.code, fields.message)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{authz, deadline, internal, not_found, not_leader};
    use super::*;

    #[test]
    fn native_not_leader() {
        let (code, _) = GatewayErrorMap::to_native(&not_leader());
        assert_eq!(code, ErrorCode::NOT_LEADER);
    }

    #[test]
    fn native_deadline() {
        let (code, _) = GatewayErrorMap::to_native(&deadline());
        assert_eq!(code, ErrorCode::DEADLINE_EXCEEDED);
    }

    #[test]
    fn native_not_found() {
        let (code, msg) = GatewayErrorMap::to_native(&not_found());
        assert_eq!(code, ErrorCode::COLLECTION_NOT_FOUND);
        assert!(msg.contains("missing_col"));
    }

    #[test]
    fn native_authz() {
        let (code, _) = GatewayErrorMap::to_native(&authz());
        assert_eq!(code, ErrorCode::AUTHORIZATION_DENIED);
    }

    #[test]
    fn native_internal() {
        let (code, _) = GatewayErrorMap::to_native(&internal());
        assert_eq!(code, ErrorCode::INTERNAL);
    }

    /// The gateway map and the native listener read one mapping, so the code
    /// a gateway caller sees is the code the wire frame carries.
    #[test]
    fn gateway_map_matches_the_wire_frame() {
        let err = Error::DataPlane(crate::bridge::envelope::ErrorCode::Unsupported {
            detail: "not here".into(),
        });
        let frame = crate::control::server::native::dispatch::error_to_native(1, &err);
        let payload = frame.error.expect("error frames carry a payload");
        let (code, message) = GatewayErrorMap::to_native(&err);
        assert_eq!(code.0, payload.ndb_code);
        assert_eq!(message, payload.message);
    }
}
