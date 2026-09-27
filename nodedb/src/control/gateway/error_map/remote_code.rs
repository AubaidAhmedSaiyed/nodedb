// SPDX-License-Identifier: BUSL-1.1

//! Numeric-`ErrorCode` fallback for the RESP surface.
//!
//! A remote peer on a newer build can mint a code this build does not know,
//! so the helper degrades to the generic shape rather than misclassifying.
//! HTTP needs no helper: `to_http` renders a remote code through its SQLSTATE.

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
        // An unrecognized remote code surfaces as the generic `ERR` prefix.
        assert_eq!(remote_code_to_resp_prefix(ErrorCode(65000)), "ERR");
    }
}
