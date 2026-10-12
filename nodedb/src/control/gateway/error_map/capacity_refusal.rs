// SPDX-License-Identifier: BUSL-1.1

//! A dispatch capacity refusal answers the retryable `SERVER_OVERLOAD` class
//! on every client surface.
//!
//! The refusal reaches a listener in two forms: raised by this node's
//! dispatcher, or relayed from a core or a peer as a Data-Plane code. Both
//! forms answer the same class, distinct from an internal error.

use nodedb_types::error::{ErrorCode, sqlstate};

use super::gateway_map::GatewayErrorMap;
use crate::Error;

/// The refusal in each form a listener can receive.
fn capacity_refusals() -> [Error; 2] {
    [
        Error::DispatchCapacity {
            scope: crate::DispatchCapacityScope::QueueFull {
                core_id: 0,
                capacity: 4,
            },
        },
        Error::DataPlane(crate::bridge::envelope::ErrorCode::DispatchCapacity {
            reason: "tenant 1 holds 4/4 in-flight requests".into(),
        }),
    ]
}

#[test]
fn pgwire_answers_57p03() {
    for err in capacity_refusals() {
        let (state, _) = GatewayErrorMap::to_pgwire(&err);
        assert_eq!(state, sqlstate::SERVER_OVERLOAD, "{err:?}");
    }
}

#[test]
fn native_frame_carries_the_server_overload_code_and_57p03() {
    for err in capacity_refusals() {
        let (code, _) = GatewayErrorMap::to_native(&err);
        assert_eq!(code, ErrorCode::SERVER_OVERLOAD, "{err:?}");

        let frame = crate::control::server::native::dispatch::error_to_native(1, &err);
        let payload = frame.error.expect("error frames carry a payload");
        assert_eq!(payload.ndb_code, ErrorCode::SERVER_OVERLOAD.0, "{err:?}");
        assert_eq!(payload.code, sqlstate::SERVER_OVERLOAD, "{err:?}");
    }
}

#[test]
fn http_answers_503() {
    for err in capacity_refusals() {
        let (status, _) = GatewayErrorMap::to_http(&err);
        assert_eq!(status, 503, "{err:?}");
    }
}

#[test]
fn resp_answers_busy() {
    for err in capacity_refusals() {
        let reply = GatewayErrorMap::to_resp(&err);
        assert!(reply.starts_with("BUSY "), "{err:?}: {reply}");
    }
}

#[test]
fn the_class_is_not_an_unclassified_failure() {
    for err in capacity_refusals() {
        assert!(
            !crate::error_classify::is_unclassified_failure(&err),
            "{err:?}"
        );
        assert_eq!(
            crate::error_classify::classify(&err).code(),
            ErrorCode::SERVER_OVERLOAD,
            "{err:?}"
        );
    }
}
