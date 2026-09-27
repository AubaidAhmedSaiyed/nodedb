// SPDX-License-Identifier: BUSL-1.1

//! Maps internal errors and DDL results onto the HTTP `ApiError` surface.

use super::super::super::super::auth::ApiError;

/// Map a DDL error to the HTTP error the client reads. The status follows the
/// SQLSTATE through the gateway status table. The code and the typed cause
/// travel in the body, as they do on native and pgwire.
pub(super) fn ddl_error_to_api(error: crate::control::server::shared::ddl::DdlError) -> ApiError {
    let status = crate::control::gateway::GatewayErrorMap::sqlstate_to_http(&error.sqlstate);
    ApiError::Coded {
        status: axum::http::StatusCode::from_u16(status)
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
        message: error.message,
        code: error.code,
        cause: error.cause,
    }
}

/// Map a gateway error to the HTTP error the client reads, through the one
/// `crate::Error` to `ApiError` conversion.
pub(super) fn gateway_error(error: crate::Error) -> ApiError {
    ApiError::from(error)
}

/// Map a Data-Plane refusal to the HTTP error the client reads. A typed
/// refusal takes the status its code maps to. Only a refusal with no code is
/// an internal error.
pub(super) fn response_error(response: &crate::bridge::envelope::Response) -> ApiError {
    match response.error_code.as_deref() {
        Some(code) => gateway_error(crate::Error::DataPlane(code.clone())),
        None => ApiError::Internal("data plane returned an error status with no error code".into()),
    }
}

#[cfg(test)]
mod tests {
    use axum::response::IntoResponse;

    use super::*;

    #[test]
    fn ddl_insufficient_privilege_maps_to_forbidden() {
        let error =
            crate::control::server::shared::ddl::DdlError::new("42501", "write permission denied");

        assert!(matches!(
            ddl_error_to_api(error),
            ApiError::Coded { status, message, code, .. }
                if status == axum::http::StatusCode::FORBIDDEN
                    && message == "write permission denied"
                    && code == nodedb_types::error::ErrorCode::AUTHORIZATION_DENIED
        ));
    }

    /// Round-trips through the actual JSON response body `into_response()`
    /// produces — not just the pre-serialization `ApiError` — so this proves
    /// the code reaches the client, not merely that the server set it.
    #[tokio::test]
    async fn ddl_error_code_survives_into_response_json() {
        let error =
            crate::control::server::shared::ddl::DdlError::new("42501", "write permission denied");

        let response = ddl_error_to_api(error).into_response();
        assert_eq!(response.status(), axum::http::StatusCode::FORBIDDEN);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON body");
        assert_eq!(
            json["code"],
            nodedb_types::error::ErrorCode::AUTHORIZATION_DENIED.to_string()
        );
        assert_eq!(json["error"], "write permission denied");
        assert!(json.get("cause").is_none(), "no cause, no cause field");
    }

    async fn response_json(error: ApiError) -> (axum::http::StatusCode, serde_json::Value) {
        let response = error.into_response();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let json = serde_json::from_slice(&body).expect("valid JSON body");
        (status, json)
    }

    /// An internal DDL error is a server fault, and its typed cause reaches
    /// the client with its own message and code.
    #[tokio::test]
    async fn internal_ddl_error_is_500_with_its_cause() {
        let cause = nodedb_types::NodeDbError::from(crate::Error::DataPlane(
            crate::bridge::envelope::ErrorCode::Unsupported {
                detail: "not on this engine".into(),
            },
        ));
        let mut error = crate::control::server::shared::ddl::DdlError::move_tenant_cutover_failed(
            "MOVE TENANT cutover failed",
        );
        error.cause = Some(Box::new(cause));

        let (status, json) = response_json(ddl_error_to_api(error)).await;
        assert_eq!(status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(json["error"], "MOVE TENANT cutover failed");
        assert_eq!(
            json["code"],
            nodedb_types::error::ErrorCode::MOVE_TENANT_CUTOVER_FAILED.to_string()
        );
        assert_eq!(
            json["cause"]["code"],
            nodedb_types::error::ErrorCode::SQL_NOT_ENABLED.to_string()
        );
        assert_eq!(json["cause"]["error"], "not on this engine");
    }

    /// An internal DDL error is a server fault, never a client error.
    #[tokio::test]
    async fn xx000_ddl_error_is_500() {
        let error = crate::control::server::shared::ddl::DdlError::internal("catalog write failed");
        let (status, _) = response_json(ddl_error_to_api(error)).await;
        assert_eq!(status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    }

    /// A feature-not-supported DDL error is 501 Not Implemented.
    #[tokio::test]
    async fn feature_not_supported_ddl_error_is_501() {
        let error = crate::control::server::shared::ddl::DdlError::new(
            "0A000",
            "changing vector index params is not supported",
        );
        let (status, json) = response_json(ddl_error_to_api(error)).await;
        assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            json["code"],
            nodedb_types::error::ErrorCode::SQL_NOT_ENABLED.to_string()
        );
    }

    /// A conflict, a constraint violation and a rate limit take their own
    /// status, never a blanket 400.
    #[test]
    fn ddl_conflicts_and_rate_limits_take_their_class_status() {
        for (state, expected) in [
            ("42P07", axum::http::StatusCode::CONFLICT),
            ("23505", axum::http::StatusCode::CONFLICT),
            ("53300", axum::http::StatusCode::TOO_MANY_REQUESTS),
            ("42P01", axum::http::StatusCode::NOT_FOUND),
        ] {
            let error = crate::control::server::shared::ddl::DdlError::new(state, "refused");
            match ddl_error_to_api(error) {
                ApiError::Coded { status, .. } => assert_eq!(status, expected, "{state}"),
                other => panic!("expected a coded error for {state}, got {other:?}"),
            }
        }
    }
}
