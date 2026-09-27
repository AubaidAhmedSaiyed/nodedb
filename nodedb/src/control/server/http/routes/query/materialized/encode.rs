// SPDX-License-Identifier: BUSL-1.1

//! Maps internal errors and DDL results onto the HTTP `ApiError` surface.

use super::super::super::super::auth::ApiError;

pub(super) fn ddl_error_to_api(error: crate::control::server::shared::ddl::DdlError) -> ApiError {
    let status = if error.sqlstate == "42501" {
        axum::http::StatusCode::FORBIDDEN
    } else {
        axum::http::StatusCode::BAD_REQUEST
    };
    ApiError::Coded {
        status,
        message: error.message,
        code: error.code,
    }
}

pub(super) fn gateway_error(error: crate::Error) -> ApiError {
    let (status, msg) = crate::control::gateway::GatewayErrorMap::to_http(&error);
    ApiError::HttpStatus(status, msg)
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
            ApiError::Coded { status, message, code }
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
    }
}
