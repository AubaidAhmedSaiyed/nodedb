// SPDX-License-Identifier: BUSL-1.1

//! The HTTP gate that holds every non-probe route until the node serves.
//!
//! The HTTP listener serves from early in boot so orchestrator probes can
//! watch startup. Until the startup gate reports [`HealthState::Ok`], only
//! these routes answer:
//!
//! - `/healthz`, which reports `starting` with `503`;
//! - `/health/*` (liveness, readiness, drain);
//! - `/metrics`.
//!
//! Every other route, an unmatched path included, returns `503` with a
//! `starting` body. The one readiness signal is the startup phase, read
//! through [`observe`]: the node is ready at [`StartupPhase::Serving`], which
//! boot enters where it opens the client protocols.
//!
//! [`StartupPhase::Serving`]: crate::control::startup::StartupPhase::Serving

use axum::extract::State;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::control::startup::health::{HealthState, observe, to_http_response};

use super::auth::AppState;

/// The error text of a route refused while the node boots.
pub const NODE_STARTING: &str = "node is starting";

/// Whether `path` is a probe or metrics route, which answers during boot.
fn answers_during_boot(path: &str) -> bool {
    path == "/healthz" || path.starts_with("/health/") || path == "/metrics"
}

/// Refuse every non-probe route until the node reaches the final phase.
pub(super) async fn startup_gate_middleware(
    State(app_state): State<AppState>,
    req: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Response {
    if answers_during_boot(req.uri().path()) {
        return next.run(req).await;
    }
    let health = observe(&app_state.shared.startup);
    if matches!(health, HealthState::Ok) {
        return next.run(req).await;
    }
    let (status, mut body) = to_http_response(&health);
    if matches!(health, HealthState::Starting { .. }) {
        body["error"] = json!(NODE_STARTING);
    }
    (status, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use tower::ServiceExt;

    use super::*;
    use crate::bridge::dispatch::Dispatcher;
    use crate::config::auth::AuthMode;
    use crate::control::startup::{ReadyGate, StartupPhase, StartupSequencer};
    use crate::control::state::SharedState;
    use crate::wal::WalManager;

    /// A router over a state whose boot reached `GatewayEnable` but not
    /// `Serving`, as between `await_cluster_ready` and the opening of the
    /// client protocols. Firing the returned gate enters `Serving`.
    fn booting_router(dir: &tempfile::TempDir) -> (axum::Router, StartupSequencer, ReadyGate) {
        let wal =
            Arc::new(WalManager::open_for_testing(&dir.path().join("gate.wal")).expect("open WAL"));
        let (dispatcher, _data_sides) = Dispatcher::new(1, 64);
        let mut shared = SharedState::new(dispatcher, wal).expect("shared state");
        let (sequencer, gate) = StartupSequencer::new();
        let serving = sequencer.register_gate(StartupPhase::Serving, "test-serving");
        sequencer
            .register_gate(StartupPhase::GatewayEnable, "test-gateway")
            .fire();
        assert_eq!(gate.current_phase(), StartupPhase::GatewayEnable);
        Arc::get_mut(&mut shared)
            .expect("state is uniquely owned here")
            .startup = Arc::clone(&gate);
        let state = AppState {
            shutdown_bus: crate::control::shutdown::ShutdownBus::new(Arc::clone(&shared.shutdown))
                .0,
            query_ctx: Arc::new(crate::control::planner::context::QueryContext::for_state(
                &shared,
            )),
            shared,
            auth_mode: AuthMode::Trust,
        };
        let router = super::super::server::build_router(state).layer(axum::Extension(
            crate::control::security::tls_policy::TransportSecurity::Cleartext,
        ));
        (router, sequencer, serving)
    }

    async fn get(router: &axum::Router, path: &str) -> (StatusCode, String) {
        let req = Request::builder()
            .method(Method::GET)
            .uri(path)
            .body(Body::empty())
            .expect("request");
        let response = router.clone().oneshot(req).await.expect("response");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn healthz_reports_starting_until_serving() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (router, _sequencer, serving) = booting_router(&dir);

        let (status, body) = get(&router, "/healthz").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = sonic_rs::from_str(&body).expect("healthz body is JSON");
        assert_eq!(body["status"], "starting");

        let (status, _) = get(&router, "/health/live").await;
        assert_eq!(status, StatusCode::OK);

        serving.fire();
        let (status, body) = get(&router, "/healthz").await;
        assert_eq!(status, StatusCode::OK, "healthz after admission: {body}");
        let body: serde_json::Value = sonic_rs::from_str(&body).expect("healthz body is JSON");
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn a_non_probe_route_is_refused_until_serving() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (router, _sequencer, serving) = booting_router(&dir);

        let (status, body) = get(&router, "/v1/status").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let parsed: serde_json::Value = sonic_rs::from_str(&body).expect("refusal body is JSON");
        assert_eq!(parsed["status"], "starting");
        assert_eq!(parsed["error"], NODE_STARTING);

        // Metrics answer during boot.
        let (_, body) = get(&router, "/metrics").await;
        assert!(!body.contains(NODE_STARTING));

        serving.fire();
        let (_, body) = get(&router, "/v1/status").await;
        assert!(!body.contains(NODE_STARTING));
    }

    /// The gate layer also wraps the fallback: an unmatched path returns 503
    /// during boot and 404 once the node serves.
    #[tokio::test]
    async fn an_unmatched_path_is_refused_until_serving_then_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (router, _sequencer, serving) = booting_router(&dir);

        let (status, body) = get(&router, "/health").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains(NODE_STARTING));

        serving.fire();
        let (status, _) = get(&router, "/health").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn probe_and_metrics_paths_answer_during_boot() {
        assert!(answers_during_boot("/healthz"));
        assert!(answers_during_boot("/health/live"));
        assert!(answers_during_boot("/health/ready"));
        assert!(answers_during_boot("/metrics"));
        assert!(!answers_during_boot("/v1/query"));
        assert!(!answers_during_boot("/healthzz"));
        assert!(!answers_during_boot("/metrics/extra"));
    }
}
