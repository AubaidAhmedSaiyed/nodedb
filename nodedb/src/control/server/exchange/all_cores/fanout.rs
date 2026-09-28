// SPDX-License-Identifier: BUSL-1.1

//! Shared per-core fan-out primitive for graph BSP/WCC superstep plans and
//! single-blob Meta ops (tenant snapshot, restore result). Used by every
//! single-blob merge path (`dispatch::single_blob_gather`, `snapshot`, `bsp`,
//! `wcc`).

use futures::future::join_all;

use crate::bridge::envelope::{Response, Status};
use crate::control::server::exchange::gather::eager_dispatch_to_all_cores;
use crate::control::server::shared::session::statement_deadline;
use crate::control::state::SharedState;
use crate::types::{DatabaseId, TenantId, TraceId, TxnId};
use nodedb_physical::physical_plan::{GraphOp, PhysicalPlan};

/// One core's outcome: its response, `None` for a `NotFound` refusal, or the
/// typed error that stopped it.
type CoreOutcome = crate::Result<Option<Response>>;

/// Shared per-core fan for a BSP/WCC superstep plan: dispatch to every local
/// core, gather bounded responses, drop `NotFound`/empty-CSR cores.
///
/// A core that fails is dropped while any other core answers. The call fails
/// only when no core answers.
pub(super) async fn gather_graph_op_all_cores(
    state: &SharedState,
    tenant_id: TenantId,
    database_id: DatabaseId,
    plan: PhysicalPlan,
    trace_id: TraceId,
    txn_id: Option<TxnId>,
    label: &'static str,
) -> crate::Result<Vec<Response>> {
    let outcomes =
        dispatch_all_cores(state, tenant_id, database_id, plan, trace_id, txn_id, label).await?;
    let mut out = Vec::with_capacity(outcomes.len());
    // First error seen across cores, kept as a TYPED error: a core cut short by
    // the statement's deadline reports the deadline, and a constraint refusal
    // keeps its own SQLSTATE.
    let mut first_error: Option<crate::Error> = None;
    for outcome in outcomes {
        match outcome {
            Ok(Some(resp)) => out.push(resp),
            Ok(None) => {}
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    if out.is_empty()
        && let Some(error) = first_error
    {
        return Err(error);
    }
    Ok(out)
}

/// Fan `plan` to every local core and require every core to answer.
///
/// Each core holds only the state its own vShards home to. A merge that
/// dropped a failed core returns that core's state as absent, so the first
/// core error fails the whole call. A `NotFound` refusal contributes nothing.
pub(super) async fn gather_every_core(
    state: &SharedState,
    tenant_id: TenantId,
    database_id: DatabaseId,
    plan: PhysicalPlan,
    trace_id: TraceId,
    label: &'static str,
) -> crate::Result<Vec<Response>> {
    let outcomes =
        dispatch_all_cores(state, tenant_id, database_id, plan, trace_id, None, label).await?;
    let mut out = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        if let Some(resp) = outcome? {
            out.push(resp);
        }
    }
    Ok(out)
}

/// Dispatch `plan` to every local core and collect each core's outcome.
///
/// Must scope `owned_vshards` to `vshard % num_cores == core_id`, or every core
/// claims sibling-homed nodes in its local CSR, duplicating them in the merge.
async fn dispatch_all_cores(
    state: &SharedState,
    tenant_id: TenantId,
    database_id: DatabaseId,
    plan: PhysicalPlan,
    trace_id: TraceId,
    txn_id: Option<TxnId>,
    label: &'static str,
) -> crate::Result<Vec<CoreOutcome>> {
    // Shared broadcast call counter (parity with gather_all_cores).
    crate::control::server::broadcast::broadcast_call_count_increment();

    // The running statement's deadline — the same instant the per-core
    // envelopes carry, so the Control-Plane wait and the Data-Plane execution
    // expire together.
    let deadline = statement_deadline(state.tuning.network.default_deadline_secs);
    let max_result_bytes = state.tuning.network.max_query_result_bytes as usize;

    let num_cores = state
        .dispatcher
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .num_cores();

    // Eager dispatch: register + dispatch to each core before awaiting any response.
    // Scope owned_vshards to `vshard % num_cores == core_id` — see doc above.
    let receivers =
        eager_dispatch_to_all_cores(state, tenant_id, database_id, trace_id, txn_id, |core_id| {
            let mut core_plan = plan.clone();
            match &mut core_plan {
                PhysicalPlan::Graph(g) => match g {
                    GraphOp::BspSuperstep(bsp) => {
                        bsp.owned_vshards
                            .retain(|v| (*v as usize) % num_cores == core_id);
                    }
                    GraphOp::WccSuperstep(wcc) => {
                        wcc.owned_vshards
                            .retain(|v| (*v as usize) % num_cores == core_id);
                    }
                    // No per-core vShard set — fanned verbatim. Exhaustive (no `_ =>`) so a
                    // new superstep variant forces a scoping decision here.
                    GraphOp::Match { .. }
                    | GraphOp::MatchContinuation { .. }
                    | GraphOp::MatchVarLenResume { .. }
                    | GraphOp::EdgePut { .. }
                    | GraphOp::EdgePutBatch { .. }
                    | GraphOp::EdgeDelete { .. }
                    | GraphOp::EdgeDeleteBatch { .. }
                    | GraphOp::ResolveEdgeDelete(_)
                    | GraphOp::Hop { .. }
                    | GraphOp::Neighbors { .. }
                    | GraphOp::NeighborsMulti { .. }
                    | GraphOp::Path { .. }
                    | GraphOp::Subgraph { .. }
                    | GraphOp::RagFusion { .. }
                    | GraphOp::Algo { .. }
                    | GraphOp::SetNodeLabels { .. }
                    | GraphOp::RemoveNodeLabels { .. }
                    | GraphOp::TemporalNeighbors { .. }
                    | GraphOp::TemporalAlgorithm { .. }
                    | GraphOp::Stats { .. } => {}
                },
                // Non-graph plans fanned verbatim. Exhaustive (no `_ =>`) to force a decision.
                PhysicalPlan::Vector(_)
                | PhysicalPlan::Document(_)
                | PhysicalPlan::Kv(_)
                | PhysicalPlan::Text(_)
                | PhysicalPlan::Columnar(_)
                | PhysicalPlan::Timeseries(_)
                | PhysicalPlan::Spatial(_)
                | PhysicalPlan::Crdt(_)
                | PhysicalPlan::Query(_)
                | PhysicalPlan::Meta(_)
                | PhysicalPlan::Array(_)
                | PhysicalPlan::ClusterArray(_)
                | PhysicalPlan::ClusterEvent(_) => {}
            }
            core_plan
        })?;

    let response_futures = receivers
        .into_iter()
        .map(|(core_id, request_id, mut rx)| async move {
            let context = format!("{label} gather on core {core_id}");
            crate::control::local_dispatch::collect_under_deadline(
                &mut rx,
                crate::control::local_dispatch::DeadlineCollect {
                    request_id,
                    deadline,
                    max_result_bytes,
                    context: &context,
                },
            )
            .await
        });

    let results: Vec<crate::Result<Response>> = join_all(response_futures).await;

    Ok(results
        .into_iter()
        .map(|result| {
            let resp = result?;
            if resp.status == Status::Error {
                // `NotFound` is an empty slice on this core, not an error.
                crate::control::local_dispatch::reject_data_plane_error(&resp)?;
                return Ok(None);
            }
            Ok(Some(resp))
        })
        .collect())
}
