// SPDX-License-Identifier: BUSL-1.1

//! Multi-op inbound array frames: per-op application and per-op results.
//!
//! Each op applies on its own. A reject stops nothing: array ops are
//! independent CRDT writes ordered by HLC, so a later op never depends on an
//! earlier one in the same frame. The result list names every op's fate.

use nodedb_array::sync::hlc::Hlc;
use nodedb_array::sync::op::ArrayOp;
use nodedb_array::sync::op_codec;
use nodedb_types::sync::wire::SyncProvenance;
use nodedb_types::sync::wire::array::{
    ArrayDeltaBatchMsg, ArrayOpOutcome, ArrayOpResult, ArrayOpResultsMsg, ArrayRejectMsg,
    ArrayRejectReason,
};
use tracing::warn;

use super::inbound::{OpApplied, OriginArrayInbound};
use super::reject::build_reject;

impl OriginArrayInbound {
    /// Handle a batch of delta ops from a Lite peer.
    ///
    /// Returns one result per entry of `op_payloads`, in the same order. An
    /// op that does not decode is rejected as `ShapeInvalid` with an all-zero
    /// HLC. The ops after any reject are still attempted.
    pub async fn handle_delta_batch(&self, msg: &ArrayDeltaBatchMsg) -> ArrayOpResultsMsg {
        let mut results = Vec::with_capacity(msg.op_payloads.len());
        for payload in &msg.op_payloads {
            let result = match op_codec::decode_op(payload) {
                Ok(op) => {
                    let provenance = self.batch_provenance(&msg.array);
                    self.apply_op_for_result(op, payload, provenance).await
                }
                Err(e) => {
                    warn!(array = %msg.array, error = %e, "array_inbound: batch decode failed");
                    ArrayOpResult::rejected(build_reject(
                        &msg.array,
                        Hlc::ZERO,
                        ArrayRejectReason::ShapeInvalid,
                        format!("batch decode error: {e}"),
                    ))
                }
            };
            results.push(result);
        }
        ArrayOpResultsMsg {
            array: msg.array.clone(),
            results,
        }
    }

    /// Apply one decoded op and report its fate as a result entry.
    pub(super) async fn apply_op_for_result(
        &self,
        op: ArrayOp,
        raw_op_bytes: &[u8],
        provenance: Option<SyncProvenance>,
    ) -> ArrayOpResult {
        let op_hlc_bytes = op.header.hlc.to_bytes();
        let outcome = self.apply_op(op, raw_op_bytes, provenance).await;
        op_result(op_hlc_bytes, outcome)
    }
}

/// The result entry for an op whose apply returned `outcome`.
fn op_result(op_hlc_bytes: [u8; 18], outcome: Result<OpApplied, ArrayRejectMsg>) -> ArrayOpResult {
    match outcome {
        Ok(OpApplied::Applied) => ArrayOpResult {
            op_hlc_bytes,
            outcome: ArrayOpOutcome::Applied,
        },
        Ok(OpApplied::Idempotent) => ArrayOpResult {
            op_hlc_bytes,
            outcome: ArrayOpOutcome::Duplicate,
        },
        Err(reject) => ArrayOpResult::rejected(reject),
    }
}
