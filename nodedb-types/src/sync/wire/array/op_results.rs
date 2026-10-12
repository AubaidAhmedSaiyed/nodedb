// SPDX-License-Identifier: Apache-2.0

//! Per-op outcomes for an array frame that carries several ops.

use serde::{Deserialize, Serialize};

use super::reject::{ArrayRejectMsg, ArrayRejectReason};

/// What Origin did with one op of a multi-op array frame.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
pub enum ArrayOpOutcome {
    /// The op committed and applied.
    Applied,
    /// Origin already held the op's HLC. Nothing changed.
    Duplicate,
    /// The op did not apply. `reason.class()` names the client action.
    Rejected {
        /// Structured reason for the rejection.
        reason: ArrayRejectReason,
        /// Human-readable detail for logs. Clients never parse it.
        detail: String,
    },
}

/// The outcome of one op, keyed by its HLC.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
pub struct ArrayOpResult {
    /// HLC of the op (18-byte layout matching
    /// `nodedb_array::sync::Hlc::to_bytes()`). All zero when the payload did
    /// not decode. The result's position in the frame still names the op.
    pub op_hlc_bytes: [u8; 18],
    /// What Origin did with the op.
    pub outcome: ArrayOpOutcome,
}

impl ArrayOpResult {
    /// The result for an op Origin refused with `reject`.
    pub fn rejected(reject: ArrayRejectMsg) -> Self {
        Self {
            op_hlc_bytes: reject.op_hlc_bytes,
            outcome: ArrayOpOutcome::Rejected {
                reason: reject.reason,
                detail: reject.detail,
            },
        }
    }
}

/// Per-op results (server → client, 0x98).
///
/// Origin sends one in answer to every `ArrayDeltaBatch`, and one when an
/// inbound snapshot completes. `results` holds one entry per op, in the
/// order the ops arrived. Each op applies on its own: a reject never stops
/// the ops after it.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
pub struct ArrayOpResultsMsg {
    /// Name of the array the frame targeted.
    pub array: String,
    /// One result per op, in arrival order.
    pub results: Vec<ArrayOpResult>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_via_msgpack() {
        let msg = ArrayOpResultsMsg {
            array: "scores".into(),
            results: vec![
                ArrayOpResult {
                    op_hlc_bytes: [1; 18],
                    outcome: ArrayOpOutcome::Applied,
                },
                ArrayOpResult {
                    op_hlc_bytes: [2; 18],
                    outcome: ArrayOpOutcome::Duplicate,
                },
                ArrayOpResult {
                    op_hlc_bytes: [0; 18],
                    outcome: ArrayOpOutcome::Rejected {
                        reason: ArrayRejectReason::ShapeInvalid,
                        detail: "decode error".into(),
                    },
                },
            ],
        };
        let encoded = zerompk::to_msgpack_vec(&msg).expect("encode");
        let decoded: ArrayOpResultsMsg = zerompk::from_msgpack(&encoded).expect("decode");
        assert_eq!(msg, decoded);
    }

    #[test]
    fn a_reject_becomes_a_result_with_its_hlc_and_reason() {
        let result = ArrayOpResult::rejected(ArrayRejectMsg {
            array: "a".into(),
            op_hlc_bytes: [9; 18],
            reason: ArrayRejectReason::EngineUnavailable,
            detail: "queue full".into(),
        });
        assert_eq!(result.op_hlc_bytes, [9; 18]);
        assert_eq!(
            result.outcome,
            ArrayOpOutcome::Rejected {
                reason: ArrayRejectReason::EngineUnavailable,
                detail: "queue full".into(),
            }
        );
    }
}
