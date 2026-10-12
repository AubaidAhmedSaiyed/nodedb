// SPDX-License-Identifier: Apache-2.0

//! Refusal of an advisory array request: an ack or a catch-up request.

use serde::{Deserialize, Serialize};

use super::reject::{ArrayRejectClass, ArrayRejectMsg, ArrayRejectReason};

/// The advisory request Origin refused.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
#[repr(u8)]
pub enum ArrayAdvisoryKind {
    /// An `ArrayAck` (0x95).
    Ack = 1,
    /// An `ArrayCatchupRequest` (0x97).
    CatchupRequest = 2,
}

/// Advisory refusal (server → client, 0x99).
///
/// Names no local op, so it never touches the client's push window.
/// `reason.class()` names the client action, as for `ArrayRejectMsg`.
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
pub struct ArrayAdvisoryRefusedMsg {
    /// Name of the array the request named.
    pub array: String,
    /// The refused request.
    pub request: ArrayAdvisoryKind,
    /// Structured reason for the refusal.
    pub reason: ArrayRejectReason,
    /// Human-readable detail for logs. Clients never parse it.
    pub detail: String,
}

impl ArrayAdvisoryRefusedMsg {
    /// The refusal of `request` for the reason and detail `reject` carries.
    /// The reject's HLC is dropped: an advisory request names no op.
    pub fn from_reject(request: ArrayAdvisoryKind, reject: ArrayRejectMsg) -> Self {
        Self {
            array: reject.array,
            request,
            reason: reject.reason,
            detail: reject.detail,
        }
    }

    /// The action a client takes for this refusal.
    pub fn class(&self) -> ArrayRejectClass {
        self.reason.class()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_via_msgpack() {
        let msg = ArrayAdvisoryRefusedMsg {
            array: "scores".into(),
            request: ArrayAdvisoryKind::CatchupRequest,
            reason: ArrayRejectReason::PermissionDenied,
            detail: "no read permission".into(),
        };
        let encoded = zerompk::to_msgpack_vec(&msg).expect("encode");
        let decoded: ArrayAdvisoryRefusedMsg = zerompk::from_msgpack(&encoded).expect("decode");
        assert_eq!(msg, decoded);
    }

    #[test]
    fn a_reject_becomes_a_refusal_with_its_reason_and_class() {
        let refusal = ArrayAdvisoryRefusedMsg::from_reject(
            ArrayAdvisoryKind::Ack,
            ArrayRejectMsg {
                array: "a".into(),
                op_hlc_bytes: [4; 18],
                reason: ArrayRejectReason::PermissionDenied,
                detail: "denied".into(),
            },
        );
        assert_eq!(refusal.array, "a");
        assert_eq!(refusal.request, ArrayAdvisoryKind::Ack);
        assert_eq!(refusal.reason, ArrayRejectReason::PermissionDenied);
        assert_eq!(refusal.class(), ArrayRejectClass::Permanent);
    }
}
