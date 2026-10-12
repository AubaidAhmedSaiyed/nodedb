// SPDX-License-Identifier: Apache-2.0

//! Array CRDT rejection wire message, reason codes, and retry classes.

use serde::{Deserialize, Serialize};

/// What the sender of a refused array op does next.
///
/// Every [`ArrayRejectReason`] maps to exactly one class through
/// [`ArrayRejectReason::class`], so a client decides without reading `detail`.
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
pub enum ArrayRejectClass {
    /// Nothing applied and nothing is wrong with the op. Keep it queued and
    /// re-send it unchanged after a short backoff, in this session.
    Transient = 1,
    /// Origin lacks the op's schema. Send `ArraySchema` for the array, then
    /// re-send the op.
    SchemaRequired = 2,
    /// The session holds no current authenticated identity. Complete the
    /// handshake or token refresh, then re-send.
    Reauthenticate = 3,
    /// The reject names this replica's catch-up cursor, not a local op. Send
    /// `ArrayCatchupRequest`. The pending queue is unchanged.
    CatchUp = 4,
    /// The op never applies. Remove it from the queue and dead-letter it.
    Permanent = 5,
}

/// Reason codes for array op rejection (Origin → Lite).
///
/// Each reason has one fixed [`ArrayRejectClass`]. Origin derives the reason
/// from the typed error, never from error text.
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
pub enum ArrayRejectReason {
    /// The op's schema HLC is newer than Origin's copy of the schema.
    SchemaTooNew = 1,
    /// The target array has no schema on Origin.
    ArrayUnknown = 2,
    /// The op payload failed decoding or shape validation.
    ShapeInvalid = 3,
    /// The cluster gave the op no verdict: capacity refusal, leader change,
    /// lost quorum, deadline, rate gate, or an unknown commit outcome.
    /// Array apply dedups by HLC, so a re-send is safe.
    EngineUnavailable = 4,
    /// This replica's catch-up cursor is below Origin's op-log GC floor.
    RetentionFloor = 5,
    /// The engine refused the op on its merits: constraint, type, fence,
    /// encoding, or another error the same bytes reproduce.
    EngineRefused = 6,
    /// The session identity lacks the permission the op needs.
    PermissionDenied = 7,
    /// The frame arrived before the session authenticated.
    Unauthenticated = 8,
    /// The session's inbound snapshot buffers are full. Other assemblies
    /// release them when they complete.
    SnapshotCapacity = 9,
    /// Origin discarded the session's partial snapshot buffers after a
    /// fault. Re-send the whole snapshot, header first.
    SnapshotBufferLost = 10,
}

impl ArrayRejectReason {
    /// Every reason, in discriminant order.
    pub const ALL: [Self; 10] = [
        Self::SchemaTooNew,
        Self::ArrayUnknown,
        Self::ShapeInvalid,
        Self::EngineUnavailable,
        Self::RetentionFloor,
        Self::EngineRefused,
        Self::PermissionDenied,
        Self::Unauthenticated,
        Self::SnapshotCapacity,
        Self::SnapshotBufferLost,
    ];

    /// The action a client takes for this reason.
    pub fn class(self) -> ArrayRejectClass {
        match self {
            Self::EngineUnavailable | Self::SnapshotCapacity | Self::SnapshotBufferLost => {
                ArrayRejectClass::Transient
            }
            Self::SchemaTooNew | Self::ArrayUnknown => ArrayRejectClass::SchemaRequired,
            Self::Unauthenticated => ArrayRejectClass::Reauthenticate,
            Self::RetentionFloor => ArrayRejectClass::CatchUp,
            Self::ShapeInvalid | Self::EngineRefused | Self::PermissionDenied => {
                ArrayRejectClass::Permanent
            }
        }
    }

    /// Convert a raw `u8` discriminant to `ArrayRejectReason`.
    ///
    /// Returns `None` for unrecognized values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::SchemaTooNew),
            2 => Some(Self::ArrayUnknown),
            3 => Some(Self::ShapeInvalid),
            4 => Some(Self::EngineUnavailable),
            5 => Some(Self::RetentionFloor),
            6 => Some(Self::EngineRefused),
            7 => Some(Self::PermissionDenied),
            8 => Some(Self::Unauthenticated),
            9 => Some(Self::SnapshotCapacity),
            10 => Some(Self::SnapshotBufferLost),
            _ => None,
        }
    }
}

/// Array reject message (server → client, 0x96).
///
/// Origin sends one per refused frame. The originating Lite uses
/// `op_hlc_bytes` to find the refused op and `reason.class()` to decide
/// whether to re-send, re-announce the schema, or drop it.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
pub struct ArrayRejectMsg {
    /// Name of the array the rejected op targeted. Empty when the frame
    /// body did not decode or the session is unauthenticated.
    pub array: String,
    /// HLC of the rejected op (18-byte layout matching
    /// `nodedb_array::sync::Hlc::to_bytes()`). All zero when the op did not
    /// decode, so the reject names no single op.
    pub op_hlc_bytes: [u8; 18],
    /// Structured reason for the rejection.
    pub reason: ArrayRejectReason,
    /// Human-readable detail for logs. Clients never parse it.
    pub detail: String,
}

impl ArrayRejectMsg {
    /// The action a client takes for this reject.
    pub fn class(&self) -> ArrayRejectClass {
        self.reason.class()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_via_msgpack() {
        let msg = ArrayRejectMsg {
            array: "leaderboard".to_string(),
            op_hlc_bytes: [0x11u8; 18],
            reason: ArrayRejectReason::EngineRefused,
            detail: "unique constraint violated".to_string(),
        };
        let encoded = zerompk::to_msgpack_vec(&msg).expect("encode");
        let decoded: ArrayRejectMsg = zerompk::from_msgpack(&encoded).expect("decode");
        assert_eq!(msg, decoded);
    }

    #[test]
    fn reject_reason_round_trips() {
        for variant in ArrayRejectReason::ALL {
            let disc = variant as u8;
            let recovered = ArrayRejectReason::from_u8(disc);
            assert_eq!(recovered, Some(variant), "failed for discriminant {disc}");
        }
    }

    #[test]
    fn reject_reason_unknown_returns_none() {
        assert_eq!(ArrayRejectReason::from_u8(0), None);
        assert_eq!(ArrayRejectReason::from_u8(11), None);
        assert_eq!(ArrayRejectReason::from_u8(255), None);
    }

    #[test]
    fn every_reason_has_its_documented_class() {
        let expected = [
            (
                ArrayRejectReason::SchemaTooNew,
                ArrayRejectClass::SchemaRequired,
            ),
            (
                ArrayRejectReason::ArrayUnknown,
                ArrayRejectClass::SchemaRequired,
            ),
            (ArrayRejectReason::ShapeInvalid, ArrayRejectClass::Permanent),
            (
                ArrayRejectReason::EngineUnavailable,
                ArrayRejectClass::Transient,
            ),
            (ArrayRejectReason::RetentionFloor, ArrayRejectClass::CatchUp),
            (
                ArrayRejectReason::EngineRefused,
                ArrayRejectClass::Permanent,
            ),
            (
                ArrayRejectReason::PermissionDenied,
                ArrayRejectClass::Permanent,
            ),
            (
                ArrayRejectReason::Unauthenticated,
                ArrayRejectClass::Reauthenticate,
            ),
            (
                ArrayRejectReason::SnapshotCapacity,
                ArrayRejectClass::Transient,
            ),
            (
                ArrayRejectReason::SnapshotBufferLost,
                ArrayRejectClass::Transient,
            ),
        ];
        assert_eq!(expected.len(), ArrayRejectReason::ALL.len());
        for (reason, class) in expected {
            assert_eq!(reason.class(), class, "{reason:?}");
        }
    }

    #[test]
    fn a_reject_reports_the_class_of_its_reason() {
        let msg = ArrayRejectMsg {
            array: "a".into(),
            op_hlc_bytes: [0; 18],
            reason: ArrayRejectReason::EngineUnavailable,
            detail: String::new(),
        };
        assert_eq!(msg.class(), ArrayRejectClass::Transient);
    }
}
