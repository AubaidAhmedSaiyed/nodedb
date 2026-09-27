// SPDX-License-Identifier: BUSL-1.1

//! VShardEnvelope RPC glue.
//!
//! The VShardEnvelope carries graph BSP, timeseries scatter-gather, migration,
//! retention, and archival messages. The inner VShardMessageType determines
//! the handler. The envelope bytes are passed through raw (already serialized
//! in their own binary format).
//!
//! A handler that refuses with a typed Data-Plane verdict answers with a
//! [`VShardRefusal`] frame instead of a response envelope.

use super::data_plane_error::DataPlaneErrorCode;
use super::discriminants::{RPC_VSHARD_ENVELOPE, RPC_VSHARD_REFUSAL};
use super::header::write_frame;
use super::raft_rpc::RaftRpc;
use crate::error::{ClusterError, Result};

/// A typed Data-Plane verdict that answers a VShardEnvelope request.
#[derive(Debug, Clone, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct VShardRefusal {
    pub code: DataPlaneErrorCode,
}

pub(super) fn encode_vshard_envelope(bytes: &[u8], out: &mut Vec<u8>) -> Result<()> {
    write_frame(RPC_VSHARD_ENVELOPE, bytes, out)
}

pub(super) fn decode_vshard_envelope(payload: &[u8]) -> Result<RaftRpc> {
    // VShardEnvelope is already in its own binary format — pass through raw.
    Ok(RaftRpc::VShardEnvelope(payload.to_vec()))
}

pub(super) fn encode_vshard_refusal(msg: &VShardRefusal, out: &mut Vec<u8>) -> Result<()> {
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(msg).map_err(|e| ClusterError::Codec {
        detail: format!("rkyv serialize VShardRefusal: {e}"),
    })?;
    write_frame(RPC_VSHARD_REFUSAL, &bytes, out)
}

pub(super) fn decode_vshard_refusal(payload: &[u8]) -> Result<RaftRpc> {
    let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(payload.len());
    aligned.extend_from_slice(payload);
    let refusal =
        rkyv::from_bytes::<VShardRefusal, rkyv::rancor::Error>(&aligned).map_err(|e| {
            ClusterError::Codec {
                detail: format!("rkyv deserialize VShardRefusal: {e}"),
            }
        })?;
    Ok(RaftRpc::VShardRefusal(refusal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster_epoch::ClusterEpochState;
    use crate::rpc_codec::{decode, encode};

    #[test]
    fn a_refusal_survives_the_wire() {
        let code = DataPlaneErrorCode::Unsupported {
            detail: "not on this engine".into(),
        };
        let epoch = ClusterEpochState::default();
        let rpc = RaftRpc::VShardRefusal(VShardRefusal { code: code.clone() });
        let encoded = encode(&rpc, &epoch).expect("encode");
        match decode(&encoded, &epoch).expect("decode") {
            RaftRpc::VShardRefusal(refusal) => assert_eq!(refusal.code, code),
            other => panic!("decoded the wrong variant: {other:?}"),
        }
    }
}
