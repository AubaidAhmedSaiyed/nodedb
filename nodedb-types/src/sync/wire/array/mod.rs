// SPDX-License-Identifier: Apache-2.0

//! Array CRDT sync wire envelopes.
//!
//! Wire types carry opaque msgpack-encoded payloads (encoded by
//! `nodedb-array::sync::op_codec`) plus the minimum routing fields
//! (array name, HLC bytes). The wire crate does not depend on
//! `nodedb-array`; engine code on the receiving side decodes
//! payloads via `nodedb-array::sync::op_codec::decode_op` etc.

pub mod ack;
pub mod advisory_refused;
pub mod catchup_request;
pub mod delta;
pub mod delta_batch;
pub mod op_results;
pub mod reject;
pub mod schema;
pub mod snapshot;
pub mod snapshot_chunk;

pub use ack::ArrayAckMsg;
pub use advisory_refused::{ArrayAdvisoryKind, ArrayAdvisoryRefusedMsg};
pub use catchup_request::ArrayCatchupRequestMsg;
pub use delta::ArrayDeltaMsg;
pub use delta_batch::ArrayDeltaBatchMsg;
pub use op_results::{ArrayOpOutcome, ArrayOpResult, ArrayOpResultsMsg};
pub use reject::{ArrayRejectClass, ArrayRejectMsg, ArrayRejectReason};
pub use schema::ArraySchemaSyncMsg;
pub use snapshot::ArraySnapshotMsg;
pub use snapshot_chunk::ArraySnapshotChunkMsg;
