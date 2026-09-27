// SPDX-License-Identifier: BUSL-1.1

//! Wire mirror of the typed error a shard-side handler answers with.
//!
//! A `VShardEnvelope` handler that fails answers with a
//! [`VShardRefusal`](super::VShardRefusal) frame carrying this mirror. The
//! caller rebuilds the same `ClusterError`, so its retry and reroute logic
//! sees the shard's own error instead of a closed stream.

pub mod convert;
pub mod raft;
pub mod wire;

pub use raft::RaftErrorWire;
pub use wire::ShardErrorWire;
