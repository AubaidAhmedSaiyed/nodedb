// SPDX-License-Identifier: BUSL-1.1

//! Pre-warm the QUIC peer cache after `TransportBind` so the
//! first replicated request after boot doesn't pay a cold
//! connect. Slots into the startup sequencer between
//! `TransportBind` and `WarmPeers` phases. Also registers fan-out targets
//! with the transport before an RPC.

pub mod register;
pub mod report;
pub mod warm;

pub(crate) use register::register_peers_from_topology;
pub use report::PeerWarmReport;
pub use warm::warm_known_peers;
