// SPDX-License-Identifier: BUSL-1.1

//! QUIC transport settings for Raft RPCs, and peer liveness.
//!
//! The QUIC idle timeout is the RPC timeout. A peer that sends nothing for
//! that long, keep-alives included, is dead: its connection closes and every
//! pending read on it fails. A live peer's keep-alives hold the connection
//! open while a slow handler runs, so a long read gets its whole budget.

use std::time::Duration;

use nodedb_types::config::tuning::ClusterTransportTuning;

use crate::error::{ClusterError, Result};

/// Smallest RPC timeout a node starts with.
///
/// The RPC timeout is also the QUIC idle timeout. A shorter one drops a live
/// peer whose keep-alives meet a short scheduling stall.
pub const MIN_RPC_TIMEOUT: Duration = Duration::from_millis(500);

/// Shortest keep-alive interval, so a short RPC timeout cannot flood a peer.
const MIN_KEEP_ALIVE: Duration = Duration::from_millis(100);

/// Keep-alives sent per idle timeout. Several must be lost in a row before
/// a live peer reads as silent.
const KEEP_ALIVES_PER_IDLE_TIMEOUT: u32 = 5;

/// The keep-alive interval for `rpc_timeout`: a fifth of it, at least 100 ms.
pub fn keep_alive_interval(rpc_timeout: Duration) -> Duration {
    (rpc_timeout / KEEP_ALIVES_PER_IDLE_TIMEOUT).max(MIN_KEEP_ALIVE)
}

/// Transport config tuned for Raft RPCs in a datacenter.
///
/// Peer liveness is the QUIC idle timeout, set to `rpc_timeout`. Keep-alives
/// hold a live connection open while a handler runs past it. A peer silent
/// for `rpc_timeout` closes the connection, and every pending read on it fails.
pub fn raft_transport_config(
    tuning: &ClusterTransportTuning,
    rpc_timeout: Duration,
) -> Result<quinn::TransportConfig> {
    let idle_timeout =
        quinn::IdleTimeout::try_from(rpc_timeout).map_err(|e| ClusterError::Config {
            detail: format!(
                "rpc timeout {}ms does not fit a QUIC idle timeout: {e}",
                rpc_timeout.as_millis()
            ),
        })?;
    let mut config = quinn::TransportConfig::default();
    // Raft RPCs use bidi streams: one per request-response pair.
    config.max_concurrent_bidi_streams(quinn::VarInt::from_u32(tuning.quic_max_bi_streams));
    // Also allow uni streams for future migration/snapshot streaming.
    config.max_concurrent_uni_streams(quinn::VarInt::from_u32(tuning.quic_max_uni_streams));
    // Datacenter tuning: generous windows, low RTT estimate.
    config.receive_window(quinn::VarInt::from_u32(tuning.quic_receive_window));
    config.send_window(u64::from(tuning.quic_send_window));
    config.stream_receive_window(quinn::VarInt::from_u32(tuning.quic_stream_receive_window));
    config.initial_rtt(Duration::from_micros(100));
    config.keep_alive_interval(Some(keep_alive_interval(rpc_timeout)));
    config.max_idle_timeout(Some(idle_timeout));
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_keep_alive_is_a_fifth_of_the_rpc_timeout() {
        assert_eq!(
            keep_alive_interval(Duration::from_secs(5)),
            Duration::from_secs(1)
        );
        assert_eq!(
            keep_alive_interval(Duration::from_millis(500)),
            Duration::from_millis(100)
        );
    }

    #[test]
    fn the_keep_alive_never_drops_below_its_floor() {
        assert_eq!(
            keep_alive_interval(Duration::from_millis(300)),
            MIN_KEEP_ALIVE
        );
    }

    #[test]
    fn the_idle_timeout_follows_the_rpc_timeout() {
        let tuning = ClusterTransportTuning::default();
        assert!(raft_transport_config(&tuning, Duration::from_millis(300)).is_ok());
        assert!(raft_transport_config(&tuning, Duration::from_secs(5)).is_ok());
    }

    #[test]
    fn an_idle_timeout_past_the_quic_range_is_refused() {
        let tuning = ClusterTransportTuning::default();
        assert!(matches!(
            raft_transport_config(&tuning, Duration::MAX),
            Err(ClusterError::Config { .. })
        ));
    }
}
