// SPDX-License-Identifier: BUSL-1.1

//! Process-wide count of overload-class answers that reached a client, by
//! protocol.
//!
//! Each protocol counts at the one place it turns an error into wire bytes,
//! so a capacity refusal the server waits out itself is never counted. The
//! counters are a process-wide static because those places hold no shared
//! state: the pgwire error handler, the native frame encoder, the HTTP
//! response builder and the RESP serializer. The label is the protocol only.

use std::sync::atomic::{AtomicU64, Ordering};

/// The protocol that sent an overload answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverloadProtocol {
    /// SQLSTATE `57P03` in a pgwire `ErrorResponse`.
    Pgwire,
    /// SQLSTATE `57P03` in a native error frame.
    Native,
    /// `503` with `Retry-After` for a `57P03` error.
    Http,
    /// A `BUSY` error reply.
    Resp,
}

impl OverloadProtocol {
    /// Every protocol, in exposition order.
    pub const ALL: [Self; 4] = [Self::Pgwire, Self::Native, Self::Http, Self::Resp];

    /// The Prometheus `protocol` label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Pgwire => "pgwire",
            Self::Native => "native",
            Self::Http => "http",
            Self::Resp => "resp",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Pgwire => 0,
            Self::Native => 1,
            Self::Http => 2,
            Self::Resp => 3,
        }
    }
}

/// Overload answers sent since process start, one counter per protocol.
#[derive(Debug)]
pub struct OverloadResponses {
    sent: [AtomicU64; OverloadProtocol::ALL.len()],
}

static OVERLOAD_RESPONSES: OverloadResponses = OverloadResponses::new();

/// The process-wide overload answer counters.
pub fn overload_responses() -> &'static OverloadResponses {
    &OVERLOAD_RESPONSES
}

impl OverloadResponses {
    pub const fn new() -> Self {
        Self {
            sent: [const { AtomicU64::new(0) }; OverloadProtocol::ALL.len()],
        }
    }

    /// Count one overload answer sent over `protocol`.
    pub fn record(&self, protocol: OverloadProtocol) {
        self.sent[protocol.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Overload answers sent over `protocol`.
    pub fn sent(&self, protocol: OverloadProtocol) -> u64 {
        self.sent[protocol.index()].load(Ordering::Relaxed)
    }

    /// Append the `nodedb_server_overload_responses_total` counter.
    pub fn write_prometheus(&self, out: &mut String) {
        use std::fmt::Write as _;

        let _ = out.write_str(
            "# HELP nodedb_server_overload_responses_total Overload-class answers \
             (SQLSTATE 57P03, HTTP 503, RESP BUSY) that reached a client.\n\
             # TYPE nodedb_server_overload_responses_total counter\n",
        );
        for protocol in OverloadProtocol::ALL {
            let _ = writeln!(
                out,
                r#"nodedb_server_overload_responses_total{{protocol="{}"}} {}"#,
                protocol.label(),
                self.sent(protocol)
            );
        }
    }
}

impl Default for OverloadResponses {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_protocol_counts_apart() {
        let responses = OverloadResponses::new();
        responses.record(OverloadProtocol::Http);
        responses.record(OverloadProtocol::Http);
        responses.record(OverloadProtocol::Resp);

        assert_eq!(responses.sent(OverloadProtocol::Pgwire), 0);
        assert_eq!(responses.sent(OverloadProtocol::Native), 0);
        assert_eq!(responses.sent(OverloadProtocol::Http), 2);
        assert_eq!(responses.sent(OverloadProtocol::Resp), 1);
    }

    #[test]
    fn exposition_has_one_series_per_protocol() {
        let responses = OverloadResponses::new();
        responses.record(OverloadProtocol::Native);
        let mut out = String::new();
        responses.write_prometheus(&mut out);

        assert!(
            out.contains("# TYPE nodedb_server_overload_responses_total counter"),
            "{out}"
        );
        for (label, count) in [("pgwire", 0), ("native", 1), ("http", 0), ("resp", 0)] {
            let series =
                format!(r#"nodedb_server_overload_responses_total{{protocol="{label}"}} {count}"#);
            assert!(out.contains(&series), "{out}");
        }
        assert_eq!(
            out.lines().count(),
            2 + OverloadProtocol::ALL.len(),
            "{out}"
        );
    }
}
