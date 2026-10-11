// SPDX-License-Identifier: BUSL-1.1

//! The QUIC endpoint a node serves and dials Raft RPCs on.
//!
//! A build with the `failpoints` feature binds the endpoint over a UDP socket
//! that drops every datagram while the node's `transport::blackhole` fail
//! point is armed. The node then looks partitioned to its peers while its
//! process keeps running. Any other build binds a plain UDP socket.

use std::net::SocketAddr;

use crate::error::{ClusterError, Result};

/// Node-scoped fail point that cuts a node off the network while armed.
pub const BLACKHOLE_FAIL_POINT: &str = "transport::blackhole";

/// A bound QUIC endpoint that accepts Raft RPC connections.
#[derive(Debug)]
pub struct RaftListener {
    endpoint: quinn::Endpoint,
    local_addr: SocketAddr,
}

impl RaftListener {
    /// Bind the endpoint of node `node_id` on `addr`.
    pub fn bind(node_id: u64, addr: SocketAddr, config: quinn::ServerConfig) -> Result<Self> {
        let endpoint =
            bind_endpoint(node_id, addr, config).map_err(|e| ClusterError::Transport {
                detail: format!("bind {addr}: {e}"),
            })?;
        let local_addr = endpoint.local_addr().map_err(|e| ClusterError::Transport {
            detail: format!("local address of {addr}: {e}"),
        })?;
        Ok(Self {
            endpoint,
            local_addr,
        })
    }

    /// Accept the next incoming QUIC connection.
    pub async fn accept(&self) -> Result<quinn::Connection> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| ClusterError::Transport {
                detail: "endpoint closed".into(),
            })?;
        incoming.await.map_err(|e| ClusterError::Transport {
            detail: format!("accept: {e}"),
        })
    }

    /// The local address the endpoint is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// The endpoint, which also dials outbound connections.
    pub fn endpoint(&self) -> &quinn::Endpoint {
        &self.endpoint
    }
}

#[cfg(not(feature = "failpoints"))]
fn bind_endpoint(
    _node_id: u64,
    addr: SocketAddr,
    config: quinn::ServerConfig,
) -> std::io::Result<quinn::Endpoint> {
    quinn::Endpoint::server(config, addr)
}

#[cfg(feature = "failpoints")]
fn bind_endpoint(
    node_id: u64,
    addr: SocketAddr,
    config: quinn::ServerConfig,
) -> std::io::Result<quinn::Endpoint> {
    let socket = std::net::UdpSocket::bind(addr)?;
    let runtime =
        quinn::default_runtime().ok_or_else(|| std::io::Error::other("no async runtime found"))?;
    let inner = runtime.wrap_udp_socket(socket)?;
    let socket = std::sync::Arc::new(blackhole::BlackholeSocket::new(node_id, inner));
    quinn::Endpoint::new_with_abstract_socket(
        quinn::EndpointConfig::default(),
        Some(config),
        socket,
        runtime,
    )
}

#[cfg(feature = "failpoints")]
mod blackhole {
    use std::io::{self, IoSliceMut};
    use std::net::SocketAddr;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use nodedb_types::fail_point::{FailScope, lookup};
    use quinn::udp::{RecvMeta, Transmit};
    use quinn::{AsyncUdpSocket, UdpPoller};

    use super::BLACKHOLE_FAIL_POINT;

    /// A UDP socket that drops every datagram, both ways, while its node's
    /// blackhole fail point is armed.
    #[derive(Debug)]
    pub(super) struct BlackholeSocket {
        node_id: u64,
        inner: Arc<dyn AsyncUdpSocket>,
    }

    impl BlackholeSocket {
        pub(super) fn new(node_id: u64, inner: Arc<dyn AsyncUdpSocket>) -> Self {
            Self { node_id, inner }
        }

        fn cut_off(&self) -> bool {
            lookup(FailScope::Node(self.node_id), BLACKHOLE_FAIL_POINT).is_some()
        }
    }

    impl AsyncUdpSocket for BlackholeSocket {
        fn create_io_poller(self: Arc<Self>) -> Pin<Box<dyn UdpPoller>> {
            Arc::clone(&self.inner).create_io_poller()
        }

        fn try_send(&self, transmit: &Transmit) -> io::Result<()> {
            if self.cut_off() {
                return Ok(());
            }
            self.inner.try_send(transmit)
        }

        fn poll_recv(
            &self,
            cx: &mut Context,
            bufs: &mut [IoSliceMut<'_>],
            meta: &mut [RecvMeta],
        ) -> Poll<io::Result<usize>> {
            // Received datagrams are drained and dropped until the inner
            // socket has none left, so it registers the waker.
            loop {
                match self.inner.poll_recv(cx, bufs, meta) {
                    Poll::Ready(Ok(_)) if self.cut_off() => {}
                    other => return other,
                }
            }
        }

        fn local_addr(&self) -> io::Result<SocketAddr> {
            self.inner.local_addr()
        }

        fn max_transmit_segments(&self) -> usize {
            self.inner.max_transmit_segments()
        }

        fn max_receive_segments(&self) -> usize {
            self.inner.max_receive_segments()
        }

        fn may_fragment(&self) -> bool {
            self.inner.may_fragment()
        }
    }
}
