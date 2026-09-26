// SPDX-License-Identifier: BUSL-1.1

//! A TCP address bound early and opened for connections late.
//!
//! Boot binds every protocol address before it waits for the node to become
//! ready, so a port conflict fails boot while nothing is exposed. A bound
//! socket that is not listening refuses each connection attempt at once. A
//! listening socket completes the TCP handshake in the kernel before any
//! accept runs, so a client would wait out the whole boot. Boot therefore
//! calls [`ReservedSocket::listen`] only once the node can serve.

use std::net::SocketAddr;

use tokio::net::{TcpListener, TcpSocket};

/// Pending-connection queue length, the same value
/// `tokio::net::TcpListener::bind` uses.
const LISTEN_BACKLOG: u32 = 1024;

/// A TCP socket bound to its address but not yet listening.
#[derive(Debug)]
pub struct ReservedSocket {
    socket: TcpSocket,
    addr: SocketAddr,
}

impl ReservedSocket {
    /// Bind `addr` without listening.
    ///
    /// The bind fails when another socket listens on `addr`. `SO_REUSEADDR`
    /// is set on Unix, as `TcpListener::bind` does, so a restart can rebind
    /// while old connections sit in `TIME_WAIT`.
    pub fn bind(addr: SocketAddr) -> crate::Result<Self> {
        let bind_error = |e: std::io::Error| crate::Error::Config {
            detail: format!("bind {addr}: {e}"),
        };
        let socket = if addr.is_ipv4() {
            TcpSocket::new_v4()
        } else {
            TcpSocket::new_v6()
        }
        .map_err(bind_error)?;
        #[cfg(not(windows))]
        socket.set_reuseaddr(true).map_err(bind_error)?;
        socket.bind(addr).map_err(bind_error)?;
        let addr = socket.local_addr().map_err(bind_error)?;
        Ok(Self { socket, addr })
    }

    /// The bound address. For port `0` it holds the port the OS assigned.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Start listening. Connection attempts are accepted from here on.
    ///
    /// Fails when another socket started listening on the same address after
    /// this one was bound.
    pub fn listen(self) -> crate::Result<TcpListener> {
        let addr = self.addr;
        self.socket
            .listen(LISTEN_BACKLOG)
            .map_err(|e| crate::Error::Config {
                detail: format!("listen on {addr}: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn loopback_any_port() -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], 0))
    }

    /// A reserved socket refuses connections at once, so a client connecting
    /// during boot never waits in the kernel's accept queue.
    #[tokio::test]
    async fn a_reserved_socket_refuses_connections_until_it_listens() {
        let reserved = ReservedSocket::bind(loopback_any_port()).expect("bind");
        let addr = reserved.local_addr();
        assert_ne!(addr.port(), 0);

        let refused =
            tokio::time::timeout(Duration::from_secs(5), tokio::net::TcpStream::connect(addr))
                .await
                .expect("a refusal is immediate");
        assert!(
            refused.is_err(),
            "a reserved socket must refuse connections"
        );

        let listener = reserved.listen().expect("listen");
        let (connected, accepted) =
            tokio::join!(tokio::net::TcpStream::connect(addr), listener.accept());
        connected.expect("connect after listen");
        accepted.expect("accept after listen");
    }

    /// An address another socket listens on cannot be reserved.
    #[tokio::test]
    async fn an_address_in_use_cannot_be_reserved() {
        let occupied = TcpListener::bind(loopback_any_port())
            .await
            .expect("occupy a port");
        let addr = occupied.local_addr().expect("occupied addr");
        assert!(ReservedSocket::bind(addr).is_err());
    }
}
