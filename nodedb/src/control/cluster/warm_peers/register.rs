// SPDX-License-Identifier: BUSL-1.1

//! Register target nodes' addresses with the transport before a fan-out.

use std::collections::BTreeSet;

use crate::control::state::SharedState;

/// Register each target node's address with the transport from the live cluster
/// topology (idempotent). Makes a fan-out robust to a peer the transport has
/// not warmed yet — without it `send_rpc` to an unregistered (but
/// topology-known) node fails with `NodeUnreachable`. Self IS registered too:
/// a coordinator that also owns one of the targets dispatches to itself via
/// `send_rpc`, which loops back through the local QUIC endpoint and runs the
/// same handler (an extra local hop, functionally correct). Missing topology /
/// address for a node is left alone so the subsequent `send_rpc` surfaces the
/// typed `NodeUnreachable` rather than this silently masking it.
pub(crate) fn register_peers_from_topology(
    state: &SharedState,
    transport: &nodedb_cluster::NexarTransport,
    nodes: &BTreeSet<u64>,
) {
    let Some(topology) = state.cluster_topology.as_ref() else {
        return;
    };
    let topo = topology.read().unwrap_or_else(|p| p.into_inner());
    for &node in nodes {
        if let Some(info) = topo.get_node(node)
            && let Some(addr) = info.socket_addr()
        {
            transport.register_peer(node, addr);
        }
    }
}
