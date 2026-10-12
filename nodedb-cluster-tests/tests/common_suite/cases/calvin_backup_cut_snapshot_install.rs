// SPDX-License-Identifier: BUSL-1.1

//! A replica that catches up by `InstallSnapshot` past a backup cut's
//! barrier takes that cut at once.
//!
//! 1. A 3-node cluster with a low log-compaction threshold holds one
//!    collection in data group `G`. The replica `F` neither leads `G` by
//!    preference nor leads the metadata group.
//! 2. `F` stops. A live node takes a backup cut at watermark `W`: every data
//!    group applies the cut's barrier on the live nodes.
//! 3. Writes compact `G`'s log past the barrier. `F` restarts and can catch
//!    up only by a snapshot install, so it never applies the barrier entry.
//! 4. `F` takes the same cut at `W`, as a remote source node of a backup
//!    does. The snapshot carried the barrier, so the cut finishes at once.
//!    Without it, `F` waits for a barrier no leader places again and the cut
//!    fails with "Retry the backup".

use std::time::{Duration, Instant};

use crate::common::cluster_harness::{TestCluster, TestClusterNode, wait_for};

/// Low enough that a few dozen single-row inserts compact `G`'s log.
const COMPACTION_THRESHOLD: u64 = 4;

const COLLECTION: &str = "cut_install_docs";

/// Most rows the test writes to compact `G` past the barrier.
const MAX_ROWS: usize = 200;

const STEP_DEADLINE: Duration = Duration::from_secs(30);
const STEP_POLL: Duration = Duration::from_millis(50);

/// Well inside the cut's 30 s statement deadline: a cut that waits for a
/// barrier no leader places runs to that deadline.
const CUT_BUDGET: Duration = Duration::from_secs(20);

fn node_by_id(cluster: &TestCluster, node_id: u64) -> &TestClusterNode {
    cluster
        .nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .unwrap_or_else(|| panic!("node {node_id} is live"))
}

/// The preferred leader of `group_id`, from `node`'s routing table.
fn preferred_leader(node: &TestClusterNode, group_id: u64) -> u64 {
    let routing = node
        .shared
        .cluster_routing
        .as_ref()
        .expect("cluster routing")
        .read()
        .unwrap_or_else(|p| p.into_inner());
    nodedb_cluster::rebalancer::preferred_leaders(&routing)
        .get(&group_id)
        .copied()
        .unwrap_or_else(|| panic!("group {group_id} has a preferred leader"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_snapshot_installed_replica_takes_a_cut_whose_barrier_it_skipped() {
    let mut cluster = TestCluster::spawn_three_with_compaction_threshold(COMPACTION_THRESHOLD)
        .await
        .expect("3-node cluster with a low compaction threshold");
    cluster
        .exec_ddl_on_any_leader(&format!(
            "CREATE COLLECTION {COLLECTION} (id TEXT PRIMARY KEY, payload TEXT) \
             WITH (engine='document_strict')"
        ))
        .await
        .expect("create the collection");
    cluster.wait_for_preferred_leaders(STEP_DEADLINE).await;
    cluster.wait_for_full_apply_convergence(STEP_DEADLINE).await;

    let probe = &cluster.nodes[0];
    let group_id = probe
        .group_id_for_collection(COLLECTION)
        .expect("the collection maps to a data group");
    let group_leader = preferred_leader(probe, group_id);
    let metadata_leader = probe.metadata_group_leader();
    let lagging_index = cluster
        .nodes
        .iter()
        .position(|node| node.node_id != group_leader && node.node_id != metadata_leader)
        .expect("a third node leads neither group");
    let lagging_id = cluster.nodes[lagging_index].node_id;
    assert!(
        cluster.nodes[lagging_index].hosts_data_group(group_id),
        "node {lagging_id} replicates group {group_id}"
    );

    let stopped = cluster
        .stop_member(lagging_index)
        .await
        .expect("stop the lagging replica");

    // A live node takes the cut. Every group applies its barrier there.
    let cutter = node_by_id(&cluster, group_leader);
    let watermark = cutter.shared.hlc_clock.now().wall_ns;
    nodedb::control::backup::cut::cut_at(&cutter.shared, watermark)
        .await
        .unwrap_or_else(|e| panic!("the cut on node {group_leader}: {e}"));
    let barrier_floor = cutter.shared.applied_index_watcher(group_id).current();

    // Compact the group's log past the barrier.
    let mut written = 0;
    while cutter.group_snapshot_index(group_id) <= barrier_floor {
        assert!(
            written < MAX_ROWS,
            "group {group_id} did not compact past index {barrier_floor} after {written} \
             writes: {}",
            cutter.group_status_line(group_id)
        );
        cutter
            .client
            .simple_query(&format!(
                "INSERT INTO {COLLECTION} (id, payload) VALUES ('r{written}', 'v')"
            ))
            .await
            .unwrap_or_else(|e| panic!("insert r{written}: {e}"));
        written += 1;
        if written % 10 == 0 {
            tokio::time::sleep(STEP_POLL).await;
        }
    }

    cluster
        .restart_member(stopped)
        .await
        .expect("restart the lagging replica");
    let lagging = node_by_id(&cluster, lagging_id);
    wait_for(
        "the lagging replica installs a snapshot past the barrier",
        STEP_DEADLINE,
        STEP_POLL,
        || lagging.local_snapshot_index_for_group(group_id) > barrier_floor,
    )
    .await;
    cluster.wait_for_preferred_leaders(STEP_DEADLINE).await;
    assert!(
        !lagging.leads_group(group_id),
        "node {lagging_id} leads group {group_id}: its own driver could place the barrier"
    );

    let started = Instant::now();
    let cut = tokio::time::timeout(
        CUT_BUDGET,
        nodedb::control::backup::cut::cut_at(&lagging.shared, watermark),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "node {lagging_id}: the cut at {watermark} still waits after {CUT_BUDGET:?}: {}",
            lagging.group_status_line(group_id)
        )
    });
    cut.unwrap_or_else(|e| panic!("node {lagging_id}: the cut at {watermark}: {e}"));
    assert!(
        !lagging.leads_group(group_id),
        "node {lagging_id} led group {group_id} during the cut after {:?}",
        started.elapsed()
    );
    cluster.shutdown().await;
}
