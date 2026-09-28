// SPDX-License-Identifier: BUSL-1.1

//! The metadata leader's lease table.
//!
//! One table exists per leadership term. It records, for each node that
//! renewed with this leader, when its lease ends and what its last report
//! covered. It also records the floors: per group, the highest index of any
//! authorization change a barrier registered. A lease is granted only to a
//! report that covers every floor.
//!
//! A barrier releases once, for every target, each node holding an unexpired
//! lease reported coverage of it. Leases granted by an earlier leader are not
//! in the table. They end within one lease duration of this leader taking
//! over, so a barrier also waits until then.
//!
//! A leader that is the only voter of the metadata group can pin its own
//! lease. A pinned lease has no expiry: every barrier waits for the pinned
//! node's coverage, however late its renewal runs. The caller reports on
//! every renewal and barrier whether the leader is still the only voter. The
//! first report that it is not removes the pin.
//!
//! The table is pure: callers pass the clock, so every rule is testable.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nodedb_cluster::GroupCoverage;

/// What a node's last renewal reported and when its lease ends.
#[derive(Debug, Default)]
struct HolderRecord {
    /// End of the lease this leader granted, if any.
    expires_at: Option<Instant>,
    /// Coverage by group, from the last renewal.
    coverage: HashMap<u64, u64>,
}

impl HolderRecord {
    fn covers(&self, group_id: u64, index: u64) -> bool {
        self.coverage
            .get(&group_id)
            .is_some_and(|through| *through >= index)
    }
}

/// The answer to a renewal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenewDecision {
    Granted,
    Withheld,
}

/// Where a barrier stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierState {
    /// No node can plan against state older than the targets.
    Released,
    /// Waiting for a report or an expiry. Nothing changes on its own before
    /// `until`, except a renewal. `until` is `None` while the pinned holder
    /// has not covered the targets: only its renewal, or the end of the pin,
    /// can release the barrier then.
    Waiting { until: Option<Instant> },
    /// The floors of this term are not loaded yet.
    NotReady,
}

/// The lease table of one leadership term.
#[derive(Debug)]
pub struct LeaseTable {
    term: u64,
    leader_since: Instant,
    floors_ready: bool,
    floors: HashMap<u64, u64>,
    holders: HashMap<u64, HolderRecord>,
    /// The holder whose lease has no expiry while the leader stays the only
    /// voter of the metadata group.
    pinned: Option<u64>,
}

impl LeaseTable {
    /// A table for `term`, whose leadership this node observed at `now`.
    pub fn new(term: u64, now: Instant) -> Self {
        Self {
            term,
            leader_since: now,
            floors_ready: false,
            floors: HashMap::new(),
            holders: HashMap::new(),
            pinned: None,
        }
    }

    pub fn term(&self) -> u64 {
        self.term
    }

    pub fn floors_ready(&self) -> bool {
        self.floors_ready
    }

    /// Load the floors this term starts from: an index per group at or above
    /// every change acknowledged before the term.
    pub fn load_floors(&mut self, floors: &[GroupCoverage]) {
        self.raise_floors(floors);
        self.floors_ready = true;
    }

    /// Raise the floors to cover `targets`.
    pub fn raise_floors(&mut self, targets: &[GroupCoverage]) {
        for target in targets {
            let floor = self.floors.entry(target.group_id).or_insert(0);
            *floor = (*floor).max(target.through);
        }
    }

    /// Record a renewal from `node_id` and decide on its lease.
    pub fn renew(
        &mut self,
        node_id: u64,
        coverage: &[GroupCoverage],
        now: Instant,
        lease: Duration,
    ) -> RenewDecision {
        let record = self.holders.entry(node_id).or_default();
        record.coverage = coverage
            .iter()
            .map(|report| (report.group_id, report.through))
            .collect();
        let covered = self
            .floors
            .iter()
            .all(|(group_id, floor)| record.covers(*group_id, *floor));
        if !self.floors_ready || !covered {
            return RenewDecision::Withheld;
        }
        record.expires_at = Some(now + lease);
        RenewDecision::Granted
    }

    /// Record whether the leader is still the only voter of the metadata
    /// group. A leader that is not removes the pin.
    pub fn observe_sole_voter(&mut self, sole_voter: bool) {
        if !sole_voter {
            self.pinned = None;
        }
    }

    /// Pin the lease of `node_id`, which a renewal just granted while the
    /// leader was the only voter.
    pub fn pin(&mut self, node_id: u64) {
        self.pinned = Some(node_id);
    }

    /// Whether the lease of `node_id` is pinned.
    pub fn is_pinned(&self, node_id: u64) -> bool {
        self.pinned == Some(node_id)
    }

    /// Every group whose floor `coverage` does not reach, as
    /// `(group_id, floor, reported)`. `reported` is `None` for a group the
    /// report omits.
    pub fn shortfall(&self, coverage: &[GroupCoverage]) -> Vec<(u64, u64, Option<u64>)> {
        let mut short: Vec<(u64, u64, Option<u64>)> = self
            .floors
            .iter()
            .filter_map(|(group_id, floor)| {
                let reported = coverage
                    .iter()
                    .find(|report| report.group_id == *group_id)
                    .map(|report| report.through);
                (reported.is_none_or(|through| through < *floor))
                    .then_some((*group_id, *floor, reported))
            })
            .collect();
        short.sort_unstable();
        short
    }

    /// Where a barrier on `targets` stands at `now`.
    pub fn barrier(
        &self,
        targets: &[GroupCoverage],
        now: Instant,
        lease: Duration,
    ) -> BarrierState {
        if !self.floors_ready {
            return BarrierState::NotReady;
        }
        let mut until: Option<Instant> = None;
        let mut wait_for = |instant: Instant| {
            until = Some(until.map_or(instant, |current: Instant| current.min(instant)));
        };
        let mut wait_for_pinned = false;
        let earlier_leases_end = self.leader_since + lease;
        if now < earlier_leases_end {
            wait_for(earlier_leases_end);
        }
        for (node_id, record) in &self.holders {
            let pinned = self.pinned == Some(*node_id);
            let live_until = record.expires_at.filter(|end| *end > now);
            if !pinned && live_until.is_none() {
                continue;
            }
            let covered = targets
                .iter()
                .all(|target| record.covers(target.group_id, target.through));
            if covered {
                continue;
            }
            match live_until {
                Some(expires_at) if !pinned => wait_for(expires_at),
                _ => wait_for_pinned = true,
            }
        }
        if wait_for_pinned {
            return BarrierState::Waiting { until: None };
        }
        match until {
            Some(until) => BarrierState::Waiting { until: Some(until) },
            None => BarrierState::Released,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEASE: Duration = Duration::from_millis(150);

    fn cover(group_id: u64, through: u64) -> GroupCoverage {
        GroupCoverage { group_id, through }
    }

    /// A table past the window of earlier leaders' leases.
    fn settled_table(start: Instant) -> LeaseTable {
        let mut table = LeaseTable::new(4, start);
        table.load_floors(&[cover(0, 10)]);
        table
    }

    #[test]
    fn nothing_is_granted_or_released_before_the_floors_load() {
        let now = Instant::now();
        let mut table = LeaseTable::new(1, now);
        assert_eq!(
            table.renew(2, &[cover(0, 99)], now, LEASE),
            RenewDecision::Withheld
        );
        assert_eq!(table.barrier(&[], now, LEASE), BarrierState::NotReady);
    }

    #[test]
    fn the_shortfall_names_each_uncovered_floor() {
        let mut table = settled_table(Instant::now());
        table.raise_floors(&[cover(7, 19)]);
        assert_eq!(
            table.shortfall(&[cover(0, 10), cover(7, 12)]),
            vec![(7, 19, Some(12))]
        );
        assert_eq!(table.shortfall(&[cover(7, 19)]), vec![(0, 10, None)]);
        assert!(table.shortfall(&[cover(0, 10), cover(7, 19)]).is_empty());
    }

    #[test]
    fn a_report_below_a_floor_is_withheld() {
        let start = Instant::now();
        let mut table = settled_table(start);
        assert_eq!(
            table.renew(2, &[cover(0, 9)], start, LEASE),
            RenewDecision::Withheld
        );
        assert_eq!(
            table.renew(2, &[cover(0, 10)], start, LEASE),
            RenewDecision::Granted
        );
        // A group the report omits counts as uncovered.
        table.raise_floors(&[cover(5, 1)]);
        assert_eq!(
            table.renew(2, &[cover(0, 10)], start, LEASE),
            RenewDecision::Withheld
        );
    }

    #[test]
    fn a_barrier_waits_out_the_leases_of_earlier_leaders() {
        let start = Instant::now();
        let table = settled_table(start);
        assert_eq!(
            table.barrier(&[cover(0, 5)], start, LEASE),
            BarrierState::Waiting {
                until: Some(start + LEASE)
            }
        );
        assert_eq!(
            table.barrier(&[cover(0, 5)], start + LEASE, LEASE),
            BarrierState::Released
        );
    }

    #[test]
    fn a_barrier_releases_on_coverage_or_expiry() {
        let start = Instant::now();
        let mut table = settled_table(start);
        let now = start + LEASE;
        assert_eq!(
            table.renew(2, &[cover(0, 10)], now, LEASE),
            RenewDecision::Granted
        );
        let target = [cover(0, 12)];
        table.raise_floors(&target);
        // Node 2 holds a lease and has not covered index 12.
        assert_eq!(
            table.barrier(&target, now, LEASE),
            BarrierState::Waiting {
                until: Some(now + LEASE)
            }
        );
        // Its renewal below the new floor is withheld, and its lease is not
        // extended.
        assert_eq!(
            table.renew(2, &[cover(0, 11)], now, LEASE),
            RenewDecision::Withheld
        );
        // Covering the target releases the barrier at once.
        assert_eq!(
            table.renew(2, &[cover(0, 12)], now, LEASE),
            RenewDecision::Granted
        );
        assert_eq!(table.barrier(&target, now, LEASE), BarrierState::Released);

        // A node that never covers releases the barrier when its lease ends.
        let later = [cover(0, 20)];
        table.raise_floors(&later);
        assert_eq!(
            table.barrier(&later, now, LEASE),
            BarrierState::Waiting {
                until: Some(now + LEASE)
            }
        );
        assert_eq!(
            table.barrier(&later, now + LEASE, LEASE),
            BarrierState::Released
        );
        // Once expired, it gets no lease back without covering the floor.
        assert_eq!(
            table.renew(2, &[cover(0, 12)], now + LEASE, LEASE),
            RenewDecision::Withheld
        );
    }

    /// A pinned lease holds a barrier past its bounded expiry. Its renewal
    /// can run arbitrarily late without any barrier releasing behind it.
    #[test]
    fn a_pinned_lease_holds_a_barrier_past_its_expiry() {
        let start = Instant::now();
        let mut table = settled_table(start);
        let now = start + LEASE;
        table.observe_sole_voter(true);
        assert_eq!(
            table.renew(1, &[cover(0, 10)], now, LEASE),
            RenewDecision::Granted
        );
        table.pin(1);
        let target = [cover(0, 12)];
        table.raise_floors(&target);

        // Ten lease durations pass with no renewal. The bounded lease ended
        // long ago, but the barrier still waits for node 1.
        let starved = now + LEASE * 10;
        table.observe_sole_voter(true);
        assert_eq!(
            table.barrier(&target, starved, LEASE),
            BarrierState::Waiting { until: None }
        );

        // A late renewal that covers the target releases it.
        assert_eq!(
            table.renew(1, &[cover(0, 12)], starved, LEASE),
            RenewDecision::Granted
        );
        assert_eq!(
            table.barrier(&target, starved, LEASE),
            BarrierState::Released
        );
    }

    /// Once the leader is not the only voter, the pin ends and the lease
    /// expires on the clock again.
    #[test]
    fn a_second_voter_ends_the_pin() {
        let start = Instant::now();
        let mut table = settled_table(start);
        let now = start + LEASE;
        assert_eq!(
            table.renew(1, &[cover(0, 10)], now, LEASE),
            RenewDecision::Granted
        );
        table.pin(1);
        assert!(table.is_pinned(1));
        let target = [cover(0, 12)];
        table.raise_floors(&target);

        let starved = now + LEASE * 10;
        table.observe_sole_voter(false);
        assert!(!table.is_pinned(1));
        assert_eq!(
            table.barrier(&target, starved, LEASE),
            BarrierState::Released
        );
    }
}
