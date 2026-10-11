// SPDX-License-Identifier: BUSL-1.1

//! A replica that takes a snapshot before it takes log entries again.
//!
//! - [`ReplicaState::WantsSnapshot`]: its data matches its log, and state
//!   its host keeps beside the log lags. It refuses entries so its leader
//!   sends a snapshot. It still campaigns and leads.
//! - [`ReplicaState::AwaitingSnapshot`]: its data diverged. Only a leader
//!   sends a snapshot, so a leader awaiting one would wait forever. Entering
//!   this state steps a leader or candidate down, and the election paths
//!   refuse to start while it holds. A leader hands its term to the voter
//!   [`RaftNode::hand_off_target`] names, when one qualifies.

use tracing::info;

use crate::message::TimeoutNowRequest;
use crate::node::core::RaftNode;
use crate::state::{NodeRole, ReplicaState};
use crate::storage::LogStorage;

impl<S: LogStorage> RaftNode<S> {
    /// Set what this replica needs before it takes log entries again. A
    /// leader or candidate whose data diverged steps down.
    pub fn set_replica_state(&mut self, state: ReplicaState) {
        self.replica = state;
        if !state.may_lead() {
            self.stand_down_for_snapshot();
        }
    }

    /// Whether this replica refuses log entries until a snapshot installs.
    pub fn snapshot_required(&self) -> bool {
        self.replica.wants_snapshot()
    }

    /// Whether this replica holds the state it serves and leads from.
    pub fn replica_state(&self) -> ReplicaState {
        self.replica
    }

    /// Leave a leadership or candidacy this replica can no longer hold.
    fn stand_down_for_snapshot(&mut self) {
        let term = self.hard_state.current_term;
        match self.role {
            NodeRole::Leader => {
                let target = self.hand_off_target();
                if let Some(target) = target {
                    // The target campaigns at once. Voters grant it although
                    // they still hear this leader, so no lease holds past
                    // here.
                    self.lease.revoke();
                    self.ready.timeout_now.push((
                        target,
                        TimeoutNowRequest {
                            term,
                            leader_id: self.config.node_id,
                            group_id: self.config.group_id,
                        },
                    ));
                }
                info!(
                    node = self.config.node_id,
                    group = self.config.group_id,
                    term,
                    ?target,
                    "replica awaits a snapshot; stepping down from leader"
                );
                self.become_follower(term);
            }
            NodeRole::Candidate => self.become_follower(term),
            NodeRole::Follower => self.pre_vote = None,
            NodeRole::Learner | NodeRole::Observer => {}
        }
    }

    /// The voter peer this leader hands its leadership to. `None` off the
    /// leader, or when no voter qualifies.
    ///
    /// A target answered whole this term and holds every committed entry. A
    /// voter missing a committed entry cannot win an election. Of the
    /// qualified voters, the highest match goes first. One at the log
    /// frontier wins its campaign against every voter.
    pub fn hand_off_target(&self) -> Option<u64> {
        let commit = self.volatile.commit_index;
        self.config
            .peers
            .iter()
            .copied()
            .filter(|&peer| self.peer_answered_whole(peer))
            .filter_map(|peer| Some((peer, self.match_index_for(peer)?)))
            .filter(|&(_, matched)| matched >= commit)
            .max_by_key(|&(_, matched)| matched)
            .map(|(peer, _)| peer)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crate::message::{AppendEntriesRequest, AppendEntriesResponse, RequestVoteResponse};
    use crate::node::core::RaftNode;
    use crate::node::leader_lease::UNTRACKED_ROUND;
    use crate::state::{NodeRole, ReplicaState};
    use crate::storage::MemStorage;
    use crate::test_support::{force_election, test_config};

    fn leader_3voter() -> RaftNode<MemStorage> {
        let mut node = RaftNode::new(test_config(1, vec![2, 3]), MemStorage::new());
        force_election(&mut node);
        let _ = node.take_ready();
        node.handle_request_vote_response(
            2,
            &RequestVoteResponse {
                term: 1,
                vote_granted: true,
            },
        );
        assert_eq!(node.role(), NodeRole::Leader);
        let _ = node.take_ready();
        node
    }

    fn reply(last_log_index: u64, replica: ReplicaState) -> AppendEntriesResponse {
        AppendEntriesResponse {
            term: 1,
            success: !replica.wants_snapshot(),
            last_log_index,
            round: UNTRACKED_ROUND,
            replica,
        }
    }

    #[test]
    fn a_replica_awaiting_a_snapshot_never_campaigns() {
        let mut node = RaftNode::new(test_config(2, vec![1, 3]), MemStorage::new());
        node.set_replica_state(ReplicaState::AwaitingSnapshot);
        assert_eq!(node.replica_state(), ReplicaState::AwaitingSnapshot);

        force_election(&mut node);
        let ready = node.take_ready();
        assert!(ready.pre_vote_requests.is_empty(), "no pre-vote round");
        assert!(ready.vote_requests.is_empty(), "no vote request");
        assert_eq!(node.role(), NodeRole::Follower);
        assert_eq!(node.current_term(), 0);

        node.set_replica_state(ReplicaState::Whole);
        force_election(&mut node);
        assert_eq!(
            node.role(),
            NodeRole::Candidate,
            "a whole replica campaigns"
        );
    }

    #[test]
    fn a_sole_voter_awaiting_a_snapshot_never_elects_itself() {
        let mut node = RaftNode::new(test_config(1, vec![]), MemStorage::new());
        node.set_replica_state(ReplicaState::AwaitingSnapshot);
        node.election_deadline_override(Instant::now() - Duration::from_millis(1));
        node.tick();
        assert_eq!(node.role(), NodeRole::Follower);
        assert_eq!(node.leader_id(), 0);
    }

    #[test]
    fn a_transfer_never_makes_an_awaiting_replica_campaign() {
        let mut node = RaftNode::new(test_config(2, vec![1, 3]), MemStorage::new());
        node.handle_append_entries(&AppendEntriesRequest {
            term: 1,
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            group_id: 1,
            round: 1,
            replicated_floor: 0,
        });
        node.set_replica_state(ReplicaState::AwaitingSnapshot);
        node.handle_timeout_now(&crate::message::TimeoutNowRequest {
            term: 1,
            leader_id: 1,
            group_id: 1,
        });
        assert_eq!(node.role(), NodeRole::Follower);
        assert_eq!(node.current_term(), 1);
        assert!(node.take_ready().vote_requests.is_empty());
    }

    #[test]
    fn a_leader_given_the_requirement_steps_down_and_hands_off() {
        let mut leader = leader_3voter();
        let frontier = leader.last_log_index();
        leader.handle_append_entries_response(2, &reply(frontier, ReplicaState::Whole));
        let _ = leader.take_ready();

        leader.set_replica_state(ReplicaState::AwaitingSnapshot);
        assert_eq!(leader.role(), NodeRole::Follower);
        assert_eq!(leader.leader_id(), 0);
        let ready = leader.take_ready();
        assert_eq!(ready.timeout_now.len(), 1);
        let (target, request) = ready.timeout_now[0].clone();
        assert_eq!(target, 2);
        assert_eq!((request.term, request.leader_id), (1, 1));

        leader.election_deadline_override(Instant::now() - Duration::from_millis(1));
        leader.tick();
        assert_eq!(leader.role(), NodeRole::Follower, "it never campaigns back");
    }

    #[test]
    fn a_leader_never_hands_off_to_a_peer_awaiting_a_snapshot() {
        let mut leader = leader_3voter();
        let frontier = leader.last_log_index();
        // Peer 2 matched the frontier, then asked for a snapshot. Peer 3
        // never answered.
        leader.handle_append_entries_response(2, &reply(frontier, ReplicaState::Whole));
        leader.handle_append_entries_response(2, &reply(frontier, ReplicaState::AwaitingSnapshot));
        let _ = leader.take_ready();

        leader.set_replica_state(ReplicaState::AwaitingSnapshot);
        assert_eq!(leader.role(), NodeRole::Follower);
        assert!(leader.take_ready().timeout_now.is_empty());
    }

    /// A replica that wants a snapshot only for state beside its log, such
    /// as a lost Calvin base, still campaigns and leads, and a leader that
    /// takes that state keeps leading.
    #[test]
    fn a_replica_that_wants_a_snapshot_still_campaigns_and_leads() {
        let mut node = RaftNode::new(test_config(1, vec![2, 3]), MemStorage::new());
        node.set_replica_state(ReplicaState::WantsSnapshot);
        assert!(node.snapshot_required());
        force_election(&mut node);
        assert_eq!(node.role(), NodeRole::Candidate);
        node.handle_request_vote_response(
            2,
            &RequestVoteResponse {
                term: 1,
                vote_granted: true,
            },
        );
        assert_eq!(node.role(), NodeRole::Leader);

        let mut leader = leader_3voter();
        leader.set_replica_state(ReplicaState::WantsSnapshot);
        assert_eq!(leader.role(), NodeRole::Leader);
        assert!(leader.take_ready().timeout_now.is_empty());

        let mut sole = RaftNode::new(test_config(1, vec![]), MemStorage::new());
        sole.set_replica_state(ReplicaState::WantsSnapshot);
        sole.election_deadline_override(Instant::now() - Duration::from_millis(1));
        sole.tick();
        assert_eq!(sole.role(), NodeRole::Leader);
    }

    #[test]
    fn a_stricter_state_wins() {
        assert_eq!(
            ReplicaState::AwaitingSnapshot.raised_to(ReplicaState::WantsSnapshot),
            ReplicaState::AwaitingSnapshot
        );
        assert_eq!(
            ReplicaState::Whole.raised_to(ReplicaState::WantsSnapshot),
            ReplicaState::WantsSnapshot
        );
    }

    #[test]
    fn a_candidate_given_the_requirement_stands_down() {
        let mut node = RaftNode::new(test_config(1, vec![2, 3]), MemStorage::new());
        force_election(&mut node);
        assert_eq!(node.role(), NodeRole::Candidate);
        node.set_replica_state(ReplicaState::AwaitingSnapshot);
        assert_eq!(node.role(), NodeRole::Follower);
        node.handle_request_vote_response(
            2,
            &RequestVoteResponse {
                term: 1,
                vote_granted: true,
            },
        );
        assert_eq!(
            node.role(),
            NodeRole::Follower,
            "a late grant elects nothing"
        );
    }

    /// A voter not heard this term is never a hand-off target.
    #[test]
    fn a_voter_not_heard_this_term_is_no_hand_off_target() {
        let mut leader = leader_3voter();
        assert_eq!(leader.hand_off_target(), None);
        leader.handle_append_entries_response(3, &reply(0, ReplicaState::AwaitingSnapshot));
        assert_eq!(leader.hand_off_target(), None, "voter 2 is still unheard");
    }

    /// The target holds every committed entry, and the highest match goes
    /// first.
    #[test]
    fn the_hand_off_target_holds_every_committed_entry() {
        let mut node = RaftNode::new(test_config(1, vec![2, 3, 4, 5]), MemStorage::new());
        force_election(&mut node);
        let _ = node.take_ready();
        for peer in [2, 3] {
            node.handle_request_vote_response(
                peer,
                &RequestVoteResponse {
                    term: 1,
                    vote_granted: true,
                },
            );
        }
        assert_eq!(node.role(), NodeRole::Leader);
        let frontier = node.last_log_index();
        for peer in [2, 3, 4] {
            node.handle_append_entries_response(peer, &reply(frontier, ReplicaState::Whole));
        }
        assert_eq!(node.commit_index(), frontier);
        node.handle_append_entries_response(5, &reply(0, ReplicaState::Whole));
        for peer in [2, 3] {
            node.handle_append_entries_response(
                peer,
                &reply(frontier, ReplicaState::AwaitingSnapshot),
            );
        }
        assert_eq!(node.hand_off_target(), Some(4));

        node.handle_append_entries_response(4, &reply(frontier, ReplicaState::WantsSnapshot));
        assert_eq!(
            node.hand_off_target(),
            None,
            "voter 5 answered whole but misses a committed entry"
        );

        node.handle_append_entries_response(5, &reply(frontier, ReplicaState::Whole));
        assert_eq!(node.hand_off_target(), Some(5));
    }
}
