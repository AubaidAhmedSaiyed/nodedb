// SPDX-License-Identifier: BUSL-1.1

//! The two bounds a linearizable read runs under.
//!
//! - [`StatementDeadline`] bounds the whole read. This node's wait to apply
//!   through the read index follows it.
//! - [`ReadIndexDeadline`] bounds one quorum round for a read index: the
//!   transport RPC timeout, never past the statement. A partitioned leader
//!   is refused at this bound, not at the statement deadline.
//!
//! Each is its own type, so a caller cannot pass one where the other belongs.

use std::time::{Duration, Instant};

pub use crate::control::server::shared::session::StatementDeadline;
use crate::control::state::SharedState;

/// When one quorum round for a read index gives up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReadIndexDeadline(Instant);

impl ReadIndexDeadline {
    /// `rpc_timeout` from now, never past `statement`.
    pub fn within(statement: StatementDeadline, rpc_timeout: Duration) -> Self {
        Self(statement.instant().min(Instant::now() + rpc_timeout))
    }

    /// The bound `state`'s transport tuning gives a round started now, never
    /// past `statement`.
    pub fn of_round(state: &SharedState, statement: StatementDeadline) -> Self {
        Self::within(
            statement,
            Duration::from_secs(state.tuning.cluster_transport.rpc_timeout_secs),
        )
    }

    /// The deadline as an instant.
    pub fn instant(self) -> Instant {
        self.0
    }

    /// Time left until the deadline, zero once it passed.
    pub fn remaining(self) -> Duration {
        self.0.saturating_duration_since(Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_round_bound_is_the_rpc_timeout_inside_a_long_statement() {
        let statement = StatementDeadline::after(Duration::from_secs(30));
        let before = Instant::now();
        let round = ReadIndexDeadline::within(statement, Duration::from_secs(1));
        assert!(round.instant() >= before + Duration::from_secs(1));
        assert!(round.instant() < statement.instant());
    }

    #[test]
    fn the_round_bound_never_passes_the_statement() {
        let statement = StatementDeadline::after(Duration::from_millis(10));
        let round = ReadIndexDeadline::within(statement, Duration::from_secs(5));
        assert_eq!(round.instant(), statement.instant());
    }
}
