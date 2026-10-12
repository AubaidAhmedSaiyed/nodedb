// SPDX-License-Identifier: BUSL-1.1

//! The cut barriers the apply loop raises commit HLCs above, beyond the ones
//! it applies itself: those the system catalog held at boot, and those a
//! data-group snapshot install carried.
//!
//! The apply loop merges every change into its own floors. A change bumps
//! the generation, so the loop reads the shared floors only after one.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::seed::RecordedCut;

/// Per group, barrier log index to watermark, and a generation that moves
/// on every change.
#[derive(Debug, Default)]
pub struct RecordedCuts {
    floors: Mutex<BTreeMap<u64, BTreeMap<u64, u64>>>,
    generation: AtomicU64,
}

impl RecordedCuts {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add every cut of `cuts`.
    pub fn record(&self, cuts: &[RecordedCut]) {
        let mut floors = self.floors.lock().unwrap_or_else(|p| p.into_inner());
        for cut in cuts {
            let watermark = floors
                .entry(cut.group_id)
                .or_default()
                .entry(cut.barrier_index)
                .or_insert(0);
            *watermark = (*watermark).max(cut.hlc);
        }
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Replace `group_id`'s cuts with `cuts`, the group's durable rows after
    /// a snapshot install.
    pub fn replace_group(&self, group_id: u64, cuts: &[RecordedCut]) {
        let mut floors = self.floors.lock().unwrap_or_else(|p| p.into_inner());
        let group = floors.entry(group_id).or_default();
        group.clear();
        for cut in cuts.iter().filter(|cut| cut.group_id == group_id) {
            let watermark = group.entry(cut.barrier_index).or_insert(0);
            *watermark = (*watermark).max(cut.hlc);
        }
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Moves on every change. Read it before [`Self::all`]: a change after
    /// the read moves it again.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Every cut, in group and index order.
    pub fn all(&self) -> Vec<RecordedCut> {
        self.floors
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .flat_map(|(group_id, group)| {
                group.iter().map(|(barrier_index, hlc)| RecordedCut {
                    group_id: *group_id,
                    barrier_index: *barrier_index,
                    hlc: *hlc,
                })
            })
            .collect()
    }

    /// The highest watermark of a cut of `group_id` below `log_index`.
    pub fn watermark_below(&self, group_id: u64, log_index: u64) -> Option<u64> {
        self.floors
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&group_id)?
            .range(..log_index)
            .map(|(_, hlc)| *hlc)
            .max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(group_id: u64, barrier_index: u64, hlc: u64) -> RecordedCut {
        RecordedCut {
            group_id,
            barrier_index,
            hlc,
        }
    }

    #[test]
    fn an_install_replaces_only_its_group() {
        let cuts = RecordedCuts::new();
        let start = cuts.generation();
        cuts.record(&[cut(1, 10, 100), cut(2, 5, 50)]);
        assert_ne!(cuts.generation(), start);
        assert_eq!(cuts.watermark_below(1, 11), Some(100));
        assert_eq!(cuts.watermark_below(1, 10), None, "the barrier's own place");

        let before = cuts.generation();
        cuts.replace_group(1, &[cut(1, 30, 300), cut(2, 9, 90)]);
        assert_ne!(cuts.generation(), before);
        assert_eq!(cuts.all(), [cut(1, 30, 300), cut(2, 5, 50)]);
    }
}
