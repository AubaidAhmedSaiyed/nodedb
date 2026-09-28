// SPDX-License-Identifier: BUSL-1.1

//! Gate handles for the [`StartupSequencer`].
//!
//! Two complementary types:
//!
//! - [`StartupGate`] — a shared, cheaply-cloneable read handle that any
//!   Control Plane code can hold to observe the current phase or `await`
//!   a specific phase before proceeding.
//! - [`ReadyGate`] — a single-use write handle returned by
//!   [`StartupSequencer::register_gate`]. When a subsystem completes its
//!   startup work it calls [`ReadyGate::fire`]. If the subsystem fails it
//!   calls [`ReadyGate::fail`]. Dropping a [`ReadyGate`] without firing it
//!   automatically transitions the sequencer to `Failed` — a dropped gate
//!   that never fired would otherwise deadlock startup forever.
//!
//! [`StartupSequencer`]: super::startup_sequencer::StartupSequencer
//! [`StartupSequencer::register_gate`]: super::startup_sequencer::StartupSequencer::register_gate

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Weak};

use tokio::sync::watch;

use super::error::StartupError;
use super::phase::StartupPhase;
use super::startup_sequencer::SequencerState;

// ---------------------------------------------------------------------------
// GateId
// ---------------------------------------------------------------------------

/// Opaque numeric identifier assigned to each registered gate.
///
/// Used internally to track which gates have fired for a given phase.
/// Visible to callers only via the `subsystem` name they supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct GateId(pub(super) u64);

// ---------------------------------------------------------------------------
// StartupGate
// ---------------------------------------------------------------------------

/// Shared read handle into the [`StartupSequencer`].
///
/// Listeners and other Control Plane code hold an `Arc<StartupGate>` and
/// call [`await_phase`] to block until the sequencer has reached (or
/// passed) a target phase. The gate is cancel-safe: dropping an
/// in-progress `await_phase` future and re-polling from `select!` does
/// not miss a subsequent advance.
///
/// [`StartupSequencer`]: super::startup_sequencer::StartupSequencer
/// [`await_phase`]: StartupGate::await_phase
#[derive(Debug, Clone)]
pub struct StartupGate {
    pub(super) rx: watch::Receiver<SequencerSnapshot>,
}

/// Lightweight snapshot of the sequencer broadcast on every phase change.
#[derive(Debug, Clone)]
pub struct SequencerSnapshot {
    /// Current phase. Increases monotonically. Jumps to `Failed` on any
    /// subsystem failure.
    pub phase: StartupPhase,
    /// Non-`None` when the sequencer has entered `Failed`. Contains the
    /// error that caused the failure, wrapped in an `Arc` so all waiters
    /// share the allocation.
    pub failed: Option<Arc<StartupError>>,
}

impl StartupGate {
    pub(super) fn new(rx: watch::Receiver<SequencerSnapshot>) -> Self {
        Self { rx }
    }

    /// Create a gate that is pre-fired at [`StartupPhase::Serving`], the
    /// final phase.
    ///
    /// Used by test helpers that construct a [`SharedState`] without a real
    /// [`StartupSequencer`]. Any call to [`await_phase`] on this gate returns
    /// immediately regardless of the requested phase.
    ///
    /// [`await_phase`]: StartupGate::await_phase
    pub fn pre_fired() -> Arc<Self> {
        let (tx, rx) = watch::channel(SequencerSnapshot {
            phase: StartupPhase::Serving,
            failed: None,
        });
        // Keep the sender alive inside the gate so the receiver never sees
        // the channel as closed and returns `AlreadyTerminated`.
        let gate = Arc::new(Self { rx });
        // The sender is dropped intentionally: no further phase changes will
        // occur. The already-received value (Serving) is what all
        // `await_phase` callers will see.
        drop(tx);
        gate
    }

    /// Wait until the sequencer has reached `phase` or a later phase.
    ///
    /// Returns `Ok(())` when the target phase is reached. Returns
    /// `Err(StartupError::SubsystemFailed{..})` (or another
    /// `StartupError` variant stored on the snapshot) if the sequencer
    /// entered `Failed` before reaching the target. Returns
    /// `Err(StartupError::AlreadyTerminated)` if the watch channel is
    /// closed (all `StartupSequencer` senders dropped).
    ///
    /// # Cancel safety
    ///
    /// Cancel-safe. The underlying `watch::Receiver::changed` call is
    /// cancel-safe, and the snapshot is re-read on every wake.
    pub async fn await_phase(&self, phase: StartupPhase) -> Result<(), StartupError> {
        // Clone to get a mutable receiver without borrowing `self`.
        let mut rx = self.rx.clone();

        loop {
            let snap = rx.borrow_and_update().clone();

            // If the sequencer has failed, return the error immediately.
            if let Some(err) = snap.failed {
                return Err((*err).clone());
            }

            // Target reached (or passed).
            if snap.phase >= phase {
                return Ok(());
            }

            // Wait for the next change.
            if rx.changed().await.is_err() {
                // Sender dropped — no further advances possible.
                return Err(StartupError::AlreadyTerminated);
            }
        }
    }

    /// Non-blocking snapshot of the current phase.
    pub fn current_phase(&self) -> StartupPhase {
        self.rx.borrow().phase
    }

    /// Non-blocking check for failure. Returns the stored error if the
    /// sequencer has entered `Failed`, or `None` if startup is still
    /// progressing (or completed successfully).
    pub fn is_failed(&self) -> Option<Arc<StartupError>> {
        self.rx.borrow().failed.clone()
    }
}

// ---------------------------------------------------------------------------
// ReadyGate
// ---------------------------------------------------------------------------

/// Single-use write handle for a registered startup gate.
///
/// Obtained from [`StartupSequencer::register_gate`]. The owning subsystem
/// calls [`fire`] when it has completed its startup work, or [`fail`] if
/// it encountered an unrecoverable error. If the `ReadyGate` is dropped
/// without either being called, the `Drop` implementation automatically
/// calls `fail` with a [`StartupError::GateDroppedWithoutFire`] — a
/// silent hang would otherwise deadlock startup forever.
///
/// [`StartupSequencer::register_gate`]: super::startup_sequencer::StartupSequencer::register_gate
/// [`fire`]: ReadyGate::fire
/// [`fail`]: ReadyGate::fail
pub struct ReadyGate {
    pub(super) id: GateId,
    pub(super) phase: StartupPhase,
    pub(super) subsystem: String,
    pub(super) sequencer: Weak<Mutex<SequencerState>>,
    pub(super) fired: AtomicBool,
    /// Sender side of the watch channel — held here so we can broadcast
    /// phase changes from `fire`.
    pub(super) phase_tx: Arc<watch::Sender<SequencerSnapshot>>,
}

impl std::fmt::Debug for ReadyGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadyGate")
            .field("id", &self.id)
            .field("phase", &self.phase)
            .field("subsystem", &self.subsystem)
            .field("fired", &self.fired.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ReadyGate {
    /// Report that this subsystem has successfully completed its startup
    /// work for the registered phase.
    ///
    /// Idempotent: calling `fire` a second time is a no-op. The sequencer
    /// advances to the next phase only when all registered gates for the
    /// current phase have fired.
    pub fn fire(&self) {
        // Idempotent: if already fired, do nothing.
        if self.fired.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(state_arc) = self.sequencer.upgrade() else {
            // Sequencer already dropped — startup is long over.
            return;
        };
        let mut state = match state_arc.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                tracing::error!(
                    subsystem = %self.subsystem,
                    "StartupSequencer mutex poisoned when firing gate — proceeding with recovery"
                );
                poisoned.into_inner()
            }
        };
        state.fire_gate(self.id, self.phase, &self.phase_tx);
    }

    /// Report that this subsystem encountered an unrecoverable error
    /// during startup. The sequencer immediately enters `Failed` and all
    /// waiters wake with an error.
    pub fn fail(&self, reason: impl Into<String>) {
        // Mark as fired so Drop doesn't emit a second, confusing error.
        self.fired.store(true, Ordering::Release);

        let err = StartupError::SubsystemFailed {
            phase: self.phase,
            subsystem: self.subsystem.clone(),
            reason: reason.into(),
        };
        let Some(state_arc) = self.sequencer.upgrade() else {
            return;
        };
        let mut state = match state_arc.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                tracing::error!(
                    subsystem = %self.subsystem,
                    "StartupSequencer mutex poisoned when failing gate"
                );
                poisoned.into_inner()
            }
        };
        state.set_failed(err, &self.phase_tx);
    }
}

impl Drop for ReadyGate {
    /// Auto-fail the sequencer if this gate was never fired.
    ///
    /// A subsystem that panics or returns early without calling `fire` or
    /// `fail` would leave the sequencer waiting forever. The `Drop` impl
    /// converts the silent hang into a loud, descriptive failure.
    fn drop(&mut self) {
        if self.fired.load(Ordering::Acquire) {
            return;
        }
        // Mark fired so the drop is idempotent if somehow called twice.
        self.fired.store(true, Ordering::Release);

        let err = StartupError::GateDroppedWithoutFire {
            phase: self.phase,
            subsystem: self.subsystem.clone(),
        };
        tracing::error!(
            subsystem = %self.subsystem,
            phase = ?self.phase,
            "ReadyGate dropped without firing — startup sequencer transitioning to Failed"
        );
        let Some(state_arc) = self.sequencer.upgrade() else {
            return;
        };
        let Ok(mut state) = state_arc.lock() else {
            return;
        };
        state.set_failed(err, &self.phase_tx);
    }
}
