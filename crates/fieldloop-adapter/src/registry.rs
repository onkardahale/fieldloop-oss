//! The adapter selection registry — map an embodiment name to its adapter.
//!
//! Dispatch choice: a **closed [`Embodiment`] enum** with an exhaustive `match`,
//! rather than a `HashMap<String, Box<dyn EmbodimentAdapter>>`. The reason is that a
//! map registry lets a new robot type be forgotten silently — a missing entry is a
//! runtime `None`, discovered in production. With a closed enum, adding a new
//! embodiment means adding a variant, and every `match` on it (here and anywhere
//! downstream) stops compiling until the new variant is handled. That turns "we
//! forgot to wire up the new robot" from a runtime gap into a compile error, which is
//! the whole point of making onboarding a bounded task.

use crate::adapter::EmbodimentAdapter;
use crate::reference::SixDofArmAdapter;

/// The closed set of embodiments Fieldloop knows how to handle.
///
/// Each variant is one onboarded robot type. Adding a robot type is intentionally a
/// core change here: add one variant, wire it into [`Embodiment::adapter`] and
/// [`Embodiment::name`], then the compiler forces every exhaustive match in this
/// crate to acknowledge it. This chooses compile-time exhaustiveness over an
/// out-of-tree plugin escape hatch because the failure mode we care most about is
/// "a robot was declared but only half wired", and a closed enum turns that from a
/// production surprise into a build break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Embodiment {
    /// The reference 6-DOF joint-position arm.
    SixDofArm,
}

impl Embodiment {
    /// Resolve an embodiment to a boxed adapter instance.
    ///
    /// Returns a `Box<dyn EmbodimentAdapter>` so callers can hold any embodiment's
    /// adapter behind one type. The exhaustive match means a new variant must be
    /// given an adapter here before the crate compiles.
    #[must_use]
    pub fn adapter(self) -> Box<dyn EmbodimentAdapter> {
        match self {
            Embodiment::SixDofArm => Box::new(SixDofArmAdapter),
        }
    }

    /// Every known embodiment. The single source of truth the name-based selection
    /// and the lock-step test both iterate, so a new variant is added in exactly one
    /// place and is automatically selectable and tested.
    pub const ALL: &'static [Embodiment] = &[Embodiment::SixDofArm];

    /// The canonical name this embodiment is selected by. Kept in lock-step with the
    /// adapter's own self-reported embodiment name via the test in this module, so a
    /// new robot is not fully onboarded until both the enum and the adapter agree on
    /// one stable string.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Embodiment::SixDofArm => "six_dof_arm",
        }
    }
}

/// Select an adapter by its embodiment name, or `None` if no embodiment matches.
///
/// `None` is a deliberate, explicit answer for an unknown robot type — a caller must
/// handle it (for example, reject the rollout) rather than silently proceeding with a
/// wrong adapter. The match over [`Embodiment`] names is exhaustive, so a newly added
/// embodiment is automatically selectable here.
#[must_use]
pub fn select_adapter(name: &str) -> Option<Box<dyn EmbodimentAdapter>> {
    // Iterate the closed variant set so adding a variant cannot leave it unselectable.
    Embodiment::ALL
        .iter()
        .copied()
        .find(|e| e.name() == name)
        .map(Embodiment::adapter)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every embodiment's registry name matches the name its own adapter reports, so
    /// selection by name can never resolve to an adapter that disagrees about its own
    /// identity.
    #[test]
    fn registry_name_matches_adapter_self_report() {
        for &e in Embodiment::ALL {
            assert_eq!(e.name(), e.adapter().embodiment());
        }
    }
}
