//! `Se3Pose` — a minimal rigid-body pose in a named reference frame.
//!
//! The spatial attribution tier binds an outcome to a rollout that was at the *same
//! place* when same-boot monotonic time cannot bind them (e.g. the rollout and the
//! outcome are from different robot boots). For that it needs a comparable pose on
//! both sides. This type is that comparable pose: a translation `(x, y, z)` plus a
//! unit quaternion `(qw, qx, qy, qz)` for orientation — a full SE(3) rigid transform,
//! kept deliberately minimal (seven `f64`s, no matrices, no library dependency) so it
//! is cheap to construct in the robot hot path and trivially serializable.
//!
//! A pose is only comparable to another pose **in the same `frame_id`**. A pose in a
//! robot-local base frame and a pose in a map frame name different origins, so
//! comparing their coordinates is meaningless; carrying the frame label next to the
//! pose (see the `frame_id` field on the rollout / outcome) lets the spatial tier
//! refuse to compare across frames rather than silently bind two unrelated places.
//!
//! Distance is defined on the translation only: "co-located" means the two bodies
//! occupied the same point in space within an embodiment-scale epsilon. Orientation
//! is carried for completeness (and so a future tier can use it) but is not part of
//! the co-location test — two events at the same point with different headings are
//! still the same place on the floor.

use serde::{Deserialize, Serialize};

/// A rigid-body pose: translation in meters plus a unit quaternion orientation,
/// expressed in a named reference frame (carried separately, next to the pose).
///
/// `f64` (not `f32`) because the co-location epsilon can be centimeters while map
/// coordinates can be tens of meters from the origin, and the subtraction must not
/// lose precision at that dynamic range. The quaternion is assumed unit-normalized by
/// the producer; this type does not re-normalize (it has no business logic), it only
/// stores and compares.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Se3Pose {
    /// Translation X in meters, in the pose's frame.
    pub x: f64,
    /// Translation Y in meters, in the pose's frame.
    pub y: f64,
    /// Translation Z in meters, in the pose's frame.
    pub z: f64,
    /// Orientation quaternion scalar part `w`.
    pub qw: f64,
    /// Orientation quaternion `x`.
    pub qx: f64,
    /// Orientation quaternion `y`.
    pub qy: f64,
    /// Orientation quaternion `z`.
    pub qz: f64,
}

impl Se3Pose {
    /// Construct a pose from a translation and a quaternion. The quaternion should be
    /// unit-normalized by the caller; this constructor stores it verbatim.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64, qw: f64, qx: f64, qy: f64, qz: f64) -> Self {
        Self {
            x,
            y,
            z,
            qw,
            qx,
            qy,
            qz,
        }
    }

    /// A translation-only pose with the identity (no-rotation) quaternion
    /// `(w=1, x=y=z=0)` — convenient when only position matters, which is exactly the
    /// case for the co-location test.
    #[must_use]
    pub const fn at(x: f64, y: f64, z: f64) -> Self {
        Self::new(x, y, z, 1.0, 0.0, 0.0, 0.0)
    }

    /// Euclidean distance in meters between the two translations.
    ///
    /// This is the co-location metric the spatial tier thresholds against its
    /// embodiment epsilon. Orientation is intentionally NOT part of it: two events at
    /// the same floor point with different headings are the same place. Computed in
    /// `f64` so a centimeter epsilon stays meaningful even tens of meters from the
    /// frame origin.
    #[must_use]
    pub fn translation_distance(&self, other: &Se3Pose) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance is zero to itself, symmetric, and equals the straight-line gap on a
    /// single axis — so the co-location threshold means literal meters-apart.
    #[test]
    fn translation_distance_is_euclidean() {
        let a = Se3Pose::at(0.0, 0.0, 0.0);
        let b = Se3Pose::at(3.0, 4.0, 0.0);
        assert_eq!(a.translation_distance(&a), 0.0);
        assert_eq!(a.translation_distance(&b), 5.0);
        assert_eq!(b.translation_distance(&a), 5.0);
    }

    /// Orientation does NOT affect the co-location distance: two poses at the same
    /// point with different quaternions are distance zero apart.
    #[test]
    fn orientation_does_not_affect_distance() {
        let a = Se3Pose::new(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0);
        let b = Se3Pose::new(1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0);
        assert_eq!(a.translation_distance(&b), 0.0);
    }

    /// A pose round-trips through JSON unchanged, so the additive schema field is a
    /// safe wire/storage append.
    #[test]
    fn pose_round_trips_through_serde_json() {
        let p = Se3Pose::new(1.5, -2.0, 0.25, 0.5, 0.5, 0.5, 0.5);
        let back: Se3Pose = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(p, back);
    }
}
