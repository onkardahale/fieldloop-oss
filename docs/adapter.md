# Onboarding a robot type

A new robot type ("embodiment") is onboarded through configuration, not code changes. Two TOML
sections, both validated at load (an illegal value fails at startup, never mid-request).

## Attribution windows

How far back each outcome kind may bind to a decision is tuned per robot type. A collision must be
clock-safe (it can only bind when the rollout and outcome share a boot and are comparable on the
monotonic clock), so its window requires that flag.

```toml
[embodiments.warehouse_pick_arm]
action_space = "joint_position"

[embodiments.warehouse_pick_arm.attribution.collision]
window_ms = 150
requires_monotonic_colocation = true

[embodiments.warehouse_pick_arm.attribution.teleop_takeover]
window_ms = 5000
requires_monotonic_colocation = false
```

A pair with no declared window returns "not configured" rather than a silent default — the caller
decides what to do, instead of inheriting a window never meant for it.

## Custom failure labels

Your team's failures are domain-specific (`failed_grasp`, `bad_dock_alignment`,
`barcode_scan_failed`), but analytics needs a controlled vocabulary. A label **binds** a custom name
to a real outcome kind and failure class from the closed taxonomy, plus an optional task, severity,
and window override:

```toml
[labels.failed_grasp]
outcome_kind  = "downstream_failure"   # one of the fixed kinds
failure_class = "manipulation"         # one of the fixed classes
task          = "pick_can"
severity      = "standard"             # standard | critical
attribution_window_ms = 2000
```

The custom label is then queryable as itself *and* rolls up into `manipulation`. Binding to a kind or
class outside the taxonomy is a load error — the controlled vocabulary cannot be polluted by a free
string. The label name and its mapping are open; the taxonomy it maps into is fixed.

## Metrics

Custom scored metrics (a boolean outcome, a float return with an optimization direction) are declared
the same way under `[metrics.<name>]`.

See `Config::example()` for a complete, valid file, and `docs/adr/0001-tenant-query-builder.md` for
the validate-once-at-load model.
