# Changelog

## 0.2.0 — 2026-06-12

First PyPI release: `pip install fieldloop`. abi3 wheels for Linux x86_64/aarch64, macOS
arm64, and Windows; sdist for everything else. The wheel runs `fieldloop demo` with no Rust
toolchain.

### Added

- **MCAP import.** `fieldloop attribute run.mcap --map mapping.toml` reads a recorded MCAP
  plus a topic-mapping TOML (which topics are decisions, which are outcomes, which clock to
  use) and attributes the file end to end. The reader is validated on a real rosbag2 file
  (`libmcap`, ROS 2). Single-boot per file.
- **`fieldloop doctor`.** Checks a mapping against a file's actual topics and flags any topic
  whose source clock diverges from the recorder's (a sensor on a different time base). Exits
  non-zero on a problem, so it can gate a pipeline.
- **`fieldloop` CLI.** `demo`, `init`, `attribute`, `doctor`, `curate`, `view`,
  `import-lerobot`. Scriptable exit codes (0 ok / 1 engine rejected the inputs / 2 bad file).
- **`fieldloop import-lerobot`.** Turns a recorded LeRobot dataset into rollouts and
  outcomes, validated end to end through the real loader on a public Hub dataset.
- **Optional extras.** `fieldloop[viz]` adds `fieldloop view` (Rerun timeline of attributed
  incidents) and `fieldloop demo --viz`. `fieldloop[lerobot]` adds the LeRobot loader.
- **Bundled showcase recording.** `fieldloop demo` attributes a ~3,400-message MCAP with a
  takeover, two e-stops (one splitting credit across a near-simultaneous pair), and a
  collision correctly left unattributed because it fell outside its window.
- **Attribution gauntlet** (`bench/`): a synthetic benchmark scoring the engine against a
  naive nearest-temporal baseline on confounded scenarios. Numbers are measured, never typed.

### Fixed

- Join dedup key pinned to FNV-1a (stable across Rust releases); credit conserved exactly;
  the safety gate now keys on join method, not a float-equality on confidence.
- Recorder fsyncs on rotation and bounds the safety channel instead of growing unboundedly.
- Agent batches are checkpointed (crash-safe, idempotent retries) and require a valid MCAP
  end-magic before declaring a file finished.
- Ingest suppresses the duplicate landed event on a write-time duplicate.

### Notes

- Confidence is a scored recency/coverage signal, not a field-calibrated probability. A
  calibration curve fits from your own confirmed labels once enough exist.
- FieldLoop does not claim automatic root cause. It builds an evidence package — method,
  confidence, and the bound decision — for a human to confirm.
