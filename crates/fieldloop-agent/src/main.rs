//! The deployable on-robot sidecar: it drives the agent's store-and-forward sync on a
//! loop AND closes the capture->ingest loop by registering the uploaded rollouts with the
//! cloud gateway.
//!
//! Each pass does the real thing end to end:
//! 1. PUT every finished `.mcap` file's bytes into object storage through the real
//!    [`fieldloop_agent::S3Uploader`] (so a sync actually ships bytes, never discards
//!    them), and
//! 2. assemble the gateway's `POST /v1/ingest` body from the drained rollout metadata plus
//!    the blob pointers of the files the cloud just *confirmed* it received, and POST it to
//!    the real gateway over HTTP — so the rollouts are registered, not merely uploaded as
//!    opaque bytes.
//!
//! The real cloud transport (the AWS S3 SDK + the HTTP client) lives behind the optional
//! `s3` feature so the default build of this crate stays socket-free; production and the
//! live stack build with `--features s3`. Built WITHOUT that feature, `main` compiles but
//! refuses to run with a clear message rather than shipping a stub that discards bytes —
//! there is no fake uploader here, because a byte-discarding stub would falsely report
//! success while losing every recording.

#[cfg(feature = "s3")]
mod runner {
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::thread::sleep;
    use std::time::Duration;

    use fieldloop_agent::{
        Agent, DirSource, S3Uploader, UploadedBlob, build_ingest_request_json, sha256_hex,
    };
    use fieldloop_types::Rollout;

    /// Everything the deployable binary needs from the environment, resolved once at
    /// startup so a misconfiguration fails fast with a named variable instead of deep
    /// inside the first sync.
    ///
    /// The S3 credentials/bucket/endpoint themselves are read by [`S3Uploader::from_env`];
    /// this struct holds the additional config the binary needs to find recordings, find
    /// the drained rollout metadata, and reach the gateway.
    struct Config {
        /// Where the recorder writes finished `<class>-<seq>.mcap` files.
        data_dir: PathBuf,
        /// Where the persisted upload manifest lives so progress survives restarts.
        state_dir: PathBuf,
        /// Newline-delimited JSON file of drained [`Rollout`] records that capture flushed
        /// to disk; each line is one rollout in its canonical serde form. The agent reads
        /// these (rather than holding a live capture handle) because it is a *separate*
        /// sidecar process from the control loop that mints rollouts.
        rollouts_path: PathBuf,
        /// Base URL of the cloud ingest gateway, e.g. `https://ingest.fieldloop.example`.
        /// The binary POSTs to `<gateway_url>/v1/ingest`.
        gateway_url: String,
        /// Bearer token the gateway resolves to this robot's identity; the manifest the
        /// binary sends must claim the identity this token maps to or the gateway 403s.
        gateway_token: String,
        /// Tenant id placed in the ingest manifest; must match what `gateway_token`
        /// resolves to (the gateway cross-checks and 403s a mismatch).
        tenant_id: String,
        /// Robot id placed in the ingest manifest; cross-checked like `tenant_id`.
        robot_id: String,
    }

    impl Config {
        /// Resolve config from CLI args (`<data-dir> <state-dir>`, with deployment
        /// defaults) and the environment, returning a named-variable error for anything
        /// required-but-missing so a half-configured unit fails fast and legibly.
        fn from_args_and_env() -> Result<Config, String> {
            let mut args = env::args().skip(1);
            let data_dir = args.next().map_or_else(
                || PathBuf::from("/var/lib/fieldloop/recordings"),
                PathBuf::from,
            );
            let state_dir = args
                .next()
                .map_or_else(|| PathBuf::from("/var/lib/fieldloop/agent"), PathBuf::from);
            let rollouts_path = env::var("FIELDLOOP_ROLLOUTS_FILE")
                .map_or_else(|_| data_dir.join("rollouts.jsonl"), PathBuf::from);
            let gateway_url = env::var("FIELDLOOP_GATEWAY_URL")
                .map_err(|_| "FIELDLOOP_GATEWAY_URL is not set".to_string())?;
            let gateway_token = env::var("FIELDLOOP_GATEWAY_TOKEN")
                .map_err(|_| "FIELDLOOP_GATEWAY_TOKEN is not set".to_string())?;
            let tenant_id = env::var("FIELDLOOP_TENANT_ID")
                .map_err(|_| "FIELDLOOP_TENANT_ID is not set".to_string())?;
            let robot_id = env::var("FIELDLOOP_ROBOT_ID")
                .map_err(|_| "FIELDLOOP_ROBOT_ID is not set".to_string())?;
            Ok(Config {
                data_dir,
                state_dir,
                rollouts_path,
                gateway_url,
                gateway_token,
                tenant_id,
                robot_id,
            })
        }
    }

    /// Read the drained rollouts capture flushed to `path`, one canonical-JSON [`Rollout`]
    /// per non-empty line.
    ///
    /// A missing file yields an empty list (no rollouts to register yet is normal on a
    /// fresh robot), but a present-and-malformed line is a hard error: silently skipping a
    /// rollout would drop it from ingest, so the binary surfaces the parse failure instead
    /// of registering a partial batch.
    fn read_rollouts(path: &Path) -> Result<Vec<Rollout>, String> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("reading rollouts file {}: {e}", path.display())),
        };
        let mut rollouts = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let rollout: Rollout = serde_json::from_str(line).map_err(|e| {
                format!(
                    "parsing rollout on line {} of {}: {e}",
                    line_no + 1,
                    path.display()
                )
            })?;
            rollouts.push(rollout);
        }
        Ok(rollouts)
    }

    /// Build the gateway's blob claims from only the files the cloud *confirmed* it
    /// received, re-reading each from local disk to compute the checksum/size of the exact
    /// bytes that were uploaded.
    ///
    /// Pointers are built from `done_keys` (manifest status `Done`) rather than from every
    /// `.mcap` in the directory so a claim always describes an object the gateway can HEAD
    /// in the bucket: claiming a blob the uploader has not finished landing would make the
    /// gateway's verify step fail. A done file that has since been deleted locally is
    /// skipped with a warning rather than aborting the whole POST.
    fn confirmed_blobs(agent: &Agent<S3Uploader, DirSource>, data_dir: &Path) -> Vec<UploadedBlob> {
        let mut blobs = Vec::new();
        for key in agent.done_keys() {
            // The agent's upload key is the bare filename, so the local path is the data dir
            // joined with the key.
            let path = data_dir.join(&key);
            match fs::read(&path) {
                Ok(bytes) => blobs.push(UploadedBlob {
                    object_key: key,
                    content_sha256: sha256_hex(&bytes),
                    size: bytes.len() as u64,
                }),
                Err(e) => eprintln!(
                    "fieldloop-agent: confirmed file {} is no longer readable locally ({e}); \
                     skipping its blob claim",
                    path.display()
                ),
            }
        }
        blobs
    }

    /// Assemble and POST the `/v1/ingest` request to the real gateway, returning an error
    /// for any non-2xx response (or transport failure) so an unregistered batch is surfaced
    /// and retried on the next pass rather than silently dropped.
    ///
    /// This is the real network call: it builds the same body shape the gateway decodes
    /// (via [`build_ingest_request_json`]) and sends it with the bearer token the gateway
    /// resolves to this robot. It deliberately does NOT fabricate success when the gateway
    /// is unreachable — a connect/HTTP error becomes an `Err` the caller logs and retries.
    fn register_with_gateway(
        client: &reqwest::blocking::Client,
        cfg: &Config,
        rollouts: &[Rollout],
        blobs: &[UploadedBlob],
        batch_id: &str,
    ) -> Result<(), String> {
        let body =
            build_ingest_request_json(&cfg.tenant_id, &cfg.robot_id, batch_id, rollouts, blobs);
        let url = format!("{}/v1/ingest", cfg.gateway_url.trim_end_matches('/'));
        let response = client
            .post(&url)
            .bearer_auth(&cfg.gateway_token)
            .json(&body)
            .send()
            .map_err(|e| format!("POST {url} failed: {e}"))?;
        let status = response.status();
        if status.is_success() {
            Ok(())
        } else {
            // Surface the real gateway error (e.g. 403 identity mismatch, 422 blob
            // verification failure) so it is logged and the batch is retried, never
            // recorded as a fake success.
            let detail = response.text().unwrap_or_default();
            Err(format!("gateway returned {status}: {detail}"))
        }
    }

    /// Derive a deterministic batch id for the rollouts a pass is registering.
    ///
    /// The gateway treats `batch_id` as the idempotency key: a re-delivered batch with the
    /// same id is a no-op, so a retried POST must reuse it. Hashing the sorted rollout ids
    /// gives the same id for the same set of rollouts across retries (so a failed POST
    /// retried next pass is idempotent), and a different id once new rollouts appear.
    fn batch_id_for(rollouts: &[Rollout]) -> String {
        let mut ids: Vec<String> = rollouts.iter().map(|r| r.id.to_string()).collect();
        ids.sort();
        format!("batch-{}", sha256_hex(ids.join(",").as_bytes()))
    }

    /// Construct the real components and drive sync+register on a loop.
    pub fn run() -> Result<(), String> {
        let cfg = Config::from_args_and_env()?;

        // The REAL S3 uploader: a sync PUTs bytes to the configured bucket/endpoint. This
        // replaces the old byte-discarding stub, so a sync now actually ships bytes.
        let uploader = S3Uploader::from_env()?;
        // Treat a file as finished once it has been unchanged for 5 seconds, so an
        // in-progress recording is never uploaded mid-write.
        let source = DirSource::new(&cfg.data_dir, Duration::from_secs(5));
        let mut agent = Agent::new(uploader, source, &cfg.state_dir)
            .map_err(|e| format!("failed to start agent: {e}"))?;

        // One blocking HTTP client reused across passes (connection pooling).
        let client = reqwest::blocking::Client::builder()
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;

        // Drive sync+register on a loop. Each pass is independent and crash-safe: a SIGKILL
        // between passes resumes from the persisted manifest, and a re-POST of an
        // already-registered batch is an idempotent no-op on the gateway.
        loop {
            match agent.sync_once() {
                Ok(stats) => {
                    if stats.pending > 0 || stats.done > 0 {
                        eprintln!(
                            "fieldloop-agent: {} pending, {} done, {} bytes uploaded",
                            stats.pending, stats.done, stats.bytes_uploaded
                        );
                    }
                }
                Err(e) => eprintln!("fieldloop-agent: sync error (will retry): {e}"),
            }

            // After shipping bytes, register the drained rollouts with the gateway, pointing
            // each at the blobs the cloud confirmed. Only attempt a POST when there is
            // something to register, so an idle robot makes no empty calls.
            match read_rollouts(&cfg.rollouts_path) {
                Ok(rollouts) if !rollouts.is_empty() => {
                    let blobs = confirmed_blobs(&agent, &cfg.data_dir);
                    let batch_id = batch_id_for(&rollouts);
                    match register_with_gateway(&client, &cfg, &rollouts, &blobs, &batch_id) {
                        Ok(()) => eprintln!(
                            "fieldloop-agent: registered {} rollouts ({} blob pointers) as {}",
                            rollouts.len(),
                            blobs.len(),
                            batch_id
                        ),
                        Err(e) => {
                            eprintln!("fieldloop-agent: ingest POST failed (will retry): {e}");
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => eprintln!("fieldloop-agent: cannot read rollouts (will retry): {e}"),
            }

            sleep(Duration::from_secs(5));
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use fieldloop_types::{
            BootId, BoundedBlob, EpisodeId, MonoClock, PayloadRef, PolicyVersion, RobotId,
            RobotIdentity, Rollout, TenantId,
        };

        fn sample_rollout(step: u32) -> Rollout {
            let robot = RobotIdentity::new(TenantId::new("acme"), RobotId::new("r1"));
            let clock = MonoClock {
                boot_id: BootId::new(),
                mono_ns: 1,
                ts_wall_ns: 1,
            };
            Rollout::new(
                robot,
                EpisodeId::new(),
                step,
                clock,
                PolicyVersion::new("pol@v1+abc123def456"),
                "sha256:w".into(),
                "arm6dof".into(),
                "pick".into(),
                PayloadRef::none(),
                PayloadRef::none(),
                BoundedBlob::empty(),
                1,
            )
        }

        /// The binary's rollout reader round-trips the canonical `Rollout` serde form the
        /// gateway also decodes: capture flushes one JSON line per rollout, the binary
        /// reads them back, and a missing file is an empty (not failing) read.
        #[test]
        fn reads_drained_rollouts_from_jsonl() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rollouts.jsonl");

            // Missing file: empty, not an error (a fresh robot has none yet).
            assert!(read_rollouts(&path).unwrap().is_empty());

            // Write two rollouts as the binary expects capture to flush them, with a blank
            // line that must be tolerated.
            let r0 = sample_rollout(0);
            let r1 = sample_rollout(1);
            let lines = format!(
                "{}\n\n{}\n",
                serde_json::to_string(&r0).unwrap(),
                serde_json::to_string(&r1).unwrap()
            );
            std::fs::write(&path, lines).unwrap();

            let read = read_rollouts(&path).unwrap();
            assert_eq!(
                read,
                vec![r0, r1],
                "reader must round-trip the drained rollouts"
            );
        }

        /// A malformed line is a hard error (never a silently-dropped rollout), naming the
        /// line so a misconfigured flush is debuggable.
        #[test]
        fn malformed_rollout_line_is_an_error() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rollouts.jsonl");
            std::fs::write(&path, "{not valid json}\n").unwrap();
            let err = read_rollouts(&path).unwrap_err();
            assert!(
                err.contains("line 1"),
                "error must name the offending line: {err}"
            );
        }

        /// The binary's wiring assembles a well-formed `/v1/ingest` body from a drained
        /// batch + confirmed blob pointers: the same `build_ingest_request_json` the gateway
        /// decodes, carrying the manifest identity, one rollout per input, and the blob
        /// claim attached as each rollout's observation_ref — proving the POST body the
        /// deployable binary would send is the real wire shape, not a hand-built fake. The
        /// actual PUT/POST round-trip needs the live MinIO+gateway stack to execute.
        #[test]
        fn wiring_builds_well_formed_ingest_body() {
            let rollouts = vec![sample_rollout(0), sample_rollout(1)];
            let blobs = vec![UploadedBlob {
                object_key: "sensor-0.mcap".into(),
                content_sha256: sha256_hex(b"bytes"),
                size: 5,
            }];
            let batch_id = batch_id_for(&rollouts);
            let body = build_ingest_request_json("acme", "r1", &batch_id, &rollouts, &blobs);

            assert_eq!(body["manifest"]["tenant_id"], serde_json::json!("acme"));
            assert_eq!(body["manifest"]["robot_id"], serde_json::json!("r1"));
            assert_eq!(body["manifest"]["batch_id"], serde_json::json!(batch_id));
            assert_eq!(body["rollouts"].as_array().unwrap().len(), 2);
            assert_eq!(body["blobs"].as_array().unwrap().len(), 1);
            assert_eq!(
                body["rollouts"][0]["observation_ref"]["object_key"],
                serde_json::json!("sensor-0.mcap")
            );
        }

        /// The batch id is stable for the same set of rollouts (so a retried POST reuses
        /// the gateway's idempotency key) and changes once the rollout set changes.
        #[test]
        fn batch_id_is_stable_per_rollout_set() {
            let a = vec![sample_rollout(0), sample_rollout(1)];
            // Same rollouts in the other order must yield the same id (ids are sorted).
            let a_rev = vec![a[1].clone(), a[0].clone()];
            assert_eq!(batch_id_for(&a), batch_id_for(&a_rev));

            let b = vec![sample_rollout(0)];
            assert_ne!(
                batch_id_for(&a),
                batch_id_for(&b),
                "a different rollout set must get a different batch id"
            );
        }
    }
}

#[cfg(feature = "s3")]
fn main() {
    if let Err(e) = runner::run() {
        eprintln!("fieldloop-agent: {e}");
        std::process::exit(1);
    }
}

/// Built without the `s3` feature there is no real cloud transport compiled in, so the
/// binary refuses to run rather than ship a stub that would discard the recorded bytes and
/// falsely report success. Building this way is still useful (it type-checks the crate and
/// keeps the default build socket-free); deploying requires `--features s3`.
#[cfg(not(feature = "s3"))]
fn main() {
    eprintln!(
        "fieldloop-agent: built without the `s3` feature, so no real uploader/gateway client \
         is compiled in. Rebuild with `--features s3` to deploy. Refusing to run a stub that \
         would discard recordings."
    );
    std::process::exit(1);
}
