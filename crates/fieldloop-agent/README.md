# fieldloop-agent

On-robot **sidecar agent** that watches *finished* recorded files and ships them
to the cloud over a flaky network with **store-and-forward, resumable,
Safety-first** uploads that **never drop a file**.

The agent owns the network so the control loop and recorder never touch
upload/retry. It is resilient to disconnects and restarts.

## How it works

- **`Uploader` trait** abstracts the cloud (real S3/HTTP comes later). It is
  resumable: `upload(key, data, start_offset)` returns `Done`,
  `Interrupted { uploaded_to }`, or `Failed`.
- **`FileSource` trait** yields only *finished* files. `DirSource` scans a
  directory for `*.mcap`, parses the class from the `<class>-<seq>.mcap` prefix,
  and treats a file as finished only once it has been stable for a quiet period —
  so a file still being written is skipped.
- **`Agent::sync_once`** discovers finished files, skips ones already `Done`,
  sorts Safety → Meta → Sensor, uploads each pending file from its recorded
  offset, and persists a JSON manifest after every file. A crash mid-sync resumes
  exactly where it left off; a `Done` file is never re-uploaded.

## Running as a service

Build the binary and run it on a loop via the sample systemd unit
(`fieldloop-agent.service`, `Type=simple`):

```
cargo build --release -p fieldloop-agent
sudo cp target/release/fieldloop-agent /usr/local/bin/
sudo cp crates/fieldloop-agent/fieldloop-agent.service /etc/systemd/system/
sudo systemctl enable --now fieldloop-agent
```

The binary calls `sync_once()` on a loop, sleeping between passes. Because every
pass is crash-safe (progress lives in the persisted manifest), `Restart=always`
can kill and restart the service at any moment without losing or duplicating an
upload. The bundled uploader in `main.rs` is a placeholder that accepts
everything; swap in the real S3/HTTP `Uploader` implementation when it lands.
