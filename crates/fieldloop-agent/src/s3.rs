//! A real S3-compatible [`Uploader`] that ships the agent's bytes to an object store.
//!
//! This is the production destination for store-and-forward: the agent reads each
//! finished `.mcap` file and calls [`Uploader::upload`], and this impl PUTs those exact
//! bytes to a real bucket/key over the S3 API. It works against AWS S3 and against any
//! S3-compatible server (MinIO, Ceph, ...) because it lets the caller override the
//! endpoint URL and force path-style addressing, which is what MinIO needs.
//!
//! ## Why `aws-sdk-s3`
//! The official AWS SDK is chosen over a lighter third-party crate (`rust-s3`/`minio`)
//! because it is the same client a production cloud deployment will already use for IAM,
//! credentials, retries, and signing — depending on it here means the bytes the e2e test
//! delivers travel the exact code path production uses, with no second S3 implementation
//! to keep in sync. It is heavier, which is exactly why it sits behind the optional `s3`
//! feature: the default agent build (and the offline closed-loop gate) never compiles it
//! and never pulls the AWS crates.
//!
//! ## Why this blocks on an async client from a sync trait
//! [`Uploader::upload`] is synchronous (the agent's sync loop is a plain method), but the
//! AWS SDK is async. This impl owns its own small Tokio runtime and drives each PUT to
//! completion with `block_on`, so a synchronous caller gets a finished upload without the
//! whole agent having to become async. Each `upload` is a single bounded PUT, so blocking
//! the calling thread for its duration is acceptable for the sidecar.

use crate::{UploadOutcome, Uploader};

use aws_sdk_s3::config::{Credentials, Region};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::{Client, Config};

/// An [`Uploader`] that PUTs bytes to an S3-compatible bucket.
///
/// Holds the configured client and bucket name so every `upload` targets the same
/// destination. The synchronous [`Uploader`] trait drives the async AWS client by
/// building a throwaway runtime on a worker thread per PUT (see `upload`), so the
/// uploader stores no runtime and is safe to drop from any context — including inside an
/// async test or handler, where dropping a stored runtime would panic.
#[derive(Debug)]
pub struct S3Uploader {
    /// The configured S3 client (endpoint, region, credentials, path-style).
    client: Client,
    /// The bucket every object is PUT into.
    bucket: String,
}

impl S3Uploader {
    /// Build an uploader against an S3-compatible endpoint with explicit credentials.
    ///
    /// `endpoint_url` points the client at the server (e.g. `http://localhost:9000` for a
    /// local MinIO); when it is `Some`, path-style addressing is forced because MinIO and
    /// most non-AWS servers do not serve the `<bucket>.<host>` virtual-host form. `region`
    /// is required by the SDK's signer even for a non-AWS server, so a placeholder like
    /// `us-east-1` is fine. Credentials are passed explicitly rather than read from the
    /// environment so the agent's identity is unambiguous.
    ///
    /// # Errors
    /// Infallible today (client construction opens no socket); returns `Result` so the
    /// signature is stable if a future credential/endpoint validation can fail.
    pub fn new(
        endpoint_url: Option<&str>,
        region: &str,
        access_key: &str,
        secret_key: &str,
        bucket: impl Into<String>,
    ) -> Result<S3Uploader, String> {
        let creds = Credentials::new(access_key, secret_key, None, None, "fieldloop-agent");
        let mut builder = Config::builder()
            .region(Region::new(region.to_string()))
            .credentials_provider(creds)
            // Force path-style: a MinIO/Ceph server addresses objects as
            // `<endpoint>/<bucket>/<key>`, not the AWS virtual-host `<bucket>.<host>`,
            // so without this a non-AWS endpoint would 404/connect-fail.
            .force_path_style(true)
            .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest());
        if let Some(url) = endpoint_url {
            builder = builder.endpoint_url(url);
        }
        let client = Client::from_conf(builder.build());

        Ok(S3Uploader {
            client,
            bucket: bucket.into(),
        })
    }

    /// Build an uploader from environment variables, the way a deployed sidecar is
    /// configured: `S3_ENDPOINT` (optional; set for MinIO, unset for real AWS),
    /// `AWS_REGION` (defaults to `us-east-1`), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    /// and `S3_BUCKET`. Reading from the environment keeps credentials out of the binary
    /// and lets the live test point the same code at a throwaway MinIO.
    ///
    /// # Errors
    /// Returns an error string if `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, or
    /// `S3_BUCKET` is missing, or the runtime cannot be built.
    pub fn from_env() -> Result<S3Uploader, String> {
        let endpoint = std::env::var("S3_ENDPOINT").ok();
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        let access_key = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| "AWS_ACCESS_KEY_ID is not set".to_string())?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| "AWS_SECRET_ACCESS_KEY is not set".to_string())?;
        let bucket = std::env::var("S3_BUCKET").map_err(|_| "S3_BUCKET is not set".to_string())?;
        S3Uploader::new(
            endpoint.as_deref(),
            &region,
            &access_key,
            &secret_key,
            bucket,
        )
    }
}

impl Uploader for S3Uploader {
    fn upload(&self, key: &str, data: &[u8], start_offset: u64) -> UploadOutcome {
        // The agent passes the whole file plus a resume offset; PUT the remaining slice so
        // the stored object is the complete file. A single PUT writes the whole object, so
        // a successful PUT either completes the upload or (on error) makes no progress —
        // there is no partial-object state to resume into, hence only Done/Failed below.
        let slice = &data[start_offset as usize..];
        let body = ByteStream::from(slice.to_vec());
        // Run the PUT on a separate OS thread that owns the runtime. A plain
        // `runtime.block_on` panics if this method is itself called from within another
        // Tokio runtime (e.g. the gateway's async handler driving the uploader), because a
        // runtime cannot be entered re-entrantly. A dedicated thread has no ambient
        // runtime, so `block_on` there is always safe.
        // Collapse the SDK's large error type to a bare bool INSIDE the async block, before
        // it crosses the thread/closure boundary, so the value carried out is tiny.
        let ok = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    // Build the runtime here, on this dedicated thread, rather than storing
                    // one on the struct: a stored runtime would be dropped wherever the
                    // uploader is dropped (e.g. inside an async test/handler), and dropping
                    // a runtime from an async context panics. Built and dropped here, it
                    // never touches the caller's async context.
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("build per-PUT tokio runtime");
                    runtime.block_on(async {
                        self.client
                            .put_object()
                            .bucket(&self.bucket)
                            // Have S3/MinIO compute and STORE the object's SHA-256 so the
                            // gateway can read it back on HEAD and confirm the bytes match
                            // the agent's claim. Without this the server stores no checksum
                            // and the gateway could only check size — letting same-size but
                            // different bytes register as a valid pointer. The SDK computes
                            // the digest from the body and the server persists it as
                            // `x-amz-checksum-sha256`.
                            .checksum_algorithm(aws_sdk_s3::types::ChecksumAlgorithm::Sha256)
                            .key(key)
                            .body(body)
                            .send()
                            .await
                            .is_ok()
                    })
                })
                .join()
                .expect("S3 PUT thread panicked")
        });
        if ok {
            // The whole object landed: the agent can mark the file done.
            UploadOutcome::Done
        } else {
            // Any transport/credential/bucket error means no progress was made; report
            // Failed so the agent keeps the file pending and retries on a later sync,
            // never dropping it.
            UploadOutcome::Failed
        }
    }
}
