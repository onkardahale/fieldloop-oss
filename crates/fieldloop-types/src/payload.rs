//! Payload pointers & bounded inline blobs.
//!
//! Sensor payloads (MCAP/video/point-cloud/tensors) stay in the customer's own
//! object store and never enter Fieldloop's metadata plane — the customer owns
//! their bytes, and moving terabytes of sensor data through Fieldloop would be both
//! a cost and a trust problem. So this crate's rows carry **pointers**
//! ([`PayloadRef`]), never bytes. The few small inline free-form fields that the
//! open SDK can set ([`BoundedBlob`]) are byte-bounded and re-validated at the
//! gateway, because any field a robot can fill is hostile until checked. These two
//! types make that distinction explicit in the type system.

use serde::{Deserialize, Serialize};

/// A pointer to bytes living in the customer's own object store.
///
/// Never the bytes themselves — the customer keeps their sensor data; Fieldloop
/// only indexes pointers to it. For MCAP this is a segment uuid + a message range;
/// modeled here as an opaque object key plus an optional byte/message range. The
/// metadata plane indexes these; byte-touching work runs on customer-VPC workers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadRef {
    /// Object key in the customer bucket (chosen by the gateway, not the robot, so
    /// a robot cannot dictate where bytes land). Empty string == "no payload
    /// referenced" (e.g. an action with no separate blob).
    pub object_key: String,
    /// Optional `[start, end)` range within the referenced object (e.g. MCAP
    /// message range or byte range). `None` means the whole object.
    pub range: Option<ByteRange>,
    /// SHA-256 of the referenced bytes, when known. Uploads are checksum-required
    /// so a consumer can verify integrity before decoding the bytes in a sandbox.
    pub content_sha256: Option<String>,
}

impl PayloadRef {
    /// An empty pointer (no referenced payload).
    #[must_use]
    pub fn none() -> Self {
        Self {
            object_key: String::new(),
            range: None,
            content_sha256: None,
        }
    }

    /// True iff this references no object.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.object_key.is_empty()
    }
}

/// A half-open `[start, end)` range within a referenced object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

/// A small inline free-form blob carried in the metadata plane.
///
/// Used for `context`, outcome `payload`, etc. — fields the open SDK can set, and
/// therefore hostile until validated. They are byte-bounded at the SDK and
/// re-validated at the ingest gateway so a robot can't blow up the metadata plane
/// with an oversized or malformed field. We keep the raw bytes plus the
/// schema_version they claim; this types crate does not enforce the bound (it has
/// no I/O), but it names the field as bounded so every consumer knows it must
/// validate length and charset before trusting it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundedBlob {
    /// The raw inline bytes (typically small JSON). MUST be length-checked
    /// against the per-`(tenant,robot)` byte budget at the boundary.
    pub bytes: Vec<u8>,
    /// Declared schema version of the blob contents. This is itself an open-SDK
    /// field, so it is untrusted until validated.
    pub schema_version: Option<String>,
}

impl BoundedBlob {
    /// An empty blob.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            schema_version: None,
        }
    }

    /// Byte length — what the boundary must check against the budget.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True iff the blob carries no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}
