//! Python binding over [`fieldloop_import::diagnose`] — the `fieldloop doctor` checks.
//!
//! Thin like the rest of this crate: it parses the mapping, runs the file-import doctor,
//! and projects the [`Diagnosis`] onto a plain dict. The checks (mapping coverage, clock
//! sanity) live in `fieldloop-import`, never here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use fieldloop_import::{
    DEFAULT_CLOCK_SKEW_THRESHOLD_NS, DiagnosedRole, Diagnosis, MappingConfig, diagnose,
};

/// Check an MCAP file against a topic-mapping before attribution.
///
/// `mcap_bytes` is the raw bytes of an MCAP recording; `mapping_toml` is the topic-mapping
/// TOML. Returns a dict: `ok` (no problems), `clock_skew_threshold_ns`, `missing_topics`
/// (declared in the mapping but absent from the file), `skewed_topics` (source clock
/// diverges from the recorder clock past the threshold), and `topics` (one entry per topic
/// in the file: `topic`, `role`, `message_count`, `log_time_min_ns`, `log_time_max_ns`,
/// `max_clock_skew_ns`). `clock_skew_threshold_ns` overrides the 1-second default. Raises
/// `ValueError` on an invalid mapping or an undecodable MCAP.
#[pyfunction]
#[pyo3(signature = (mcap_bytes, mapping_toml, *, clock_skew_threshold_ns=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn doctor<'py>(
    py: Python<'py>,
    mcap_bytes: &[u8],
    mapping_toml: &str,
    clock_skew_threshold_ns: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    let mapping = MappingConfig::from_toml(mapping_toml)
        .map_err(|e| PyValueError::new_err(format!("invalid mapping: {e}")))?;
    let diagnosis = diagnose(
        mcap_bytes,
        &mapping,
        clock_skew_threshold_ns.unwrap_or(DEFAULT_CLOCK_SKEW_THRESHOLD_NS),
    )
    .map_err(|e| PyValueError::new_err(format!("could not read MCAP: {e}")))?;
    diagnosis_to_pydict(py, &diagnosis)
}

/// Snake_case spelling of a [`DiagnosedRole`], so Python callers read a plain string rather
/// than an opaque variant.
fn role_str(role: DiagnosedRole) -> &'static str {
    match role {
        DiagnosedRole::Decision => "decision",
        DiagnosedRole::Outcome => "outcome",
        DiagnosedRole::Unmapped => "unmapped",
    }
}

/// Project a [`Diagnosis`] onto the flat dict the CLI and Python callers read.
fn diagnosis_to_pydict<'py>(py: Python<'py>, dx: &Diagnosis) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("ok", !dx.has_problems())?;
    result.set_item("clock_skew_threshold_ns", dx.clock_skew_threshold_ns)?;
    result.set_item("missing_topics", dx.missing_topics.clone())?;
    result.set_item("skewed_topics", dx.skewed_topics.clone())?;

    let topics = PyList::empty(py);
    for t in &dx.topics {
        let td = PyDict::new(py);
        td.set_item("topic", &t.topic)?;
        td.set_item("role", role_str(t.role))?;
        td.set_item("message_count", t.message_count)?;
        td.set_item("log_time_min_ns", t.log_time_min_ns)?;
        td.set_item("log_time_max_ns", t.log_time_max_ns)?;
        td.set_item("max_clock_skew_ns", t.max_clock_skew_ns)?;
        topics.append(td)?;
    }
    result.set_item("topics", topics)?;
    Ok(result)
}
