//! Shared test helpers for the `fieldloop-import` crate. Compiled only under `cfg(test)`
//! so none of this reaches production or the public API surface.

use std::collections::BTreeMap;

/// Write an in-memory MCAP with one channel per distinct topic and one message per entry.
///
/// The `messages` slice is `(topic, log_time, publish_time, data)`. Using separate
/// `log_time` and `publish_time` lets clock-skew tests set them independently; callers
/// that do not care about skew pass `publish_time = log_time`. No schema is declared
/// (encoding is `json`) because the import and doctor logic never inspects message
/// payloads — timestamps and the channel's topic name are the only fields it reads.
pub(crate) fn write_mcap(messages: &[(&str, u64, u64, &[u8])]) -> Vec<u8> {
    let mut writer = mcap::WriteOptions::new()
        .library("fieldloop-import-test")
        .create(std::io::Cursor::new(Vec::new()))
        .expect("create writer");
    let mut channels: BTreeMap<&str, u16> = BTreeMap::new();
    for (seq, (topic, log_time, publish_time, data)) in messages.iter().enumerate() {
        let channel_id = *channels.entry(topic).or_insert_with(|| {
            writer
                .add_channel(0, topic, "json", &BTreeMap::new())
                .expect("add channel")
        });
        let header = mcap::records::MessageHeader {
            channel_id,
            sequence: u32::try_from(seq).unwrap(),
            log_time: *log_time,
            publish_time: *publish_time,
        };
        writer
            .write_to_known_channel(&header, data)
            .expect("write message");
    }
    writer.finish().expect("finish");
    writer.into_inner().into_inner()
}
