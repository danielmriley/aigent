pub mod events;

use chrono::Utc;

use crate::events::AuditEvent;

pub fn memory_event(details: impl Into<String>) -> AuditEvent {
    AuditEvent {
        kind: "memory".to_string(),
        details: details.into(),
        timestamp: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_event_kind_and_details() {
        let evt = memory_event("belief recorded");
        assert_eq!(evt.kind, "memory");
        assert_eq!(evt.details, "belief recorded");
    }

    #[test]
    fn memory_event_timestamp_is_recent() {
        let before = Utc::now();
        let evt = memory_event("test");
        let after = Utc::now();
        assert!(evt.timestamp >= before && evt.timestamp <= after);
    }

    #[test]
    fn memory_event_from_string_owned() {
        let detail = String::from("owned string");
        let evt = memory_event(detail);
        assert_eq!(evt.details, "owned string");
    }

    #[test]
    fn audit_event_serde_roundtrip() {
        let evt = memory_event("roundtrip test");
        let json = serde_json::to_string(&evt).unwrap();
        let back: events::AuditEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind, evt.kind);
        assert_eq!(back.details, evt.details);
        assert_eq!(back.timestamp, evt.timestamp);
    }
}
