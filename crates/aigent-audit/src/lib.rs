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
