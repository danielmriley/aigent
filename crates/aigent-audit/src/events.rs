use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub kind: String,
    pub details: String,
    pub timestamp: DateTime<Utc>,
}
