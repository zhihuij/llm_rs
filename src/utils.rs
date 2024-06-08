use std::time::{SystemTime, UNIX_EPOCH};

pub fn time_in_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("get time error")
        .as_millis()
}
