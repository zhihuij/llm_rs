pub const I32_SIZE: usize = std::mem::size_of::<i32>();
pub const F32_SIZE: usize = std::mem::size_of::<f32>();

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const GGML_MAX_DIMS: usize = 4;
pub const GGUF_DEFAULT_ALIGNMENT: u32 = 32;
