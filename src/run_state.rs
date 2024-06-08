use std::io::Result;

use crate::model::ModelConfig;

#[derive(Debug)]
pub struct RunState {
    // current wave of activations
    pub x: Vec<f32>,   // activation at current time stamp (dim,)
    pub xb: Vec<f32>,  // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>, // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,  // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,   // query (dim,)
    // pub k: Vec<f32>,      // key (dim,)
    // pub v: Vec<f32>,      // value (dim,)
    // pub att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<f32>, // output logits
    // kv cache
    pub key_cache: Vec<f32>,   // (layer, seq_len, dim)
    pub value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    // Constructor: Open or create a file for message store.
    pub fn new(model_config: &ModelConfig) -> Result<Self> {
        let kv_dim = (model_config.dim * model_config.n_kv_heads) / model_config.n_heads;

        let mut x: Vec<f32> = Vec::new();
        x.resize_with(model_config.dim, || 0.0);
        let mut xb: Vec<f32> = Vec::new();
        xb.resize_with(model_config.dim, || 0.0);
        let mut xb2: Vec<f32> = Vec::new();
        xb2.resize_with(model_config.dim, || 0.0);
        let mut hb: Vec<f32> = Vec::new();
        hb.resize_with(model_config.hidden_dim, || 0.0);
        let mut hb2: Vec<f32> = Vec::new();
        hb2.resize_with(model_config.hidden_dim, || 0.0);

        let mut q: Vec<f32> = Vec::new();
        q.resize_with(model_config.dim, || 0.0);
        // let mut k: Vec<f32> = Vec::new();
        // k.resize_with(model_config.dim, || 0.0);
        // let mut v: Vec<f32> = Vec::new();
        // v.resize_with(model_config.dim, || 0.0);

        let mut key_cache: Vec<f32> = Vec::new();
        key_cache.resize_with(
            model_config.n_layers * model_config.seq_len * kv_dim,
            || 0.0,
        );
        let mut value_cache: Vec<f32> = Vec::new();
        value_cache.resize_with(
            model_config.n_layers * model_config.seq_len * kv_dim,
            || 0.0,
        );
        // let mut att: Vec<f32> = Vec::new();
        // att.resize_with(model_config.n_heads * model_config.seq_len, || 0.0);
        let mut logits: Vec<f32> = Vec::new();
        logits.resize_with(model_config.vocab_size, || 0.0);

        Ok(RunState {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            // k,
            // v,
            // att,
            logits,
            key_cache,
            value_cache,
        })
    }
}
