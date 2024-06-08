use std::fs::OpenOptions;
use std::io::Result;

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use rayon::prelude::*;

use crate::{
    constants::{F32_SIZE, I32_SIZE},
    run_state::RunState,
};

#[derive(Debug)]
pub struct ModelConfig {
    pub dim: usize,        // transformer dimension
    pub hidden_dim: usize, // for ffn layers
    pub n_layers: usize,   // number of layers
    pub n_heads: usize,    // number of query heads
    pub n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize,    // max sequence length
}

impl ModelConfig {
    // Constructor: Open or create a file for message store.
    pub fn new(mapped_file: &Mmap) -> Result<Self> {
        // Read the values from the array at the specified positions
        let mut offset = 0;
        let dim = LittleEndian::read_i32(&mapped_file[offset..offset + 4]) as usize;
        offset += 4;
        let hidden_dim = LittleEndian::read_i32(&mapped_file[offset..offset + 4]) as usize;
        offset += 4;
        let n_layers = LittleEndian::read_i32(&mapped_file[offset..offset + 4]) as usize;
        offset += 4;
        let n_heads = LittleEndian::read_i32(&mapped_file[offset..offset + 4]) as usize;
        offset += 4;
        let n_kv_heads = LittleEndian::read_i32(&mapped_file[offset..offset + 4]) as usize;
        offset += 4;
        let vocab_size = LittleEndian::read_i32(&mapped_file[offset..offset + 4]) as usize;
        offset += 4;
        let seq_len = LittleEndian::read_i32(&mapped_file[offset..offset + 4]) as usize;

        Ok(ModelConfig {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        })
    }
}

#[derive(Debug)]
pub struct ModelWeights {
    // token embedding table
    pub token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: Vec<f32>, // (layer, dim, n_heads * head_size)
    pub wk: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    pub wv: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    pub wo: Vec<f32>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: Vec<f32>, // (layer, hidden_dim, dim)
    pub w2: Vec<f32>, // (layer, dim, hidden_dim)
    pub w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<f32>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Option<Vec<f32>>,
}

impl ModelWeights {
    // Constructor: Open or create a file for message store.
    pub fn new(
        mapped_file: &Mmap,
        model_config: &ModelConfig,
        shared_weights: i32,
    ) -> Result<Self> {
        let head_size = model_config.dim / model_config.n_heads;
        let mut offset = I32_SIZE * 7; // ModeConfig has 7 field
        let token_embedding_table = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.vocab_size * model_config.dim,
            offset,
        );
        offset += token_embedding_table.len() * F32_SIZE;
        let rms_att_weight = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.dim,
            offset,
        );
        offset += rms_att_weight.len() * F32_SIZE;
        let wq = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.dim * (model_config.n_heads * head_size),
            offset,
        );
        offset += wq.len() * F32_SIZE;
        let wk = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.dim * (model_config.n_kv_heads * head_size),
            offset,
        );
        offset += wk.len() * F32_SIZE;
        let wv = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.dim * (model_config.n_kv_heads * head_size),
            offset,
        );
        offset += wv.len() * F32_SIZE;
        let wo = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * (model_config.n_heads * head_size) * model_config.dim,
            offset,
        );
        offset += wo.len() * F32_SIZE;
        let rms_ffn_weight = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.dim,
            offset,
        );
        offset += rms_ffn_weight.len() * F32_SIZE;
        let w1 = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.dim * model_config.hidden_dim,
            offset,
        );
        offset += w1.len() * F32_SIZE;
        let w2 = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.hidden_dim * model_config.dim,
            offset,
        );
        offset += w2.len() * F32_SIZE;
        let w3 = ModelWeights::read_f32_vec(
            mapped_file,
            model_config.n_layers * model_config.dim * model_config.hidden_dim,
            offset,
        );
        offset += w3.len() * F32_SIZE;
        let rms_final_weight = ModelWeights::read_f32_vec(mapped_file, model_config.dim, offset);
        offset += rms_final_weight.len() * F32_SIZE;
        offset += model_config.seq_len * head_size / 2 * F32_SIZE; // skip what used to be freq_cis_real (for RoPE)
        offset += model_config.seq_len * head_size / 2 * F32_SIZE; // skip what used to be freq_cis_imag (for RoPE)

        let wcls = if shared_weights > 0 {
            None
        } else {
            Some(ModelWeights::read_f32_vec(
                mapped_file,
                model_config.vocab_size * model_config.dim,
                offset,
            ))
        };

        Ok(ModelWeights {
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        })
    }

    fn read_f32_vec(buf: &[u8], vec_size: usize, offset: usize) -> Vec<f32> {
        let mut result_vec: Vec<f32> = Vec::with_capacity(vec_size);
        for i in 0..vec_size {
            result_vec.push(LittleEndian::read_f32(
                &buf[(i * F32_SIZE) + offset..((i + 1) * F32_SIZE) + offset],
            ));
        }
        result_vec
    }
}

#[derive(Debug)]
pub struct Transformer {
    pub config: ModelConfig, // the hyperparameters of the architecture (the blueprint)
    pub weights: ModelWeights, // the weights of the model
    pub state: RunState,     // buffers for the "wave" of activations in the forward pass
}

impl Transformer {
    pub fn new(checkpoint: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(checkpoint)
            .expect(&format!("Couldn't open file {}\n", checkpoint));
        let mapped_file = unsafe { Mmap::map(&file).unwrap() };

        let model_config = ModelConfig::new(&mapped_file).unwrap();

        let shared_weights = if model_config.vocab_size > 0 { 1 } else { 0 };

        let model_weights = ModelWeights::new(&mapped_file, &model_config, shared_weights).unwrap();
        let run_state = RunState::new(&model_config).unwrap();

        Ok(Transformer {
            config: model_config,
            weights: model_weights,
            state: run_state,
        })
    }

    // ----------------------------------------------------------------------------
    // neural net blocks; the dynamics of the Transformer
    fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
        let mut ss: f32 = 0.0;
        for j in 0..size {
            ss += x[j] * x[j];
        }
        ss /= size as f32;
        ss += 1e-5 as f32;
        ss = 1.0 as f32 / ss.sqrt();
        // normalize and scale
        for j in 0..size {
            o[j] = weight[j] * (ss * x[j]);
        }
    }

    pub fn softmax(x: &mut [f32], size: usize) {
        // find max value (for numerical stability)
        let mut max_val: f32 = x[0];
        for i in 1..size {
            if x[i] > max_val {
                max_val = x[i];
            }
        }
        // exp and sum
        let mut sum: f32 = 0.0;
        for i in 0..size {
            x[i] = (x[i] - max_val).exp();
            sum += x[i];
        }
        // normalize
        for i in 0..size {
            x[i] /= sum;
        }
    }

    fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        // for i in 0..d {
        //     let mut val: f32 = 0.0;
        //     for j in 0..n {
        //         val += w[i * n + j] * x[j];
        //     }
        //     xout[i] = val;
        // }

        // TODO: Rayon
        let parallel: Vec<usize> = (0..d).collect();
        let result: Vec<(usize, f32)> = parallel
            .par_iter()
            .map(|i| {
                let mut val: f32 = 0.0;
                for j in 0..n {
                    val += w[i * n + j] * x[j];
                }
                (*i, val)
            })
            .collect();
        result.iter().for_each(|(index, val)| xout[*index] = *val)
    }

    fn single_head_attn(
        p: &ModelConfig,
        h: usize,
        head_size: usize,
        pos: usize,
        loff: usize,
        kv_dim: usize,
        kv_mul: usize,
        s_q: &[f32],
        s_key_cache: &[f32],
        s_value_cache: &[f32],
    ) -> Vec<f32> {
        // get the query vector for this head
        let q = &s_q[(h * head_size)..((h + 1) * head_size)];
        // attention scores for this head
        let mut att = &mut vec![0.0; p.seq_len][0..p.seq_len];

        // iterate over all timesteps, including the current one
        for t in 0..(pos + 1) {
            // get the key vector for this head and at this timestep
            let k = &s_key_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..];
            // calculate the attention score as the dot product of q and k
            let mut score: f32 = 0.0;
            for i in 0..head_size {
                score += q[i] * k[i];
            }
            score /= (head_size as f32).sqrt();
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        Transformer::softmax(&mut att, pos + 1);

        // weighted sum of the values, store back into xb
        let mut xb = vec![0.0; head_size];
        for t in 0..(pos + 1) {
            // get the value vector for this head and at this timestep
            let v = &s_value_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..];
            // get the attention weight for this timestep
            let a = att[t];

            // accumulate the weighted value into xb
            for i in 0..head_size {
                xb[i] += a * v[i];
            }
        }
        return xb;
    }

    fn copy_vec(xout: &mut [f32], xin: &[f32], size: usize) {
        for i in 0..size {
            xout[i] = xin[i];
        }
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> &mut Vec<f32> {
        // a few convenience variables
        let p = &self.config;
        let w = &self.weights;
        let s = &mut self.state;

        let x = &mut s.x;
        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
        let hidden_dim = p.hidden_dim;
        let head_size = dim / p.n_heads;

        // copy the token embedding into x
        Transformer::copy_vec(x, &w.token_embedding_table[token * dim..], dim);

        for l in 0..p.n_layers {
            // attention rmsnorm
            Transformer::rmsnorm(&mut s.xb, x, &w.rms_att_weight[l * dim..], dim);

            let loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            let s_k = &mut s.key_cache[(loff + pos * kv_dim)..];
            let s_v = &mut s.value_cache[(loff + pos * kv_dim)..];

            // qkv matmuls for this position
            Transformer::matmul(&mut s.q, &s.xb, &w.wq[(l * dim * dim)..], dim, dim);
            Transformer::matmul(s_k, &s.xb, &w.wk[(l * dim * dim)..], dim, kv_dim);
            Transformer::matmul(s_v, &s.xb, &w.wv[(l * dim * dim)..], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / (10000.0 as f32).powf(head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn = if i < kv_dim { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
                for v in 0..rotn {
                    let vec = if v == 0 { &mut s.q } else { &mut *s_k };
                    let v0 = vec[i];
                    let v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // multihead attention. iterate over all heads
            let parallel: Vec<usize> = (0..p.n_heads).collect();
            let result: Vec<(usize, Vec<f32>)> = parallel
                .par_iter()
                .map(|h| {
                    let single_head = Transformer::single_head_attn(
                        p,
                        *h,
                        head_size,
                        pos,
                        loff,
                        kv_dim,
                        kv_mul,
                        &s.q,
                        &s.key_cache,
                        &s.value_cache,
                    );
                    (*h, single_head)
                })
                .collect();
            result.iter().for_each(|(h, single_head)| {
                single_head.iter().enumerate().for_each(|(i, v)| {
                    s.xb[h * head_size + i] = *v;
                });
            });

            // for h in 0..p.n_heads {
            //     // get the query vector for this head
            //     let q = &s.q[(h * head_size)..((h + 1) * head_size)];
            //     // attention scores for this head
            //     let mut att = &mut vec![0.0; p.seq_len][0..p.seq_len];
            //     // iterate over all timesteps, including the current one
            //     for t in 0..(pos + 1) {
            //         // get the key vector for this head and at this timestep
            //         let k = &s.key_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..];
            //         // calculate the attention score as the dot product of q and k
            //         let mut score: f32 = 0.0;
            //         for i in 0..head_size {
            //             score += q[i] * k[i];
            //         }
            //         score /= (head_size as f32).sqrt();
            //         // save the score to the attention buffer
            //         att[t] = score;
            //     }

            //     // softmax the scores to get attention weights, from 0..pos inclusively
            //     Transformer::softmax(&mut att, pos + 1);

            //     // weighted sum of the values, store back into xb
            //     let xb = &mut s.xb[(h * head_size)..((h + 1) * head_size)];
            //     for i in 0..head_size {
            //         xb[i] = 0.0;
            //     }
            //     for t in 0..(pos + 1) {
            //         // get the value vector for this head and at this timestep
            //         let v = &s.value_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..];
            //         // get the attention weight for this timestep
            //         let a = att[t];

            //         // accumulate the weighted value into xb
            //         for i in 0..head_size {
            //             xb[i] += a * v[i];
            //         }
            //     }
            // }

            // final matmul to get the output of the attention
            Transformer::matmul(&mut s.xb2, &s.xb, &w.wo[(l * dim * dim)..], dim, dim);

            // residual connection back into x
            for i in 0..dim {
                x[i] += s.xb2[i];
            }

            // ffn rmsnorm
            Transformer::rmsnorm(&mut s.xb, &x, &w.rms_ffn_weight[(l * dim)..], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            Transformer::matmul(
                &mut s.hb,
                &s.xb,
                &w.w1[l * dim * hidden_dim..],
                dim,
                hidden_dim,
            );
            Transformer::matmul(
                &mut s.hb2,
                &s.xb,
                &w.w3[l * dim * hidden_dim..],
                dim,
                hidden_dim,
            );

            // SwiGLU non-linearity
            for i in 0..hidden_dim {
                let mut val = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= 1.0 as f32 / (1.0 as f32 + (-val).exp());
                // elementwise multiply with w3(x)
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // final matmul to get the output of the ffn
            Transformer::matmul(
                &mut s.xb,
                &s.hb,
                &w.w2[l * dim * hidden_dim..],
                hidden_dim,
                dim,
            );

            // residual connection
            for i in 0..dim {
                x[i] += s.xb[i];
            }
        }

        // final rmsnorm
        // TODO: avoid the copy
        let input: Vec<f32> = x.clone();
        Transformer::rmsnorm(x, &input, &w.rms_final_weight, dim);

        // classifier into logits
        let wcls = match &w.wcls {
            Some(wcls) => wcls,
            None => &w.token_embedding_table,
        };

        Transformer::matmul(&mut s.logits, x, wcls, p.dim, p.vocab_size);
        return &mut s.logits;
    }
}
