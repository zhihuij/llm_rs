use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use rand::Rng;
use regex::Regex;
use std::fs::OpenOptions;
use std::io::{self, Result, Write};
use std::time::{SystemTime, UNIX_EPOCH};
use std::usize;

pub const I32_SIZE: usize = std::mem::size_of::<i32>();
pub const F32_SIZE: usize = std::mem::size_of::<f32>();

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
    pub att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
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
        let mut att: Vec<f32> = Vec::new();
        att.resize_with(model_config.n_heads * model_config.seq_len, || 0.0);
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
            att,
            logits,
            key_cache,
            value_cache,
        })
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

    fn softmax(x: &mut [f32], size: usize) {
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
        for i in 0..d {
            let mut val: f32 = 0.0;
            for j in 0..n {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
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
            for h in 0..p.n_heads {
                // get the query vector for this head
                let q = &s.q[(h * head_size)..((h + 1) * head_size)];
                // attention scores for this head
                let mut att = &mut s.att[(h * p.seq_len)..((h + 1) * p.seq_len)];
                // iterate over all timesteps, including the current one
                for t in 0..(pos + 1) {
                    // get the key vector for this head and at this timestep
                    let k = &s.key_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..];
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
                let xb = &mut s.xb[(h * head_size)..((h + 1) * head_size)];
                for i in 0..head_size {
                    xb[i] = 0.0;
                }
                for t in 0..(pos + 1) {
                    // get the value vector for this head and at this timestep
                    let v = &s.value_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..];
                    // get the attention weight for this timestep
                    let a = att[t];

                    // accumulate the weighted value into xb
                    for i in 0..head_size {
                        xb[i] += a * v[i];
                    }
                }
            }

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

pub struct TokenIndex {
    pub str: String,
    pub id: usize,
}

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub vocab_scores: Vec<f32>,
    pub sorted_vocab: Vec<TokenIndex>,
    pub vocab_size: usize,
    pub max_token_length: usize,
    pub byte_pieces: Vec<u8>, // stores all single-byte strings
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(tokenizer_path)
            .expect(&format!("Couldn't open file {}\n", tokenizer_path));
        let mapped_file = unsafe { Mmap::map(&file).unwrap() };

        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        let mut vocab_scores: Vec<f32> = Vec::with_capacity(vocab_size);
        let mut sorted_vocab: Vec<TokenIndex> = Vec::with_capacity(vocab_size);

        let mut offset = 0;
        let max_token_length =
            LittleEndian::read_u32(&mapped_file[offset..offset + I32_SIZE]) as usize;
        offset += I32_SIZE;

        for i in 0..vocab_size {
            vocab_scores.push(LittleEndian::read_f32(
                &mapped_file[offset..offset + F32_SIZE],
            ));
            offset += F32_SIZE;
            let str_len = LittleEndian::read_i32(&mapped_file[offset..offset + I32_SIZE]) as usize;
            offset += I32_SIZE;

            let token_str = std::str::from_utf8(&mapped_file[offset..offset + str_len])
                .unwrap()
                .to_string();
            vocab.push(token_str.clone());
            sorted_vocab.push(TokenIndex {
                str: token_str,
                id: i,
            });
            offset += str_len as usize;
        }

        sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));

        let mut byte_pieces: Vec<u8> = Vec::new();
        byte_pieces.resize_with(512, || 0);
        for i in 0..256 {
            byte_pieces[i * 2] = i as u8;
            byte_pieces[i * 2 + 1] = b'\0';
        }

        Ok(Tokenizer {
            vocab,
            vocab_scores,
            sorted_vocab,
            vocab_size,
            max_token_length,
            byte_pieces,
        })
    }

    pub fn decode(&self, prev_token: i32, token: i32) -> String {
        let piece = self.vocab.get(token as usize).unwrap();
        let mut piece_copy = piece.clone();
        let first = piece.as_bytes().first().unwrap();
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if prev_token == 1 && *first == b' ' {
            piece_copy.remove(0);
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        match self.find_0x(&piece_copy) {
            Some(byte_val) => {
                let c = self.byte_pieces.get(byte_val as usize * 2).unwrap();
                char::from_u32(*c as u32).unwrap().to_string()
            }
            None => piece_copy,
        }
    }

    pub fn encode(&self, text: String, bos: i8, eos: i8) -> Vec<i32> {
        // encode the string text (input) into an upper-bound preallocated tokens[] array
        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)

        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        // let mut str_buffer = String::with_capacity(self.max_token_length as usize * 2 + 1 + 2);
        let mut tokens: Vec<i32> = Vec::new();

        // add optional BOS (=1) token, if desired
        if bos != 0 {
            tokens.push(1);
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if text.len() != 0 {
            let dummy_prefix = self.str_lookup(&(" ".to_string()));
            tokens.push(dummy_prefix);
        }

        for c in text.chars() {
            let c_str = c.to_string();
            let id = self.str_lookup(&c_str);
            if id != -1 {
                tokens.push(id);
            } else {
                let utf8_bytes = c_str.as_bytes();
                for b in utf8_bytes {
                    let id = *b as i32 + 3;
                    tokens.push(id);
                }
            }
        }

        loop {
            let mut best_score: f32 = -1e10;
            let mut best_id: i32 = -1;
            let mut best_idx: i32 = -1;
            for idx in 0..tokens.len() - 1 {
                let mut str_buffer = String::new();
                let a = tokens.get(idx).unwrap();
                let b = tokens.get(idx + 1).unwrap();
                str_buffer.push_str(self.vocab.get(*a as usize).unwrap());
                str_buffer.push_str(self.vocab.get(*b as usize).unwrap());

                let id = self.str_lookup(&str_buffer);
                if id != -1 {
                    let score = self.vocab_scores.get(id as usize).unwrap();
                    if *score > best_score {
                        best_score = *score;
                        best_id = id;
                        best_idx = idx as i32;
                    }
                }
            }

            if best_idx == -1 {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx as usize] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for idx2 in best_idx + 1..tokens.len() as i32 - 1 {
                tokens[idx2 as usize] = tokens[idx2 as usize + 1];
            }
            tokens.pop();
        }

        if eos != 0 {
            tokens.push(2);
        }

        tokens
    }

    fn str_lookup(&self, str: &String) -> i32 {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        let result = self
            .sorted_vocab
            .binary_search_by(|probe| probe.str.cmp(str));
        match result {
            Ok(index) => self.sorted_vocab.get(index).unwrap().id as i32,
            Err(_) => -1,
        }
    }

    fn find_0x(&self, input: &String) -> Option<u8> {
        let re = Regex::new(r"<0x([0-9a-fA-F]{2})>").unwrap();
        if let Some(caps) = re.captures(input) {
            if let Some(hex_str) = caps.get(1) {
                return Some(u8::from_str_radix(hex_str.as_str(), 16).unwrap());
            }
        }
        None
    }
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

pub struct ProbIndex {
    pub prob: f32,
    pub index: usize,
}

pub struct Sampler {
    pub vocab_size: usize,
    pub probindex: Vec<ProbIndex>, // buffer used in top-p sampling
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, rng_state: u64) -> Result<Self> {
        // buffer only used with nucleus sampling; may not need but it's ~small
        let mut probindex: Vec<ProbIndex> = Vec::new();
        probindex.resize_with(vocab_size, || ProbIndex {
            prob: 0.0,
            index: 0,
        });
        Ok(Sampler {
            vocab_size,
            probindex,
            temperature,
            topp,
            rng_state,
        })
    }

    pub fn sample(&mut self, logits: &mut Vec<f32>) -> usize {
        // sample the token given the logits and some hyperparameters
        let mut next = 0;
        if self.temperature == 0.0 {
            // greedy argmax sampling: take the token with the highest probability
            next = Sampler::sample_argmax(logits, self.vocab_size);
        } else {
            // apply the temperature to the logits
            for q in 0..self.vocab_size {
                logits[q] /= self.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            Transformer::softmax(logits, self.vocab_size);
            // flip a (float) coin (this is our source of entropy for sampling)
            let coin: f32 = rand::thread_rng().gen();
            // we sample from this distribution to get the next token
            if self.topp <= 0.0 || self.topp >= 1.0 {
                // simply sample from the predicted probability distribution
                next = Sampler::sample_mult(logits, self.vocab_size, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = Sampler::sample_topp(
                    logits,
                    self.vocab_size,
                    self.topp,
                    &mut self.probindex,
                    coin,
                );
            }
        }
        return next;
    }

    pub fn sample_argmax(probabilities: &Vec<f32>, n: usize) -> usize {
        // return the index that has the highest probability
        let mut max_i = 0;
        let mut max_p = probabilities[0];
        for i in 1..n {
            if probabilities[i] > max_p {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        return max_i;
    }

    pub fn sample_mult(probabilities: &Vec<f32>, n: usize, coin: f32) -> usize {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        let mut cdf = 0.0;
        for i in 0..n {
            cdf += probabilities[i];
            if coin < cdf {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    pub fn sample_topp(
        probabilities: &Vec<f32>,
        n: usize,
        topp: f32,
        probindex: &mut Vec<ProbIndex>,
        coin: f32,
    ) -> usize {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        let mut n0 = 0;
        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        let cutoff = (1.0 - topp) / (n - 1) as f32;
        for i in 0..n {
            if probabilities[i] >= cutoff {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0 += 1;
            }
        }

        let select_token = &mut probindex[0..n0];
        select_token.sort_by(|a, b| b.prob.total_cmp(&a.prob));

        // truncate the list where cumulative probability exceeds topp
        let mut cumulative_prob = 0.0;
        let mut last_idx = n0 - 1; // in case of rounding errors consider all elements
        for i in 0..n0 {
            cumulative_prob += probindex[i].prob;
            if cumulative_prob > topp {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0.0;
        for i in 0..last_idx + 1 {
            cdf += probindex[i].prob;
            if r < cdf {
                return probindex[i].index;
            }
        }
        return probindex[last_idx].index; // in case of rounding errors
    }
}

fn time_in_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("get time error")
        .as_millis()
}

fn genrate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: String,
    steps: usize,
) {
    // encode the (string) prompt into tokens sequence
    let prompt_tokens = tokenizer.encode(prompt, 1, 0);
    if prompt_tokens.len() < 1 {
        panic!("something is wrong, expected at least 1 prompt token\n");
    }
    let num_prompt_tokens = prompt_tokens.len();

    // start the main loop
    let mut start = 0; // used to time our code, only initialized after first iteration
    let mut next: i32; // will store the next token in the sequence
    let mut token = prompt_tokens[0]; // kick off with the first token in the prompt
    let mut pos = 0; // position in the sequence
    while pos < steps {
        // forward the transformer to get logits for the next token
        let logits = transformer.forward(token as usize, pos);
        // println!("Generate logits: {:?}", &logits[..10]);

        // advance the state machine
        if pos < num_prompt_tokens - 1 {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler.sample(logits) as i32;
        }
        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        let piece = tokenizer.decode(token, next);
        print!("{}", piece);
        io::stdout().flush().unwrap();
        token = next;
        // init the timer here because the first iteration can be slower
        if start == 0 {
            start = time_in_ms();
        }
    }
    println!("");
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end = time_in_ms();
        println!(
            "achieved tok/s: {:?}\n",
            (pos - 1) as f64 / (end - start) as f64 * 1000.0,
        );
    }
}

fn main() {
    let temperature: f32 = 1.0; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    let topp: f32 = 0.9; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    let steps: usize = 256; // number of steps to run for
    let rng_seed: u64 = 1234; // seed rng with time by default

    let checkpoint = "/Users/winpro/Documents/AIApp/llama2.c/stories15M.bin";
    let tokenizer_path = "/Users/winpro/Documents/AIApp/llama2.c/tokenizer.bin";

    let mut transformer = Transformer::new(checkpoint).unwrap();
    // println!("Model config: {:?}", transformer.config);
    // println!(
    //     "Model weights - token_embedding_table: {:?}",
    //     &transformer.weights.token_embedding_table[..20]
    // );
    // println!(
    //     "Model weights - rms_att_weight: {:?}",
    //     &transformer.weights.rms_att_weight[..20]
    // );

    let tokenizer = Tokenizer::new(tokenizer_path, transformer.config.vocab_size).unwrap();
    // println!("Tokenizer vocab: {:?}", &tokenizer.vocab[..10]);
    // println!(
    //     "Tokenizer vocab_scores: {:?}",
    //     &tokenizer.vocab_scores[..10]
    // );

    let mut sampler =
        Sampler::new(transformer.config.vocab_size, temperature, topp, rng_seed).unwrap();

    let prompt = "I believe the meaning of life is".to_string();
    genrate(&mut transformer, &tokenizer, &mut sampler, prompt, steps);
}

#[cfg(test)]
mod tests {
    use crate::Tokenizer;
    use regex::Regex;

    #[test]
    fn test_tokenizer() {
        let tokenizer_path = "/Users/winpro/Documents/AIApp/llama2.c/tokenizer.bin";
        let vocab_size = 32000;
        let tokenizer = Tokenizer::new(tokenizer_path, vocab_size).unwrap();

        let expect0 = vec![1];
        let result0 = tokenizer.encode("".to_string(), 1, 0);
        assert_eq!(expect0, result0, "equal");

        let expect = vec![1, 306, 4658, 278, 6593, 310, 2834, 338];
        let result = tokenizer.encode("I believe the meaning of life is".to_string(), 1, 0);
        assert_eq!(expect, result, "equal");

        let expect2 = vec![
            1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871,
        ];
        let result2 = tokenizer.encode(
            "Simply put, the theory of relativity states that ".to_string(),
            1,
            0,
        );
        assert_eq!(expect2, result2, "equal");

        let expect3 = vec![
            1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13,
            4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871,
        ];
        let result3 = tokenizer.encode(
            "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ".to_string(),
            1,
            0,
        );
        assert_eq!(expect3, result3, "equal");

        let expect4 = vec![
            1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449,
            276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318,
            13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706,
            923, 968, 1149,
        ];
        let result4 = tokenizer.encode(
            "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>".to_string(),
            1,
            0,
        );
        assert_eq!(expect4, result4, "equal");
    }

    #[test]
    fn test_regex() {
        let piece = "<0x0A>";
        let re = Regex::new(r"<0x([0-9a-fA-F]{2})>").unwrap();

        if let Some(caps) = re.captures(piece) {
            if let Some(hex_str) = caps.get(1) {
                let byte_val = u8::from_str_radix(hex_str.as_str(), 16).unwrap();
                println!("Parsed byte value: {}", byte_val);
            }
        } else {
            println!("No match found in the input string.");
        }
    }
}
