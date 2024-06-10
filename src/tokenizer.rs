use std::fs::OpenOptions;
use std::io::Result;

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use regex::Regex;

use crate::{
    constants::{F32_SIZE, I32_SIZE},
    gguf::{gguf_context, gguf_find_key, gguf_get_val_str, gguf_value},
};

#[derive(Debug)]
pub struct TokenIndex {
    pub str: String,
    pub id: usize,
}

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub vocab_scores: Vec<f32>,
    pub sorted_vocab: Vec<TokenIndex>,
    pub vocab_size: usize,
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
        // let max_token_length =
        //     LittleEndian::read_u32(&mapped_file[offset..offset + I32_SIZE]) as usize;
        offset += I32_SIZE; // skip max_token_length

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
            byte_pieces,
        })
    }

    pub fn from_gguf(ctx: &gguf_context, vocab_size: usize) -> Result<Self> {
        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        let mut vocab_scores: Vec<f32> = Vec::with_capacity(vocab_size);
        let mut sorted_vocab: Vec<TokenIndex> = Vec::with_capacity(vocab_size);

        let key_id_result = gguf_find_key(&ctx, "tokenizer.ggml.tokens");
        let token_arr;
        if let Some(key_id) = key_id_result {
            let value = &ctx.kv[key_id].value;
            if let gguf_value::array(arr) = value {
                token_arr = arr;
            } else {
                panic!("invalid value type {:?}", value)
            }
        } else {
            panic!("invalid token key type");
        };

        for i in 0..vocab_size {
            vocab_scores.push(0.0 as f32);
            let token_str = gguf_get_val_str(&token_arr[i]);
            vocab.push(token_str.clone());
            sorted_vocab.push(TokenIndex {
                str: token_str.clone(),
                id: i,
            });
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
            // c_str = "Ġ".to_string();
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

#[cfg(test)]
mod tests {
    use crate::{gguf::load_gguf_file, Tokenizer};
    use regex::Regex;

    #[test]
    fn test_tokenizer_gguf() {
        let checkpoint = "/Users/winpro/Documents/AIApp/models/qwen2-0_5b-instruct-fp16.gguf";
        let ctx = load_gguf_file(checkpoint);

        let vocab_size = 151936;
        let tokenizer = Tokenizer::from_gguf(&ctx, vocab_size).unwrap();

        // let expect0 = vec![1];
        // let result0 = tokenizer.encode("".to_string(), 1, 0);
        // assert_eq!(expect0, result0, "equal");

        let expect = vec![40, 4411, 279, 7290, 315, 2272, 374];
        let result = tokenizer.encode("I believe the meaning of life is".to_string(), 0, 0);
        assert_eq!(expect, result, "equal");

        // let expect2 = vec![105043, 100165, 30];
        // let result2 = tokenizer.encode("你是谁?".to_string(), 0, 0);
        // assert_eq!(expect2, result2, "equal");
    }

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
