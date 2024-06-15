use std::{
    collections::{HashMap, HashSet},
    io::Result,
};

use regex::Regex;

use crate::gguf::{gguf_context, gguf_find_key, gguf_get_val_i32, gguf_get_val_str, gguf_value};

// 使用宏来简化 ord() 函数的调用
macro_rules! ord {
    ($c: expr) => {
        $c as u8
    };
}

// const PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

#[derive(Debug)]
pub struct BpeTokenizer {
    pub vocab: Vec<String>,
    pub decoder: HashMap<usize, String>,
    pub vocab_size: usize,
    pub byte_encoder: HashMap<u8, char>,
    pub byte_decoder: HashMap<char, u8>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub speciak_tokens: HashMap<String, usize>,
}

impl BpeTokenizer {
    pub fn from_gguf(ctx: &gguf_context, vocab_size: usize) -> Result<Self> {
        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        let mut decoder: HashMap<usize, String> = HashMap::new();
        let mut speciak_tokens: HashMap<String, usize> = HashMap::new();

        let tokens_id_result = gguf_find_key(&ctx, "tokenizer.ggml.tokens");
        let token_arr;
        if let Some(key_id) = tokens_id_result {
            let value = &ctx.kv[key_id].value;
            if let gguf_value::array(arr) = value {
                token_arr = arr;
            } else {
                panic!("invalid value type {:?}", value)
            }
        } else {
            panic!("invalid token key type");
        };

        let tokens_type_result = gguf_find_key(&ctx, "tokenizer.ggml.token_type");
        let token_type_arr;
        if let Some(key_id) = tokens_type_result {
            let value = &ctx.kv[key_id].value;
            if let gguf_value::array(arr) = value {
                token_type_arr = arr;
            } else {
                panic!("invalid value type {:?}", value)
            }
        } else {
            panic!("invalid token key type");
        };

        for i in 0..vocab_size {
            let token_str = gguf_get_val_str(&token_arr[i]);
            let token_type = gguf_get_val_i32(&token_type_arr[i]);
            vocab.push(token_str.clone());
            decoder.insert(i, token_str.clone());

            if token_type != 1 {
                // not normal token
                speciak_tokens.insert(token_str.clone(), i);
            }
        }

        let byte_encoder: HashMap<u8, char> = BpeTokenizer::bytes_to_unicode();
        let mut byte_decoder = HashMap::new();
        for (key, value) in &byte_encoder {
            byte_decoder.insert(*value, *key);
        }

        let merges_id_result = gguf_find_key(&ctx, "tokenizer.ggml.merges");
        let merge_arr;
        if let Some(key_id) = merges_id_result {
            let value = &ctx.kv[key_id].value;
            if let gguf_value::array(arr) = value {
                merge_arr = arr;
            } else {
                panic!("invalid value type {:?}", value)
            }
        } else {
            panic!("invalid token key type");
        };

        let mut bpe_ranks: HashMap<(String, String), usize> = HashMap::new();
        for i in 0..merge_arr.len() {
            let merge_str = gguf_get_val_str(&merge_arr[i]);
            let pair: Vec<&str> = merge_str.split(" ").collect();
            bpe_ranks.insert((pair[0].to_string(), pair[1].to_string()), i);
        }

        Ok(BpeTokenizer {
            vocab,
            decoder,
            vocab_size,
            byte_encoder,
            byte_decoder,
            bpe_ranks,
            speciak_tokens: speciak_tokens,
        })
    }

    pub fn decode(&self, token_ids: &Vec<usize>) -> String {
        let token_list = self.convert_ids_to_tokens(token_ids);
        return self.convert_tokens_to_string(&token_list);
    }

    fn convert_ids_to_tokens(&self, ids: &Vec<usize>) -> Vec<String> {
        let mut token_vec: Vec<String> = Vec::new();
        ids.iter().for_each(|index| {
            let token_result = self.decoder.get(index);
            if let Some(token) = token_result {
                token_vec.push(token.clone());
            }
        });

        return token_vec;
    }

    fn convert_tokens_to_string(&self, tokens: &Vec<String>) -> String {
        let text = tokens.join("");
        let mut byte_array = Vec::new();
        text.chars().for_each(|c| {
            let result = self.byte_decoder.get(&c);
            if let Some(b) = result {
                byte_array.push(*b);
            }
        });

        match String::from_utf8(byte_array) {
            Ok(s) => return s,
            Err(_) => panic!("something is wrong, expected some str\n"),
        }
    }

    pub fn encode(&self, text: String, bos: i8, eos: i8) -> Vec<usize> {
        let re = Regex::new(PRETOKENIZE_REGEX).unwrap();

        let pre_split = BpeTokenizer::split_by_markers(&text);
        let mut token_list: Vec<String> = Vec::new();

        pre_split.iter().for_each(|str| {
            if str.starts_with("<|") {
                token_list.push(str.clone());
            } else {
                let mut result: Vec<String> = re
                    .captures_iter(str)
                    .map(|cap| cap.get(0).unwrap().as_str().to_string())
                    .collect();
                token_list.append(&mut result);
            }
        });

        let mut bpe_tokens = Vec::new();

        token_list.iter().for_each(|token| {
            let specical_result = self.speciak_tokens.get(token);
            if let Some(id) = specical_result {
                bpe_tokens.push(*id);
                return;
            }

            let bytes = token.as_bytes();
            let mut vec_str = Vec::new();
            bytes.iter().for_each(|b| {
                if let Some(c) = self.byte_encoder.get(b) {
                    vec_str.push(*c);
                }
            });

            let token_str = vec_str.into_iter().collect::<String>();
            let bpe_result = self.bpe(&token_str);

            bpe_result.iter().for_each(|token| {
                let result = self.vocab.iter().position(|s| s == token);
                match result {
                    Some(index) => bpe_tokens.push(index),
                    None => println!("can't find token: {:?}", token),
                }
            });
        });

        return bpe_tokens;
    }

    pub fn get_pairs(word: &Vec<String>) -> HashSet<(&String, &String)> {
        let mut pairs = HashSet::new();
        let first_char = &word[0];
        let mut prev_char = first_char;
        for char in word[1..].into_iter() {
            pairs.insert((prev_char, char));
            prev_char = char;
        }
        pairs
    }

    pub fn bpe(&self, token: &String) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        let mut pairs = BpeTokenizer::get_pairs(&word);

        let mut bpe_result = Vec::new();

        if pairs.len() == 0 {
            bpe_result.push(token.clone());
            return bpe_result;
        }

        loop {
            let bigram_result = pairs.iter().min_by_key(|pair| {
                self.bpe_ranks
                    .get(&(pair.0.to_string(), pair.1.to_string()))
                    .unwrap_or(&usize::MAX)
            });

            if let Some(bigram) = bigram_result {
                if self
                    .bpe_ranks
                    .get(&(bigram.0.clone(), bigram.1.clone()))
                    .is_none()
                {
                    break;
                }

                let (first, second) = bigram;
                let mut new_word: Vec<String> = Vec::new();
                let mut i = 0;

                while i < word.len() {
                    if let Some(position) = word.iter().skip(i).position(|c| c == *first) {
                        word[i..i + position]
                            .iter()
                            .for_each(|c| new_word.push(c.to_string()));
                        i = i + position;
                    } else {
                        word[i..].iter().for_each(|c| new_word.push(c.to_string()));
                        break;
                    }

                    if word[i] == **first && i < word.len() - 1 && word[i + 1] == **second {
                        new_word.push(format!("{}{}", first, second));
                        i += 2;
                    } else {
                        new_word.push(word[i].to_string());
                        i += 1;
                    }
                }

                word = new_word.iter().cloned().collect::<Vec<String>>();
                if word.len() == 1 {
                    break;
                } else {
                    pairs = BpeTokenizer::get_pairs(&word)
                }
            } else {
                break;
            }
        }

        return word;
    }

    pub fn bytes_to_unicode() -> HashMap<u8, char> {
        let basic_ascii: Vec<u8> = (ord!('!')..ord!('~') + 1).into_iter().collect();
        let extended_ascii: Vec<u8> = (ord!('¡')..ord!('¬') + 1).into_iter().collect();
        let additional_chars: Vec<u8> = (ord!('®')..=ord!('ÿ')).into_iter().collect();

        let mut bs = vec![];
        bs.extend(basic_ascii);
        bs.extend(extended_ascii);
        bs.extend(additional_chars);

        let mut cs: Vec<u32> = bs.iter().map(|x| *x as u32).collect();
        let mut n = 0;
        for b in 0..=u8::MAX {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push((2 as u32).pow(8) + n as u32);
                n += 1;
            }
        }

        let mut map = HashMap::new();
        for (&b, &c) in bs.iter().zip(cs.iter()) {
            map.insert(b, std::char::from_u32(c as u32).unwrap());
        }

        map
    }

    fn split_by_markers(input: &String) -> Vec<String> {
        let mut result = Vec::new();
        let chars = input.chars().collect::<Vec<_>>();
        let mut start = 0;
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '<' && i < chars.len() - 1 && chars[i + 1] == '|' {
                if i > 0 && i > start {
                    result.push(chars[start..i].iter().cloned().collect());
                }
                start = i;
                i += 2;
                continue;
            }
            if chars[i] == '|' && i < chars.len() - 1 && chars[i + 1] == '>' {
                result.push(chars[start..i + 2].iter().cloned().collect());
                start = i + 2;
                i += 2;
                continue;
            }

            i += 1;
        }
        if start < chars.len() {
            result.push(chars[start..chars.len()].iter().cloned().collect())
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{bpe_tokenizer::BpeTokenizer, gguf::load_gguf_file};

    #[test]
    fn test_tokenizer_gguf() {
        let checkpoint = "/Users/winpro/Documents/AIApp/models/qwen2-0_5b-instruct-fp16.gguf";
        let ctx = load_gguf_file(checkpoint);

        let vocab_size = 151936;
        let tokenizer = BpeTokenizer::from_gguf(&ctx, vocab_size).unwrap();

        let expect = vec![40, 4411, 279, 7290, 315, 2272, 374];
        let result = tokenizer.encode("I believe the meaning of life is".to_string(), 0, 0);
        assert_eq!(expect, result, "equal");

        let expect2 = vec![61443, 18947, 40820, 17177, 109547, 107018];
        let result2 = tokenizer.encode("写个二分查找算法".to_string(), 0, 0);
        assert_eq!(expect2, result2, "equal");

        let expect3 = "I believe the meaning of life is".to_string();
        let result3 = tokenizer.decode(&expect);
        assert_eq!(expect3, result3, "equal");

        let expect4 = "写个二分查找算法".to_string();
        let result4 = tokenizer.decode(&expect2);
        assert_eq!(expect4, result4, "equal");

        let expect5 = vec![
            151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198,
            61443, 18947, 40820, 17177, 109547, 107018, 151645, 198, 151644, 77091, 198,
        ];
        let result5 = tokenizer.encode(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n写个二分查找算法<|im_end|>\n<|im_start|>assistant\n".to_string(),
            0,
            0,
        );
        assert_eq!(expect5, result5, "equal");
    }
}
