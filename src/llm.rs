use std::io::{self, Write};
use std::usize;

mod constants;
mod gguf;
mod model;
mod run_state;
mod sample;
mod tokenizer;
mod utils;

use gguf::load_gguf_file;
use model::Transformer;
use sample::Sampler;
use tokenizer::Tokenizer;

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
            start = utils::time_in_ms();
        }
    }
    println!("");
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end = utils::time_in_ms();
        println!(
            "achieved tok/s: {:?}\n",
            (pos - 1) as f64 / (end - start) as f64 * 1000.0,
        );
    }
}

fn main() {
    let temperature: f32 = 0.0; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    let topp: f32 = 0.9; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    let steps: usize = 256; // number of steps to run for
    let rng_seed: u64 = 1234; // seed rng with time by default

    // let model_path = "/Users/winpro/Documents/AIApp/llama2.c/stories15M.bin";
    let model_path = "/Users/winpro/Documents/AIApp/models/qwen2-0_5b-instruct-fp16.gguf";
    // let tokenizer_path = "/Users/winpro/Documents/AIApp/llama2.c/tokenizer.bin";

    // let mut transformer = Transformer::new(model_path).unwrap();
    // let tokenizer = Tokenizer::new(tokenizer_path, 32000).unwrap();
    let ctx = load_gguf_file(model_path);
    let mut transformer = Transformer::from_gguf(&ctx).unwrap();
    let tokenizer = Tokenizer::from_gguf(&ctx, transformer.config.vocab_size).unwrap();

    println!("Model config: {:?}", &transformer.config);
    println!(
        "Model weights - token_embedding_table: {:?}",
        &transformer.weights.token_embedding_table[..20]
    );
    println!(
        "Model weights - rms_att_weight: {:?}",
        &transformer.weights.rms_att_weight[..20]
    );
    println!("Tokenizer vocab: {:?}", &tokenizer.vocab[..10]);
    println!(
        "Tokenizer sorted vocab: {:?}",
        &tokenizer.sorted_vocab[..10]
    );

    let mut sampler =
        Sampler::new(transformer.config.vocab_size, temperature, topp, rng_seed).unwrap();

    let prompt = "I believe the meaning of life is".to_string();
    genrate(&mut transformer, &tokenizer, &mut sampler, prompt, steps);
}
