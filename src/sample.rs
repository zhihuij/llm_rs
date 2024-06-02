use std::io::Result;

use rand::Rng;

use crate::model::Transformer;

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
        let next: usize;
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
