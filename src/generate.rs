//! Text generation with temperature, top-k, and top-p (nucleus) sampling.
//!
//! The [`generate`] function runs the autoregressive loop and supports:
//! - **Greedy** decoding (`temperature = 0`)
//! - **Temperature** scaling
//! - **Top-K** filtering
//! - **Top-P** (nucleus) filtering
//! - Early stopping on the EOT token
//! - An optional **KV cache** (`use_kv_cache`) so each step only processes the
//!   newest token instead of re-running the whole sequence

use burn::prelude::*;

use serde::{Deserialize, Serialize};

use std::time::{Duration, Instant};

use crate::model::Gpt2Model;
use crate::tokenizer::{EOT_TOKEN, Gpt2Tokenizer};

/// Configuration for autoregressive text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of **new** tokens to generate (not counting the prompt).
    pub max_new_tokens: usize,

    /// Temperature for logit scaling.
    /// - `1.0` → unchanged distribution
    /// - `< 1.0` → sharper (more deterministic)
    /// - `> 1.0` → flatter (more random)
    /// - `0.0` → greedy argmax
    pub temperature: f64,

    /// If set, keep only the `k` highest-probability tokens before sampling.
    pub top_k: Option<usize>,

    /// If set, keep the smallest set of tokens whose cumulative probability
    /// is at least `p` (nucleus sampling).
    pub top_p: Option<f64>,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Stop early when the EOT token is generated.
    pub stop_on_eot: bool,

    /// Reuse cached attention keys/values between steps.
    ///
    /// With the cache on, the prompt is processed once ("prefill") and each
    /// later step feeds only the newest token. With it off, every step re-runs
    /// the model over the full sequence. Both produce the same tokens while the
    /// sequence fits in the context window.
    #[serde(default = "default_use_kv_cache")]
    pub use_kv_cache: bool,
}

fn default_use_kv_cache() -> bool {
    true
}

/// Timing information collected by [`generate_with_stats`].
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Number of prompt tokens fed to the model.
    pub prompt_tokens: usize,
    /// Number of new tokens generated.
    pub new_tokens: usize,
    /// Time until the first new token was produced (includes the prompt prefill).
    pub time_to_first_token: Duration,
    /// Total wall-clock time of the generation loop.
    pub total_time: Duration,
}

impl GenerationStats {
    /// Throughput of the decode phase (tokens after the first), in tokens/second.
    pub fn decode_tokens_per_sec(&self) -> f64 {
        let decode = self.total_time.saturating_sub(self.time_to_first_token).as_secs_f64();
        if self.new_tokens > 1 && decode > 0.0 {
            (self.new_tokens - 1) as f64 / decode
        } else {
            0.0
        }
    }

    /// End-to-end throughput, in new tokens/second.
    pub fn total_tokens_per_sec(&self) -> f64 {
        let total = self.total_time.as_secs_f64();
        if total > 0.0 { self.new_tokens as f64 / total } else { 0.0 }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 200,
            temperature: 0.8,
            top_k: Some(50),
            top_p: None,
            seed: 42,
            stop_on_eot: true,
            use_kv_cache: true,
        }
    }
}

/// Generate text autoregressively.
///
/// # Arguments
/// - `model`     — Initialised [`Gpt2Model`]
/// - `tokenizer` — GPT-2 BPE tokenizer
/// - `prompt`    — Starting text
/// - `config`    — Sampling configuration
/// - `device`    — Target device
///
/// # Returns
/// The full generated string (prompt + new tokens).
pub fn generate<B: Backend>(
    model: &Gpt2Model<B>,
    tokenizer: &Gpt2Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    device: &B::Device,
) -> String {
    generate_with_stats(model, tokenizer, prompt, config, device).0
}

/// Like [`generate`], but also returns timing statistics.
pub fn generate_with_stats<B: Backend>(
    model: &Gpt2Model<B>,
    tokenizer: &Gpt2Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    device: &B::Device,
) -> (String, GenerationStats) {
    use rand::{rngs::StdRng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(config.seed);
    let max_seq_len = model.max_seq_len();

    // Encode prompt
    let mut token_ids: Vec<u32> = tokenizer.encode(prompt);

    if token_ids.is_empty() {
        token_ids.push(tokenizer.eot_token());
    }

    let mut stats = GenerationStats {
        prompt_tokens: token_ids.len(),
        ..Default::default()
    };
    let start = Instant::now();

    let mut kv_cache = config.use_kv_cache.then(|| model.new_kv_cache());

    // Autoregressive loop
    for step in 0..config.max_new_tokens {
        // Forward pass — get logits for last position [1, vocab_size]
        let logits = match kv_cache.as_mut() {
            Some(cache) => {
                if cache.len() >= max_seq_len {
                    // Context window full: positions can't shift inside the cache,
                    // so restart it from the most recent half-window of tokens.
                    cache.clear();
                }
                // Feed only the tokens the cache has not seen. After a clear (or on
                // the first step) this re-prefills from a window that fits.
                let new_ids = if cache.is_empty() {
                    let window = if step == 0 { max_seq_len } else { max_seq_len / 2 };
                    &token_ids[token_ids.len().saturating_sub(window)..]
                } else {
                    &token_ids[token_ids.len() - 1..]
                };
                model.forward_cached(ids_to_tensor::<B>(new_ids, device), cache)
            }
            None => {
                // Re-run the full (cropped) sequence every step.
                let window = &token_ids[token_ids.len().saturating_sub(max_seq_len)..];
                model.forward_last(ids_to_tensor::<B>(window, device))
            }
        };

        // Extract as f32 vec
        let mut logits_vec: Vec<f32> = logits
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to convert logits to f32");

        // Temperature scaling
        if config.temperature > 0.0 && config.temperature != 1.0 {
            let inv_temp = (1.0 / config.temperature) as f32;
            for l in logits_vec.iter_mut() {
                *l *= inv_temp;
            }
        }

        // Top-K filtering
        if let Some(k) = config.top_k {
            if k < logits_vec.len() {
                top_k_filter(&mut logits_vec, k);
            }
        }

        // Top-P (nucleus) filtering
        if let Some(p) = config.top_p {
            top_p_filter(&mut logits_vec, p as f32);
        }

        // Sampling
        let next_token = if config.temperature == 0.0 {
            // Greedy argmax
            logits_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0) as u32
        } else {
            sample_from_logits(&logits_vec, &mut rng)
        };

        if step == 0 {
            stats.time_to_first_token = start.elapsed();
        }
        stats.new_tokens += 1;

        // Stream the token's text representation
        let token_text = tokenizer.decode(&[next_token]);
        print!("{}", token_text);
        use std::io::Write;
        let _ = std::io::stdout().flush();

        token_ids.push(next_token);

        if config.stop_on_eot && next_token == EOT_TOKEN {
            break;
        }
    }

    stats.total_time = start.elapsed();
    (tokenizer.decode(&token_ids), stats)
}

/// Build a `[1, T]` Int tensor from token ids.
fn ids_to_tensor<B: Backend>(ids: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(ids_i64, [ids.len()]), device)
        .unsqueeze::<2>()
}

// Sampling helpers

/// Set logits below the k-th largest to −∞ in-place.
fn top_k_filter(logits: &mut Vec<f32>, k: usize) {
    let mut sorted = logits.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let threshold = sorted[k - 1];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Apply nucleus (top-p) filtering in-place.
fn top_p_filter(logits: &mut Vec<f32>, p: f32) {
    // Build (index, logit) sorted by logit descending
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Softmax over sorted logits
    let max_l = indexed[0].1;
    let mut sum_exp = 0.0f32;
    let exps: Vec<f32> = indexed
        .iter()
        .map(|(_, l)| {
            let e = (l - max_l).exp();
            sum_exp += e;
            e
        })
        .collect();

    // Accumulate until cumulative probability >= p
    let mut cum_prob = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (rank, &(_, _)) in indexed.iter().enumerate() {
        cum_prob += exps[rank] / sum_exp;
        if cum_prob >= p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    // Mask out everything below the nucleus
    let nucleus: std::collections::HashSet<usize> = indexed[..cutoff_idx]
        .iter()
        .map(|(i, _)| *i)
        .collect();

    for (i, l) in logits.iter_mut().enumerate() {
        if !nucleus.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Sample a token index from logits using categorical sampling.
fn sample_from_logits<R: rand::Rng>(logits: &[f32], rng: &mut R) -> u32 {
    // Stable softmax
    let max_l = logits
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut exps: Vec<f32> = logits
        .iter()
        .map(|&l| if l.is_finite() { (l - max_l).exp() } else { 0.0 })
        .collect();

    let sum: f32 = exps.iter().sum();
    for e in exps.iter_mut() {
        *e /= sum;
    }

    // Inverse CDF sampling
    let u: f32 = rng.r#gen::<f32>();
    let mut cum = 0.0f32;
    for (i, &prob) in exps.iter().enumerate() {
        cum += prob;
        if u <= cum {
            return i as u32;
        }
    }
    (exps.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_filter() {
        let mut logits = vec![1.0, 3.0, 2.0, 0.5, 4.0];
        top_k_filter(&mut logits, 2);
        // Top-2 are 4.0 and 3.0; rest → -inf
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert_eq!(logits[1], 3.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
        assert_eq!(logits[4], 4.0);
    }

    #[test]
    fn test_top_p_filter_keeps_top() {
        let mut logits = vec![100.0, -1.0, -1.0, -1.0];
        top_p_filter(&mut logits, 0.9);
        // First logit dominates; rest should be masked
        assert_eq!(logits[0], 100.0);
        for &l in &logits[1..] {
            assert!(l.is_infinite() && l < 0.0);
        }
    }

    #[test]
    fn test_sample_from_logits_deterministic() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0);
        let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 100.0, f32::NEG_INFINITY];
        // Only index 2 has non-zero probability
        assert_eq!(sample_from_logits(&logits, &mut rng), 2);
    }

    fn tiny_model() -> crate::model::Gpt2Model<burn::backend::NdArray<f32>> {
        // Real GPT-2 vocab so tokenizer ids are valid embedding indices.
        let cfg = crate::config::Gpt2Config::new()
            .with_vocab_size(50257)
            .with_max_seq_len(16)
            .with_n_embd(32)
            .with_n_layer(2)
            .with_n_head(4)
            .with_dropout(0.0);
        cfg.init(&Default::default())
    }

    fn cfg(use_kv_cache: bool, max_new_tokens: usize, temperature: f64) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            temperature,
            top_k: Some(20),
            top_p: None,
            seed: 7,
            stop_on_eot: false,
            use_kv_cache,
        }
    }

    #[test]
    fn test_kv_cache_matches_no_cache() {
        let model = tiny_model();
        let tok = Gpt2Tokenizer::new().unwrap();
        let device = Default::default();
        // Prompt is a few tokens; 8 new tokens stays within max_seq_len = 16.
        for temperature in [0.0, 0.9] {
            let (with, s1) = generate_with_stats(&model, &tok, "Hi", &cfg(true, 8, temperature), &device);
            let (without, s2) = generate_with_stats(&model, &tok, "Hi", &cfg(false, 8, temperature), &device);
            assert_eq!(with, without, "temperature {temperature}");
            assert_eq!(s1.new_tokens, 8);
            assert_eq!(s2.new_tokens, 8);
        }
    }

    #[test]
    fn test_generation_past_context_window() {
        // 40 new tokens with max_seq_len = 16 must not panic in either mode.
        let model = tiny_model();
        let tok = Gpt2Tokenizer::new().unwrap();
        let device = Default::default();
        for use_kv_cache in [true, false] {
            let (_, stats) = generate_with_stats(&model, &tok, "Hi", &cfg(use_kv_cache, 40, 0.0), &device);
            assert_eq!(stats.new_tokens, 40);
        }
    }
}
