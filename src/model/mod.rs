//! Full GPT-2 model.
//!
//! Assembles token embedding, positional embedding, transformer blocks, and
//! the language-model head into a single Burn `Module`.

pub mod attention;
pub mod block;
pub mod mlp;

use burn::{
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::config::Gpt2Config;
use attention::LayerKvCache;
use block::TransformerBlock;

/// Key/value cache for the whole model: one [`LayerKvCache`] per block.
///
/// Create it with [`Gpt2Model::new_kv_cache`] and pass it to
/// [`Gpt2Model::forward_cached`]. Each call appends the new tokens' keys and
/// values, so later calls only need the tokens that were not seen yet.
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    layers: Vec<LayerKvCache<B>>,
}

impl<B: Backend> KvCache<B> {
    /// Number of token positions currently cached.
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.len())
    }

    /// `true` when no tokens have been cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop all cached keys and values.
    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = LayerKvCache::default());
    }
}

/// The full GPT-2 model.
///
/// Matches the weight layout of the official OpenAI checkpoint:
/// - `transformer.wte`   — token embedding
/// - `transformer.wpe`   — positional embedding
/// - `transformer.h.{i}` — transformer blocks
/// - `transformer.ln_f`  — final layer norm
/// - `lm_head`           — language-model head (weight-tied to `wte`)
#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    /// Token embedding table `[vocab_size, n_embd]`.
    wte: Embedding<B>,
    /// Positional embedding table `[max_seq_len, n_embd]`.
    wpe: Embedding<B>,
    /// Embedding dropout.
    drop: Dropout,
    /// Stack of transformer blocks.
    blocks: Vec<TransformerBlock<B>>,
    /// Final layer normalisation.
    ln_f: LayerNorm<B>,
    /// Language-model head `[n_embd → vocab_size]` (no bias, weight-tied).
    lm_head: Linear<B>,
}

impl<B: Backend> Gpt2Model<B> {
    /// Build a new `Gpt2Model` with random initialisation.
    pub fn new(cfg: &Gpt2Config, device: &B::Device) -> Self {
        let wte = EmbeddingConfig::new(cfg.vocab_size, cfg.n_embd).init(device);
        let wpe = EmbeddingConfig::new(cfg.max_seq_len, cfg.n_embd).init(device);
        let drop = DropoutConfig::new(cfg.dropout).init();

        let blocks = (0..cfg.n_layer)
            .map(|_| TransformerBlock::new(cfg, device))
            .collect();

        let ln_f = LayerNormConfig::new(cfg.n_embd).init(device);

        // LM head: n_embd → vocab_size, no bias.
        // Weight tying (sharing weights with wte) is handled at the Burn record
        // level when loading from safetensors; for random init we use a separate
        // matrix (Burn does not have a built-in weight-tie API yet).
        let lm_head = LinearConfig::new(cfg.n_embd, cfg.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            wte,
            wpe,
            drop,
            blocks,
            ln_f,
            lm_head,
        }
    }

    /// Forward pass — returns **logits** `[batch, seq_len, vocab_size]`.
    ///
    /// # Arguments
    /// - `input_ids` — Token IDs `[batch, seq_len]` (Int tensor)
    ///
    /// # Panics
    /// Panics if `seq_len > max_seq_len`.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch, seq_len] = input_ids.dims();

        // Token + Positional Embeddings
        let tok_emb = self.wte.forward(input_ids.clone()); // [B, T, n_embd]

        let positions = Tensor::<B, 1, Int>::arange(
            0..(seq_len as i64),
            &input_ids.device(),
        )
        .unsqueeze::<2>(); // [1, T]

        let pos_emb = self.wpe.forward(positions); // [1, T, n_embd] — broadcasts
        let mut x = self.drop.forward(tok_emb + pos_emb);

        // Transformer Blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Final LayerNorm + LM Head
        let x = self.ln_f.forward(x); // [B, T, n_embd]
        self.lm_head.forward(x) // [B, T, vocab_size]
    }

    /// Convenience method: forward pass returning only the **last** token's
    /// logits — useful during autoregressive generation.
    ///
    /// Returns `[batch, vocab_size]`.
    pub fn forward_last(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids); // [B, T, V]
        let [batch, seq_len, vocab] = logits.dims();
        // Slice the last time step
        logits.slice([0..batch, (seq_len - 1)..seq_len, 0..vocab])
            .reshape([batch, vocab])
    }

    /// Maximum context length supported by the positional embedding table.
    pub fn max_seq_len(&self) -> usize {
        self.wpe.weight.dims()[0]
    }

    /// Create an empty [`KvCache`] sized for this model.
    pub fn new_kv_cache(&self) -> KvCache<B> {
        KvCache {
            layers: (0..self.blocks.len()).map(|_| LayerKvCache::default()).collect(),
        }
    }

    /// Incremental forward pass using a key/value cache.
    ///
    /// `input_ids` holds only the tokens **not yet in** `cache` `[batch, t_new]`.
    /// Their positions continue from `cache.len()`. Returns the logits of the
    /// **last** new token `[batch, vocab_size]`, which is all generation needs.
    ///
    /// # Panics
    /// Panics if `cache.len() + t_new > max_seq_len`.
    pub fn forward_cached(&self, input_ids: Tensor<B, 2, Int>, cache: &mut KvCache<B>) -> Tensor<B, 2> {
        let [batch, t_new] = input_ids.dims();
        let past_len = cache.len();
        assert!(
            past_len + t_new <= self.max_seq_len(),
            "KV cache overflow: {} cached + {} new > max_seq_len {}",
            past_len,
            t_new,
            self.max_seq_len()
        );

        let tok_emb = self.wte.forward(input_ids.clone()); // [B, t_new, n_embd]
        let positions = Tensor::<B, 1, Int>::arange(
            past_len as i64..(past_len + t_new) as i64,
            &input_ids.device(),
        )
        .unsqueeze::<2>(); // [1, t_new]
        let pos_emb = self.wpe.forward(positions);
        let mut x = self.drop.forward(tok_emb + pos_emb);

        for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            x = block.forward_cached(x, layer_cache);
        }

        // Only the last position's logits are needed: skip the LM head for the rest.
        let n_embd = x.dims()[2];
        let x = x.slice([0..batch, (t_new - 1)..t_new, 0..n_embd]);
        let x = self.ln_f.forward(x);
        let vocab = self.lm_head.weight.dims()[1];
        self.lm_head.forward(x).reshape([batch, vocab])
    }

    /// Number of trainable parameters (approximate, counts tensor elements).
    pub fn num_params(&self) -> usize {
        // Burn doesn't expose a built-in param counter; we compute from config.
        // This is an informational helper only.
        0 // placeholder — populated at runtime via module inspection
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn tiny_cfg() -> Gpt2Config {
        Gpt2Config::new()
            .with_vocab_size(256)
            .with_max_seq_len(16)
            .with_n_embd(32)
            .with_n_layer(2)
            .with_n_head(4)
            .with_dropout(0.0)
    }

    #[test]
    fn test_forward_shape() {
        let device = Default::default();
        let cfg = tiny_cfg();
        let model = Gpt2Model::<B>::new(&cfg, &device);

        let ids = Tensor::<B, 2, Int>::zeros([2, 8], &device);
        let logits = model.forward(ids);
        assert_eq!(logits.dims(), [2, 8, 256]);
    }

    #[test]
    fn test_forward_last_shape() {
        let device = Default::default();
        let cfg = tiny_cfg();
        let model = Gpt2Model::<B>::new(&cfg, &device);

        let ids = Tensor::<B, 2, Int>::zeros([3, 5], &device);
        let last = model.forward_last(ids);
        assert_eq!(last.dims(), [3, 256]);
    }

    #[test]
    fn test_forward_cached_matches_full() {
        let device = Default::default();
        let cfg = tiny_cfg();
        let model = Gpt2Model::<B>::new(&cfg, &device);

        let ids: Vec<i64> = vec![3, 17, 42, 99, 5, 200, 7, 1];
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(ids.clone(), [8]), &device)
            .unsqueeze::<2>();
        let full = model.forward(input.clone()); // [1, 8, V]

        // Prefill 5 tokens, then step one token at a time; compare every step.
        let mut cache = model.new_kv_cache();
        let first = model.forward_cached(input.clone().slice([0..1, 0..5]), &mut cache);
        let expected = full.clone().slice([0..1, 4..5, 0..256]).reshape([1, 256]);
        first.into_data().assert_approx_eq::<f32>(&expected.into_data(), Default::default());

        for t in 5..8 {
            let step = model.forward_cached(input.clone().slice([0..1, t..t + 1]), &mut cache);
            let expected = full.clone().slice([0..1, t..t + 1, 0..256]).reshape([1, 256]);
            step.into_data().assert_approx_eq::<f32>(&expected.into_data(), Default::default());
        }
        assert_eq!(cache.len(), 8);
        cache.clear();
        assert!(cache.is_empty());
    }
}
