//! Causal self-attention module for GPT-2.
//!
//! This is a **custom** implementation — not a thin wrapper around
//! `burn::nn::MultiHeadAttention` — so we can match GPT-2's exact weight
//! layout (single `c_attn` projection for Q/K/V) and causal masking semantics.

use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::softmax,
};

use crate::config::Gpt2Config;

/// Multi-head causal (decoder-only) self-attention.
///
/// Weight layout matches OpenAI's GPT-2 checkpoint:
/// - `c_attn` — combined Q/K/V projection `[n_embd → 3*n_embd]`
/// - `c_proj` — output projection `[n_embd → n_embd]`
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    /// Combined Q, K, V projection (split at runtime).
    c_attn: Linear<B>,
    /// Output projection.
    c_proj: Linear<B>,
    /// Attention-weight dropout.
    attn_drop: Dropout,
    /// Residual dropout.
    resid_drop: Dropout,
    /// Number of attention heads.
    n_head: usize,
    /// Head dimension: n_embd / n_head.
    head_dim: usize,
    /// Maximum sequence length (for pre-computing causal mask shape).
    max_seq_len: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    /// Construct a new `CausalSelfAttention` layer.
    pub fn new(cfg: &Gpt2Config, device: &B::Device) -> Self {
        let n_embd = cfg.n_embd;
        let head_dim = cfg.head_dim();

        // Single matrix that projects input to Q, K, V concatenated.
        let c_attn = LinearConfig::new(n_embd, 3 * n_embd)
            .with_bias(cfg.bias)
            .init(device);

        let c_proj = LinearConfig::new(n_embd, n_embd)
            .with_bias(cfg.bias)
            .init(device);

        let attn_drop = DropoutConfig::new(cfg.dropout).init();
        let resid_drop = DropoutConfig::new(cfg.dropout).init();

        Self {
            c_attn,
            c_proj,
            attn_drop,
            resid_drop,
            n_head: cfg.n_head,
            head_dim,
            max_seq_len: cfg.max_seq_len,
        }
    }

    /// Forward pass (no caching) — used for training and full-sequence scoring.
    ///
    /// # Arguments
    /// - `x` — Input tensor `[batch, seq_len, n_embd]`
    ///
    /// # Returns
    /// Output tensor `[batch, seq_len, n_embd]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (q, k, v) = self.project_qkv(x);
        self.attend(q, k, v, 0)
    }

    /// Forward pass with a key/value cache, used for incremental decoding.
    ///
    /// `x` holds only the **new** tokens `[batch, t_new, n_embd]`. Their keys and
    /// values are appended to `cache`, and the new queries attend over every
    /// cached position plus themselves. On the first call pass an empty cache
    /// (the "prefill" step); afterwards feed one token at a time.
    ///
    /// # Returns
    /// Output tensor `[batch, t_new, n_embd]`
    pub fn forward_cached(&self, x: Tensor<B, 3>, cache: &mut LayerKvCache<B>) -> Tensor<B, 3> {
        let (q, k_new, v_new) = self.project_qkv(x);
        let [batch, n_head, t_new, head_dim] = k_new.dims();
        let past_len = cache.len;
        let total = past_len + t_new;

        // Allocate the full-context buffers on first use, then write the new
        // rows in place instead of re-concatenating the whole history.
        let device = k_new.device();
        let shape = [batch, n_head, self.max_seq_len, head_dim];
        let k_buf = cache.k.take().unwrap_or_else(|| Tensor::zeros(shape, &device));
        let v_buf = cache.v.take().unwrap_or_else(|| Tensor::zeros(shape, &device));
        let rows = [0..batch, 0..n_head, past_len..total, 0..head_dim];
        let k_buf = k_buf.slice_assign(rows.clone(), k_new);
        let v_buf = v_buf.slice_assign(rows, v_new);

        // Attend over the filled prefix only.
        let filled = [0..batch, 0..n_head, 0..total, 0..head_dim];
        let k = k_buf.clone().slice(filled.clone());
        let v = v_buf.clone().slice(filled);
        cache.k = Some(k_buf);
        cache.v = Some(v_buf);
        cache.len = total;

        self.attend(q, k, v, past_len)
    }

    /// Project `x` to Q, K, V in multi-head form `[B, n_head, T, head_dim]`.
    fn project_qkv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _n_embd] = x.dims();

        // 1. Combined QKV projection
        // x: [B, T, n_embd]  →  qkv: [B, T, 3*n_embd]
        let qkv = self.c_attn.forward(x);

        // Split along last dimension into Q, K, V each [B, T, n_embd]
        let n_embd = self.n_head * self.head_dim;
        let q = qkv.clone().slice([0..batch, 0..seq_len, 0..n_embd]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq_len, n_embd..2 * n_embd]);
        let v = qkv.slice([0..batch, 0..seq_len, 2 * n_embd..3 * n_embd]);

        // 2. Reshape to multi-head form
        // [B, T, n_embd] → [B, T, n_head, head_dim] → [B, n_head, T, head_dim]
        let split = |t: Tensor<B, 3>| {
            t.reshape([batch, seq_len, self.n_head, self.head_dim])
                .swap_dims(1, 2)
        };
        (split(q), split(k), split(v))
    }

    /// Scaled dot-product attention with a causal mask, then output projection.
    ///
    /// `q` covers query positions `offset .. offset + T_q`; `k`/`v` cover key
    /// positions `0 .. T_k` (with `T_k = offset + T_q`).
    fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let [batch, _, q_len, _] = q.dims();
        let k_len = k.dims()[2];

        // 3. Scaled dot-product attention + causal mask
        // scores: [B, n_head, T_q, T_k]
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.transpose()) / scale;

        // Causal mask: a query may not see keys from its future. With a single
        // new query token every cached key is in its past, so no mask is needed.
        let scores = if q_len > 1 {
            let mask = self.causal_mask_offset(q_len, k_len, offset, &scores.device());
            scores + mask
        } else {
            scores
        };

        // Softmax over the last (key) dimension
        let attn_weights = softmax(scores, 3); // [B, n_head, T_q, T_k]
        let attn_weights = self.attn_drop.forward(attn_weights);

        // Weighted sum of values
        // [B, n_head, T_q, T_k] × [B, n_head, T_k, head_dim] → [B, n_head, T_q, head_dim]
        let context = attn_weights.matmul(v);

        // 4. Re-assemble heads
        // [B, n_head, T_q, head_dim] → [B, T_q, n_head, head_dim] → [B, T_q, n_embd]
        let context = context
            .swap_dims(1, 2)
            .reshape([batch, q_len, self.n_head * self.head_dim]);

        // 5. Output projection
        let out = self.c_proj.forward(context);
        self.resid_drop.forward(out)
    }

    /// Build a lower-triangular causal mask of shape `[1, 1, T, T]`.
    ///
    /// Positions where `j > i` are set to `−∞` (–1e9 in f32 to avoid
    /// NaN in softmax with fp16).
    #[cfg_attr(not(test), allow(dead_code))]
    fn causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        self.causal_mask_offset(seq_len, seq_len, 0, device)
    }

    /// Build a causal mask of shape `[1, 1, T_q, T_k]` for queries at absolute
    /// positions `offset .. offset + T_q` against keys at `0 .. T_k`.
    ///
    /// `mask[i, j] = −1e9` when `j > offset + i`, otherwise `0`.
    fn causal_mask_offset(
        &self,
        q_len: usize,
        k_len: usize,
        offset: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let rows = Tensor::<B, 1, Int>::arange(offset as i64..(offset + q_len) as i64, device)
            .reshape([q_len, 1]);
        let cols = Tensor::<B, 1, Int>::arange(0..(k_len as i64), device)
            .reshape([1, k_len]);

        // Boolean mask: True where col > row (future positions)
        let future_mask = cols.greater(rows); // [T_q, T_k]

        // Convert: True → -1e9, False → 0.0
        let mask_2d = Tensor::<B, 2>::zeros([q_len, k_len], device)
            .mask_fill(future_mask, -1e9_f32);

        // Expand to [1, 1, T_q, T_k] for broadcasting over batch and heads
        mask_2d.unsqueeze::<4>()
    }
}

/// Cached keys and values for one attention layer.
///
/// The buffers are preallocated to `[batch, n_head, max_seq_len, head_dim]` on
/// first use; only the first `len` positions hold valid data.
#[derive(Debug, Clone)]
pub struct LayerKvCache<B: Backend> {
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
    len: usize,
}

impl<B: Backend> Default for LayerKvCache<B> {
    fn default() -> Self {
        Self { k: None, v: None, len: 0 }
    }
}

impl<B: Backend> LayerKvCache<B> {
    /// Number of cached sequence positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when nothing has been cached yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_attention_output_shape() {
        let device = Default::default();
        let cfg = crate::config::Gpt2Config::new()
            .with_n_embd(64)
            .with_n_head(4)
            .with_n_layer(1)
            .with_max_seq_len(16)
            .with_dropout(0.0);

        let attn = CausalSelfAttention::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::zeros([2, 8, 64], &device);
        let out = attn.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_causal_mask_shape() {
        let device = Default::default();
        let cfg = crate::config::Gpt2Config::new()
            .with_n_embd(64)
            .with_n_head(4)
            .with_n_layer(1)
            .with_max_seq_len(16)
            .with_dropout(0.0);

        let attn = CausalSelfAttention::<B>::new(&cfg, &device);
        let mask = attn.causal_mask(8, &device);
        assert_eq!(mask.dims(), [1, 1, 8, 8]);
    }

    #[test]
    fn test_cached_attention_matches_full() {
        let device = Default::default();
        let cfg = crate::config::Gpt2Config::new()
            .with_n_embd(32)
            .with_n_head(4)
            .with_n_layer(1)
            .with_max_seq_len(16)
            .with_dropout(0.0);

        let attn = CausalSelfAttention::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::random([2, 7, 32], burn::tensor::Distribution::Default, &device);
        let full = attn.forward(x.clone());

        // Prefill 4 tokens, then feed the remaining 3 one at a time.
        let mut cache = LayerKvCache::default();
        let mut parts = vec![attn.forward_cached(x.clone().slice([0..2, 0..4, 0..32]), &mut cache)];
        for t in 4..7 {
            parts.push(attn.forward_cached(x.clone().slice([0..2, t..t + 1, 0..32]), &mut cache));
        }
        assert_eq!(cache.len(), 7);
        let cached = Tensor::cat(parts, 1);
        cached.into_data().assert_approx_eq::<f32>(&full.into_data(), Default::default());
    }
}
