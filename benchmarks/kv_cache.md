# KV cache benchmarks

GPT-2 small (124M), greedy decoding (`--temperature 0`), batch size 1, on an
Apple Silicon Mac. Two short prompts: prompt 0 is
"The future of artificial intelligence is" (6 tokens), prompt 1 is
"Once upon a time, in a small village by the sea," (13 tokens).

Reproduce with:

```bash
benchmarks/bench_kv.sh <label> "metal" "50 200" 3
python3 benchmarks/summarize.py benchmarks/results/<label>.csv
```

Raw per-run numbers are in `benchmarks/results/*.csv`. Every row stores a hash
of the generated text; "same output" compares it with the no-cache run.

## 1. KV cache on vs off (v1)

Metal backend, median of 3 runs after one warm-up run per configuration
(`results/v1-metal-baseline.csv`):

| prompt | new tokens | no cache, tok/s | KV cache, tok/s | speedup | same output |
|---|---|---|---|---|---|
| 0 | 50  | 28.8 | 62.7 | 2.2x | yes |
| 0 | 200 | 52.1 | 71.4 | 1.4x | yes |
| 1 | 50  | 36.1 | 65.9 | 1.8x | yes |
| 1 | 200 | 54.9 | 73.0 | 1.3x | yes |

(total tokens/second, including the prompt prefill)

Earlier single runs on the CPU (NdArray) and WGPU backends, before warm-up runs
were added to the script (`results/initial-cpu-wgpu.csv`):

| backend | new tokens | no cache, tok/s | KV cache, tok/s | speedup |
|---|---|---|---|---|
| CPU  | 50  | 3.6–4.6 | 5.6–6.1 | 1.3–1.6x |
| CPU  | 200 | 1.9–2.3 | 5.5–6.3 | 2.8–2.9x |
| WGPU | 50  | 14–21   | 42–59   | 2.8–3.0x |
| WGPU | 200 | 20–21   | 68–69   | 3.3–3.5x |

Notes:

- GPU kernels are compiled per tensor shape and macOS keeps a shader cache
  across processes, so cold first runs are much slower. The script therefore
  does one unrecorded warm-up run per configuration.
- With the cache, decode speed plateaus near 74 tok/s (about 13.5 ms/token).
  That is far above the arithmetic cost of one GPT-2 small token, so the decode
  step is dominated by per-kernel launch/synchronisation overhead, not by
  attention over the cache.
- The CPU-side sampling path sorts all 50,257 logits for top-k every step. It
  costs about 1 ms/token (74 → 79 tok/s with `--top-k 0`).

## 2. Further cache variations

Candidates, roughly in order of expected benefit for short, simple queries:

1. **Preallocated cache buffer.** Allocate `[B, H, max_seq_len, D]` once per
   layer and write each new key/value row in place (`slice_assign`), instead of
   `Tensor::cat`, which reallocates and copies the whole history every step.
2. **Fixed-shape (bucketed) attention.** Attend over the cache padded to a
   multiple of a bucket size (e.g. 64), masking unused slots. Kernel shapes then
   change once per bucket rather than every token, so compiled kernels and
   fusion plans are reused.
3. **Half-precision (f16) cache.** Store K/V in f16 to halve cache memory and
   bandwidth. Matters at long contexts and large batches; adds cast ops.
4. **Prefix (prompt) cache reuse.** Keep the KV cache of a shared prompt
   prefix across `generate` calls so repeated prompts skip prefill. Helps TTFT
   in a chat/REPL loop; does nothing for a single one-shot CLI call.
5. **Smarter context-window overflow.** GPT-2's absolute positions prevent a
   true rolling cache; the current policy re-prefills the last half window when
   1024 tokens are reached. Alternatives trade re-prefill frequency against
   context kept. Only matters past 1024 tokens.
6. **Batched cache for multiple prompts.** Generate several prompts at once
   with one batched cache (with left-padding masks) to raise throughput.

Variations 4–6 cannot change single-prompt, short-generation numbers, so the
experiments below cover 1–3.

## 3. Experiments

All runs: Metal, greedy, median of 3 after a warm-up run; decode tok/s.

### 3.1 Preallocated cache buffer: rejected

`results/v2-prealloc.csv`, `results/v2-prealloc-long.csv`, `results/v1-long.csv`

| new tokens | prompt | v1 (`Tensor::cat`) | preallocated | same output |
|---|---|---|---|---|
| 50  | 0 | 67.9 | 70.3 | yes |
| 50  | 1 | 70.1 | 71.0 | yes |
| 200 | 0 | 73.9 | 75.4 | yes |
| 200 | 1 | 74.3 | 75.1 | yes |
| 900 | 0 | 71.5 | 70.4 | yes |
| 900 | 1 | 71.0 | 69.2 | yes |

No measurable gain: at these lengths the copied history is a few MB per step,
which is negligible next to per-kernel launch overhead. It also allocates the
full 1024-position buffer (about 75 MB for GPT-2 small) for every generation,
so it was reverted.

For reference, at 900 new tokens the cache gives 3.1–3.4x over no cache
(21–23 vs 71 tok/s), since the uncached cost grows with sequence length.

### 3.2 Bucketed fixed-shape cache (64 positions): kept, marginal gain

The cache grows in blocks of 64 zero-padded positions; decode steps attend
over the whole block with the causal mask hiding the padding, so kernel shapes
stay fixed for 64 steps at a time. Prefill attends over the prompt directly
(a first version that also padded during prefill added 50–100 ms to the time
to first token: `results/v3-bucket64*.csv`).

Compared against a fresh v1 run from the same session (`results/v1-rerun.csv`),
total tok/s, median of 5 (900-token rows: median of 3, v1 from `v1-long.csv`):

| new tokens | prompt | v1 | bucketed | TTFT v1 → bucketed (ms) | same output |
|---|---|---|---|---|---|
| 50  | 0 | 66.3 | 74.3 | 55 → 46 | yes |
| 50  | 1 | 71.1 | 72.2 | 54 → 48 | yes |
| 200 | 0 | 73.8 | 74.7 | 54 → 51 | yes |
| 200 | 1 | 73.7 | 73.7 | 53 → 54 | yes |
| 900 | 0 | 71.2 | 73.2 | 54 → 51 | yes |
| 900 | 1 | 70.6 | 72.9 | 62 → 50 | yes |

A 0–3% gain in most cases, never slower. This mostly rules out per-length
kernel recompilation as the main cost: the decode step is bound by fixed
per-kernel launch and synchronisation overhead. Memory cost is at most 63
extra positions per layer.

### 3.3 Half-precision (f16) cache storage: rejected

K/V stored as f16 and cast back to f32 for attention, on top of 3.2
(`results/v4-f16*.csv`; code in `experiments/f16-kv-cache.patch`).

| new tokens | prompt | bucketed f32 | bucketed f16 | same output |
|---|---|---|---|---|
| 50  | 0 | 74.3 | 70.7 | yes |
| 50  | 1 | 72.2 | 71.0 | yes |
| 200 | 0 | 74.7 | 74.3 | yes |
| 200 | 1 | 73.7 | 74.3 | yes |
| 900 | 0 | 73.2 | 72.4 | yes |
| 900 | 1 | 72.9 | 75.9 | yes |

(total tok/s) No consistent change: GPT-2 small at batch size 1 is not limited
by cache bandwidth, and the extra cast kernels offset the smaller reads. It also
fails on the CPU (NdArray) backend, which has no f16 support. It would only pay
off for much larger models, batches or contexts.

## 4. Conclusions

- The KV cache itself is the big win: 1.3–3.4x on Metal and up to 2.9x on the
  CPU, growing with output length, with identical output.
- Of the cache refinements, only bucketed fixed-shape attention helped, and only
  slightly (0–3%).
- The remaining decode cost (about 13 ms/token, ~74 tok/s) is per-kernel launch
  and host synchronisation overhead, not cache work. Larger wins would come from
  outside the cache: cheaper top-k selection on the host (about 7%, measured
  above), on-device greedy argmax to avoid copying all 50,257 logits back each
  step, and fewer/fused kernels per layer.
