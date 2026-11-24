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
