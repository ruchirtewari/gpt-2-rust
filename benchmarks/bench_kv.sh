#!/usr/bin/env bash
# Benchmark generation speed with the KV cache on and off.
#
# Usage: benchmarks/bench_kv.sh <label> [devices] [token-counts] [reps] [kv-modes]
#   label        name for the results file: benchmarks/results/<label>.csv
#   devices      space-separated list (default: "metal")
#   token-counts space-separated list of --max-new-tokens values (default: "50 200")
#   reps         repetitions per configuration (default: 3)
#   kv-modes     which --kv-cache values to run (default: "false true")
#
# Each configuration gets one unrecorded warm-up run, then <reps> measured runs.
# Uses greedy decoding so outputs are deterministic; each row records a hash of
# the generated text so runs can be checked for identical output.
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL=${1:?label required}
DEVICES=${2:-metal}
NS=${3:-50 200}
REPS=${4:-3}
KVS=${5:-false true}
BIN=./target/release/gpt-2-rust
OUT=benchmarks/results/$LABEL.csv
PROMPTS=(
  "The future of artificial intelligence is"
  "Once upon a time, in a small village by the sea,"
)

cargo build --release --quiet
ERR=$(mktemp)
trap 'rm -f "$ERR"' EXIT
echo "label,device,prompt,n,kv_cache,rep,prompt_tokens,new_tokens,ttft_ms,total_ms,decode_tok_s,total_tok_s,output_sha" > "$OUT"

for dev in $DEVICES; do
  for pi in "${!PROMPTS[@]}"; do
    for n in $NS; do
      for kv in $KVS; do
        # Unrecorded warm-up: GPU kernels are compiled per tensor shape and the
        # OS keeps a shader cache across processes, so the first run is slower.
        "$BIN" --generate --device "$dev" --temperature 0 --max-new-tokens "$n" \
          --kv-cache "$kv" --prompt "${PROMPTS[$pi]}" >/dev/null 2>&1
        for rep in $(seq 1 "$REPS"); do
          text=$("$BIN" --generate --device "$dev" --temperature 0 --max-new-tokens "$n" \
                   --kv-cache "$kv" --stats --prompt "${PROMPTS[$pi]}" 2>"$ERR") || {
            echo "run failed: $dev p$pi n=$n kv=$kv" >&2; cat "$ERR" >&2; exit 1; }
          sha=$(printf '%s' "$text" | shasum | cut -c1-12)
          s=$(grep -o '\[stats\].*' "$ERR")
          get() { sed -E "s/.* $1=([^ ]+).*/\1/" <<<"$s"; }
          echo "$LABEL,$dev,$pi,$n,$kv,$rep,$(get prompt_tokens),$(get new_tokens),$(get ttft_ms),$(get total_ms),$(get decode_tok_s),$(get total_tok_s),$sha" >> "$OUT"
        done
      done
    done
  done
done
python3 benchmarks/summarize.py "$OUT"
