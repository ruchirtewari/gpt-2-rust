"""Summarize one or more bench_kv.sh CSV files as a markdown table (medians)."""
import csv, statistics, sys
from collections import defaultdict

rows = [r for path in sys.argv[1:] for r in csv.DictReader(open(path))]
groups = defaultdict(list)
for r in rows:
    groups[(r["label"], r["device"], r["prompt"], int(r["n"]), r["kv_cache"])].append(r)

def med(rs, k):
    return statistics.median(float(r[k]) for r in rs)

print("| label | device | prompt | new tokens | kv_cache | decode tok/s | total tok/s | TTFT ms | total ms | same output as no-cache |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(groups):
    label, dev, p, n, kv = key
    rs = groups[key]
    shas = {r["output_sha"] for r in rs}
    # Reference: no-cache runs for the same device/prompt/length in any file.
    ref = [r for k, rs2 in groups.items() if k[1:] == (dev, p, n, "false") for r in rs2]
    same = "n/a" if not ref or kv == "false" else ("yes" if shas == {r["output_sha"] for r in ref} else "NO")
    print(f"| {label} | {dev} | {p} | {n} | {kv} | {med(rs,'decode_tok_s'):.1f} | {med(rs,'total_tok_s'):.1f} "
          f"| {med(rs,'ttft_ms'):.0f} | {med(rs,'total_ms'):.0f} | {same} |")
