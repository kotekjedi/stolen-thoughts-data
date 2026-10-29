#!/usr/bin/env python3
"""Per-model summary of Stolen Thoughts data: rows, recovered traces, exact and within-1% token matches,
plus the one-trace-per-problem pick (the row with the smallest |token_diff|).

    python scripts/summarize.py demo_data
    python scripts/summarize.py path/to/stolen-thoughts/data/hle
"""
import glob, os, sys
import pyarrow.parquet as pq

COLS = ["problem_id", "vendor", "model", "reasoning_effort", "reasoning_tokens", "recovered_reasoning_tokens", "token_diff"]


def rows(paths):
    for p in paths:
        for f in sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)) if os.path.isdir(p) else [p]:
            yield from pq.read_table(f, columns=COLS).to_pylist()


def main(paths):
    per_model, best = {}, {}
    for r in rows(paths):
        s = per_model.setdefault((r["vendor"], r["model"], r["reasoning_effort"]), [0, 0, 0, 0])
        d = r["token_diff"]
        s[0] += 1
        s[1] += r["recovered_reasoning_tokens"] is not None
        s[2] += d == 0
        s[3] += d is not None and abs(d) <= 0.01 * r["reasoning_tokens"]
        if d is not None:  # one trace per (model, problem): the row closest to the reported length
            k = (r["model"], r["problem_id"])
            if k not in best or abs(d) < abs(best[k]):
                best[k] = d
    print(f"{'vendor':10} {'model':24} {'effort':8} {'rows':>6} {'recovered':>10} {'exact':>6} {'within 1%':>10}")
    for (v, m, e), s in sorted(per_model.items()):
        print(f"{v:10} {m:24} {e:8} {s[0]:6} {s[1]:10} {s[2]:6} {s[3]:10}")
    t = [sum(s[i] for s in per_model.values()) for i in range(4)]
    print(f"{'total':44} {t[0]:6} {t[1]:10} {t[2]:6} {t[3]:10}")
    print(f"\none trace per (model, problem): {len(best)} traces, {sum(d == 0 for d in best.values())} exact")


if __name__ == "__main__":
    main(sys.argv[1:] or ["demo_data"])
