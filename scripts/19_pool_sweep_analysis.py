#!/usr/bin/env python3
"""Analyze the C7 pool-kernel ablation sweep.

Reads results/plan2/pool_sweep_k{N}_ctx16k.csv (Llama, synthetic_json, ctx 16k,
n=50, configs {snapkv-only, snapkv-no-key, snapkv-no-key-nomax}, budgets
{0.05,0.1,0.2,0.3,0.5}) and pivots EM accuracy by (pool_kernel x budget) per
config. The headline SnapKV default is kernel=7 = 0.20/0.34/0.70/0.80/0.88 on
the snapkv-only baseline (SDPA path); this sweep brackets it with k in
{3,5,11,15} to show the maxpool kernel is the load-bearing knob (C7).
"""
import csv
import glob
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.5]
# k=7 headline (SDPA path), snapkv-only, synthetic_json ctx16k, from §9/§14.2.
HEADLINE_K7 = {0.05: 0.20, 0.1: 0.34, 0.2: 0.70, 0.3: 0.80, 0.5: 0.88}


def load():
    # acc[config][kernel][budget] -> (n_correct, n_total)
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    paths = sorted(glob.glob(os.path.join(ROOT, "results/plan2/pool_sweep_k*_ctx16k.csv")))
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                cfg = row["config"]
                k = int(row["pool_kernel"])
                b = round(float(row["budget_frac"]), 3)
                cell = acc[cfg][k][b]
                cell[0] += int(row["correct"])
                cell[1] += 1
    return acc, paths


def fmt(cfg_acc, kernels):
    """Return markdown table: rows=kernel, cols=budget; values=EM."""
    lines = []
    hdr = "| pool_k | " + " | ".join(f"B={b}" for b in BUDGETS) + " | n |"
    sep = "|" + "---|" * (len(BUDGETS) + 2)
    lines.append(hdr)
    lines.append(sep)
    for k in kernels:
        cells = []
        n_seen = 0
        for b in BUDGETS:
            nc, nt = cfg_acc[k].get(b, [0, 0])
            n_seen = max(n_seen, nt)
            cells.append(f"{nc / nt:.2f}" if nt else "—")
        tag = " **(SnapKV default)**" if k == 7 else ""
        lines.append(f"| {k}{tag} | " + " | ".join(cells) + f" | {n_seen} |")
    return "\n".join(lines)


def main():
    acc, paths = load()
    kernels = sorted({k for cfg in acc.values() for k in cfg})
    print(f"Loaded {len(paths)} sweep CSVs: kernels {kernels}\n")

    order = ["snapkv-only", "snapkv-no-key", "snapkv-no-key-nomax"]
    captions = {
        "snapkv-only": "C7 curve — SnapKV baseline (windowed mass + maxpool=k)",
        "snapkv-no-key": "combined method (no-key) + maxpool=k",
        "snapkv-no-key-nomax": "control — maxpool DISABLED (k has no effect)",
    }
    for cfg in order:
        if cfg not in acc:
            continue
        print(f"### {cfg} — {captions[cfg]}")
        print(fmt(acc[cfg], kernels))
        print()

    # C7 verdict: monotone-ish peak near k=7? Compare swept kernels to headline.
    print("### C7 readout (snapkv-only vs k=7 headline)")
    print(f"k=7 headline (not in sweep): "
          + " ".join(f"{HEADLINE_K7[b]:.2f}" for b in BUDGETS)
          + "  (B=0.05..0.50)")
    so = acc.get("snapkv-only", {})
    for k in kernels:
        deltas = []
        for b in BUDGETS:
            nc, nt = so[k].get(b, [0, 0])
            if nt:
                deltas.append(nc / nt - HEADLINE_K7[b])
        mean_d = sum(deltas) / len(deltas) if deltas else float("nan")
        print(f"  k={k:>2}: mean Δ vs k=7 = {mean_d:+.3f}")


if __name__ == "__main__":
    main()
