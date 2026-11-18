"""Compare domain profiles: do domains use different experts, and how much
routing does a top-N% expert set retain?

    python scripts/analyze_overlap.py [--fraction 0.30] [--profiles profiles]

Expert sets are always chosen on half A of a domain's prompts and scored on
half B (of the same or another domain), so nothing is measured in-sample.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_moe.router_hooks import top_fraction_mask  # noqa: E402


def retention(values: np.ndarray, mask: torch.Tensor) -> float:
    """Share of `values` [layers, experts] that falls on masked experts, averaged over layers."""
    v = torch.as_tensor(values, dtype=torch.float64)
    return float(((v * mask).sum(1) / v.sum(1)).mean())


def jaccard(m1: torch.Tensor, m2: torch.Tensor) -> float:
    return float(((m1 & m2).sum(1) / (m1 | m2).sum(1)).mean())


def table(title: str, rows: list[str], cols: list[str], cell) -> None:
    w = max(len(r) for r in rows + cols) + 2
    print(f"\n{title}")
    print(" " * w + "".join(f"{c:>{w}}" for c in cols))
    for r in rows:
        print(f"{r:<{w}}" + "".join(f"{cell(r, c):>{w}.2f}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=Path, default=Path(__file__).resolve().parents[1] / "profiles")
    ap.add_argument("--fraction", type=float, default=0.30)
    args = ap.parse_args()

    P = {p.stem: np.load(p) for p in sorted(args.profiles.glob("*.npz")) if not p.stem.startswith("_")}
    names = list(P)
    f = args.fraction
    n_layers, n_experts = P[names[0]]["pick_counts"].shape
    print(f"{len(names)} domains, {n_layers} layers x {n_experts} experts, keep fraction {f:.0%}")
    for n in names:
        print(f"  {n}: {int(P[n]['num_prompts'])} prompts, {int(P[n]['tokens_seen'][0])} content tokens")

    set_a = {n: top_fraction_mask(P[n]["pick_counts_a"], f) for n in names}
    set_b = {n: top_fraction_mask(P[n]["pick_counts_b"], f) for n in names}
    set_all = {n: top_fraction_mask(P[n]["pick_counts"], f) for n in names}

    table(
        f"1. Jaccard overlap of top-{f:.0%} expert sets (random = {f / (2 - f):.2f}; "
        "diagonal = same domain, half A vs half B = noise ceiling)",
        names, names,
        lambda r, c: jaccard(set_a[r], set_b[r]) if r == c else jaccard(set_all[r], set_all[c]),
    )  # fmt: skip
    table(
        f"2. Share of PICKS kept: row's top-{f:.0%} set (from half A) applied to column's held-out half B "
        f"(random = {f:.2f})",
        names, names, lambda r, c: retention(P[c]["pick_counts_b"], set_a[r]),
    )  # fmt: skip
    table(
        "3. Same, but share of full router PROBABILITY MASS kept",
        names, names, lambda r, c: retention(P[c]["prob_mass_b"], set_a[r]),
    )  # fmt: skip

    # One universal set: pool per-domain pick frequencies (half A) so every domain counts equally.
    def freq(n, key):
        c = torch.as_tensor(P[n][key], dtype=torch.float64)
        return c / c.sum(1, keepdim=True)

    pooled_a = sum(freq(n, "pick_counts_a") for n in names)
    print("\n4. Keep-fraction sweep, share of held-out picks kept (prob mass in brackets)")
    print(f"{'keep':>6} | {'own-domain set':>22} | {'universal set':>22} | {'other-domain set':>22}")
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        uni = top_fraction_mask(pooled_a, frac)
        sa = {n: top_fraction_mask(P[n]["pick_counts_a"], frac) for n in names}
        own = np.mean([retention(P[n]["pick_counts_b"], sa[n]) for n in names])
        own_m = np.mean([retention(P[n]["prob_mass_b"], sa[n]) for n in names])
        un = np.mean([retention(P[n]["pick_counts_b"], uni) for n in names])
        un_m = np.mean([retention(P[n]["prob_mass_b"], uni) for n in names])
        oth = np.mean([retention(P[c]["pick_counts_b"], sa[r]) for r in names for c in names if r != c])
        oth_m = np.mean([retention(P[c]["prob_mass_b"], sa[r]) for r in names for c in names if r != c])
        print(f"{frac:>6.0%} | {own:>14.2f} ({own_m:.2f}) | {un:>14.2f} ({un_m:.2f}) | {oth:>14.2f} ({oth_m:.2f})")

    # Where in the network does domain specialisation live?
    print(f"\n5. By layer band (keep {f:.0%}): own-domain vs other-domain share of held-out picks kept")
    for lo in range(0, n_layers, 5):
        sl = slice(lo, lo + 5)
        own = np.mean([retention(P[n]["pick_counts_b"][sl], set_a[n][sl]) for n in names])
        oth = np.mean([retention(P[c]["pick_counts_b"][sl], set_a[r][sl]) for r in names for c in names if r != c])
        print(f"  layers {lo:>2}-{lo + 4:<2}  own {own:.2f}   other {oth:.2f}   gap {own - oth:+.2f}")


if __name__ == "__main__":
    main()
