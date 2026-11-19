"""Summarise the ten-domain experiment: fingerprint stability, pack quality, the break-even curve, token vs domain.

    python scripts/analyze_domains.py

Reads profiles/wiki_*.npz, results/domain_loss.json and (if present)
profiles/expert_tokens.json. Prints markdown tables and writes
results/domain_summary.json and results/break_even.png.

Excess loss = loss of a configuration minus the full model's loss on the same
held-out text, in nats per token; 0 means as good as all 128 experts.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.expert_packs import load_manifest, pack_mask  # noqa: E402

WIKI = ["art", "history", "biology", "geology", "politics", "computer_science", "economics", "chemistry", "physics", "philosophy"]


def jacc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def main() -> None:
    base = load_manifest(ROOT / "expert_packs" / "base.json")[1]
    out: dict = {}

    # 1. Fingerprint stability: a pack picked on half A of a domain's prompts vs one picked on half B.
    packs, halves = {}, {}
    for d in WIKI:
        z = np.load(ROOT / "profiles" / f"wiki_{d}.npz")
        mk = lambda k: pack_mask(torch.as_tensor(z[k]).double(), base, 51 / 128).numpy()  # noqa: E731
        packs[d], halves[d] = mk("pick_counts"), (mk("pick_counts_a"), mk("pick_counts_b"))
    within = {d: jacc(*halves[d]) for d in WIKI}
    between = {d: float(np.mean([jacc(packs[d], packs[o]) for o in WIKI if o != d])) for d in WIKI}
    out["stability"] = {"within": within, "between_mean": between}
    print("## Fingerprint stability (pack-expert overlap, Jaccard)\n\n| Domain | Half A vs half B | vs other domains (mean) |\n|---|---|---|")
    for d in WIKI:
        print(f"| {d} | {within[d]:.2f} | {between[d]:.2f} |")
    print(f"| **mean** | **{np.mean(list(within.values())):.2f}** | **{np.mean(list(between.values())):.2f}** |\n")

    lp = ROOT / "results" / "domain_loss.json"
    if not lp.exists():
        print("(no domain_loss.json yet)")
    else:
        res = json.loads(lp.read_text(encoding="utf-8"))["loss"]
        full = res["full"]
        ex = lambda c, d: res[c][d] - full[d]  # noqa: E731
        rand = [c for c in res if c.startswith("rand:")]

        # 2. Own pack vs base plate alone vs random 25 vs the best other domain's pack.
        print("## Excess loss at 51 experts/layer (nats/token vs full model)\n\n| Domain | Full loss | Base only | Random 25 | Own pack | Best other pack |\n|---|---|---|---|---|---|")
        rows = {}
        for d in WIKI:
            others = {o: ex(f"pack:{o}", d) for o in WIKI if o != d and f"pack:{o}" in res}
            r = float(np.mean([ex(c, d) for c in rand])) if rand else float("nan")
            bo = min(others, key=others.get) if others else None
            rows[d] = {"full": full[d], "base": ex("base", d), "random": r, "own": ex(f"pack:{d}", d),
                       "best_other": bo, "best_other_excess": others.get(bo, float("nan"))}  # fmt: skip
            print(f"| {d} | {full[d]:.3f} | {rows[d]['base']:+.3f} | {r:+.3f} | {rows[d]['own']:+.3f} | {bo} {rows[d]['best_other_excess']:+.3f} |")
        out["packs"] = rows
        m = lambda k: np.mean([rows[d][k] for d in WIKI])  # noqa: E731
        print(f"| **mean** | | **{m('base'):+.3f}** | **{m('random'):+.3f}** | **{m('own'):+.3f}** | **{m('best_other_excess'):+.3f}** |\n")

        # Cross-domain matrix: rows = pack, columns = text.
        out["matrix"] = {p: {d: ex(f"pack:{p}", d) for d in WIKI} for p in WIKI if f"pack:{p}" in res}

        # 3. Break-even: one fixed model shared by k domains, scored on its member domains.
        curve = {}
        for c in res:
            if not c.startswith("mix:"):
                continue
            mem = c[4:].split("+")
            curve.setdefault(len(mem), []).append(float(np.mean([ex(c, d) for d in mem])))
        own = m("own")
        rnd = m("random")
        print("## Break-even: one model shared by k domains (mean excess on its domains)\n\n| k | Shared model | Own pack (k=1) | Share of the pack's gain over random kept |\n|---|---|---|---|")
        print(f"| 1 | {own:+.3f} | {own:+.3f} | 100% |")
        kept = {}
        for k in sorted(curve):
            v = float(np.mean(curve[k]))
            kept[k] = (rnd - v) / (rnd - own) if rnd != own else float("nan")
            print(f"| {k} | {v:+.3f} (n={len(curve[k])}) | {own:+.3f} | {kept[k] * 100:.0f}% |")
        out["break_even"] = {"own": own, "random": rnd, "curve": {k: float(np.mean(v)) for k, v in curve.items()},
                             "runs": curve, "gain_kept": kept}  # fmt: skip
        print()
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            ks = [1] + sorted(curve)
            ax.plot(ks, [own] + [float(np.mean(curve[k])) for k in sorted(curve)], "o-", label="one model shared by k domains")
            for k in curve:
                ax.scatter([k] * len(curve[k]), curve[k], s=10, color="gray", alpha=0.6)
            ax.axhline(own, ls="--", color="C2", label="each domain's own pack")
            ax.axhline(rnd, ls=":", color="C3", label="base + random experts")
            ax.axhline(m("base"), ls="-.", color="C7", label="base plate alone")
            ax.set_xlabel("domains served by one 51-expert/layer model (k)")
            ax.set_ylabel("excess loss vs full model (nats/token)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(ROOT / "results" / "break_even.png")
        except ImportError:
            print("(matplotlib missing: no chart)")

    # 4. Token vs domain.
    tp = ROOT / "profiles" / "expert_tokens.json"
    if tp.exists():
        t = json.loads(tp.read_text(encoding="utf-8"))
        dom = [x["domain"]["frac"] for x in t["mi"]]
        sur = [x["surface"]["frac"] for x in t["mi"]]
        print("## What the router's choice tracks (share of expert-choice entropy explained)\n\n| Layers | Domain (15) | Token type (6) |\n|---|---|---|")
        for a, b in ((0, 10), (10, 20), (20, 30)):
            print(f"| {a + 1}-{b} | {np.mean(dom[a:b]) * 100:.1f}% | {np.mean(sur[a:b]) * 100:.1f}% |")
        out["token_vs_domain"] = {"domain_frac": dom, "surface_frac": sur}

    (ROOT / "results" / "domain_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
