"""Evaluate route-B packs on the held-out MMLU questions (data/test/*.jsonl).

    python scripts/eval_packs.py [--packs packs] [--limit 20]

For every test question (5 domains x 200, the same set step 4 used):

    base            base plate alone (also the Gemma 4 E4B baseline for step 5)
    +<pack>         one pack forced on (cross-domain matrix: knowledge vs format)
    all_forced      every pack on at once (worst-case interference)
    all_gated       every pack installed, the gates decide (the real setup)
    all_gated_soft  same, but each pack weighted by its gate probability

and every gate's probability, which gives the "I don't have a map for that"
detection numbers for any install subset.

Writes results/routeB_eval.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.base_plate import BasePlate, format_mc  # noqa: E402
from edge_moe.packs import PackSet  # noqa: E402

DOMAINS = ["medicine", "law", "math", "computer_science", "history"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", type=Path, default=ROOT / "packs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8, help="questions per batch (x conditions rows)")
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "routeB_eval.json")
    args = ap.parse_args()

    plate = BasePlate()
    ps = PackSet(plate)
    names = [ps.load_pack(p) for p in sorted(args.packs.glob("*.safetensors"))]
    print("packs:", names, {n: ps.gates[n].threshold for n in names}, flush=True)
    conds = ["base"] + [f"+{n}" for n in names] + ["all_forced", "all_gated", "all_gated_soft"]
    letters = torch.tensor(plate.letter_ids[:4], device=plate.device)

    results = {"packs": names, "conditions": conds, "domains": {}}
    t0 = time.time()
    for domain in DOMAINS:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"{domain}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = rows[: args.limit]
        pred = {c: [] for c in conds}
        probs = {n: [] for n in names}
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            B = len(chunk)
            enc = plate.encode([plate.prompt_text(format_mc(r["question"], r["choices"])) for r in chunk], args.max_tokens)
            ids, att = enc["input_ids"], enc["attention_mask"]
            with torch.inference_mode():
                # pass 1: base answer + gate features (the gate layer runs before any pack layer)
                h = ps.forward_hidden(ids, att, mode="features")
                last = att.sum(1) - 1
                lg = plate.logits(h[torch.arange(B, device=ids.device), last])
                pred["base"] += lg[:, letters].argmax(-1).tolist()
                p = {n: ps.gates[n].prob(ps.last_features) for n in names}
                for n in names:
                    probs[n] += p[n].tolist()
                # pass 2: every other condition, one row block each, per-row pack weights
                others = conds[1:]
                C = len(others)
                w = {n: torch.zeros(C, B, device=ids.device) for n in names}
                for ci, c in enumerate(others):
                    for n in names:
                        if c == f"+{n}" or c == "all_forced":
                            w[n][ci] = 1.0
                        elif c == "all_gated":
                            w[n][ci] = (p[n] >= ps.gates[n].threshold).float()
                        elif c == "all_gated_soft":
                            w[n][ci] = p[n]
                weights = {n: v.reshape(-1) for n, v in w.items()}
                ids2, att2 = ids.repeat(C, 1), att.repeat(C, 1)
                h2 = ps.forward_hidden(ids2, att2, mode="forced", forced=weights)
                lg2 = plate.logits(h2[torch.arange(C * B, device=ids.device), att2.sum(1) - 1])
                ans = lg2[:, letters].argmax(-1).view(C, B)
                for ci, c in enumerate(others):
                    pred[c] += ans[ci].tolist()
        gold = [r["answer"] for r in rows]
        acc = {c: sum(int(a == g) for a, g in zip(pred[c], gold)) / len(gold) for c in conds}
        fire = {n: sum(int(x >= ps.gates[n].threshold) for x in probs[n]) / len(gold) for n in names}
        results["domains"][domain] = {"n": len(rows), "accuracy": acc, "gate_fire_rate": fire,
                                      "pred": pred, "gold": gold, "gate_probs": probs}  # fmt: skip
        print(f"== {domain} (n={len(rows)}, {time.time() - t0:.0f}s): "
              + "  ".join(f"{c} {a:.3f}" for c, a in acc.items()), flush=True)  # fmt: skip
        print("   gates fire: " + "  ".join(f"{n} {f:.2f}" for n, f in fire.items()), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
