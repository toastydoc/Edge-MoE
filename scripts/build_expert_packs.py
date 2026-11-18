"""Choose the 26B base plate and domain expert packs from router profiles.

    python scripts/build_expert_packs.py --analyze
    python scripts/build_expert_packs.py --base 0.2 --total 0.5

--analyze prints, for several base sizes, how much of each domain's routing
(top-8 picks of the full model) the installed experts cover. Packs are chosen on
half A of each profile and coverage is scored on half B, so it is out-of-sample.
Computer science and history get no pack: they show what an uncovered topic
looks like.

Without --analyze, writes expert_packs/base.json and expert_packs/<domain>.json
(packs chosen on the whole profile) and prints the memory footprint.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.expert_packs import base_mask, counts, pack_mask, save_manifest, size_gb  # noqa: E402
from edge_moe.loading import MODEL_ID, SafetensorsIndex, resolve_model_dir  # noqa: E402

PACKS = ["medicine", "law", "math"]
UNSEEN = ["computer_science", "history"]


def coverage(c, mask):
    return float((c * mask).sum() / c.sum())


def shared_gb():
    idx = SafetensorsIndex(resolve_model_dir(MODEL_ID))
    total, emb = 0, 0
    for name, (_, _, shape, _) in idx.entries.items():
        if ".experts." in name or "vision" in name or "audio" in name:
            continue
        n = int(torch.tensor(shape).prod()) if shape else 1
        total += n
        if "embed_tokens" in name:
            emb += n
    return total * 2 / 1e9, emb * 2 / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--base", type=float, default=0.2)
    ap.add_argument("--total", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "expert_packs")
    args = ap.parse_args()

    uni = counts(ROOT / "profiles_base" / "universal.npz")
    prof = {d: ROOT / "profiles" / f"{d}.npz" for d in PACKS + UNSEEN}

    if args.analyze:
        print("coverage of each domain's routing on held-out half B (packs chosen on half A)")
        for total in (0.4, 0.5):
            for b in (0.1, 0.15, 0.2, 0.25, 0.3):
                base = base_mask(uni, b)
                packs = {d: pack_mask(counts(prof[d], "pick_counts_a"), base, total) for d in PACKS}
                allp = base | packs["medicine"] | packs["law"] | packs["math"]
                row = []
                for d in PACKS + UNSEEN:
                    cb = counts(prof[d], "pick_counts_b")
                    own = coverage(cb, base | packs[d]) if d in packs else float("nan")
                    row.append(f"{d[:7]:>7} base {coverage(cb, base):.2f} own {own:.2f} all3 {coverage(cb, allp):.2f}")
                print(f"base {b:.2f} -> {total:.2f}: all3 = {float(allp.float().mean()):.2f} of experts | " + " | ".join(row))
        own_only = {d: coverage(counts(prof[d], "pick_counts_b"), base_mask(counts(prof[d], "pick_counts_a"), 0.5)) for d in PACKS + UNSEEN}
        print("reference, own top 50% (step 4 'own50'): " + "  ".join(f"{d} {v:.2f}" for d, v in own_only.items()))
        return

    base = base_mask(uni, args.base)
    meta = {"model": MODEL_ID, "base_fraction": args.base, "total_fraction": args.total}
    save_manifest(args.out_dir / "base.json", "base", "base", base, **meta)
    sizes = {"base": size_gb(base)}
    union = base.clone()
    for d in PACKS:
        p = pack_mask(counts(prof[d]), base, args.total)
        save_manifest(args.out_dir / f"{d}.json", d, "pack", p, **meta)
        sizes[d] = size_gb(p)
        union |= p
    sizes["all"] = size_gb(union)
    sizes["full"] = size_gb(torch.ones_like(base))
    sh, emb = shared_gb()
    sizes["shared_bf16_gb"], sizes["embeddings_bf16_gb"] = sh, emb
    (args.out_dir / "sizes.json").write_text(json.dumps(sizes, indent=1), encoding="utf-8")
    print(f"non-expert weights {sh:.2f}GB bf16 (of which embeddings {emb:.2f}GB)")
    for k, v in sizes.items():
        if isinstance(v, dict):
            print(f"{k:>9}: {v['experts']:5d} experts  {v['bf16_gb']:6.2f}GB bf16  {v['int4_gb']:5.2f}GB int4")


if __name__ == "__main__":
    main()
