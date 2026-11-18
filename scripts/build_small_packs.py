"""Pack-size and base-size sweep: 4-bit pack files for law and math.

    python scripts/build_small_packs.py [--base 0.2] [--totals 0.25 0.3 0.35 0.4]
    python scripts/build_small_packs.py --rest      # every expert not in base/med/law/math

A pack at total T adds the domain's most-used experts not in the base until
base + pack = T per layer, so smaller packs are subsets of larger ones.
Base 0.2 reuses quant/base.safetensors; another base B writes quant/base_b<BB>
and names packs <domain>_b<BB>_t<TT>. --rest writes quant/rest.safetensors so
base + medicine + law + math + rest = the full 26B.
Results are logged to quant/small_packs.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.expert_packs import base_mask, counts, load_manifest, pack_mask  # noqa: E402
from edge_moe.quant_experts import build_pack_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=0.2)
    ap.add_argument("--totals", type=float, nargs="+", default=[0.25, 0.3, 0.35, 0.4])
    ap.add_argument("--domains", nargs="+", default=["law", "math"])
    ap.add_argument("--rest", action="store_true")
    args = ap.parse_args()
    out_path = ROOT / "quant" / "small_packs.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    def build(name, mask, **extra):
        meta = build_pack_file(mask, ROOT / "quant" / f"{name}.safetensors", name)
        out[name] = {**meta, **extra, "experts_per_layer": float(mask.sum(1).float().mean())}
        print(f"{name}: {meta['experts']} experts ({out[name]['experts_per_layer']:.0f}/layer), {meta['file_gb']:.2f}GB", flush=True)
        out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    if args.rest:
        union = load_manifest(ROOT / "expert_packs" / "base.json")[1].clone()
        for d in ("medicine", "law", "math"):
            union |= load_manifest(ROOT / "expert_packs" / f"{d}.json")[1]
        build("rest", ~union)
        return

    if abs(args.base - 0.2) < 1e-9:
        base, tag = load_manifest(ROOT / "expert_packs" / "base.json")[1], ""
    else:
        tag = f"_b{round(args.base * 100)}"
        base = base_mask(counts(ROOT / "profiles_base" / "universal.npz"), args.base)
        build(f"base{tag}", base, base_fraction=args.base)
    for d in args.domains:
        c = counts(ROOT / "profiles" / f"{d}.npz")
        for t in args.totals:
            build(f"{d}{tag}_t{round(t * 100)}", pack_mask(c, base, t), base_fraction=args.base, total_fraction=t)


if __name__ == "__main__":
    main()
