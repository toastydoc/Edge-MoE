"""Quantise the base plate and each domain pack into 4-bit pack files.

    python scripts/quant_packs.py

Reads expert_packs/<name>.json manifests, writes quant/<name>.safetensors and
quant/manifest.json (file sizes, mean relative weight error).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.expert_packs import load_manifest  # noqa: E402
from edge_moe.quant_experts import build_pack_file  # noqa: E402


def main() -> None:
    out = {}
    for name in ("base", "medicine", "law", "math"):
        _, mask = load_manifest(ROOT / "expert_packs" / f"{name}.json")
        meta = build_pack_file(mask, ROOT / "quant" / f"{name}.safetensors", name)
        out[name] = meta
        print(f"{name}: {meta['experts']} experts, {meta['file_gb']:.2f}GB on disk, "
              f"relative weight error {meta['rel_error_mean']:.3f}", flush=True)  # fmt: skip
    (ROOT / "quant" / "manifest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
