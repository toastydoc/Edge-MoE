"""Export an expert pack as a standalone Hugging Face checkpoint: a smaller, valid Gemma 4 MoE.

    python scripts/export_pack_checkpoint.py --domain regulation --budget 64 [--check]

The pack (shared core + the domain's most used experts, `budget` per layer) is cut out of
google/gemma-4-26B-A4B-it. Per layer, the fused expert tensors (experts.gate_up_proj,
experts.down_proj) and the router (router.proj.weight rows, router.per_expert_scale) keep
only the pack's experts, in ascending expert order; config num_experts becomes `budget`.
Everything else (attention, shared MLP, norms, embeddings, vision/audio towers) is copied.

Why this is exact: the Gemma 4 router takes a softmax over all experts, keeps the top 8 and
renormalises them. Renormalised top-k weights only depend on ratios between the chosen
experts' scores, so dropping the rows of experts that are never allowed gives the same
choices and weights as masking them (what the prototype does). --check verifies this on the
real router weights.

Output: exports/gemma4-26b-<domain>-<budget>e/ (bf16 safetensors, config, tokenizer, and
pack_manifest.json with the original expert ids per layer). Ready for llama.cpp's
convert_hf_to_gguf.py.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.expert_packs import counts, load_manifest, pack_mask  # noqa: E402

MODEL_ID = "google/gemma-4-26B-A4B-it"
SHARD_BYTES = 4 * 1024**3


def profile_path(domain: str) -> Path:
    for name in (f"wiki_{domain}", domain):
        p = ROOT / "profiles" / f"{name}.npz"
        if p.exists():
            return p
    raise SystemExit(f"no profile for {domain!r} in profiles/")


def pack_ids(domain: str, budget: int) -> list[list[int]]:
    """The same selection the judged runs use: shared core | pack_mask(profile, core, budget/128)."""
    base = load_manifest(ROOT / "expert_packs" / "base.json")[1]
    mask = base | pack_mask(counts(profile_path(domain)), base, budget / 128)
    ids = [torch.nonzero(mask[l]).flatten().tolist() for l in range(mask.shape[0])]
    sizes = {len(i) for i in ids}
    if sizes != {budget}:
        raise SystemExit(f"pack has {sorted(sizes)} experts per layer, expected exactly {budget}")
    return ids


def layer_of(name: str) -> int | None:
    if "language_model.layers." not in name:
        return None
    return int(name.split("language_model.layers.")[1].split(".")[0])


def check_router(src: Path, idx: dict, ids: list[list[int]], n_tokens: int = 4096) -> None:
    """Masked full router vs sliced router on random hidden states, every layer: same experts, same weights."""
    torch.manual_seed(0)
    worst = 0.0
    for l, keep in enumerate(ids):
        pre = f"model.language_model.layers.{l}.router."
        t = {}
        for k in ("proj.weight", "per_expert_scale", "scale"):
            with safe_open(str(src / idx[pre + k]), "pt") as f:
                t[k] = f.get_tensor(pre + k).float()
        h = torch.randn(n_tokens, t["scale"].shape[0])
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * t["scale"] * h.shape[-1] ** -0.5
        keep_t = torch.tensor(keep)
        # prototype: softmax over all experts, mask to the pack, top 8, renormalise
        p = torch.softmax(h @ t["proj.weight"].T, -1)
        allowed = torch.zeros(p.shape[-1], dtype=torch.bool)
        allowed[keep_t] = True
        w1, i1 = torch.topk(p.masked_fill(~allowed, 0), 8, -1)
        w1 = w1 / w1.sum(-1, keepdim=True) * t["per_expert_scale"][i1]
        # exported: softmax over the pack's rows only, top 8, renormalise, map back to original ids
        q = torch.softmax(h @ t["proj.weight"][keep_t].T, -1)
        w2, i2 = torch.topk(q, 8, -1)
        w2 = w2 / w2.sum(-1, keepdim=True) * t["per_expert_scale"][keep_t][i2]
        i2 = keep_t[i2]
        o1, o2 = i1.sort(-1), i2.sort(-1)
        if not torch.equal(o1.values, o2.values):
            raise SystemExit(f"layer {l}: different experts chosen")
        d = (w1.gather(-1, o1.indices) - w2.gather(-1, o2.indices)).abs().max().item()
        worst = max(worst, d)
    print(f"router check: all {len(ids)} layers choose identical experts; max weight difference {worst:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="verify the sliced routers against the masked ones first")
    args = ap.parse_args()

    src = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*"]))
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    ids = pack_ids(args.domain, args.budget)
    if args.check:
        check_router(src, idx, ids)
    out = args.out or ROOT / "exports" / f"gemma4-26b-{args.domain}-{args.budget}e"
    out.mkdir(parents=True, exist_ok=True)

    shard, shard_bytes, n_shard, weight_map, total = {}, 0, 0, {}, 0

    def flush() -> None:
        nonlocal shard, shard_bytes, n_shard
        if not shard:
            return
        n_shard += 1
        name = f"model-{n_shard:05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        print(f"  wrote {name} ({shard_bytes / 1e9:.1f} GB)", flush=True)
        shard, shard_bytes = {}, 0

    for file in sorted(set(idx.values())):
        with safe_open(str(src / file), "pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                l = layer_of(name)
                if l is not None and (name.endswith((".experts.gate_up_proj", ".experts.down_proj", ".router.proj.weight", ".router.per_expert_scale"))):
                    if t.shape[0] != 128:
                        raise SystemExit(f"{name}: expected 128 experts on dim 0, got {tuple(t.shape)}")
                    t = t.index_select(0, torch.tensor(ids[l])).contiguous()
                shard[name] = t
                shard_bytes += t.numel() * t.element_size()
                total += t.numel() * t.element_size()
                if shard_bytes >= SHARD_BYTES:
                    flush()
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))

    cfg = json.loads((src / "config.json").read_text())
    (cfg["text_config"] if "text_config" in cfg else cfg)["num_experts"] = args.budget
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for p in src.iterdir():
        if p.suffix in (".json", ".jinja") and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(p, out / p.name)
    (out / "pack_manifest.json").write_text(json.dumps({
        "source": MODEL_ID, "domain": args.domain, "experts_per_layer": args.budget,
        "selection": "shared core (expert_packs/base.json) + the domain's most used experts (pack_mask)",
        "original_expert_ids": ids}, indent=1))  # fmt: skip
    print(f"done: {out} ({total / 1e9:.1f} GB bf16, {args.budget} of 128 experts per layer)")


if __name__ == "__main__":
    main()
