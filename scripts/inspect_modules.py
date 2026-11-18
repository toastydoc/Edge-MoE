"""Build the real 26B architecture with no weights (meta device) and report
the router / expert modules and the parameter budget. Downloads config.json only.

    python scripts/inspect_modules.py
"""

import sys
from pathlib import Path

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_moe.router_hooks import find_routers  # noqa: E402

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-26B-A4B-it"

config = AutoConfig.from_pretrained(MODEL_ID)
with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(config)

routers = find_routers(model)
print(f"{type(model).__name__}: {len(routers)} routers")
print("first:", routers[0][0], "| last:", routers[-1][0])
r = routers[0][1]
print(f"router.proj: Linear({r.proj.in_features} -> {r.proj.out_features}), top_k={r.config.top_k_experts}")

layer0 = model.get_submodule(routers[0][0].rsplit(".", 1)[0])
print("layer children:", [n for n, _ in layer0.named_children()])
for n, p in layer0.experts.named_parameters():
    print(f"experts.{n}: {tuple(p.shape)}")

groups = {"routed experts": 0, "shared expert (mlp)": 0, "router": 0, "attention": 0, "embeddings": 0, "vision": 0, "other": 0}
for n, p in model.named_parameters():
    if n.startswith("lm_head") and config.tie_word_embeddings:
        continue  # tied to embed_tokens; not stored separately in the checkpoint
    if "vision" in n:
        k = "vision"
    elif ".experts." in n:
        k = "routed experts"
    elif ".mlp." in n:
        k = "shared expert (mlp)"
    elif ".router." in n:
        k = "router"
    elif ".self_attn." in n:
        k = "attention"
    elif "embed_tokens" in n:
        k = "embeddings"
    else:
        k = "other"
    groups[k] += p.numel()

total = sum(groups.values())
print(f"\n{'group':22} {'params':>10} {'share':>7} {'bf16 GB':>8} {'4-bit GB':>9}")
for k, v in groups.items():
    print(f"{k:22} {v / 1e9:9.3f}B {v / total:7.1%} {v * 2 / 1e9:8.2f} {v * 0.5 / 1e9:9.2f}")
print(f"{'total':22} {total / 1e9:9.3f}B")

n_exp = r.proj.out_features * len(routers)
per_expert = groups["routed experts"] / n_exp
skeleton = total - groups["routed experts"]
print(f"\nper expert: {per_expert / 1e6:.2f}M params = {per_expert * 0.5 / 1e6:.2f} MB at 4-bit")
print(f"skeleton (everything but routed experts): {skeleton / 1e9:.2f}B = {skeleton * 0.5 / 1e9:.2f} GB at 4-bit")
print(f"skeleton + 30% of experts: {(skeleton + 0.3 * groups['routed experts']) * 0.5 / 1e9:.2f} GB at 4-bit")
