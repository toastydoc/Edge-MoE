"""Expert packs for the 26B: a base plate of universal experts plus domain packs.

    base plate  = everything that is not a routed expert (attention, shared MLP,
                  embeddings, routers) + the most-used experts on universal text
    domain pack = the experts a domain uses most that the base plate lacks

A device holds the base plate and whichever packs are installed; the router is
restricted to that set (router_hooks.restrict_experts). Packs are sets of
experts, so stacking them only moves the model closer to the full 26B: no
interference by construction.

The routers ship whole with the base plate (they are tiny), so the router itself
is the pack gate: the probability mass it puts on a pack's (missing) experts
says "this question wants that pack", and mass on experts in no pack says "I
don't have a map for this".

Manifest JSON: {"name", "kind": "base"|"pack", "experts": [[ids for layer 0], ...]}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .router_hooks import top_fraction_mask

HIDDEN, MOE_FF = 2816, 704
EXPERT_PARAMS = 3 * HIDDEN * MOE_FF  # gate, up, down projections
EXPERT_BYTES_BF16 = EXPERT_PARAMS * 2
EXPERT_BYTES_INT4 = EXPERT_PARAMS * 4.5 / 8  # 4-bit weights + one bf16 scale per 32


def counts(path: str | Path, key: str = "pick_counts") -> torch.Tensor:
    return torch.as_tensor(np.load(path)[key]).double()


def base_mask(universal_counts: torch.Tensor, fraction: float) -> torch.Tensor:
    return top_fraction_mask(universal_counts, fraction)


def pack_mask(domain_counts: torch.Tensor, base: torch.Tensor, total_fraction: float) -> torch.Tensor:
    """Experts not in `base`, the domain's most used first, until base + pack = total_fraction per layer."""
    L, E = domain_counts.shape
    target = round(E * total_fraction)
    pack = torch.zeros(L, E, dtype=torch.bool)
    for l in range(L):
        need = target - int(base[l].sum())
        if need <= 0:
            continue
        ranked = domain_counts[l].masked_fill(base[l], -1).argsort(descending=True)
        pack[l, ranked[:need]] = True
    return pack


def size_gb(mask: torch.Tensor) -> dict:
    n = int(mask.sum())
    return {"experts": n, "bf16_gb": n * EXPERT_BYTES_BF16 / 1e9, "int4_gb": n * EXPERT_BYTES_INT4 / 1e9}


def save_manifest(path: str | Path, name: str, kind: str, mask: torch.Tensor, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    experts = [torch.nonzero(row).flatten().tolist() for row in mask]
    path.write_text(json.dumps({"name": name, "kind": kind, **size_gb(mask), **meta, "experts": experts}), encoding="utf-8")


def load_manifest(path: str | Path, num_experts: int = 128) -> tuple[dict, torch.Tensor]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    mask = torch.zeros(len(m["experts"]), num_experts, dtype=torch.bool)
    for l, ids in enumerate(m["experts"]):
        mask[l, ids] = True
    return m, mask


class RouterMass:
    """Per-row router probability mass on given expert sets, averaged over tokens and layers.

    Attach to the routers alongside restrict_experts; records the router's full
    softmax (before any masking), which is what the base-plate device computes.
    """

    def __init__(self, routers, sets: dict[str, torch.Tensor]):
        self.routers = routers
        self.names = list(sets)
        self.sets = torch.stack([sets[n] for n in self.names]).float()  # [S, L, E]
        self.rows = None
        self.mask = None
        self.handles = []

    def start(self, token_mask: torch.Tensor, rows: torch.Tensor | None = None):
        """token_mask [B, T]; rows: bool [B] of which batch rows to record (others ignored)."""
        self.mask = token_mask.float()
        self.rows = rows
        self.acc = torch.zeros(token_mask.shape[0], len(self.names), device=token_mask.device)
        sets = self.sets.to(token_mask.device)

        def make(l):
            def hook(module, args, output):
                probs = output[0].float().view(*self.mask.shape, -1)  # [B, T, E]
                per_tok = torch.einsum("bte,se->bts", probs, sets[:, l])
                self.acc += (per_tok * self.mask[..., None]).sum(1)

            return hook

        self.handles = [r.register_forward_hook(make(l)) for l, (_, r) in enumerate(self.routers)]

    def stop(self) -> torch.Tensor:
        for h in self.handles:
            h.remove()
        self.handles = []
        out = self.acc / (self.mask.sum(1, keepdim=True) * len(self.routers))
        return out.cpu()
