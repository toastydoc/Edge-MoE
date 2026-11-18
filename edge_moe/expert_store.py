"""All 3,840 experts of the 26B in 4-bit, held in system RAM, installed on the GPU by mask.

    store = ExpertStore.from_packs([...base, medicine, law, math, rest pack files...])
    store.install(model, mask)            # exactly these experts on the GPU, router restricted
    store.install_offload(model, slots)   # every expert reachable; `slots` per layer cached on the GPU

One store serves every 8GB configuration in the comparison, so they differ only
in which experts are on the GPU:
  fixed pack     base plate + one domain pack            (mask from pack files)
  joint          one trimmed model for all domains       (joint_mask)
  mix            base + top experts of several domains   (mix_mask)
  plan           top experts of this prompt's own router (plan_mask)
  offload        full model, experts copied in on demand (install_offload)

Pinned (page-locked) RAM lets copies to the GPU run at full PCIe speed.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .quant_experts import QuantExperts, _restrict_router, dequantize
from .router_hooks import find_routers

KEYS = ("gate_up.q", "gate_up.s", "down.q", "down.s")
SHORT = {"gate_up.q": "gu_q", "gate_up.s": "gu_s", "down.q": "dn_q", "down.s": "dn_s"}


def _shares(c: torch.Tensor) -> torch.Tensor:
    """Pick counts [L, E] -> each expert's share of its layer's picks."""
    return c / c.sum(1, keepdim=True).clamp(min=1)


def top_per_layer(score: torch.Tensor, n: int, keep: torch.Tensor | None = None) -> torch.Tensor:
    """Mask with `keep` plus the highest-scoring other experts, n per layer in total."""
    L, E = score.shape
    mask = keep.clone() if keep is not None else torch.zeros(L, E, dtype=torch.bool)
    for l in range(L):
        need = n - int(mask[l].sum())
        if need > 0:
            ranked = score[l].masked_fill(mask[l], float("-inf")).argsort(descending=True)
            mask[l, ranked[:need]] = True
    return mask


def joint_mask(profiles: list[torch.Tensor], n: int) -> torch.Tensor:
    """One fixed trimmed model: the n experts per layer with the highest mean usage share across profiles."""
    return top_per_layer(torch.stack([_shares(c) for c in profiles]).mean(0), n)


def mix_mask(base: torch.Tensor, profiles: list[torch.Tensor], n: int, weights: list[float] | None = None) -> torch.Tensor:
    """Base plate plus the experts most used across several domains (weighted), n per layer in total."""
    w = weights or [1.0] * len(profiles)
    score = sum(wi * _shares(c) for wi, c in zip(w, profiles))
    return top_per_layer(score, n, keep=base)


class RouterPlan:
    """Accumulates each router's full probability over experts for one prompt (before any masking)."""

    def __init__(self, model):
        self.routers = find_routers(model)
        self.mass = None
        self.handles = []

    def __enter__(self):
        L = len(self.routers)
        E = self.routers[0][1].proj.out_features
        self.mass = torch.zeros(L, E, device="cuda")

        def make(l):
            def hook(module, args, output):
                self.mass[l] += output[0].float().reshape(-1, E).sum(0)

            return hook

        self.handles = [r.register_forward_hook(make(l)) for l, (_, r) in enumerate(self.routers)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def plan_mask(mass: torch.Tensor, n: int, keep: torch.Tensor | None = None) -> torch.Tensor:
    """The n experts per layer the router wanted most for this prompt (always keeping `keep`, e.g. the base plate)."""
    return top_per_layer(mass.cpu(), n, keep=keep)


class OffloadExperts(nn.Module):
    """Every expert reachable: `slots` experts per layer cached on the GPU, the rest copied from pinned RAM
    when the router picks them, least recently used evicted. Counts bytes copied."""

    def __init__(self, num_experts, act_fn, store_layer: dict, slots: int, device, preload: list[int], stats: dict):
        super().__init__()
        self.num_experts, self.act_fn, self.src, self.stats = num_experts, act_fn, store_layer, stats
        self.slots = slots
        for k, v in store_layer.items():
            self.register_buffer(k, torch.empty((slots, *v.shape[1:]), dtype=v.dtype, device=device), persistent=False)
        self.lru: OrderedDict[int, int] = OrderedDict()  # expert -> slot, oldest first
        self.free = list(range(slots))
        for e in preload[:slots]:
            self._load(e)

    def _load(self, e: int) -> int:
        if e in self.lru:
            self.lru.move_to_end(e)
            return self.lru[e]
        slot = self.free.pop() if self.free else self.lru.popitem(last=False)[1]
        for k, v in self.src.items():
            getattr(self, k)[slot].copy_(v[e], non_blocking=True)
        self.stats["bytes"] += sum(v[e].numel() * v.element_size() for v in self.src.values())
        self.stats["misses"] += 1
        self.lru[e] = slot
        return slot

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        # resident experts first, so they are not evicted before use
        hit.sort(key=lambda e: e not in self.lru)
        self.stats["picks"] += len(hit)
        for e in hit:
            j = self._load(e)
            top_k_pos, token_idx = torch.where(expert_mask[e])
            x = hidden_states[token_idx]
            gate, up = nn.functional.linear(x, dequantize(self.gu_q[j], self.gu_s[j], x.dtype)).chunk(2, dim=-1)
            h = nn.functional.linear(self.act_fn(gate) * up, dequantize(self.dn_q[j], self.dn_s[j], x.dtype))
            out.index_add_(0, token_idx, (h * top_k_weights[token_idx, top_k_pos, None]).to(out.dtype))
        return out


class ExpertStore:
    def __init__(self, layers: list[dict]):
        self.layers = layers  # per layer: {"gu_q": [E, ...], "gu_s", "dn_q", "dn_s"} on CPU

    @classmethod
    def from_packs(cls, pack_files: list[str | Path], num_experts: int = 128, pin: bool = True) -> "ExpertStore":
        # On Windows, pinning ~13GB right after another process released its pinned RAM can fail
        # transiently ("resource already mapped"); wait and retry.
        for attempt in range(5):
            try:
                return cls._from_packs(pack_files, num_experts, pin)
            except torch.AcceleratorError:
                if attempt == 4:
                    raise
                import gc
                import time

                gc.collect()
                time.sleep(30)

    @classmethod
    def _from_packs(cls, pack_files, num_experts, pin) -> "ExpertStore":
        files = [safe_open(str(p), "pt") for p in pack_files]
        n_layers = sum(1 for k in files[0].keys() if k.endswith(".ids"))
        layers = []
        for l in range(n_layers):
            have = torch.zeros(num_experts, dtype=torch.bool)
            layer = {}
            for f in files:
                ids = f.get_tensor(f"layer{l}.ids").long()
                for k in KEYS:
                    t = f.get_tensor(f"layer{l}.{k}")
                    if SHORT[k] not in layer:
                        layer[SHORT[k]] = torch.empty((num_experts, *t.shape[1:]), dtype=t.dtype, pin_memory=pin)
                    layer[SHORT[k]][ids] = t
                have[ids] = True
            if not have.all():
                raise ValueError(f"layer {l}: experts {torch.nonzero(~have).flatten().tolist()[:8]}... in no pack file")
            layers.append(layer)
        return cls(layers)

    def gb(self) -> float:
        return sum(t.numel() * t.element_size() for layer in self.layers for t in layer.values()) / 1e9

    def install(self, model, mask: torch.Tensor, device="cuda") -> torch.Tensor:
        """Put exactly the experts in mask [L, E] on the GPU (replacing what was there); restrict routers."""
        for l, (rname, router) in enumerate(find_routers(model)):
            layer = model.get_submodule(rname.rsplit(".", 1)[0])
            ids = torch.nonzero(mask[l]).flatten()
            part = {"ids": ids.int(), **{k: self.layers[l][SHORT[k]][ids] for k in KEYS}}
            act = layer.experts.act_fn
            layer.experts = None
            layer.experts = QuantExperts(mask.shape[1], act, [part], device)
            _restrict_router(router, mask[l])
        torch.cuda.empty_cache()
        return mask.clone()

    def install_offload(self, model, slots: int, preload: torch.Tensor | None = None, device="cuda") -> dict:
        """Every expert reachable; `slots` per layer on the GPU, preloaded from the `preload` mask if given."""
        stats = {"bytes": 0, "misses": 0, "picks": 0}
        E = self.layers[0]["gu_q"].shape[0]
        for l, (rname, router) in enumerate(find_routers(model)):
            layer = model.get_submodule(rname.rsplit(".", 1)[0])
            pre = torch.nonzero(preload[l]).flatten().tolist() if preload is not None else []
            act = layer.experts.act_fn
            layer.experts = None
            torch.cuda.empty_cache()
            layer.experts = OffloadExperts(E, act, self.layers[l], slots, device, pre, stats)
            _restrict_router(router, torch.ones(E, dtype=torch.bool))
        torch.cuda.synchronize()
        stats.update(bytes=0, misses=0, picks=0)  # do not count the preload
        return stats
