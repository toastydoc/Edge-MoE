"""Router logging and masking hooks for Gemma 4 MoE (google/gemma-4-26B-A4B-it).

Hook point (confirmed against transformers 5.23 source and the checkpoint's
safetensors headers):

    model.model.language_model.layers[i].router      -> Gemma4TextRouter
    model.model.language_model.layers[i].experts     -> Gemma4TextExperts (3D stacked weights)
    model.model.language_model.layers[i].mlp         -> the always-on "shared expert"

Gemma4TextRouter.forward(hidden_states_flat) returns a 3-tuple:

    router_probabilities  [B*S, E]  float32 softmax over all experts
    top_k_weights         [B*S, K]  renormalised to sum 1, then * per_expert_scale
    top_k_index           [B*S, K]  the picks

The router sees a flattened [batch*seq] view, so padding has to be excluded
with a token mask supplied before each forward pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROUTER_CLASS_NAME = "Gemma4TextRouter"


def find_routers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return [(module_name, router)] in layer order.

    Matches on class name rather than a hard-coded path so it works for both
    Gemma4ForConditionalGeneration (model.language_model.layers.N.router) and
    Gemma4ForCausalLM (model.layers.N.router).
    """
    routers = [(n, m) for n, m in model.named_modules() if type(m).__name__ == ROUTER_CLASS_NAME]
    if not routers:
        raise RuntimeError(
            f"No {ROUTER_CLASS_NAME} modules found. Is this a Gemma 4 MoE checkpoint "
            "(text_config.enable_moe_block=True)?"
        )
    return routers


class RouterProfiler:
    """Tallies expert picks per (layer, expert) across forward passes.

    Usage:
        prof = RouterProfiler(model)
        with prof:
            prof.set_token_mask(inputs["attention_mask"])
            model(**inputs)
        prof.save("profiles/law.npz")

    Accumulates, all shaped [num_layers, num_experts]:
        pick_counts  how many tokens had this expert in their top-k
        prob_mass    sum of the full softmax probability (before top-k), which
                     shows how much routing mass a top-k / masked subset drops
    """

    def __init__(self, model: nn.Module):
        self.routers = find_routers(model)
        self.layer_names = [n for n, _ in self.routers]
        self.num_layers = len(self.routers)
        self.num_experts = self.routers[0][1].proj.out_features
        self.top_k = self.routers[0][1].config.top_k_experts
        self.reset()
        self._handles: list = []
        self._mask: torch.Tensor | None = None

    def reset(self) -> None:
        self.pick_counts = torch.zeros(self.num_layers, self.num_experts, dtype=torch.int64)
        self.prob_mass = torch.zeros(self.num_layers, self.num_experts, dtype=torch.float64)
        self.tokens_seen = torch.zeros(self.num_layers, dtype=torch.int64)

    def set_token_mask(self, mask: torch.Tensor | None) -> None:
        """mask: [batch, seq] bool/int, 1 = count this token. None = count all.

        Pass attention_mask to drop padding; AND it with your own mask to also
        drop BOS / chat-template tokens.
        """
        self._mask = None if mask is None else mask.reshape(-1).bool()

    def _make_hook(self, layer: int):
        def hook(module, args, output):
            probs, _, top_k_index = output
            if self._mask is not None:
                if self._mask.numel() != top_k_index.shape[0]:
                    raise RuntimeError(
                        f"token mask has {self._mask.numel()} positions but the router saw "
                        f"{top_k_index.shape[0]}; call set_token_mask() before every forward pass "
                        "(during generate() with a KV cache, pass None)."
                    )
                keep = self._mask.to(top_k_index.device)
                top_k_index = top_k_index[keep]
                probs = probs[keep]
            self.pick_counts[layer] += torch.bincount(top_k_index.reshape(-1), minlength=self.num_experts).cpu()
            self.prob_mass[layer] += probs.sum(dim=0, dtype=torch.float64).cpu()
            self.tokens_seen[layer] += top_k_index.shape[0]

        return hook

    def attach(self) -> "RouterProfiler":
        if self._handles:
            raise RuntimeError("already attached")
        self._handles = [r.register_forward_hook(self._make_hook(i)) for i, (_, r) in enumerate(self.routers)]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    __enter__ = attach

    def __exit__(self, *exc) -> None:
        self.detach()

    def pick_frequency(self) -> torch.Tensor:
        """[layers, experts] fraction of tokens that picked each expert (rows sum to top_k)."""
        return self.pick_counts.double() / self.tokens_seen.clamp(min=1).double()[:, None]

    def save(self, path: str | Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pick_counts=self.pick_counts.numpy(),
            prob_mass=self.prob_mass.numpy(),
            tokens_seen=self.tokens_seen.numpy(),
            layer_names=np.array(self.layer_names),
            top_k=self.top_k,
            **meta,
        )


def top_fraction_mask(pick_counts: torch.Tensor | np.ndarray, fraction: float = 0.30) -> torch.Tensor:
    """Per-layer bool mask [layers, experts] keeping the most-picked `fraction` of experts."""
    counts = torch.as_tensor(pick_counts)
    keep = max(1, round(counts.shape[1] * fraction))
    idx = counts.topk(keep, dim=1).indices
    return torch.zeros_like(counts, dtype=torch.bool).scatter_(1, idx, True)


@contextmanager
def restrict_experts(model: nn.Module, allowed: torch.Tensor):
    """Restrict every router to `allowed` experts, re-normalising the weights.

    allowed: bool [num_layers, num_experts], or [rows, num_layers, num_experts] to
    give every batch row its own expert set (several conditions in one forward
    pass). Disallowed experts get probability 0,
    top-k is re-taken among the allowed ones, and weights are renormalised to sum
    to 1 then multiplied by per_expert_scale -- i.e. exactly the router's own
    post-softmax logic applied to the surviving experts. The shared expert (mlp)
    is untouched.

    Enter this *before* a RouterProfiler if you want the profiler to log the
    masked picks (forward hooks run in registration order).
    """
    routers = find_routers(model)
    allowed = torch.as_tensor(allowed).bool()
    num_experts = routers[0][1].proj.out_features
    if allowed.shape[-2:] != (len(routers), num_experts) or allowed.dim() not in (2, 3):
        raise ValueError(
            f"allowed must be [{len(routers)}, {num_experts}] or [rows, {len(routers)}, {num_experts}], "
            f"got {tuple(allowed.shape)}"
        )
    top_k = routers[0][1].config.top_k_experts
    if int(allowed.sum(dim=-1).min()) < top_k:
        raise ValueError(f"every layer needs at least top_k={top_k} allowed experts")
    per_row = allowed.dim() == 3

    def make_hook(layer_allowed: torch.Tensor):  # [experts] or [rows, experts]
        def hook(module, args, output):
            probs = output[0]
            mask = layer_allowed.to(probs.device)
            if per_row:
                # the router sees [rows * seq, experts], row-major
                seq, rem = divmod(probs.shape[0], mask.shape[0])
                if rem:
                    raise RuntimeError(f"{probs.shape[0]} router positions not divisible by {mask.shape[0]} rows")
                mask = mask.repeat_interleave(seq, dim=0)
            masked = probs.masked_fill(~mask, 0.0)
            weights, index = torch.topk(masked, k=top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            weights = weights * module.per_expert_scale[index]
            return probs, weights, index

        return hook

    handles = [r.register_forward_hook(make_hook(allowed[..., i, :])) for i, (_, r) in enumerate(routers)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
