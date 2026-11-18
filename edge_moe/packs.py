"""Plug-in domain packs on a frozen base plate (route B).

A pack is a set of low-rank (LoRA) weight deltas on the base plate's linear
layers plus a gate. Packs are trained one at a time against the same frozen
base, so any subset can be installed without retraining the others.

Single-pass routing: each gate reads the base plate's hidden state after
`gate_layer` (mean over the prompt), and packs only modify layers after it.
So by the time the first pack-modified layer runs, every gate has already
decided, and one forward pass both routes and answers.

The gate catalogue can hold gates for packs that are NOT installed (they are a
few kB each). A strong uninstalled gate means "I don't have a map for that":
the device knows which pack would help without sending the question anywhere.

File format (one .safetensors per pack):
    lora.<layer>.<module>.A / .B   the deltas
    gate.mean / gate.std / gate.w / gate.b
    metadata: name, base_model, rank, alpha, gate_layer, first_pack_layer, threshold
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRA(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.to(self.A.dtype) @ self.A.T) @ self.B.T * self.scale).to(x.dtype)


@dataclass
class RouteState:
    """Per-forward pack weights: name -> float or tensor [batch]. Missing = off."""

    weights: dict = field(default_factory=dict)


class PackedLinear(nn.Module):
    def __init__(self, base: nn.Linear, state: RouteState):
        super().__init__()
        self.base = base
        self.loras = nn.ModuleDict()
        self._state = [state]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        weights = self._state[0].weights
        for name, lora in self.loras.items():
            w = weights.get(name)
            if w is None:
                continue
            d = lora(x)
            if isinstance(w, torch.Tensor):
                d = d * w.to(d.dtype).view(-1, *([1] * (d.dim() - 1)))
            elif w != 1.0:
                d = d * w
            y = y + d
        return y


@dataclass
class Gate:
    mean: torch.Tensor
    std: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor
    threshold: float = 0.5

    def prob(self, feats: torch.Tensor) -> torch.Tensor:
        z = ((feats.float() - self.mean) / self.std) @ self.w + self.b
        return torch.sigmoid(z)

    def to(self, device):
        return Gate(self.mean.to(device), self.std.to(device), self.w.to(device), self.b.to(device), self.threshold)


def train_gate(pos: torch.Tensor, neg: torch.Tensor, steps: int = 400, weight_decay: float = 1e-2) -> Gate:
    """Logistic regression: this pack's domain (pos) vs generic text (neg).

    Negatives are generic, never another pack's data, so gates stay independent.
    """
    x = torch.cat([pos, neg]).float()
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(x.device)
    mean, std = x.mean(0), x.std(0).clamp(min=1e-4)
    xs = (x - mean) / std
    w = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros((), device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    pos_weight = len(neg) / len(pos)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            xs @ w + b, y, pos_weight=torch.tensor(pos_weight, device=x.device)
        ) + weight_decay * w.pow(2).sum() / 2  # fmt: skip
        loss.backward()
        return loss

    opt.step(closure)
    return Gate(mean.detach(), std.detach(), w.detach(), b.detach())


class PackSet:
    """Wraps a BasePlate's linear layers so packs can be installed and routed."""

    def __init__(self, plate, gate_layer: int = 10, rank: int = 16, alpha: float = 32.0):
        self.plate = plate
        self.gate_layer = gate_layer
        self.first_pack_layer = gate_layer + 1
        self.rank, self.alpha = rank, alpha
        self.state = RouteState()
        self.wrapped: dict[str, PackedLinear] = {}
        for i, layer in enumerate(plate.layers):
            if i < self.first_pack_layer:
                continue
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name)
                for t in TARGETS:
                    lin = getattr(parent, t, None)
                    if not isinstance(lin, nn.Linear):
                        continue
                    if t in ("k_proj", "v_proj") and getattr(parent, "is_kv_shared_layer", False):
                        continue  # these layers reuse earlier K/V; the projections never run
                    pl = PackedLinear(lin, self.state)
                    setattr(parent, t, pl)
                    self.wrapped[f"{i}.{parent_name}.{t}"] = pl
        self.installed: list[str] = []
        self.gates: dict[str, Gate] = {}  # catalogue: may include uninstalled packs
        self.meta: dict[str, dict] = {}
        # routing
        self.mode = "off"  # off | forced | gated | features
        self.forced: dict = {}
        self.soft = False
        self.pool_mask: torch.Tensor | None = None
        self.last_features: torch.Tensor | None = None
        self.last_gate_probs: dict[str, torch.Tensor] = {}
        plate.layers[gate_layer].register_forward_hook(self._gate_hook)

    # ---- packs -------------------------------------------------------------
    def new_pack(self, name: str) -> list[nn.Parameter]:
        params = []
        for pl in self.wrapped.values():
            lora = LoRA(pl.base.in_features, pl.base.out_features, self.rank, self.alpha).to(self.plate.device)
            pl.loras[name] = lora
            params += [lora.A, lora.B]
        self.installed.append(name)
        return params

    def remove_pack(self, name: str) -> None:
        for pl in self.wrapped.values():
            pl.loras.pop(name, None)
        if name in self.installed:
            self.installed.remove(name)

    def save_pack(self, name: str, path: str | Path, gate: Gate | None, extra: dict | None = None) -> None:
        tensors = {}
        for key, pl in self.wrapped.items():
            lora = pl.loras[name]
            tensors[f"lora.{key}.A"] = lora.A.detach().to(torch.bfloat16).cpu().contiguous()
            tensors[f"lora.{key}.B"] = lora.B.detach().to(torch.bfloat16).cpu().contiguous()
        meta = {
            "name": name, "base_model": self.plate.model_id, "rank": self.rank, "alpha": self.alpha,
            "gate_layer": self.gate_layer, "first_pack_layer": self.first_pack_layer, **(extra or {}),
        }  # fmt: skip
        if gate is not None:
            for k in ("mean", "std", "w", "b"):
                tensors[f"gate.{k}"] = getattr(gate, k).float().cpu().contiguous()
            meta["threshold"] = gate.threshold
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={"meta": json.dumps(meta)})

    def load_pack(self, path: str | Path, install: bool = True) -> str:
        """Always loads the gate into the catalogue; installs the deltas only if `install`."""
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            meta = json.loads(f.metadata()["meta"])
        if meta["base_model"] != self.plate.model_id or meta["gate_layer"] != self.gate_layer:
            raise ValueError(f"pack {path} was built for {meta['base_model']} / gate layer {meta['gate_layer']}")
        t = load_file(str(path))
        name = meta["name"]
        if "gate.w" in t:
            self.gates[name] = Gate(
                t["gate.mean"], t["gate.std"], t["gate.w"], t["gate.b"], meta.get("threshold", 0.5)
            ).to(self.plate.device)  # fmt: skip
        self.meta[name] = meta
        if install:
            for key, pl in self.wrapped.items():
                lora = LoRA(pl.base.in_features, pl.base.out_features, meta["rank"], meta["alpha"])
                lora.A.data = t[f"lora.{key}.A"].float()
                lora.B.data = t[f"lora.{key}.B"].float()
                pl.loras[name] = lora.to(self.plate.device).requires_grad_(False)
            self.installed.append(name)
        return name

    # ---- routing -----------------------------------------------------------
    def _gate_hook(self, module, args, output):
        if self.mode not in ("gated", "features"):
            return
        h = output[0] if isinstance(output, tuple) else output
        m = self.pool_mask.to(h.dtype).unsqueeze(-1)
        feats = (h * m).sum(1).float() / m.sum(1).clamp(min=1).float()
        self.last_features = feats
        if self.mode == "features":
            return
        self.last_gate_probs = {n: g.prob(feats) for n, g in self.gates.items()}
        weights = {}
        for n in self.installed:
            if n not in self.last_gate_probs:
                continue
            p = self.last_gate_probs[n]
            weights[n] = p if self.soft else (p >= self.gates[n].threshold).float()
        self.state.weights = weights

    def forward_hidden(self, input_ids, attention_mask, mode: str = "off", forced: dict | None = None):
        """mode: off (base only) | forced (fixed weights per pack) | gated | features."""
        self.mode = mode
        self.pool_mask = attention_mask
        self.state.weights = dict(forced or {}) if mode == "forced" else {}
        try:
            return self.plate.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        finally:
            self.mode = "off"
            self.state.weights = {}
