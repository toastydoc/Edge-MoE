"""4-bit expert packs for the 26B, held resident on the GPU.

Pack file (safetensors), one per base plate / domain pack:
    layer<L>.ids          int32 [n]             expert ids this pack holds in layer L
    layer<L>.gate_up.q    uint8 [n, 1408, 1408] two 4-bit weights per byte
    layer<L>.gate_up.s    bf16  [n, 1408, 88]   one scale per 32 weights
    layer<L>.down.q       uint8 [n, 2816, 352]
    layer<L>.down.s       bf16  [n, 2816, 22]
Symmetric round-to-nearest, group 32 along the input dimension (4.5 bits/weight).

load_resident(pack_files) builds the 26B with every non-expert weight on the GPU
in bf16 (vision tower skipped), the installed packs' experts on the GPU in 4-bit,
and the routers restricted to the installed experts. Nothing streams from disk,
so it can generate() normally.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from .loading import MODEL_ID, SafetensorsIndex, _model_class, resolve_model_dir
from .router_hooks import find_routers

GROUP = 32


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [..., in] -> (packed uint8 [..., in/2], scale bf16 [..., in/GROUP])."""
    g = w.float().reshape(*w.shape[:-1], -1, GROUP)
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
    q = (g / scale).round().clamp(-8, 7).to(torch.int16) + 8  # 0..15
    q = q.reshape(*w.shape[:-1], -1, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(-1).to(torch.bfloat16)


def dequantize(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    lo = (packed & 15).to(torch.int8)
    hi = (packed >> 4).to(torch.int8)
    q = torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], -1) - 8
    w = q.reshape(*scale.shape, GROUP).to(dtype) * scale[..., None].to(dtype)
    return w.reshape(*packed.shape[:-1], -1)


def build_pack_file(mask: torch.Tensor, out: str | Path, name: str, model_id: str = MODEL_ID, device="cuda") -> dict:
    """Quantise the experts selected by mask [layers, experts] into one pack file."""
    index = SafetensorsIndex(resolve_model_dir(model_id))
    tensors, err = {}, []
    for l in range(mask.shape[0]):
        ids = torch.nonzero(mask[l]).flatten()
        tensors[f"layer{l}.ids"] = ids.int()
        for part in ("gate_up", "down"):
            w = index.read(f"model.language_model.layers.{l}.experts.{part}_proj", ids.numpy()).to(device)
            q, s = quantize(w)
            tensors[f"layer{l}.{part}.q"], tensors[f"layer{l}.{part}.s"] = q.cpu(), s.cpu()
            err.append(float(((dequantize(q, s).float() - w.float()).norm() / w.float().norm())))
    meta = {"name": name, "model": model_id, "group": GROUP, "experts": int(mask.sum()),
            "rel_error_mean": sum(err) / len(err)}  # fmt: skip
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={"meta": json.dumps(meta)})
    meta["file_gb"] = Path(out).stat().st_size / 1e9
    return meta


class QuantExperts(nn.Module):
    """Drop-in for Gemma4TextExperts holding only installed experts, in 4-bit."""

    def __init__(self, num_experts, act_fn, parts: list[dict], device):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = act_fn
        # packs are self-contained, so two packs may hold the same expert: keep the first copy
        seen, kept = set(), []
        for p in parts:
            keep = torch.tensor([int(i) not in seen and not seen.add(int(i)) for i in p["ids"]], dtype=torch.bool)
            kept.append({k: v[keep] for k, v in p.items()})
        parts = kept
        ids = torch.cat([p["ids"].long() for p in parts])
        local = torch.full((num_experts,), -1, dtype=torch.long)
        local[ids] = torch.arange(len(ids))
        self.local = local.tolist()
        cat = lambda k: torch.cat([p[k] for p in parts]).to(device)  # noqa: E731
        self.register_buffer("gu_q", cat("gate_up.q"))
        self.register_buffer("gu_s", cat("gate_up.s"))
        self.register_buffer("dn_q", cat("down.q"))
        self.register_buffer("dn_s", cat("down.s"))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for e in hit:
            j = self.local[e]
            if j < 0:
                raise RuntimeError(f"router picked expert {e}, which is not installed")
            top_k_pos, token_idx = torch.where(expert_mask[e])
            x = hidden_states[token_idx]
            gate, up = nn.functional.linear(x, dequantize(self.gu_q[j], self.gu_s[j], x.dtype)).chunk(2, dim=-1)
            h = nn.functional.linear(self.act_fn(gate) * up, dequantize(self.dn_q[j], self.dn_s[j], x.dtype))
            out.index_add_(0, token_idx, (h * top_k_weights[token_idx, top_k_pos, None]).to(out.dtype))
        return out


def _restrict_router(router: nn.Module, allowed: torch.Tensor) -> None:
    """Restrict a router to `allowed` experts (same rule as router_hooks.restrict_experts).

    The mask lives on router._allowed, so set_packs() / set_allowed() can change it later.
    """
    first = not hasattr(router, "_allowed")
    router._allowed = allowed.clone().to(router.proj.weight.device)
    if not first:
        return

    def hook(module, args, output):
        probs = output[0]
        masked = probs.masked_fill(~module._allowed, 0.0)
        w, idx = torch.topk(masked, k=module.config.top_k_experts, dim=-1)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-20)
        return probs, w * module.per_expert_scale[idx], idx

    router.register_forward_hook(hook)


def set_packs(model: nn.Module, pack_files: list[str | Path], device="cuda") -> torch.Tensor:
    """Install exactly these packs (replacing whatever was installed). Returns the installed mask."""
    device = torch.device(device)
    files = [safe_open(str(p), "pt") for p in pack_files]
    routers = find_routers(model)
    E = routers[0][1].proj.out_features
    installed = torch.zeros(len(routers), E, dtype=torch.bool)
    for l, (rname, router) in enumerate(routers):
        layer = model.get_submodule(rname.rsplit(".", 1)[0])
        parts = [{k: f.get_tensor(f"layer{l}.{k}") for k in ("ids", "gate_up.q", "gate_up.s", "down.q", "down.s")} for f in files]
        act = layer.experts.act_fn
        layer.experts = None  # free the old pack before allocating the new one
        layer.experts = QuantExperts(E, act, parts, device)
        for p in parts:
            installed[l, p["ids"].long()] = True
        _restrict_router(router, installed[l])
    del files
    torch.cuda.empty_cache()
    return installed


def set_allowed(model: nn.Module, mask: torch.Tensor) -> None:
    """Temporarily narrow the routers to a subset of the installed experts (mask [layers, experts])."""
    for l, (_, router) in enumerate(find_routers(model)):
        router._allowed = mask[l].to(router._allowed.device)


def load_resident(pack_files: list[str | Path], model_id: str = MODEL_ID, device="cuda", dtype=torch.bfloat16,
                  quant_dense: bool = False, quant_embed: bool = False):  # fmt: skip
    """Returns (model, tokenizer, installed mask [layers, experts]).

    quant_dense=True also puts every non-expert linear layer (attention, shared
    MLP) in 4-bit; routers, embeddings and norms stay bf16.
    """
    device = torch.device(device)
    model_dir = resolve_model_dir(model_id)
    config = AutoConfig.from_pretrained(model_dir)
    index = SafetensorsIndex(model_dir)
    with init_empty_weights():
        model = _model_class(config).from_config(config)
    for name in index.keys():
        if ".experts." in name or "vision" in name or "audio" in name:
            continue
        set_module_tensor_to_device(model, name, device, value=index.read(name), dtype=dtype)
    model.tie_weights()
    for name, buf in list(model.named_buffers()):
        if buf.device != device and not buf.is_meta:
            set_module_tensor_to_device(model, name, device, value=buf)
    for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)

    installed = set_packs(model, pack_files, device)
    if quant_dense:
        from .quant_dense import quantize_linears

        quantize_linears(model.model.language_model, device)
        torch.cuda.empty_cache()
    if quant_embed:
        from .quant_dense import quantize_embeddings

        quantize_embeddings(model, device)
        torch.cuda.empty_cache()
    left =[n for n, t in model.state_dict().items() if t.is_meta]
    if left:
        raise RuntimeError(f"weights still on meta: {left[:5]}")
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "right"
    return model, tok, installed
