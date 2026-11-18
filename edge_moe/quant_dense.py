"""4-bit linear layers, same format as the 26B expert packs (quant_experts.quantize).

QuantLinear holds a weight as packed 4-bit + bf16 group scales and dequantises
it on every forward. quantize_linears() swaps every nn.Linear under a module
for one, so a dense baseline (E4B, Gemma 3 12B) and the 26B's attention layers
get exactly the treatment the experts got: all linear layers 4-bit, embeddings
and norms bf16.
"""

from __future__ import annotations

import torch
from torch import nn

from .quant_experts import GROUP, dequantize, quantize


class QuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, device):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        q, s = quantize(lin.weight.detach().to(device))
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().to(device), requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, dequantize(self.q, self.s, x.dtype), self.bias)


class _RowView:
    """Stands in for embedding.weight: .device/.dtype/.shape, and rows dequantised on indexing."""

    def __init__(self, emb):
        self._e = emb
        self.device, self.dtype = emb.q.device, torch.bfloat16
        self.shape = (emb.num_embeddings, emb.embedding_dim)

    def __getitem__(self, key):
        rows, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        out = dequantize(self._e.q[rows], self._e.s[rows], torch.bfloat16)
        return out[(Ellipsis, *rest)] if rest else out


class QuantTiedEmbedding(nn.Module):
    """4-bit token embedding; looks up and dequantises only the rows it needs."""

    def __init__(self, emb: nn.Embedding, device, chunk: int = 16384):
        super().__init__()
        w = emb.weight.detach()
        qs = [quantize(w[i : i + chunk].to(device)) for i in range(0, w.shape[0], chunk)]
        self.register_buffer("q", torch.cat([q for q, _ in qs]))
        self.register_buffer("s", torch.cat([s for _, s in qs]))
        self.register_buffer("embed_scale", getattr(emb, "embed_scale", torch.tensor(1.0)).detach().to(device), persistent=False)
        self.num_embeddings, self.embedding_dim = w.shape

    @property
    def weight(self):
        return _RowView(self)

    def forward(self, ids):
        return dequantize(self.q[ids], self.s[ids], torch.bfloat16) * self.embed_scale.to(torch.bfloat16)


class QuantLMHead(nn.Module):
    """Output head tied to a QuantTiedEmbedding; dequantises the vocab in chunks."""

    def __init__(self, emb: QuantTiedEmbedding, chunk: int = 32768):
        super().__init__()
        self._emb = [emb]  # shared, not a second copy
        self.chunk = chunk

    @property
    def weight(self):
        return self._emb[0].s

    def forward(self, h):
        e = self._emb[0]
        return torch.cat([nn.functional.linear(h, dequantize(e.q[i : i + self.chunk], e.s[i : i + self.chunk], h.dtype))
                          for i in range(0, e.num_embeddings, self.chunk)], -1)  # fmt: skip


def quantize_embeddings(model: nn.Module, device) -> None:
    """Swap the tied embedding + lm_head of a Gemma model for one shared 4-bit table."""
    lm = model.model.language_model
    qe = QuantTiedEmbedding(lm.embed_tokens, device)
    lm.embed_tokens = qe
    model.lm_head = QuantLMHead(qe)


def quantize_linears(root: nn.Module, device, skip=("lm_head", "router")) -> tuple[int, int]:
    """Replace nn.Linear modules in place. Returns (quantised, kept in bf16) counts."""
    done = kept = 0
    for name, mod in list(root.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in skip) or mod.in_features % GROUP:
            kept += 1
            continue
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child, QuantLinear(mod, device))
        done += 1
    return done, kept
