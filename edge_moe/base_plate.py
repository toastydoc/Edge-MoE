"""The route-B base plate: a frozen Gemma 4 E4B, text only.

GPU holds the decoder layers and the main embedding (tied to the LM head),
about 9.3GB in bf16. The 2.8B-parameter per-layer embedding table (PLE) is a
pure lookup, so it stays in system RAM and only the looked-up rows are sent
to the GPU. Vision and audio towers are dropped.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE_MODEL_ID = "google/gemma-4-E4B-it"
LETTERS = "ABCDE"


class _CpuLookup(nn.Module):
    """Embedding lookup done in system RAM; the table is hidden from .to()."""

    def __init__(self, emb: nn.Module):
        super().__init__()
        self._table = [emb]  # list, so it is not a registered submodule

    @property
    def weight(self):
        return self._table[0].weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._table[0](ids.cpu()).to(ids.device)


class BasePlate:
    def __init__(self, model_id: str = BASE_MODEL_ID, device: str = "cuda"):
        full = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cpu")
        full.eval()
        for p in full.parameters():
            p.requires_grad_(False)
        lm = full.model.language_model
        lm.embed_tokens_per_layer = _CpuLookup(lm.embed_tokens_per_layer)
        lm.to(device)
        self.lm = lm
        self.lm_head = full.lm_head.to(device)  # tied to lm.embed_tokens
        self.config = lm.config
        self.softcap = self.config.final_logit_softcapping
        self.device = torch.device(device)
        self.model_id = model_id
        del full.model.vision_tower, full.model.audio_tower
        self._full = full  # keeps the CPU embedding table alive
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "right"
        self.letter_ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
        self.end_of_turn = "<turn|>"

    @property
    def layers(self) -> nn.ModuleList:
        return self.lm.layers

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.lm_head(hidden).float()
        if self.softcap:
            x = torch.tanh(x / self.softcap) * self.softcap
        return x

    def prompt_text(self, user: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

    def encode(self, texts: list[str], max_tokens: int) -> dict:
        self.tok.truncation_side = "left"
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens, add_special_tokens=False)
        return enc.to(self.device)


def format_mc(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, choices))
    return f"{question.strip()}\n\n{opts}\n\nAnswer with the letter of the correct option only."
