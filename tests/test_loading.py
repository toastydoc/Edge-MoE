"""The disk-streamed load must give the same logits and router picks as a normal load."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_moe.loading import SafetensorsIndex, input_device, load_streaming  # noqa: E402
from edge_moe.router_hooks import RouterProfiler  # noqa: E402
from tiny import tiny_text_config  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")




@torch.no_grad()
def test_streamed_matches_normal_load(tmp_path):
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(tiny_text_config()).eval()
    for p in ref.parameters():  # default init is too flat to be a meaningful comparison
        torch.nn.init.normal_(p, std=0.3)
    ref.save_pretrained(tmp_path)

    ref = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).cuda().eval()
    model, _ = load_streaming(str(tmp_path), dtype=torch.float32)
    # experts hold no real weights between forward passes
    assert all(p.is_meta for n, p in model.named_parameters() if ".experts." in n)

    ids = torch.randint(3, 512, (4, 12), device=input_device(model))
    mask = torch.ones_like(ids)
    mask[2, 7:] = 0

    profs = []
    logits = []
    for m in (ref, model):
        prof = RouterProfiler(m)
        with prof:
            prof.set_token_mask(mask)
            logits.append(m(input_ids=ids, attention_mask=mask).logits.float().cpu())
        profs.append(prof)

    assert torch.allclose(logits[0], logits[1], atol=1e-4)
    assert torch.equal(profs[0].pick_counts, profs[1].pick_counts)
    assert int(profs[1].pick_counts.sum()) == int(mask.sum()) * 4 * 3


def test_index_reads_selected_expert_rows(tmp_path):
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(tiny_text_config()).save_pretrained(tmp_path)
    index = SafetensorsIndex(tmp_path)
    name = "model.layers.1.experts.down_proj"
    full = index.read(name)
    assert full.shape == (16, 64, 16)
    assert torch.equal(index.read(name, expert_ids=[5, 2, 11]), full[[5, 2, 11]])
