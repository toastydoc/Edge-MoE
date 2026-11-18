"""Exercise the hooks against the real Gemma 4 module classes at toy size
(random weights, real config shrunk), so no 51GB download is needed."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_moe.router_hooks import RouterProfiler, find_routers, restrict_experts, top_fraction_mask  # noqa: E402

from tiny import EXPERTS, LAYERS, TOP_K, VOCAB, tiny_text_config  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(tiny_text_config()).eval()
    # Random init leaves router.proj near zero (near-uniform routing); sharpen it
    # so top-k is well separated and the tests are meaningful.
    for _, r in find_routers(m):
        torch.nn.init.normal_(r.proj.weight, std=1.0)
        torch.nn.init.uniform_(r.per_expert_scale, 0.5, 1.5)
    return m


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(1)
    ids = torch.randint(3, VOCAB, (4, 12))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0
    mask[3, 5:] = 0
    return ids, mask


def test_finds_one_router_per_layer(model):
    routers = find_routers(model)
    assert len(routers) == LAYERS
    assert all(n.endswith(".router") for n, _ in routers)


@torch.no_grad()
def test_counts_match_builtin_router_output(model, batch):
    ids, mask = batch
    prof = RouterProfiler(model)
    with prof:
        prof.set_token_mask(mask)
        out = model(input_ids=ids, attention_mask=mask, output_router_logits=True)

    n_tokens = int(mask.sum())
    assert prof.tokens_seen.tolist() == [n_tokens] * LAYERS
    assert prof.pick_counts.sum(dim=1).tolist() == [n_tokens * TOP_K] * LAYERS
    assert torch.allclose(prof.prob_mass.sum(dim=1), torch.full((LAYERS,), float(n_tokens), dtype=torch.float64))

    # Independent path: transformers' own output_router_logits (full probabilities).
    assert len(out.router_logits) == LAYERS
    for layer, probs in enumerate(out.router_logits):
        picks = probs[mask.reshape(-1).bool()].topk(TOP_K, dim=-1).indices
        expected = torch.bincount(picks.reshape(-1), minlength=EXPERTS)
        assert torch.equal(prof.pick_counts[layer], expected)


@torch.no_grad()
def test_hooks_do_not_change_outputs(model, batch):
    ids, mask = batch
    base = model(input_ids=ids, attention_mask=mask).logits
    # transformers installs its own persistent output-recorder hooks, so compare to a baseline
    before = [len(r._forward_hooks) for _, r in find_routers(model)]
    with RouterProfiler(model):
        assert [len(r._forward_hooks) for _, r in find_routers(model)] == [n + 1 for n in before]
        hooked = model(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(base, hooked)
    assert [len(r._forward_hooks) for _, r in find_routers(model)] == before


@torch.no_grad()
def test_allow_all_mask_is_identity(model, batch):
    ids, mask = batch
    base = model(input_ids=ids, attention_mask=mask).logits
    with restrict_experts(model, torch.ones(LAYERS, EXPERTS, dtype=torch.bool)):
        masked = model(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(base, masked, atol=1e-6)


@torch.no_grad()
def test_restriction_confines_picks_and_changes_output(model, batch):
    ids, mask = batch
    prof = RouterProfiler(model)
    with prof:
        prof.set_token_mask(mask)
        base = model(input_ids=ids, attention_mask=mask).logits

    allowed = top_fraction_mask(prof.pick_counts, fraction=0.5)
    assert allowed.sum(dim=1).tolist() == [EXPERTS // 2] * LAYERS

    prof.reset()
    with restrict_experts(model, allowed), prof:
        prof.set_token_mask(mask)
        masked = model(input_ids=ids, attention_mask=mask).logits

    assert int(prof.pick_counts[~allowed].sum()) == 0
    assert prof.pick_counts.sum(dim=1).tolist() == [int(mask.sum()) * TOP_K] * LAYERS
    assert not torch.allclose(base, masked, atol=1e-4)


def test_too_small_mask_rejected(model):
    allowed = torch.zeros(LAYERS, EXPERTS, dtype=torch.bool)
    allowed[:, : TOP_K - 1] = True
    with pytest.raises(ValueError):
        with restrict_experts(model, allowed):
            pass


@torch.no_grad()
def test_per_row_masks_match_separate_runs(model, batch):
    ids, mask = batch
    torch.manual_seed(2)
    sets = torch.rand(ids.shape[0], LAYERS, EXPERTS) < 0.6
    sets[:, :, :TOP_K] = True  # keep every layer legal
    sets[0] = True  # row 0 unrestricted
    with restrict_experts(model, sets):
        together = model(input_ids=ids, attention_mask=mask).logits
    for row in range(ids.shape[0]):
        with restrict_experts(model, sets[row]):
            alone = model(input_ids=ids[row : row + 1], attention_mask=mask[row : row + 1]).logits
        assert torch.allclose(together[row], alone[0], atol=1e-5)
