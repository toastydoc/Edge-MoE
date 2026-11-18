# Edge-MoE: domain expert packs for a mixture-of-experts model

**Research question.** Can an existing mixture-of-experts model be repackaged, without retraining its weights, so that one shared base and a user-selected domain pack deliver competitive answer quality on an 8 GB consumer GPU?

The idea in one line: **map the router by domain, and shed what is not in the domain.** We keep every router (gate), but only the expert weights a domain uses. Nothing is retrained, and there are no LoRAs.

The model is `google/gemma-4-26B-A4B-it`, which has 30 layers of 128 experts and routes each token to 8 of them. It is split into:

- **Base plate:** attention, embeddings, all routers, plus the experts that general text uses most (10–20% per layer).
- **Domain packs (medicine, law, math):** the experts that domain uses most and that are not already in the base. Packs stack, and an expert shared by two packs is loaded once.
- **Routers restricted to installed experts:** each router picks its top 8 among the experts present. Installing a pack only adds experts.
- **Pack selector ("gate"):** a small logistic regression over the base plate's hidden states. It picks a pack before any pack is loaded. When it fires for a domain with no pack installed, that is a missing-module signal: the device can fetch the pack or defer to an online service.

All weights are 4-bit (symmetric round-to-nearest, groups of 32, bf16 scales), and the dense baselines get exactly the same treatment. The tests ran on an RTX 5080 16 GB with a Ryzen 7 9700X and 32 GB RAM, under Windows 11.

## Results

### MMLU, 200 questions per domain, everything 4-bit (embeddings included)

| Option | Peak VRAM | law | math | Average of 5 domains | Speed |
|---|---|---|---|---|---|
| 26B base + 40% law pack | 7.1 GB | .655 | .305 | 49.7 | 3–5 tok/s |
| 26B base + 40% math pack | 7.1 GB | .605 | .590 | 60.8 | 3–5 tok/s |
| 26B base + full domain pack | 8.5 GB | .705–.750 | .680–.700 | — | 3–5 tok/s |
| 26B base + three packs | 10.5 GB | .770 | .645 | 75.0 | about 5 tok/s |
| Full 26B | 14.9 GB | .775 | .660 | 79.3 | — |
| Gemma 4 12B | 7.5 GB | .730 | .565 | 70.3 | 3.2 tok/s |
| Gemma 3 12B | 7.5 GB | .655 | .465 | 67.2 | 3.3 tok/s |
| Gemma 4 E4B | 3.2 GB | .610 | .385 | 59.0 | 7.3 tok/s |

At 8 GB the pack ties Gemma 4 12B on math, trails it by 7.5 points on law, and is much weaker outside its domain. With the full pack at 8.5 GB it matches the 12B on law and leads it by 11–14 points on math. The selectors fire on 88–100% of their own domain's questions.

### Held-out policy tasks at a fixed 8 GB: 122 eCFR tasks

The tasks are 50 extract (list the authorities a section cites), 50 cite (fill a value and cite its section) and 22 reconcile (two dated versions of a rule: which value applies on a date). They are graded automatically. Every 26B configuration at 8 GB keeps exactly **51 of 128 experts per layer** on the GPU and runs on the same 4-bit weights; they differ only in *which* experts are there.

| Configuration (8 GB unless noted) | Extract F1 | Cite value / right section | Reconcile | Peak GPU (in use / reserved) | First token, 2k / 8k | Decode | 32k prompt |
|---|---|---|---|---|---|---|---|
| Gemma 4 12B (dense) | .59 | .98 / .90 | **1.00** | 8.3 / 9.2 GB | 1.4 / 4.5 s | 3.2 tok/s | 25 s |
| Base + law pack (from MMLU law) | .39 | 1.00 / .78 | .68 | 7.7 / 8.5 GB | 1.7 / 5.9 s | 5.0 tok/s | 30 s |
| **Base + regulation pack** (non-test CFR titles) | .57 | 1.00 / **.98** | .73 | 7.7 / 8.5 GB | 1.7 / 6.0 s | 5.1 tok/s | 30 s |
| Base + regulation and math, shared budget | .55 | 1.00 / .98 | .73 | 7.7 / 8.5 GB | 1.6 / 5.8 s | 5.1 tok/s | 27 s |
| Base + reconcile pack (version-comparison prompts) | .55 | 1.00 / 1.00 | .82 | 7.7 / 8.5 GB | 1.6 / 5.4 s | 5.5 tok/s | 26 s |
| Base + regulation and reconcile, shared budget | .46 | 1.00 / 1.00 | .86 | 7.7 / 8.5 GB | 1.6 / 5.4 s | 5.5 tok/s | 27 s |
| One trimmed model for all five domains | .51 | .98 / .92 | .91 | 7.7 / 8.5 GB | 1.6 / 5.6 s | 5.4 tok/s | 27 s |
| Per-prompt plan (router picks the 51) | .56 | 1.00 / .98 | .73 | 7.7 / 8.4 GB | 3.6 / 7.5 s | 5.5 tok/s | 26 s |
| **Expert offloading** (all 128 reachable from RAM, 51 cached) | **.77** | 1.00 / .98 | **1.00** | 7.7 / 8.5 GB | 3.1 / 11.2 s | 5.4 tok/s | 57 s |
| Base + three packs (11 GB, over the limit) | .66 | 1.00 / 1.00 | 1.00 | 11.0 / 11.8 GB | 2.1 / 7.9 s | 5.2 tok/s | 37 s |

Paired comparisons on the same tasks (bootstrap 95% intervals), against the regulation pack:

- **Packs built from the real work matter.** The regulation pack beats the MMLU-law pack by 18 points on extract and 20 on citing the right section, and cites the right section more often than the 12B (.98 vs .90).
- **Adding a second domain cost nothing measurable.** Regulation plus math in the same 51 slots is within noise of regulation alone on all three task types.
- **The 12B wins reconcile** (+27 points, 6 wins and 0 losses). All six pack failures have the same shape: the version printed first governs, and the pack chooses the second one. Configurations that keep more general experts (the trimmed model, offloading, 11 GB) do not show this.
- **Offloading has the best quality at 8 GB.** It is 21 points ahead on extract and 27 on reconcile, and at least as good as the 11 GB model. The cost is the prompt: it copies about 0.85 GB from RAM per prompt token (a third of expert lookups miss the GPU cache), first token takes roughly twice as long, and a 32k prompt takes 57 s instead of 30 s. Decoding speed is unaffected.
- **A reconcile-aware pack narrowed the gap but did not close it.** It was profiled on 1,600 synthetic version-comparison prompts from the non-test titles, each with a worked answer (`scripts/build_reconcile_prompts.py`); the prompts reuse the test's question wording but none of its regulations. Alone it raised reconcile from .73 to .82 at no cost elsewhere; mixed with the regulation pack it reached .86 but lost 11 points on extract. Neither reconcile gain is outside noise (5 wins, 2–3 losses), and every remaining failure is still the same "second version" mistake, so the bias is reduced, not removed. Offloading and the 12B stay at 1.00.
- **Picking experts per prompt did not help.** One base-only router pass (0.5 s) plus a swap (1.5 s, about 190 experts changed) scored the same as the fixed regulation pack.

**Where this leaves the claim.** At 8 GB the packs are 1.6× faster than the 12B at decoding and fit under the limit, which the 12B does not quite. On quality they match the 12B on extract, beat it on citations and lose on reconcile. The decisive alternative turned out to be offloading the full 26B: same memory, equal or better quality on every task type, and only the time to first token is worse. A pack system only wins where first-token latency on long prompts matters more than reconcile-type reasoning.

An earlier 63-task run (law packs only, no limit) is kept in `results/policy_*.json` without the `_tasks_v2` suffix.

## Caveats

- **Small samples:** 200 questions per domain is about ±3 points. There are 22 reconcile tasks, so each is worth 4.5 points; differences of one or two tasks are noise.
- **Plain rounding:** round-to-nearest 4-bit hurts dense models more than the MoE; E4B lost 7 points. GPTQ, AWQ or Q4_K_M would be a fairer comparison.
- **Speed:** speeds come from a plain-PyTorch prototype that unpacks the 4-bit weights on every step. They compare the models with each other, not with what a real runtime would reach.
- **Load times:** the 26B loads from pre-quantised pack files, while the dense models are quantised as they load.
- **Memory:** "8 GB" means peak memory in use by PyTorch; the allocator reserves 0.7–0.9 GB more. Gemma 4 12B peaks at 8.3 GB, slightly over the limit.
- **Long prompts:** the earlier slow 32k prompts (91–152 s) were allocator fragmentation spilling into system RAM; freeing the cache after each prefill chunk fixed it (now 25–37 s, 57 s with offloading).
- **Offloading uses this prototype's own LRU cache** with pinned RAM, preloaded with the trimmed model's experts. A production offloading runtime would prefetch and could be faster; llama.cpp was not tested.
- **Reconcile tasks** come only from yearly penalty and threshold updates where a value sits in the paragraph text.

## Prior work

Expert pruning and skipping without retraining: Lu et al., "Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models" (ACL 2024), and work on task-specific expert pruning. What this project adds is the packaging: one shared base plate, stackable domain packs, a pack selector that runs on the base alone, and the missing-pack signal.

## Reproducing

Use Python 3.12 and CUDA 12.8 (RTX 50-series), then `pip install -r requirements.txt` (see that file for the torch index). Gemma models are gated on Hugging Face: accept the licence and run `hf auth login` first. Pack files and model weights are not in this repository; the scripts rebuild them (about 73 GB under `quant/`).

| Step | Script |
|---|---|
| Build prompt sets and test sets | `scripts/build_datasets.py`, `scripts/build_base_prompts.py` |
| Profile expert use per domain | `scripts/profile_domain.py` |
| Choose base plate and packs | `scripts/build_expert_packs.py`, `scripts/build_small_packs.py` |
| Train pack selectors | `scripts/train_expert_gates.py` |
| Write 4-bit pack files | `scripts/quant_packs.py` |
| Score the 26B from packs on MMLU | `scripts/eval_quant.py --quant-dense --quant-embed` |
| Score the dense baselines | `scripts/eval_dense_quant.py --quant-embed` |
| Build the eCFR policy tasks | `scripts/build_policy_tasks.py` |
| Profile the regulation and reconcile packs | `scripts/build_reg_prompts.py`, `scripts/build_reconcile_prompts.py`, then `scripts/profile_domain.py` |
| Run the policy tasks | `scripts/eval_policy.py --config ...` |
| The 8 GB comparison | `scripts/run_8gb_comparison.ps1` (`edge_moe/expert_store.py` holds all configurations) |

The repository includes the expert-usage profiles (`profiles*/`), the pack manifests and selectors (`expert_packs/`), the test sets (`data/test/`, `data/policy/tasks*.jsonl`) and every result with per-question outputs (`results/`). The `train_*` and `distill` scripts are earlier experiments with trained packs; the packaging approach above does not use them.

Data sources: MMLU (MIT licence), and the eCFR (US federal regulations, public domain). Model weights are subject to the [Gemma terms of use](https://ai.google.dev/gemma/terms); none are redistributed here.
