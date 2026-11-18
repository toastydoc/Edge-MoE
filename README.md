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

### Held-out policy tasks: 63 eCFR tasks, automatic grading

The tasks are extracting cited authorities, filling a value with a supporting citation, and reconciling two dated versions of a rule.

| Configuration | Peak, 8k docs | Peak, 32k prompt | Extract F1 | Cite value / right section | Reconcile | Load / cold first answer | Decode |
|---|---|---|---|---|---|---|---|
| Gemma 4 12B | 8.3 GB | 8.7 GB | .60 | 1.00 / .96 | 1.00 | 68 s / 89 s | 3.3 tok/s |
| 26B base + 40% law pack | 7.7 GB | 8.9 GB | .39 | .96 / .71 | .60 | 12 s / 25 s | 5.6 tok/s |
| 26B base + law pack | 9.0 GB | 10.2 GB | .51 | 1.00 / .79 | .73 | 13 s / 34 s | 5.4 tok/s |
| 26B, automatic routing | 9.0 GB | 10.2 GB | .51 | 1.00 / .79 | .73 | 9 s / 34 s | 5.4 tok/s |
| 26B base + all three packs | 11.0 GB | 12.3 GB | .65 | 1.00 / .83 | 1.00 | 15 s / 31 s | 5.5 tok/s |

The pack chosen from MMLU law questions does **not** carry over to regulatory work: it trails the 12B on every task type. Adding the math and medicine packs does, and reconciling dates looks like the reason. The selector picked law for all 63 tasks; it costs 0.3 s per question, and a pack swap takes about 3.4 s. The lesson is to build packs from the work the device will actually do.

## Caveats

- **Small samples:** 200 questions per domain is about ±3 points. With 15 reconcile tasks, each task is worth 6.7 points.
- **Plain rounding:** round-to-nearest 4-bit hurts dense models more than the MoE; E4B lost 7 points. GPTQ, AWQ or Q4_K_M would be a fairer comparison.
- **Speed:** speeds come from a plain-PyTorch prototype that unpacks the 4-bit weights on every step. They compare the models with each other, not with what a real runtime would reach.
- **Load times:** the 26B loads from pre-quantised pack files, while the dense models are quantised as they load.
- **Long prompts:** a 32k-token prompt is slow on the 26B (91–152 s). The cause has not been diagnosed.

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
| Run the policy tasks | `scripts/eval_policy.py --config ...` |

The repository includes the expert-usage profiles (`profiles*/`), the pack manifests and selectors (`expert_packs/`), the test sets (`data/test/`, `data/policy/tasks.jsonl`) and every result with per-question outputs (`results/`). The `train_*` and `distill` scripts are earlier experiments with trained packs; the packaging approach above does not use them.

Data sources: MMLU (MIT licence), and the eCFR (US federal regulations, public domain). Model weights are subject to the [Gemma terms of use](https://ai.google.dev/gemma/terms); none are redistributed here.
