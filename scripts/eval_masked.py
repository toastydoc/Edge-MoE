"""Step 4: score the 26B with its routers restricted to an expert subset.

    python scripts/eval_masked.py [--domains math law] [--limit 8]

For every held-out multiple-choice question the model is run once per
condition, all conditions stacked in the same batch (each row gets its own
expert mask), and the answer is the highest-scoring of the tokens A/B/C/D at
the first answer position. Zero-shot, chat format, unquantised bf16 weights.

Conditions (expert sets come from profiles/, built on disjoint questions):
    full         all 128 experts per layer (reference)
    own20/30/50  the domain's own most-picked 20/30/50%
    universal30  one set for all domains (pooled pick frequency)
    other30      the least similar domain's 30% (the "wrong bundle")
    random30     random 30% per layer (control)

Writes results/step4_<domain>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.loading import MODEL_ID, input_device, load_streaming  # noqa: E402
from edge_moe.router_hooks import restrict_experts, top_fraction_mask  # noqa: E402

LETTERS = "ABCD"


def format_question(row: dict) -> str:
    options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, row["choices"]))
    return f"{row['question'].strip()}\n\n{options}\n\nAnswer with the letter of the correct option only."


def build_conditions(domain: str, profiles: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    own = torch.as_tensor(profiles[domain])
    sets30 = {n: top_fraction_mask(c, 0.30) for n, c in profiles.items()}

    def jaccard(a, b):
        return float(((a & b).sum(1) / (a | b).sum(1)).mean())

    farthest = min((n for n in profiles if n != domain), key=lambda n: jaccard(sets30[domain], sets30[n]))
    counts = [torch.as_tensor(c, dtype=torch.float64) for c in profiles.values()]
    pooled = sum(c / c.sum(1, keepdim=True) for c in counts)
    g = torch.Generator().manual_seed(0)
    return {
        "full": torch.ones_like(own, dtype=torch.bool),
        "own20": top_fraction_mask(own, 0.20),
        "own30": sets30[domain],
        "own50": top_fraction_mask(own, 0.50),
        "universal30": top_fraction_mask(pooled, 0.30),
        f"other30({farthest})": sets30[farthest],
        "random30": top_fraction_mask(torch.rand(own.shape, generator=g), 0.30),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--position-budget", type=int, default=12000, help="max rows * seq_len per batch")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    profiles = {
        p.stem: np.load(p)["pick_counts"] for p in sorted((ROOT / "profiles").glob("*.npz")) if not p.stem.startswith("_")
    }
    domains = args.domains or list(profiles)

    model, tok = load_streaming(args.model)
    device = input_device(model)
    lm_head = model.get_output_embeddings()
    letter_ids = []
    for l in LETTERS:
        ids = tok.encode(l, add_special_tokens=False)
        assert len(ids) == 1, f"{l!r} is not a single token: {ids}"
        letter_ids.append(ids[0])
    letter_ids = torch.tensor(letter_ids, device=device)

    for domain in domains:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"{domain}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = rows[: args.limit]
        conds = build_conditions(domain, profiles)
        names = list(conds)
        masks = torch.stack([conds[n] for n in names])  # [C, L, E]
        C = len(names)

        texts = [
            tok.apply_chat_template([{"role": "user", "content": format_question(r)}], tokenize=False, add_generation_prompt=True)
            for r in rows
        ]
        lengths = [min(len(tok.encode(t, add_special_tokens=False)), args.max_tokens) for t in texts]
        order = sorted(range(len(rows)), key=lambda i: lengths[i])
        batches, cur = [], []
        for i in order:  # sorted ascending, so lengths[i] is the padded length if i joins
            if cur and (len(cur) + 1) * C * lengths[i] > args.position_budget:
                batches.append(cur)
                cur = []
            cur.append(i)
        batches.append(cur)

        pred = torch.full((len(rows), C), -1, dtype=torch.long)
        top1_is_letter = torch.zeros(len(rows), C, dtype=torch.bool)
        t0 = time.time()
        for bi, idx in enumerate(batches):
            tok.truncation_side = "left"  # keep the options and instruction if a passage is too long
            enc = tok(
                [texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_tokens, add_special_tokens=False,
            ).to(device)  # fmt: skip
            ids = enc["input_ids"].repeat_interleave(C, dim=0)
            att = enc["attention_mask"].repeat_interleave(C, dim=0)
            with torch.inference_mode(), restrict_experts(model, masks.repeat(len(idx), 1, 1)):
                hidden = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
                last = hidden[torch.arange(ids.shape[0], device=device), att.sum(1) - 1]
                logits = lm_head(last).float()
            choice = logits[:, letter_ids].argmax(-1).view(len(idx), C).cpu()
            is_letter = torch.isin(logits.argmax(-1), letter_ids).view(len(idx), C).cpu()
            for j, i in enumerate(idx):
                pred[i] = choice[j]
                top1_is_letter[i] = is_letter[j]
            print(
                f"{domain} batch {bi + 1}/{len(batches)}: {len(idx)} questions x {C} conditions, "
                f"seq {ids.shape[1]}, {time.time() - t0:.0f}s elapsed, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                flush=True,
            )

        answers = torch.tensor([r["answer"] for r in rows])
        acc = {n: float((pred[:, c] == answers).float().mean()) for c, n in enumerate(names)}
        out = {
            "domain": domain,
            "model": args.model,
            "n": len(rows),
            "conditions": names,
            "experts_kept": {n: float(conds[n].float().mean()) for n in names},
            "accuracy": acc,
            "top1_is_letter": {n: float(top1_is_letter[:, c].float().mean()) for c, n in enumerate(names)},
            "agreement_with_full": {n: float((pred[:, c] == pred[:, 0]).float().mean()) for c, n in enumerate(names)},
            "answers": answers.tolist(),
            "pred": {n: pred[:, c].tolist() for c, n in enumerate(names)},
            "subjects": [r["subject"] for r in rows],
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_limit{args.limit}" if args.limit else ""
        (args.out_dir / f"step4_{domain}{suffix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"== {domain} (n={len(rows)}): " + "  ".join(f"{n} {a:.3f}" for n, a in acc.items()), flush=True)
        print(f"   top-1 token is a letter: " + "  ".join(f"{n} {v:.2f}" for n, v in out["top1_is_letter"].items()), flush=True)


if __name__ == "__main__":
    main()
