"""Profile expert usage for one domain.

    python scripts/profile_domain.py --prompts data/prompts/law.jsonl --out profiles/law.npz

Input is JSONL, one object per line: {"prompt": "..."}.

Only the tokens of the prompt text itself are counted; the chat-template
wrapper (<bos>, turn markers, ...) is identical in every domain and would
inflate overlap. Prompts are split at random into halves A and B, tallied
separately, so an expert set chosen on A can be scored on unseen prompts in B.

Output .npz, each array [layers, experts] unless noted:
    pick_counts, prob_mass            all prompts
    pick_counts_a/_b, prob_mass_a/_b  the two halves
    tokens_seen, tokens_seen_a/_b     [layers]

Every batch re-reads all expert weights from disk, so keep batches large.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_moe.loading import MODEL_ID, input_device, load_streaming  # noqa: E402
from edge_moe.router_hooks import RouterProfiler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in args.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
    prompts = prompts[: args.limit]
    half = [i % 2 for i in range(len(prompts))]
    random.Random(0).shuffle(half)

    t0 = time.time()
    model, tokenizer = load_streaming(args.model)
    print(f"loaded in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)

    items = []  # (chat text, char span of the prompt inside it, half)
    for prompt, h in zip(prompts, half):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        start = text.find(prompt.strip())
        if start < 0:
            raise RuntimeError(f"prompt text not found verbatim in chat template output: {prompt[:60]!r}")
        items.append((text, start, start + len(prompt.strip()), h))
    items.sort(key=lambda x: len(x[0]))  # similar lengths together -> less padding

    device = input_device(model)
    profs = [RouterProfiler(model), RouterProfiler(model)]

    with profs[0], profs[1], torch.inference_mode():
        for b in range(0, len(items), args.batch_size):
            t1 = time.time()
            chunk = items[b : b + args.batch_size]
            enc = tokenizer(
                [c[0] for c in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_tokens,
                add_special_tokens=False,  # the chat template already added <bos>
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")
            starts = torch.tensor([c[1] for c in chunk])[:, None]
            ends = torch.tensor([c[2] for c in chunk])[:, None]
            content = (offsets[..., 0] >= starts) & (offsets[..., 1] <= ends) & enc["attention_mask"].bool()
            in_half = torch.tensor([c[3] for c in chunk])[:, None]
            for h, prof in enumerate(profs):
                prof.set_token_mask(content & (in_half == h))
            enc = enc.to(device)
            # base_model skips the LM head: logits over a 262k vocab cost ~0.5MB per token
            model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            print(
                f"batch {b // args.batch_size + 1}: {len(chunk)} prompts, {int(content.sum())} content tokens "
                f"of {int(enc['attention_mask'].sum())}, {time.time() - t1:.0f}s, "
                f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                flush=True,
            )

    a, b = profs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pick_counts=(a.pick_counts + b.pick_counts).numpy(),
        prob_mass=(a.prob_mass + b.prob_mass).numpy(),
        tokens_seen=(a.tokens_seen + b.tokens_seen).numpy(),
        pick_counts_a=a.pick_counts.numpy(),
        pick_counts_b=b.pick_counts.numpy(),
        prob_mass_a=a.prob_mass.numpy(),
        prob_mass_b=b.prob_mass.numpy(),
        tokens_seen_a=a.tokens_seen.numpy(),
        tokens_seen_b=b.tokens_seen.numpy(),
        layer_names=np.array(a.layer_names),
        top_k=a.top_k,
        model=args.model,
        prompts=str(args.prompts),
        num_prompts=len(prompts),
    )
    total = a.pick_counts + b.pick_counts
    print(f"saved {args.out}: {len(prompts)} prompts, {int(a.tokens_seen[0] + b.tokens_seen[0])} content tokens")
    print(f"experts never picked: {int((total == 0).sum())} of {total.numel()}")


if __name__ == "__main__":
    main()
