"""Score the 26B as base plate + expert packs, with the router as the pack gate.

    python scripts/eval_expert_packs.py [--domains math] [--limit 16] [--tag b20t50]

Conditions, stacked in one batch (each row gets its own expert set):
    full          all experts (reference)
    base          base plate only
    base+<pack>   base plate + one domain pack (medicine, law, math)
    base+all      base plate + all three packs
    gated         base + the pack the router asks for (decided from the
                  base row's router probabilities), else base alone

Gate: for each pack, the mean router probability on that pack's experts while
running on the base plate alone. Thresholds are the 95th percentile on general
questions no profile used (ARC test, CommonsenseQA validation), i.e. the gate
fires on ~5% of general questions. "unmapped" = mass on experts in no pack and
not in the base: the "I don't have a map for that" signal.

Writes results/expert_packs_<tag>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.expert_packs import RouterMass, load_manifest  # noqa: E402
from edge_moe.loading import MODEL_ID, input_device, load_streaming  # noqa: E402
from edge_moe.router_hooks import find_routers, restrict_experts  # noqa: E402

PACKS = ["medicine", "law", "math"]
LETTERS = "ABCDE"


def format_question(q, choices):
    options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, choices))
    return f"{q.strip()}\n\n{options}\n\nAnswer with the letter of the correct option only."


def calibration_items(n_arc=300, n_csqa=100):
    items = []
    for cfg, n in (("ARC-Challenge", n_arc * 2 // 3), ("ARC-Easy", n_arc - n_arc * 2 // 3)):
        for r in load_dataset("allenai/ai2_arc", cfg, split="test").select(range(n)):
            items.append(format_question(r["question"], r["choices"]["text"]))
    for r in load_dataset("tau/commonsense_qa", split="validation").select(range(n_csqa)):
        items.append(format_question(r["question"], r["choices"]["text"]))
    return items


def batches_by_length(lengths, rows_per_item, budget):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * rows_per_item * lengths[i] > budget:
            out.append(cur)
            cur = []
        cur.append(i)
    out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=["medicine", "law", "math", "computer_science", "history"])
    ap.add_argument("--packs-dir", type=Path, default=ROOT / "expert_packs")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--position-budget", type=int, default=12000)
    ap.add_argument("--fpr", type=float, default=0.05)
    args = ap.parse_args()

    _, base = load_manifest(args.packs_dir / "base.json")
    packs = {d: load_manifest(args.packs_dir / f"{d}.json")[1] for d in PACKS}
    union = base.clone()
    for p in packs.values():
        union |= p
    conds = {"full": torch.ones_like(base), "base": base}
    conds.update({f"base+{d}": base | p for d, p in packs.items()})
    conds["base+all"] = union
    names = list(conds)
    C = len(names)
    B_ROW = names.index("base")
    masks = torch.stack([conds[n] for n in names])
    gate_sets = {**packs, "unmapped": ~union, "base": base}

    model, tok = load_streaming(MODEL_ID)
    device = input_device(model)
    lm_head = model.get_output_embeddings()
    routers = find_routers(model)
    letter_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in LETTERS], device=device)
    tok.truncation_side = "left"

    def encode(texts):
        chats = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True) for t in texts]
        return tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens, add_special_tokens=False).to(device)

    def lengths_of(texts):
        return [min(len(tok.encode(t, add_special_tokens=False)) + 12, args.max_tokens) for t in texts]

    # --- gate calibration on general questions, base plate only ---
    t0 = time.time()
    calib = calibration_items() if not args.limit else calibration_items(24, 8)
    calib_mass = torch.zeros(len(calib), len(gate_sets))
    for idx in batches_by_length(lengths_of(calib), 1, args.position_budget * 2):
        enc = encode([calib[i] for i in idx])
        rm = RouterMass(routers, gate_sets)
        with torch.inference_mode(), restrict_experts(model, base):
            rm.start(enc["attention_mask"])
            try:
                model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            finally:
                m = rm.stop()
        calib_mass[idx] = m
    thr = {d: float(calib_mass[:, j].quantile(1 - args.fpr)) for j, d in enumerate(PACKS)}
    print(f"gate thresholds ({len(calib)} general questions, {time.time() - t0:.0f}s): "
          + "  ".join(f"{d} {v:.4f}" for d, v in thr.items()), flush=True)  # fmt: skip
    print("general questions, mean mass: " + "  ".join(f"{n} {v:.4f}" for n, v in zip(gate_sets, calib_mass.mean(0).tolist())), flush=True)

    results = {"conditions": names, "thresholds": thr, "fpr_target": args.fpr,
               "experts_kept": {n: float(m.float().mean()) for n, m in conds.items()},
               "calibration_mass_mean": dict(zip(gate_sets, calib_mass.mean(0).tolist())), "domains": {}}  # fmt: skip
    for domain in args.domains:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"{domain}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = rows[: args.limit]
        texts = [format_question(r["question"], r["choices"]) for r in rows]
        pred = torch.full((len(rows), C), -1, dtype=torch.long)
        mass = torch.zeros(len(rows), len(gate_sets))
        t0 = time.time()
        for bi, idx in enumerate(batches_by_length(lengths_of(texts), C, args.position_budget)):
            enc = encode([texts[i] for i in idx])
            ids = enc["input_ids"].repeat_interleave(C, dim=0)
            att = enc["attention_mask"].repeat_interleave(C, dim=0)
            rm = RouterMass(routers, gate_sets)
            with torch.inference_mode(), restrict_experts(model, masks.repeat(len(idx), 1, 1)):
                rm.start(att)
                try:
                    hidden = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
                finally:
                    m = rm.stop()
                last = hidden[torch.arange(ids.shape[0], device=device), att.sum(1) - 1]
                logits = lm_head(last).float()
            n_opts = [len(rows[i]["choices"]) for i in idx]
            sc = logits[:, letter_ids].view(len(idx), C, -1).cpu()
            for j, i in enumerate(idx):
                pred[i] = sc[j, :, : n_opts[j]].argmax(-1)
                mass[i] = m.view(len(idx), C, -1)[j, B_ROW]
            print(f"{domain} batch {bi + 1}: {len(idx)} q x {C}, seq {ids.shape[1]}, {time.time() - t0:.0f}s, "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)  # fmt: skip

        gold = torch.tensor([r["answer"] for r in rows])
        ratio = torch.stack([mass[:, j] / thr[d] for j, d in enumerate(PACKS)], 1)
        best = ratio.argmax(1)
        fired = ratio.max(1).values > 1
        gated_col = torch.where(fired, torch.tensor([names.index(f"base+{d}") for d in PACKS])[best], torch.tensor(B_ROW))
        gated = pred[torch.arange(len(rows)), gated_col]
        acc = {n: float((pred[:, c] == gold).float().mean()) for c, n in enumerate(names)}
        acc["gated"] = float((gated == gold).float().mean())
        fire = {d: float((ratio[:, j] > 1).float().mean()) for j, d in enumerate(PACKS)}
        chosen = {d: float((fired & (best == j)).float().mean()) for j, d in enumerate(PACKS)}
        chosen["none"] = float((~fired).float().mean())
        results["domains"][domain] = {
            "n": len(rows), "accuracy": acc, "gate_fire_rate": fire, "gate_choice": chosen,
            "mass_mean": dict(zip(gate_sets, mass.mean(0).tolist())),
            "gold": gold.tolist(), "pred": {n: pred[:, c].tolist() for c, n in enumerate(names)},
            "gate_mass": mass.tolist(),
        }  # fmt: skip
        print(f"== {domain} (n={len(rows)}, {time.time() - t0:.0f}s): " + "  ".join(f"{n} {a:.3f}" for n, a in acc.items()), flush=True)
        print("   gate picks: " + "  ".join(f"{d} {v:.2f}" for d, v in chosen.items())
              + " | mean mass: " + "  ".join(f"{n} {v:.4f}" for n, v in results["domains"][domain]["mass_mean"].items()), flush=True)  # fmt: skip
        out = ROOT / "results" / f"expert_packs_{args.tag}.json"
        out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
