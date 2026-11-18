"""Train one domain pack (LoRA deltas + gate) on the frozen E4B base plate.

    python scripts/train_pack.py --domain medicine [--limit 200]

Only this domain's data (data/packs/<domain>.jsonl) and generic negatives
(data/packs/generic.jsonl) are used, never another pack's data, so packs can be
trained, refined and shipped independently.

Writes packs/<domain>.safetensors and packs/<domain>.log.json.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.base_plate import LETTERS, BasePlate, format_mc  # noqa: E402
from edge_moe.packs import PackSet, train_gate  # noqa: E402

HOLDOUT = 500
LAW_PREFIX = "Which holding best fills the blank in this case excerpt?\n\n"


def load_rows(name):
    return [json.loads(l) for l in (ROOT / "data" / "packs" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def gate_text(plate, row):
    """What the gate sees: the question as a user would ask it (no dataset-specific framing)."""
    q = row["question"].removeprefix(LAW_PREFIX)
    return plate.prompt_text(format_mc(q, row["choices"]) if row["choices"] else q)


def build_examples(plate, rows, max_tokens):
    tok = plate.tok
    out, dropped = [], 0
    for r in rows:
        prompt = tok.encode(plate.prompt_text(format_mc(r["question"], r["choices"])), add_special_tokens=False)
        target = LETTERS[r["answer"]] + (f"\n\n{r['explanation']}" if r["explanation"] else "") + plate.end_of_turn
        tgt = tok.encode(target, add_special_tokens=False)
        if len(prompt) + len(tgt) > max_tokens:
            tgt = tgt[: max(1, max_tokens - len(prompt))]  # keep the answer letter, cut the explanation
            if len(prompt) + len(tgt) > max_tokens:
                dropped += 1
                continue
        out.append((prompt, tgt))
    return out, dropped


def batches(examples, token_budget, rng):
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + len(examples[i][1]))
    groups, cur = [], []
    for i in order:
        n = len(examples[i][0]) + len(examples[i][1])
        if cur and (len(cur) + 1) * n > token_budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    rng.shuffle(groups)
    return groups


def collate(examples, idx, device):
    seqs = [examples[i][0] + examples[i][1] for i in idx]
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    att = torch.zeros(len(seqs), L, dtype=torch.long)
    letter_pos, rest_pos = [], []  # (row, position that predicts the token)
    for b, i in enumerate(idx):
        p, t = examples[i]
        s = p + t
        ids[b, : len(s)] = torch.tensor(s)
        att[b, : len(s)] = 1
        letter_pos.append((b, len(p) - 1, t[0]))
        rest_pos += [(b, len(p) - 1 + k, t[k]) for k in range(1, len(t))]
    return ids.to(device), att.to(device), letter_pos, rest_pos


def ce(plate, hidden, pos, chunk=256):
    """Mean cross-entropy at the given positions. Logits over the 262k vocab are
    built `chunk` positions at a time and recomputed in backward, so they are
    never all held in memory (2k positions would be ~2GB per fp32 copy)."""
    if not pos:
        return hidden.new_zeros(()).float()
    b = torch.tensor([p[0] for p in pos], device=hidden.device)
    s = torch.tensor([p[1] for p in pos], device=hidden.device)
    y = torch.tensor([p[2] for p in pos], device=hidden.device)
    h = hidden[b, s]

    def part(hc, yc):
        return F.cross_entropy(plate.logits(hc), yc, reduction="sum")

    total = sum(
        torch.utils.checkpoint.checkpoint(part, h[i : i + chunk], y[i : i + chunk], use_reentrant=False)
        for i in range(0, len(pos), chunk)
    )
    return total / len(pos)


@torch.inference_mode()
def mc_accuracy(plate, ps, rows, weights, max_tokens=768, bs=16):
    correct = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        enc = plate.encode([plate.prompt_text(format_mc(r["question"], r["choices"])) for r in chunk], max_tokens)
        ids, att = enc["input_ids"], enc["attention_mask"]
        h = ps.forward_hidden(ids, att, mode="forced", forced=weights)
        lg = plate.logits(h[torch.arange(len(chunk), device=ids.device), att.sum(1) - 1])
        n_opts = max(len(r["choices"]) for r in chunk)
        pred = lg[:, plate.letter_ids[:n_opts]].argmax(-1).cpu()
        correct += sum(int(p) == r["answer"] for p, r in zip(pred, chunk))
    return correct / len(rows)


@torch.inference_mode()
def features(plate, ps, texts, max_tokens=768, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = plate.encode(texts[i : i + bs], max_tokens)
        ps.forward_hidden(enc["input_ids"], enc["attention_mask"], mode="features")
        out.append(ps.last_features)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--limit", type=int, default=None, help="training examples (default: all but the held-out 500)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=6144, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gate-examples", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "packs")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)

    rows = load_rows(args.domain)
    held, train_rows = rows[-HOLDOUT:], rows[:-HOLDOUT][: args.limit]
    generic = load_rows("generic")
    g_held, g_train = generic[-HOLDOUT:], generic[:-HOLDOUT]

    plate = BasePlate()
    ps = PackSet(plate, rank=args.rank)
    log = {"domain": args.domain, "args": {k: str(v) for k, v in vars(args).items()}}

    # ---- gate: this domain vs generic, on base-plate features ---------------
    t0 = time.time()
    pos = features(plate, ps, [gate_text(plate, r) for r in train_rows[: args.gate_examples]])
    neg = features(plate, ps, [gate_text(plate, r) for r in g_train[: args.gate_examples]])
    gate = train_gate(pos, neg)
    pos_h = gate.prob(features(plate, ps, [gate_text(plate, r) for r in held]))
    neg_h = gate.prob(features(plate, ps, [gate_text(plate, r) for r in g_held]))
    log["gate_heldout"] = {"tpr": float((pos_h >= 0.5).float().mean()), "fpr": float((neg_h >= 0.5).float().mean())}
    print(f"gate trained in {time.time() - t0:.0f}s: held-out {log['gate_heldout']}", flush=True)

    # ---- pack deltas ---------------------------------------------------------
    eval_rows = held[:200]
    base_acc = mc_accuracy(plate, ps, eval_rows, {})
    print(f"base plate on held-out {args.domain}: {base_acc:.3f}", flush=True)

    params = ps.new_pack(args.domain)
    examples, dropped = build_examples(plate, train_rows, args.max_tokens)
    print(f"{len(examples)} training examples ({dropped} too long, dropped); "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M pack parameters", flush=True)  # fmt: skip
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    groups = []
    for e in range(math.ceil(args.epochs)):
        g = batches(examples, args.token_budget, rng)
        groups += g if e + 1 <= args.epochs else g[: int(len(g) * (args.epochs - e))]
    total_steps = math.ceil(len(groups) / args.accum)
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )  # fmt: skip

    plate.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    plate.lm.train()
    ps.state.weights = {args.domain: 1.0}  # stays set through backward (checkpoint recompute needs it)
    t0, run_l, run_r, curve = time.time(), 0.0, 0.0, []
    for gi, idx in enumerate(groups):
        ids, att, lp, rp = collate(examples, idx, plate.device)
        hidden = plate.lm(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        l_letter, l_rest = ce(plate, hidden, lp), ce(plate, hidden, rp)
        ((l_letter + l_rest) / args.accum).backward()
        run_l += l_letter.item()
        run_r += l_rest.item()
        if (gi + 1) % args.accum == 0 or gi + 1 == len(groups):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = (gi + 1 + args.accum - 1) // args.accum
            if step % 25 == 0 or step == total_steps:
                n = 25 * args.accum if step % 25 == 0 else (gi + 1) % (25 * args.accum) or 25 * args.accum
                curve.append({"step": step, "letter_loss": run_l / n, "text_loss": run_r / n})
                print(f"step {step}/{total_steps}: letter loss {run_l / n:.3f}, explanation loss {run_r / n:.3f}, "
                      f"{time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)  # fmt: skip
                run_l = run_r = 0.0
    ps.state.weights = {}
    plate.lm.eval()
    plate.lm.gradient_checkpointing_disable()
    log["train_seconds"] = time.time() - t0
    log["curve"] = curve

    pack_acc = mc_accuracy(plate, ps, eval_rows, {args.domain: 1.0})
    log["heldout_accuracy"] = {"base": base_acc, "pack": pack_acc, "n": len(eval_rows)}
    print(f"held-out {args.domain}: base {base_acc:.3f} -> with pack {pack_acc:.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.domain}.safetensors"
    ps.save_pack(args.domain, path, gate, extra={"gate_heldout": log["gate_heldout"]})
    log["file_mb"] = path.stat().st_size / 1e6
    (args.out_dir / f"{args.domain}.log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"saved {path} ({log['file_mb']:.0f}MB)", flush=True)


if __name__ == "__main__":
    main()
