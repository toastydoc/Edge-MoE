"""Train a pack by distilling the 26B teacher into it (route A/B hybrid).

    python scripts/train_distill.py --domain medicine [--targets distill/medicine.pt]

The pack (same LoRA layout as train_pack.py) learns to match the teacher's
top-k next-token distribution on every token of the domain text, so it absorbs
what the 26B knows about the domain rather than one dataset's answer key.
Loss = KD at the answer-letter position + mean KD over all other positions.

The gate is copied from the round-1 pack file (packs/<domain>.safetensors),
which train_gates.py already upgraded; gates do not depend on the deltas.

Writes packs_distill/<domain>.safetensors and .log.json.
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
from edge_moe.base_plate import BasePlate  # noqa: E402
from edge_moe.packs import Gate, PackSet  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from teacher_targets import template_parts  # noqa: E402
from train_pack import HOLDOUT, load_rows, mc_accuracy  # noqa: E402


def kd_loss(plate, hidden, rows_pos, idx, lp, chunk=256):
    """Cross-entropy of student against teacher top-k (renormalised), chunked + recomputed."""
    if len(rows_pos) == 0:
        return hidden.new_zeros(()).float()
    b = torch.tensor([p[0] for p in rows_pos], device=hidden.device)
    s = torch.tensor([p[1] for p in rows_pos], device=hidden.device)
    h = hidden[b, s]
    t_idx = idx.to(hidden.device).long()
    t_p = torch.softmax(lp.to(hidden.device).float(), -1)  # renormalise over top-k

    def part(hc, ic, pc):
        logq = torch.log_softmax(plate.logits(hc), -1)
        return -(pc * logq.gather(-1, ic)).sum()

    total = sum(
        torch.utils.checkpoint.checkpoint(part, h[i : i + chunk], t_idx[i : i + chunk], t_p[i : i + chunk], use_reentrant=False)
        for i in range(0, len(rows_pos), chunk)
    )
    return total / len(rows_pos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--targets", type=Path, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=6144)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "packs_distill")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)

    data = torch.load(args.targets or ROOT / "distill" / f"{args.domain}.pt")
    plate = BasePlate()
    s_pre, s_mid = (plate.tok.encode(x, add_special_tokens=False) for x in template_parts(plate.tok))
    ps = PackSet(plate, rank=args.rank)

    held = load_rows(args.domain)[-HOLDOUT:][:200]
    base_acc = mc_accuracy(plate, ps, held, {})
    params = ps.new_pack(args.domain)

    def length(d):
        return len(s_pre) + len(d["user"]) + len(s_mid) + len(d["answer"])

    order = sorted(range(len(data)), key=lambda i: length(data[i]))
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * length(data[i]) > args.token_budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    all_groups = []
    for e in range(math.ceil(args.epochs)):
        g = groups[:]
        rng.shuffle(g)
        all_groups += g if e + 1 <= args.epochs else g[: int(len(g) * (args.epochs - e))]
    total_steps = math.ceil(len(all_groups) / args.accum)
    warmup = max(1, total_steps // 20)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )  # fmt: skip
    print(f"{len(data)} rows, {total_steps} steps; base plate held-out {base_acc:.3f}", flush=True)

    plate.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    plate.lm.train()
    ps.state.weights = {args.domain: 1.0}
    t0, run_l, run_r, curve = time.time(), 0.0, 0.0, []
    for gi, idx in enumerate(all_groups):
        seqs, letter_pos, rest_pos, l_idx, l_lp, r_idx, r_lp = [], [], [], [], [], [], []
        for b, i in enumerate(idx):
            d = data[i]
            u, a = d["user"].tolist(), d["answer"].tolist()
            seqs.append(s_pre + u + s_mid + a)
            pos = [len(s_pre) - 1 + j for j in range(len(u))]
            base_a = len(s_pre) + len(u) + len(s_mid) - 1
            pos += [base_a + j for j in range(len(a))]
            for t, p in enumerate(pos):
                if d["letter"] and t == len(u):  # the answer-letter position
                    letter_pos.append((b, p)); l_idx.append(d["topk_idx"][t]); l_lp.append(d["topk_lp"][t])  # fmt: skip
                else:
                    rest_pos.append((b, p)); r_idx.append(d["topk_idx"][t]); r_lp.append(d["topk_lp"][t])  # fmt: skip
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s)
            att[b, : len(s)] = 1
        hidden = plate.lm(input_ids=ids.to(plate.device), attention_mask=att.to(plate.device), use_cache=False).last_hidden_state
        l_letter = kd_loss(plate, hidden, letter_pos, torch.stack(l_idx), torch.stack(l_lp)) if letter_pos else hidden.new_zeros(()).float()
        l_rest = kd_loss(plate, hidden, rest_pos, torch.stack(r_idx), torch.stack(r_lp))
        ((l_letter + l_rest) / args.accum).backward()
        run_l += l_letter.item()
        run_r += l_rest.item()
        if (gi + 1) % args.accum == 0 or gi + 1 == len(all_groups):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = (gi + args.accum) // args.accum
            if step % 25 == 0 or step == total_steps:
                n = 25 * args.accum if step % 25 == 0 else (gi + 1) % (25 * args.accum) or 25 * args.accum
                curve.append({"step": step, "letter_kd": run_l / n, "text_kd": run_r / n})
                print(f"step {step}/{total_steps}: letter KD {run_l / n:.3f}, text KD {run_r / n:.3f}, "
                      f"{time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)  # fmt: skip
                run_l = run_r = 0.0
    ps.state.weights = {}
    plate.lm.eval()
    plate.lm.gradient_checkpointing_disable()

    pack_acc = mc_accuracy(plate, ps, held, {args.domain: 1.0})
    print(f"held-out {args.domain}: base {base_acc:.3f} -> distilled pack {pack_acc:.3f}", flush=True)
    # reuse the v2 gate from the round-1 pack file
    from safetensors import safe_open
    from safetensors.torch import load_file

    src = ROOT / "packs" / f"{args.domain}.safetensors"
    t = load_file(str(src))
    with safe_open(str(src), "pt") as f:
        thr = json.loads(f.metadata()["meta"]).get("threshold", 0.5)
    gate = Gate(t["gate.mean"], t["gate.std"], t["gate.w"], t["gate.b"], thr)
    path = args.out_dir / f"{args.domain}.safetensors"
    ps.save_pack(args.domain, path, gate, extra={"trained_by": "distillation from " + "google/gemma-4-26B-A4B-it"})
    log = {"domain": args.domain, "heldout_accuracy": {"base": base_acc, "pack": pack_acc, "n": len(held)},
           "curve": curve, "train_seconds": time.time() - t0, "rows": len(data)}  # fmt: skip
    (args.out_dir / f"{args.domain}.log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
