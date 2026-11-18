"""Record the 26B teacher's next-token distributions on distillation text.

    python scripts/teacher_targets.py --domain medicine [--limit 64]

Teacher and student use different chat templates (the 26B's generation prompt
adds an empty thought channel), so sequences are built from segments:

    teacher:  T_prefix + USER + T_mid + ANSWER + <turn|>
    student:  S_prefix + USER + S_mid + ANSWER + <turn|>

and targets are stored per token of USER + ANSWER + <turn|>, which both see
identically. Each target keeps the teacher's top-k log-probabilities.

Writes distill/<domain>.pt: list of {"user", "answer" (int32 ids), "letter" (bool),
"topk_idx" [T, k] int32, "topk_lp" [T, k] float16}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.base_plate import BASE_MODEL_ID  # noqa: E402
from edge_moe.loading import MODEL_ID, input_device, load_streaming  # noqa: E402

MARK = "⁣QQQ⁣"


def template_parts(tok):
    s = tok.apply_chat_template([{"role": "user", "content": MARK}], tokenize=False, add_generation_prompt=True)
    pre, mid = s.split(MARK)
    return pre, mid


def segments(tok, row, max_user, max_answer):
    user = tok.encode(row["user"], add_special_tokens=False)[-max_user:]
    answer = tok.encode(row["assistant"], add_special_tokens=False)[:max_answer] + tok.encode("<turn|>", add_special_tokens=False)
    return user, answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--token-budget", type=int, default=24000)
    ap.add_argument("--max-user", type=int, default=600)
    ap.add_argument("--max-answer", type=int, default=400)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "distill")
    args = ap.parse_args()

    stok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    rows = [json.loads(l) for l in (ROOT / "data" / "distill" / f"{args.domain}.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = rows[: args.limit]

    model, ttok = load_streaming(MODEL_ID)
    probe = [r["user"] + "\n" + r["assistant"] for r in rows[:200]]
    assert all(ttok.encode(p, add_special_tokens=False) == stok.encode(p, add_special_tokens=False) for p in probe), \
        "teacher and student tokenizers disagree"  # fmt: skip
    t_pre, t_mid = (ttok.encode(x, add_special_tokens=False) for x in template_parts(ttok))
    print(f"teacher template prefix {ttok.convert_ids_to_tokens(t_pre)} mid {ttok.convert_ids_to_tokens(t_mid)}", flush=True)
    device = input_device(model)
    lm_head = model.get_output_embeddings()
    softcap = model.config.get_text_config().final_logit_softcapping

    items = []
    for r in rows:
        u, a = segments(ttok, r, args.max_user, args.max_answer)
        items.append({"user": u, "answer": a, "letter": bool(r["letter"])})
    order = sorted(range(len(items)), key=lambda i: len(items[i]["user"]) + len(items[i]["answer"]))
    batches, cur = [], []
    for i in order:
        L = len(t_pre) + len(items[i]["user"]) + len(t_mid) + len(items[i]["answer"])
        if cur and (len(cur) + 1) * L > args.token_budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)

    t0 = time.time()
    for bi, idx in enumerate(batches):
        seqs, tpos = [], []
        for i in idx:
            u, a = items[i]["user"], items[i]["answer"]
            seqs.append(t_pre + u + t_mid + a)
            # position predicting each target token of user + answer
            p_u = [len(t_pre) - 1 + j for j in range(len(u))]
            base_a = len(t_pre) + len(u) + len(t_mid) - 1
            tpos.append(p_u + [base_a + j for j in range(len(a))])
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s)
            att[b, : len(s)] = 1
        with torch.inference_mode():
            hidden = model.base_model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
            for b, i in enumerate(idx):
                h = hidden[b, torch.tensor(tpos[b], device=device)]
                vals, inds = [], []
                for c in range(0, len(h), 1024):
                    lg = lm_head(h[c : c + 1024]).float()
                    if softcap:
                        lg = torch.tanh(lg / softcap) * softcap
                    lp = torch.log_softmax(lg, -1)
                    v, ix = lp.topk(args.k, dim=-1)
                    vals.append(v.half().cpu())
                    inds.append(ix.int().cpu())
                items[i]["topk_lp"] = torch.cat(vals)
                items[i]["topk_idx"] = torch.cat(inds)
        print(f"{args.domain} batch {bi + 1}/{len(batches)}: {len(idx)} rows x {L} tokens, {time.time() - t0:.0f}s", flush=True)

    out = []
    for it in items:
        out.append({
            "user": torch.tensor(it["user"], dtype=torch.int32),
            "answer": torch.tensor(it["answer"], dtype=torch.int32),
            "letter": it["letter"], "topk_idx": it["topk_idx"], "topk_lp": it["topk_lp"],
        })  # fmt: skip
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    torch.save(out, args.out_dir / f"{args.domain}{suffix}.pt")
    print(f"saved {len(out)} rows; top-k mass kept {float(torch.cat([o['topk_lp'].float().exp().sum(-1) for o in out]).mean()):.3f}")


if __name__ == "__main__":
    main()
