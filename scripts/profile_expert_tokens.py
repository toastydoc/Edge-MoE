"""What text each expert receives: the token-level pass behind the expert labels on the domain maps.

    python scripts/profile_expert_tokens.py [--per-domain 60 --max-tokens 384]

Runs a sample of every domain's profiling prompts through the full model (all
128 experts) and records, for every (layer, expert):
    - which token ids were routed to it (joint counts, kept sparse)
    - which domain the token came from
    - the token's surface class: word start, word piece, number, punctuation, whitespace, other

Writes profiles/expert_tokens.json with, per expert, its most characteristic
tokens (by lift over the overall token distribution, with a minimum count),
its surface-class and domain shares, plus per-layer mutual information
I(expert; domain) and I(expert; surface class): how much of the routing
choice each one accounts for.
"""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.loading import MODEL_ID, input_device, load_streaming  # noqa: E402
from edge_moe.router_hooks import find_routers  # noqa: E402

DOMAINS = {
    "law": "law", "regulation": "regulation", "reconcile": "reconcile", "medicine": "medicine", "math": "math",
    **{d: f"wiki_{d}" for d in ["politics", "economics", "history", "philosophy", "art", "biology", "chemistry",
                                 "geology", "physics", "computer_science"]},
}  # fmt: skip
CLASSES = ["word start", "word piece", "number", "punctuation", "whitespace", "other"]


def surface_class(piece: str) -> int:
    s = piece.replace("▁", " ")
    core = s.strip()
    if not core:
        return 4
    if any(c.isdigit() for c in core):
        return 2
    if all(not c.isalnum() for c in core):
        return 3
    if core.isalpha() or core.replace("'", "").replace("-", "").isalpha():
        return 0 if s[0] == " " else 1
    return 5


def mutual_info(joint: np.ndarray) -> tuple[float, float]:
    """I(X;Y) in bits from a joint count table, and as a fraction of H(X) (X = rows = experts)."""
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())
    hx = float(-(px[px > 0] * np.log2(px[px > 0])).sum())
    return mi, mi / hx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(0)
    names = list(DOMAINS)
    rows = []
    for di, (d, f) in enumerate(DOMAINS.items()):
        ps = [json.loads(l)["prompt"] for l in (ROOT / "data" / "prompts" / f"{f}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        rows += [(p, di) for p in rng.sample(ps, min(args.per_domain, len(ps)))]

    t0 = time.time()
    model, tok = load_streaming(MODEL_ID)
    print(f"loaded in {time.time() - t0:.0f}s; {len(rows)} prompts", flush=True)
    routers = find_routers(model)
    L, E = len(routers), routers[0][1].proj.out_features
    V = len(tok)
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    cls_of = torch.tensor([surface_class(p or "") for p in pieces])

    items = []
    for p, di in rows:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        s = text.find(p.strip())
        items.append((text, s, s + len(p.strip()), di))
    items.sort(key=lambda x: len(x[0]))

    tok_counts = [Counter() for _ in range(L)]  # key e*V + token id
    dom = np.zeros((L, E, len(names)), dtype=np.int64)
    cls = np.zeros((L, E, len(CLASSES)), dtype=np.int64)
    tok_total = Counter()
    state = {}

    def hook_for(l):
        def hook(module, a, out):
            idx = out[2][state["keep"]]  # [n, K]
            ids = state["ids"][:, None].expand_as(idx)
            keys = (idx.long() * V + ids).reshape(-1)
            u, c = torch.unique(keys, return_counts=True)
            tok_counts[l].update(dict(zip(u.tolist(), c.tolist())))
            for arr, lab, n in ((dom, state["dom"], len(names)), (cls, state["cls"], len(CLASSES))):
                j = (idx.long() * n + lab[:, None].expand_as(idx)).reshape(-1)
                arr[l] += torch.bincount(j, minlength=E * n).reshape(E, n).cpu().numpy()
        return hook

    handles = [r.register_forward_hook(hook_for(l)) for l, (_, r) in enumerate(routers)]
    dev = input_device(model)
    try:
        with torch.inference_mode():
            for b in range(0, len(items), args.batch_size):
                t1 = time.time()
                chunk = items[b : b + args.batch_size]
                enc = tok([c[0] for c in chunk], return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_tokens, add_special_tokens=False, return_offsets_mapping=True)  # fmt: skip
                off = enc.pop("offset_mapping")
                st = torch.tensor([c[1] for c in chunk])[:, None]
                en = torch.tensor([c[2] for c in chunk])[:, None]
                keep = ((off[..., 0] >= st) & (off[..., 1] <= en) & enc["attention_mask"].bool()).reshape(-1)
                full = enc["input_ids"]
                c_all = cls_of[full]
                # a letter token right after whitespace or a non-content token starts a word (e.g. at a line start)
                prev_ws = torch.ones_like(full, dtype=torch.bool)
                prev_ws[:, 1:] = (c_all[:, :-1] == 4) | ~keep.reshape(full.shape)[:, :-1]
                c_all = torch.where((c_all == 1) & prev_ws, torch.zeros_like(c_all), c_all)
                ids = full.reshape(-1)[keep]
                dl = torch.tensor([c[3] for c in chunk])[:, None].expand_as(enc["input_ids"]).reshape(-1)[keep]
                state.update(keep=keep.to(dev), ids=ids.to(dev), dom=dl.to(dev), cls=c_all.reshape(-1)[keep].to(dev))
                tok_total.update(dict(zip(*[x.tolist() for x in torch.unique(ids, return_counts=True)])))
                enc = enc.to(dev)
                model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                print(f"batch {b // args.batch_size + 1}/{math.ceil(len(items) / args.batch_size)}: {int(keep.sum())} tokens, {time.time() - t1:.0f}s", flush=True)
    finally:
        for h in handles:
            h.remove()

    N = sum(tok_total.values())
    p_tok = {t: c / N for t, c in tok_total.items()}
    experts = []
    for l in range(L):
        per = [dict() for _ in range(E)]
        for key, c in tok_counts[l].items():
            per[key // V][key % V] = c
        layer = []
        for e in range(E):
            n = sum(per[e].values())
            scored = [(c * math.log((c / n) / p_tok[t]), t, c) for t, c in per[e].items() if c >= args.min_count] if n else []
            scored.sort(reverse=True)
            top = [{"t": tok.decode([t]), "n": c, "lift": round((c / n) / p_tok[t], 1)} for _, t, c in scored[: args.top]]
            layer.append({
                "n": n,
                "top": top,
                "cls": np.round(cls[l, e] / max(1, cls[l, e].sum()), 3).tolist(),
                "dom": np.round(dom[l, e] / max(1, dom[l, e].sum()), 3).tolist(),
            })  # fmt: skip
        experts.append(layer)

    mi = [{"domain": mutual_info(dom[l]), "surface": mutual_info(cls[l])} for l in range(L)]
    out = {"domains": names, "classes": CLASSES, "tokens": N, "experts": experts,
           "mi": [{k: {"bits": round(v[0], 4), "frac": round(v[1], 4)} for k, v in m.items()} for m in mi]}  # fmt: skip
    path = ROOT / "profiles" / "expert_tokens.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)}: {N} tokens, {path.stat().st_size / 1e6:.1f}MB")
    for l in (0, L // 2, L - 1):
        print(f"layer {l + 1}: I(expert;domain) {mi[l]['domain'][0]:.3f} bits, I(expert;surface class) {mi[l]['surface'][0]:.3f} bits")


if __name__ == "__main__":
    main()
