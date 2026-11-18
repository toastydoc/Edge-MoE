"""Train "missing pack" gates for the 26B expert packs.

    python scripts/train_expert_gates.py [--limit 200]

The device runs the 26B on the base plate alone, so features are taken there:
    h<L>    mean-pooled hidden state after decoder layer L (L in LAYERS)
    router  per-layer router probability on each pack's experts and on the
            unmapped experts, layers 0..max(LAYERS)
The forward pass stops after the deepest layer used, so this costs well under a
full pass. Gates are logistic regressions (packs.train_gate) trained on the same
reference pool as the E4B gates (train_gates.build_pool: CS and history never
included). The feature set is chosen on the pool's held-out 20%, not on MMLU test.

Then: firing rates on the MMLU test questions, and gated accuracy computed from
the per-condition predictions saved by eval_expert_packs.py.

Writes expert_packs/gates.pt, results/expert_gates.json.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge_moe.expert_packs import load_manifest  # noqa: E402
from edge_moe.loading import MODEL_ID, input_device, load_streaming  # noqa: E402
from edge_moe.packs import train_gate  # noqa: E402
from edge_moe.router_hooks import find_routers, restrict_experts  # noqa: E402
from eval_expert_packs import format_question  # noqa: E402
from train_gates import ADJACENT, PACKS, build_pool  # noqa: E402

LAYERS = [5, 11, 17]
DOMAINS = ["medicine", "law", "math", "computer_science", "history"]


class _Stop(Exception):
    pass


@torch.inference_mode()
def featurize(model, tok, texts, base, sets, budget=24000, max_tokens=640):
    device = input_device(model)
    routers = find_routers(model)
    names = [n.rsplit(".", 1)[0] for n, _ in routers]
    layers = [model.get_submodule(n) for n in names]
    deepest = max(LAYERS)
    S = torch.stack(list(sets.values())).float().to(device)  # [S, L, E]
    chats = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True) for t in texts]
    lens = [min(len(tok.encode(c, add_special_tokens=False)), max_tokens) for c in chats]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lens[i] > budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)

    feats = {f"h{L}": torch.zeros(len(texts), model.config.get_text_config().hidden_size) for L in LAYERS}
    feats["router"] = torch.zeros(len(texts), (deepest + 1) * S.shape[0])
    t0 = time.time()
    for gi, idx in enumerate(groups):
        enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens, add_special_tokens=False).to(device)  # fmt: skip
        m = enc["attention_mask"].float()
        denom = m.sum(1, keepdim=True)
        rmass = torch.zeros(len(idx), deepest + 1, S.shape[0], device=device)
        handles = []

        def r_hook(l):
            def hook(mod, args, out):
                p = out[0].float().view(*m.shape, -1)
                rmass[:, l] = torch.einsum("bte,se->bs", p * m[..., None], S[:, l]) / denom

            return hook

        def h_hook(L):
            def hook(mod, args, out):
                h = (out[0] if isinstance(out, tuple) else out).float()
                feats[f"h{L}"][idx] = ((h * m[..., None]).sum(1) / denom).cpu()
                if L == deepest:
                    raise _Stop

            return hook

        for l in range(deepest + 1):
            handles.append(routers[l][1].register_forward_hook(r_hook(l)))
        for L in LAYERS:
            handles.append(layers[L].register_forward_hook(h_hook(L)))
        try:
            with restrict_experts(model, base):
                model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        except _Stop:
            pass
        finally:
            for h in handles:
                h.remove()
        feats["router"][idx] = rmass.flatten(1).cpu()
        if gi % 10 == 0 or gi + 1 == len(groups):
            print(f"  features {gi + 1}/{len(groups)} batches, {time.time() - t0:.0f}s", flush=True)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs-dir", type=Path, default=ROOT / "expert_packs")
    ap.add_argument("--eval", type=Path, default=ROOT / "results" / "expert_packs_b20t50.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    _, base = load_manifest(args.packs_dir / "base.json")
    packs = {d: load_manifest(args.packs_dir / f"{d}.json")[1] for d in PACKS}
    union = base.clone()
    for p in packs.values():
        union |= p
    sets = {**packs, "unmapped": ~union}

    pool = build_pool(random.Random(0))
    if args.limit:
        pool = random.Random(1).sample(pool, args.limit)
    labels = [l for _, l in pool]
    tests = {}
    for dom in DOMAINS:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"{dom}.jsonl").read_text(encoding="utf-8").splitlines()]
        tests[dom] = [format_question(r["question"], r["choices"]) for r in rows][: args.limit and 16]
    print(f"pool {len(pool)} questions; test {sum(map(len, tests.values()))}", flush=True)

    model, tok = load_streaming(MODEL_ID)
    tok.truncation_side = "left"
    t_texts = [t for d in DOMAINS for t in tests[d]]
    F = featurize(model, tok, [t for t, _ in pool] + t_texts, base, sets)
    n = len(pool)
    cut = {}
    off = n
    for d in DOMAINS:
        cut[d] = (off, off + len(tests[d]))
        off += len(tests[d])
    torch.save({"features": F, "labels": labels, "test_cut": cut}, ROOT / "expert_packs" / "gate_features.pt")

    feature_sets = {**{k: [k] for k in F}, "h11+router": ["h11", "router"], "h17+router": ["h17", "router"]}
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    is_val = torch.zeros(n, dtype=torch.bool)
    is_val[perm[: n // 5]] = True
    dev = input_device(model)
    del model
    torch.cuda.empty_cache()

    report = {"pool_size": n, "feature_sets": {}, "chosen": {}, "gates": {}}
    best = {}
    for fs, keys in feature_sets.items():
        X = torch.cat([F[k] for k in keys], 1).to(dev)
        report["feature_sets"][fs] = {}
        for d in PACKS:
            pos = torch.tensor([l == d for l in labels])
            neg = torch.tensor([l != d and l not in ADJACENT[d] for l in labels])
            Xp = X[:n]
            g = train_gate(Xp[(pos & ~is_val).to(dev)], Xp[(neg & ~is_val).to(dev)])
            tpr = float((g.prob(Xp[(pos & is_val).to(dev)]) >= 0.5).float().mean())
            fpr = float((g.prob(Xp[(neg & is_val).to(dev)]) >= 0.5).float().mean())
            report["feature_sets"][fs][d] = {"tpr": tpr, "fpr": fpr}
            score = tpr - fpr
            if d not in best or score > best[d][0]:
                best[d] = (score, fs, g, X)
        print(f"{fs:>11}: " + "  ".join(f"{d} TPR {v['tpr']:.2f} FPR {v['fpr']:.2f}" for d, v in report["feature_sets"][fs].items()), flush=True)

    print("\nfiring rate on MMLU test (CS and history never seen by any gate):")
    print(f"{'':10s}" + "".join(f"{d:>18s}" for d in DOMAINS))
    probs = {}
    for d in PACKS:
        _, fs, g, X = best[d]
        report["chosen"][d] = fs
        p = g.prob(X[n:]).cpu()
        probs[d] = p
        rates = {dom: float((p[a - n : b - n] >= 0.5).float().mean()) for dom, (a, b) in cut.items()}
        report["gates"][d] = {"features": fs, "pool_val": report["feature_sets"][fs][d], "mmlu_fire_rate": rates}
        print(f"{d:10s}" + "".join(f"{v:>18.2f}" for v in rates.values()) + f"   [{fs}]")
    torch.save({d: {"features": best[d][1], **{k: v.cpu() if torch.is_tensor(v) else v for k, v in vars(best[d][2]).items()}}
                for d in PACKS}, args.packs_dir / "gates.pt")  # fmt: skip

    if args.eval.exists() and not args.limit:
        ev = json.loads(args.eval.read_text(encoding="utf-8"))
        report["gated_accuracy"] = {}
        print("\ngated accuracy (fired packs installed; 2+ fired -> all packs; none -> fallback):")
        for dom, (a, b) in cut.items():
            r = ev["domains"][dom]
            gold = torch.tensor(r["gold"])
            P = {k: torch.tensor(v) for k, v in r["pred"].items()}
            fired = torch.stack([probs[d][a - n : b - n] >= 0.5 for d in PACKS], 1)
            out = {}
            for fallback in ("base", "base+all"):
                pred = P[fallback].clone()
                for i in range(len(gold)):
                    k = int(fired[i].sum())
                    if k == 1:
                        pred[i] = P[f"base+{PACKS[int(fired[i].float().argmax())]}"][i]
                    elif k > 1:
                        pred[i] = P["base+all"][i]
                out[f"none->{fallback}"] = float((pred == gold).float().mean())
            out["base+all"] = float((P["base+all"] == gold).float().mean())
            out["full"] = float((P["full"] == gold).float().mean())
            report["gated_accuracy"][dom] = out
            print(f"  {dom:>16}: " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()))
    (ROOT / "results" / ("expert_gates.json" if not args.limit else "expert_gates_smoke.json")).write_text(
        json.dumps(report, indent=1), encoding="utf-8")  # fmt: skip


if __name__ == "__main__":
    main()
