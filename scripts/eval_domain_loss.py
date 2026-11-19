"""Next-token loss on each domain's held-out Wikipedia text, for many expert selections, one model load.

    python scripts/eval_domain_loss.py [--domains art history ...] [--mix-sizes 2 3 4 5 6 8 10] [--mix-seeds 3]

Configurations (every 26B one uses the same 4-bit weights; they differ only in
which experts are on the GPU and which the router may pick):
    full            all 128 experts per layer (reference). Run through the offloading path with every
                    expert reachable, which computes the same output without holding all 15GB at once
    base            the base plate alone
    pack:<d>        base plate + domain d's pack, 51 per layer
    mix:<d1>+<d2>.. base plate + the experts the listed domains use most, 51 per layer in total:
                    one fixed model shared by k domains
    rand:<seed>     base plate + randomly chosen experts, 51 per layer (control: is a pack better than any 25?)

Every configuration is scored on every domain, so packs also give a cross-domain
matrix. Loss is mean cross-entropy (nats per token) over the first --max-tokens
tokens of each held-out article introduction, scored as the model's reply to
"Write the Wikipedia introduction for the article <title>." Gemma 4 instruct is a
chat model: on bare text, with no chat turn, it degenerates (repeats words) and
the loss says nothing useful. Writes results/domain_loss.json
after each configuration.
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
from edge_moe.expert_packs import counts, load_manifest, pack_mask  # noqa: E402
from edge_moe.expert_store import ExpertStore, mix_mask  # noqa: E402
from edge_moe.quant_experts import load_resident  # noqa: E402

WIKI = ["art", "history", "biology", "geology", "politics", "computer_science", "economics", "chemistry", "physics", "philosophy"]


def prof(d: str) -> torch.Tensor:
    p = ROOT / "profiles" / f"wiki_{d}.npz"
    return counts(p if p.exists() else ROOT / "profiles" / f"{d}.npz")


@torch.inference_mode()
def domain_loss(model, seqs: list[tuple[torch.Tensor, int]], batch: int = 8) -> float:
    """Mean next-token cross-entropy over the reply tokens (positions k.. of each sequence);
    batched with right padding, padding and the prompt excluded."""
    tot, n = 0.0, 0
    for i in range(0, len(seqs), batch):
        chunk = seqs[i : i + batch]
        T = max(len(x) for x, _ in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long, device="cuda")
        att = torch.zeros(len(chunk), T, dtype=torch.long, device="cuda")
        for j, (x, _) in enumerate(chunk):
            ids[j, : len(x)], att[j, : len(x)] = x, 1
        logits = model(input_ids=ids, attention_mask=att, logits_to_keep=0).logits
        for j, (x, k) in enumerate(chunk):
            lj = logits[j, k - 1 : len(x) - 1].float()
            tot += torch.nn.functional.cross_entropy(lj, x[k:], reduction="sum").item()
            n += len(x) - k
        del logits
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=WIKI)
    ap.add_argument("--budget", type=int, default=51)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--mix-sizes", nargs="*", type=int, default=[2, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--mix-seeds", type=int, default=3)
    ap.add_argument("--configs", nargs="*", default=None, help="explicit list instead of the default sweep")
    ap.add_argument("--out", default="domain_loss.json")
    args = ap.parse_args()

    store = ExpertStore.from_packs([ROOT / "quant" / f"{n}.safetensors" for n in ("base", "medicine", "law", "math", "rest")])
    model, tok, _ = load_resident([ROOT / "quant" / "base.safetensors"], quant_dense=True, quant_embed=True)
    model.eval()
    base = load_manifest(ROOT / "expert_packs" / "base.json")[1]
    n = args.budget

    seqs = {}
    for d in args.domains:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"wiki_{d}.jsonl").read_text(encoding="utf-8").splitlines()]
        seqs[d] = []
        for r in rows[: args.n_test]:
            pre = tok.apply_chat_template([{"role": "user", "content": f'Write the Wikipedia introduction for the article "{r["title"]}".'}],
                                          tokenize=False, add_generation_prompt=True)  # fmt: skip
            a = tok(pre, add_special_tokens=False)["input_ids"]  # the template starts with <bos>
            body = tok(r["text"], add_special_tokens=False)["input_ids"][: args.max_tokens]
            seqs[d].append((torch.tensor(a + body).cuda(), len(a)))

    if args.configs:
        configs = args.configs
    else:
        configs = ["full", "base"] + [f"pack:{d}" for d in args.domains] + [f"rand:{s}" for s in range(3)]
        rng = random.Random(0)
        for k in args.mix_sizes:
            for s in range(args.mix_seeds if k < len(args.domains) else 1):
                configs.append("mix:" + "+".join(sorted(rng.sample(args.domains, k))))

    out_path = ROOT / "results" / args.out
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"budget": n, "loss": {}}
    for c in configs:
        if c in res["loss"]:
            continue
        kind, _, arg = c.partition(":")
        if kind == "full":
            mask = torch.ones_like(base)
        elif kind == "base":
            mask = base.clone()
        elif kind == "pack":
            mask = base | pack_mask(prof(arg), base, n / 128)
        elif kind == "rand":
            g = torch.Generator().manual_seed(int(arg))
            mask = base.clone()
            for l in range(mask.shape[0]):
                free = (~base[l]).nonzero().flatten()
                mask[l, free[torch.randperm(len(free), generator=g)[: n - int(base[l].sum())]]] = True
        elif kind == "mix":
            mask = mix_mask(base, [prof(d) for d in arg.split("+")], n)
        else:
            raise SystemExit(f"unknown config {c}")
        t0 = time.time()
        if kind == "full":
            store.install_offload(model, n)
        else:
            store.install(model, mask)
        torch.cuda.reset_peak_memory_stats()
        res["loss"][c] = {d: domain_loss(model, seqs[d]) for d in args.domains}
        res.setdefault("experts_per_layer", {})[c] = float(mask.sum(1).float().mean())
        out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        torch.cuda.empty_cache()
        mean = sum(res["loss"][c].values()) / len(args.domains)
        print(f"{c:60} mean loss {mean:.4f}  peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
