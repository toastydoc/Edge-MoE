"""Answers for the blind-judged test: the same questions answered by the full model and by 8GB configurations.

    python scripts/gen_judge_answers.py [--per-domain 30] [--configs full own shared]

Questions come from each domain's held-out test articles (never profiled):
"In <domain>, what is <title>? Explain it in one paragraph for an educated reader."

Configurations (all the same 4-bit Gemma 4 26B-A4B weights):
    full     all 128 experts, through the offloading path (exact full-model output)
    own      base plate + the question's own domain pack, 51 per layer (swap packs per domain)
    shared   one fixed 51-per-layer model built for all ten domains (no swapping)
    gemma4_12b   the dense Gemma 4 12B, 4-bit the same way (the best model that fits the same device)

Greedy decoding, batched. Writes results/judge/questions.jsonl and
results/judge/answers_<config>.jsonl (answers_<config>_n<budget>.jsonl when --budget is not 51); skips
questions already answered. GPU memory held by the weights of each configuration goes to results/judge/memory.json.
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
from edge_moe.expert_packs import counts, load_manifest, pack_mask  # noqa: E402
from edge_moe.expert_store import ExpertStore, mix_mask  # noqa: E402
from edge_moe.quant_experts import load_resident  # noqa: E402

WIKI = ["art", "history", "biology", "geology", "politics", "computer_science", "economics", "chemistry", "physics", "philosophy"]
OUT = ROOT / "results" / "judge"


def questions(per_domain: int) -> list[dict]:
    path = OUT / "questions.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    rng = random.Random(0)
    qs = []
    for d in WIKI:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"wiki_{d}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if not r["title"].startswith(("List", "Lists"))]
        for r in rng.sample(rows, min(per_domain, len(rows))):
            field = d.replace("_", " ")
            qs.append({"id": f"{d}:{r['title']}", "domain": d, "title": r["title"],
                       "question": f"In {field}, what is {r['title']}? Explain it in one paragraph for an educated reader."})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    return qs


@torch.inference_mode()
def answer(model, tok, qs: list[dict], batch: int, max_new: int) -> list[dict]:
    out = []
    for i in range(0, len(qs), batch):
        chunk = qs[i : i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False, add_generation_prompt=True) for q in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        t0 = time.time()
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        dt = time.time() - t0
        for q, row in zip(chunk, g[:, enc["input_ids"].shape[1] :]):
            text = tok.decode(row, skip_special_tokens=True).strip()
            out.append({"id": q["id"], "answer": text, "tokens": int((row != tok.pad_token_id).sum())})
        torch.cuda.empty_cache()
        print(f"  {i + len(chunk)}/{len(qs)}  {dt:.0f}s", flush=True)
    return out


DENSE = {"gemma4_12b": "google/gemma-4-12B-it", "gemma4_e4b": "google/gemma-4-E4B-it"}


def log_memory(name: str) -> None:
    """GPU memory held by the weights of this configuration (before any activations)."""
    torch.cuda.synchronize()
    mem_path = OUT / "memory.json"
    mem = json.loads(mem_path.read_text(encoding="utf-8")) if mem_path.exists() else {}
    mem[name] = round(torch.cuda.memory_allocated() / 1e9, 2)
    mem_path.write_text(json.dumps(mem, indent=1), encoding="utf-8")


def run_dense(cfg: str, qs: list[dict], args) -> None:
    from eval_dense_quant import load
    from transformers import AutoTokenizer

    path = OUT / f"answers_{cfg}.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [q for q in qs if q["id"] not in done][: args.limit]
    if not todo:
        return
    print(f"=== {cfg}: {len(todo)} questions", flush=True)
    model, _ple, _ = load(DENSE[cfg], bf16=False, quant_embed=True)
    tok = AutoTokenizer.from_pretrained(DENSE[cfg])
    tok.padding_side = "left"
    log_memory(cfg)
    res = answer(model, tok, todo, args.batch, args.max_new)
    with path.open("a", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=30)
    ap.add_argument("--configs", nargs="+", default=["own", "shared", "full"])
    ap.add_argument("--budget", type=int, default=51)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--limit", type=int, default=None, help="dense configs only: answer at most N questions (smoke test)")
    args = ap.parse_args()

    qs = questions(args.per_domain)
    for cfg in [c for c in args.configs if c in DENSE]:
        run_dense(cfg, qs, args)
    args.configs = [c for c in args.configs if c not in DENSE]
    if not args.configs:
        return
    store = ExpertStore.from_packs([ROOT / "quant" / f"{n}.safetensors" for n in ("base", "medicine", "law", "math", "rest")])
    model, tok, _ = load_resident([ROOT / "quant" / "base.safetensors"], quant_dense=True, quant_embed=True)
    model.eval()
    tok.padding_side = "left"
    base = load_manifest(ROOT / "expert_packs" / "base.json")[1]
    prof = lambda d: counts(ROOT / "profiles" / f"wiki_{d}.npz")  # noqa: E731
    n = args.budget

    for cfg in args.configs:
        name = cfg if n == 51 or cfg == "full" else f"{cfg}_n{n}"
        path = OUT / f"answers_{name}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        todo = [q for q in qs if q["id"] not in done]
        if not todo:
            continue
        print(f"=== {cfg}: {len(todo)} questions", flush=True)
        groups = {d: [q for q in todo if q["domain"] == d] for d in WIKI} if cfg == "own" else {"all": todo}
        if cfg == "full":
            store.install_offload(model, n)
        elif cfg == "shared":
            store.install(model, mix_mask(base, [prof(d) for d in WIKI], n))
        for d, group in groups.items():
            if not group:
                continue
            if cfg == "own":
                store.install(model, base | pack_mask(prof(d), base, n / 128))
            if cfg != "full":
                log_memory(name)
            res = answer(model, tok, group, args.batch, args.max_new)
            with path.open("a", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
