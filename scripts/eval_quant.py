"""Run the 26B from 4-bit pack files, fully resident on the GPU, and measure it.

    python scripts/eval_quant.py --install base medicine law math [--domains ...] [--limit N]

Reports GPU memory after loading, peak during scoring, MMLU accuracy (same
prompts and scoring as eval_expert_packs.py) and greedy generation speed.
Writes results/quant_<installed>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge_moe.quant_experts import load_resident  # noqa: E402
from eval_expert_packs import LETTERS, batches_by_length, format_question  # noqa: E402

GEN_PROMPTS = [
    "Explain how the kidneys regulate blood pressure.",
    "What is the difference between a contract being void and voidable?",
    "Prove that the square root of 2 is irrational.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", nargs="+", default=["base", "medicine", "law", "math"])
    ap.add_argument("--domains", nargs="*", default=["medicine", "law", "math", "computer_science", "history"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--position-budget", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--quant-dense", action="store_true", help="attention / shared MLP in 4-bit too")
    ap.add_argument("--quant-embed", action="store_true", help="tied embedding / lm_head in 4-bit too")
    args = ap.parse_args()

    t0 = time.time()
    model, tok, installed = load_resident([ROOT / "quant" / f"{n}.safetensors" for n in args.install],
                                          quant_dense=args.quant_dense, quant_embed=args.quant_embed)  # fmt: skip
    torch.cuda.synchronize()
    res = {"installed": args.install, "quant_dense": args.quant_dense, "quant_embed": args.quant_embed,
           "experts_installed": int(installed.sum()),
           "load_seconds": time.time() - t0, "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
           "gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9}  # fmt: skip
    print(f"loaded {args.install}: {res['experts_installed']} experts, {res['gpu_allocated_gb']:.2f}GB allocated "
          f"({res['gpu_reserved_gb']:.2f}GB reserved), {res['load_seconds']:.0f}s", flush=True)  # fmt: skip
    device = model.get_input_embeddings().weight.device
    lm_head = model.get_output_embeddings()
    letter_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in LETTERS], device=device)
    tok.truncation_side = "left"
    torch.cuda.reset_peak_memory_stats()

    res["accuracy"], res["pred"] = {}, {}
    for domain in args.domains:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"{domain}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = rows[: args.limit]
        chats = [tok.apply_chat_template([{"role": "user", "content": format_question(r["question"], r["choices"])}],
                                         tokenize=False, add_generation_prompt=True) for r in rows]  # fmt: skip
        lens = [min(len(tok.encode(c, add_special_tokens=False)), args.max_tokens) for c in chats]
        pred = [-1] * len(rows)
        t1 = time.time()
        for idx in batches_by_length(lens, 1, args.position_budget):
            enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_tokens, add_special_tokens=False).to(device)  # fmt: skip
            with torch.inference_mode():
                h = model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
                last = h[torch.arange(len(idx), device=device), enc["attention_mask"].sum(1) - 1]
                sc = lm_head(last).float()[:, letter_ids].cpu()
            for j, i in enumerate(idx):
                pred[i] = int(sc[j, : len(rows[i]["choices"])].argmax())
        acc = sum(p == r["answer"] for p, r in zip(pred, rows)) / len(rows)
        res["accuracy"][domain], res["pred"][domain] = acc, pred
        print(f"== {domain} (n={len(rows)}, {time.time() - t1:.0f}s): {acc:.3f}", flush=True)
    res["peak_gpu_scoring_gb"] = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    speeds = []
    for p in GEN_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(device)  # fmt: skip
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.gen_tokens, do_sample=False)
        torch.cuda.synchronize()
        n = out.shape[1] - enc["input_ids"].shape[1]
        speeds.append(n / (time.time() - t1))
        print(f"generate: {n} tokens at {speeds[-1]:.1f} tok/s | {tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)[:160]!r}", flush=True)
    res["gen_tokens_per_s"] = sum(speeds) / len(speeds)
    res["peak_gpu_generate_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak GPU: scoring {res['peak_gpu_scoring_gb']:.2f}GB, generating {res['peak_gpu_generate_gb']:.2f}GB; "
          f"{res['gen_tokens_per_s']:.1f} tok/s", flush=True)  # fmt: skip
    tag = "_".join(args.install) + ("_qdense" if args.quant_dense else "") + ("_qembed" if args.quant_embed else "") + (f"_limit{args.limit}" if args.limit else "")
    (ROOT / "results" / f"quant_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
