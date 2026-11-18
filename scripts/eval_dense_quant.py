"""4-bit dense baselines, scored exactly like eval_quant.py scores the 26B packs.

    python scripts/eval_dense_quant.py --model google/gemma-4-E4B-it [--limit N] [--bf16]

Every linear layer of the language model goes to 4-bit in the expert-pack format
(symmetric, group 32); embeddings, norms and lm_head (tied to the embedding) stay
bf16 -- the same treatment as the 26B. For E4B the per-layer embedding table
(PLE, a pure lookup) stays in system RAM, as in the bf16 baseline.

Writes results/dense_<model>_<int4|bf16>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge_moe.base_plate import _CpuLookup  # noqa: E402
from edge_moe.quant_dense import quantize_embeddings, quantize_linears  # noqa: E402
from eval_expert_packs import LETTERS, batches_by_length, format_question  # noqa: E402
from eval_quant import GEN_PROMPTS  # noqa: E402


def load(model_id, bf16, device="cuda", quant_embed=False):
    full = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cpu")
    full.eval()
    for p in full.parameters():
        p.requires_grad_(False)
    for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio", "multi_modal_projector"):
        if hasattr(full.model, attr):
            setattr(full.model, attr, None)
    lm = full.model.language_model
    keep_cpu = None
    if hasattr(lm, "embed_tokens_per_layer"):
        keep_cpu = lm.embed_tokens_per_layer
        lm.embed_tokens_per_layer = _CpuLookup(keep_cpu)
    counts = (0, 0) if bf16 else quantize_linears(lm, device)
    torch.cuda.empty_cache()
    if quant_embed:
        quantize_embeddings(full, device)
    full.to(device)  # lm_head is tied to lm.embed_tokens
    return full, keep_cpu, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--bf16", action="store_true", help="skip quantisation (sanity check against earlier bf16 numbers)")
    ap.add_argument("--domains", nargs="*", default=["medicine", "law", "math", "computer_science", "history"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--position-budget", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--quant-embed", action="store_true", help="tied embedding / lm_head in 4-bit too")
    args = ap.parse_args()

    t0 = time.time()
    model, _ple, (nq, nk) = load(args.model, args.bf16, quant_embed=args.quant_embed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side, tok.truncation_side = "right", "left"
    torch.cuda.synchronize()
    res = {"model": args.model, "format": "bf16" if args.bf16 else "int4", "linears_quantised": nq,
           "linears_bf16": nk, "load_seconds": time.time() - t0,
           "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9}  # fmt: skip
    print(f"loaded {args.model} ({res['format']}): {nq} linears 4-bit, {nk} kept, "
          f"{res['gpu_allocated_gb']:.2f}GB allocated, {res['load_seconds']:.0f}s", flush=True)  # fmt: skip
    device = model.get_input_embeddings().weight.device
    letter_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in LETTERS], device=device)
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
                h = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
                last = h[torch.arange(len(idx), device=device), enc["attention_mask"].sum(1) - 1]
                sc = model.lm_head(last).float()[:, letter_ids].cpu()
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
    res["quant_embed"] = args.quant_embed
    tag = (args.model.split("/")[-1] + "_" + res["format"] + ("_qembed" if args.quant_embed else "")
           + (f"_limit{args.limit}" if args.limit else ""))  # fmt: skip
    (ROOT / "results" / f"dense_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
