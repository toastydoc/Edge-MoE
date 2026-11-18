"""Time each 1024-token prefill chunk of a long prompt, with GPU memory, to find where long prompts slow down.

    python scripts/diag_prefill.py --tokens 32768 [--install base law_t40]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.quant_experts import load_resident  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--install", nargs="*", default=["base", "law_t40"])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--empty-cache", action="store_true")
    args = ap.parse_args()
    model, tok, _ = load_resident([ROOT / "quant" / f"{n}.safetensors" for n in args.install], quant_dense=True, quant_embed=True)
    print("attn impl:", model.config.get_text_config()._attn_implementation, flush=True)
    import json

    docs = [json.loads(l)["document"] for l in (ROOT / "data/policy/tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = tok("\n\n".join(docs), return_tensors="pt", add_special_tokens=False)["input_ids"][:, : args.tokens].cuda()
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config.get_text_config())
    n = ids.shape[1]
    with torch.inference_mode():
        for s in range(0, n, args.chunk):
            e = min(s + args.chunk, n)
            torch.cuda.synchronize()
            t = time.time()
            model(input_ids=ids[:, s:e], attention_mask=torch.ones_like(ids[:, :e]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            if args.empty_cache:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"{e:>6} tok  {time.time() - t:6.2f}s  alloc {torch.cuda.memory_allocated() / 1e9:5.2f}GB  "
                  f"reserved {torch.cuda.memory_reserved() / 1e9:5.2f}GB  peak {torch.cuda.max_memory_allocated() / 1e9:5.2f}GB", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
