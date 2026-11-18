"""Run the eCFR policy tasks on one configuration and measure quality, VRAM and latency.

    python scripts/eval_policy.py --config gemma4-12b
    python scripts/eval_policy.py --config 26b --install base law            # matching pack
    python scripts/eval_policy.py --config 26b --install base law_t40        # 8GB-sized pack
    python scripts/eval_policy.py --config 26b --install base medicine law math   # all resident
    python scripts/eval_policy.py --config 26b-routed                        # gate loads one pack

Every model is 4-bit throughout (linear layers, experts, embeddings). Greedy
decoding. Per task: time to first token, total time, tokens generated, peak
GPU memory. Cold = model load + the first task; warm = the median of the rest.
"26b-routed" keeps only the base plate resident; for each task the gates read
the base plate's hidden states and the most likely pack is loaded from disk
when it differs from the one installed (swap time recorded).
Then a long-context probe: prefill 16k and 32k tokens + 8 new tokens.

Writes results/policy_<config>.json.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge_moe.packs import Gate  # noqa: E402
from edge_moe.policy_scoring import score  # noqa: E402
from edge_moe.quant_experts import load_resident, set_allowed, set_packs  # noqa: E402
from edge_moe.router_hooks import find_routers  # noqa: E402

MAX_NEW = {"extract": 160, "cite": 96, "reconcile": 96}
PACKS = ["medicine", "law", "math"]
SYSTEM_HINT = "Answer from the provided regulations only. Be concise."


class _FirstToken(LogitsProcessor):
    def __init__(self):
        self.t = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            torch.cuda.synchronize()
            self.t = time.time()
        return scores


class _Stop(Exception):
    pass


def chunked_generate(model, input_ids, max_new_tokens, chunk=1024, logits_processor=None):
    """Prefill all but the last prompt token in `chunk`-token pieces into a KV cache, then
    generate. Local runtimes (llama.cpp, vLLM) prefill in chunks, so peak memory is not
    set by one giant attention pass over the whole prompt."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config.get_text_config())
    n = input_ids.shape[1]
    with torch.inference_mode():
        for s in range(0, n - 1, chunk):
            e = min(s + chunk, n - 1)
            model(input_ids=input_ids[:, s:e], attention_mask=torch.ones_like(input_ids[:, :e]),
                  past_key_values=cache, use_cache=True, logits_to_keep=1)  # fmt: skip
            # each chunk allocates slightly larger buffers than the last; without this the
            # caching allocator's pool grows past the card's memory and Windows spills it to
            # system RAM (32k prompt: 24GB reserved for 7.6GB in use, ~9x slower chunks)
            torch.cuda.empty_cache()
        return model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), past_key_values=cache,
                              max_new_tokens=max_new_tokens, do_sample=False, logits_processor=logits_processor)  # fmt: skip


def gate_features(model, enc, layers=(5, 17)):
    """Mean-pooled hidden states after the given decoder layers, base plate only."""
    names = [n.rsplit(".", 1)[0] for n, _ in find_routers(model)]
    feats, handles = {}, []
    m = enc["attention_mask"].float()

    def hook(L):
        def f(mod, args, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            feats[f"h{L}"] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True))
            if L == max(layers):
                raise _Stop

        return f

    for L in layers:
        handles.append(model.get_submodule(names[L]).register_forward_hook(hook(L)))
    try:
        with torch.inference_mode():
            model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["gemma4-12b", "gemma3-12b", "e4b", "26b", "26b-routed", "26b-store"])
    ap.add_argument("--install", nargs="*", default=["base", "law"])
    ap.add_argument("--select", default="joint",
                    help="26b-store: pack:<domain> | mix:<d1>+<d2> | joint | plan | offload")  # fmt: skip
    ap.add_argument("--budget", type=int, default=51, help="26b-store: experts per layer on the GPU")
    ap.add_argument("--tasks", default="tasks.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", type=int, nargs="*", default=[16384, 32768])
    ap.add_argument("--gate-tokens", type=int, default=2048)
    args = ap.parse_args()
    tasks = [json.loads(l) for l in (ROOT / "data" / "policy" / args.tasks).read_text(encoding="utf-8").splitlines()]
    # interleave types so the cold task and the warm median are not one type
    by = {t: [x for x in tasks if x["type"] == t] for t in MAX_NEW}
    tasks = [x for grp in zip(*by.values()) for x in grp] + [x for g in by.values() for x in g[min(map(len, by.values())):]]
    tasks = tasks[: args.limit] if args.limit else tasks

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gates, current, plan, offload_stats = None, None, False, None
    if args.config == "26b-store":
        from edge_moe.expert_packs import counts, load_manifest, pack_mask
        from edge_moe.expert_store import ExpertStore, joint_mask, mix_mask

        # build the pinned store before the model: filling it after the model loaded crashed intermittently on Windows
        store = ExpertStore.from_packs([ROOT / "quant" / f"{n}.safetensors" for n in ("base", "medicine", "law", "math", "rest")])
        print(f"expert store: {store.gb():.1f}GB in pinned RAM", flush=True)
        model, tok, _ = load_resident([ROOT / "quant" / "base.safetensors"], quant_dense=True, quant_embed=True)
        base_mask = load_manifest(ROOT / "expert_packs" / "base.json")[1]
        prof = lambda d: counts(ROOT / ("profiles_base/universal.npz" if d == "universal" else f"profiles/{d}.npz"))  # noqa: E731
        kind, _, arg = args.select.partition(":")
        n = args.budget
        if kind == "pack":
            installed = store.install(model, base_mask | pack_mask(prof(arg), base_mask, n / 128))
        elif kind == "mix":
            installed = store.install(model, mix_mask(base_mask, [prof(d) for d in arg.split("+")], n))
        elif kind == "joint":
            ds = arg.split("+") if arg else ["universal", "medicine", "law", "math", "regulation"]
            installed = store.install(model, joint_mask([prof(d) for d in ds], n))
        elif kind == "plan":
            installed, plan = store.install(model, base_mask | pack_mask(prof("law"), base_mask, n / 128)), True
        elif kind == "offload":
            offload_stats = store.install_offload(model, n, preload=joint_mask([prof(d) for d in ("universal", "medicine", "law", "math", "regulation")], n))
        else:
            raise SystemExit(f"unknown --select {args.select}")
        if offload_stats is None:
            print(f"installed {int(installed.sum())} experts ({installed.sum(1).float().mean():.0f}/layer)", flush=True)
        name = f"26b_{args.select.replace(':', '-').replace('+', '-')}_n{n}"
    elif args.config.startswith("26b"):
        install = ["base"] if args.config == "26b-routed" else args.install
        model, tok, installed = load_resident([ROOT / "quant" / f"{n}.safetensors" for n in install], quant_dense=True, quant_embed=True)
        base_mask = installed.clone()
        if args.config == "26b-routed":
            g = torch.load(ROOT / "expert_packs" / "gates.pt")
            gates = {d: (v["features"], Gate(v["mean"], v["std"], v["w"], v["b"], v["threshold"]).to("cuda")) for d, v in g.items()}
        name = args.config if args.config == "26b-routed" else "26b_" + "_".join(args.install)
    else:
        from eval_dense_quant import load
        from transformers import AutoTokenizer

        mid = {"gemma4-12b": "google/gemma-4-12B-it", "gemma3-12b": "google/gemma-3-12b-it", "e4b": "google/gemma-4-E4B-it"}[args.config]
        model, _ple, _ = load(mid, bf16=False, quant_embed=True)
        tok = AutoTokenizer.from_pretrained(mid)
        name = args.config
    torch.cuda.synchronize()
    load_s = time.time() - t0
    device = model.get_input_embeddings().weight.device
    print(f"[{name}] loaded in {load_s:.0f}s, {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated", flush=True)

    rows = []
    for i, task in enumerate(tasks):
        content = f"{SYSTEM_HINT}\n\n{task['document']}\n\n{task['question']}"
        enc = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(device)  # fmt: skip
        n_in = enc["input_ids"].shape[1]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        route = {}
        if gates:
            set_allowed(model, base_mask)
            g_enc = {k: v[:, -args.gate_tokens :] for k, v in enc.items()}
            f = gate_features(model, g_enc)
            probs = {d: float(gt.prob(f[fs])[0]) for d, (fs, gt) in gates.items()}
            pick = max(probs, key=probs.get)
            torch.cuda.synchronize()
            route = {"gate_probs": probs, "fired": [d for d, p in probs.items() if p >= 0.5], "pack": pick, "gate_s": time.time() - start}
            if pick != current:
                ts = time.time()
                installed = set_packs(model, [ROOT / "quant" / f"{n}.safetensors" for n in ("base", pick)])
                torch.cuda.synchronize()
                route["swap_s"] = time.time() - ts
                current = pick
            set_allowed(model, installed)
        if plan:
            # ask the router what it wants: one base-plate pass over the end of the prompt, recording
            # each router's full probabilities, then install base + the experts it wanted most
            from edge_moe.expert_store import RouterPlan, plan_mask

            set_allowed(model, base_mask)
            with RouterPlan(model) as rp, torch.inference_mode():
                model.model(input_ids=enc["input_ids"][:, -args.gate_tokens :])
            want = plan_mask(rp.mass, args.budget, keep=base_mask)
            torch.cuda.synchronize()
            route = {"plan_s": time.time() - start, "plan_changed": int((want ^ installed).sum()) // 2}
            ts = time.time()
            installed = store.install(model, want)
            torch.cuda.synchronize()
            route["swap_s"] = time.time() - ts
        if offload_stats is not None:
            b0, m0, p0 = offload_stats["bytes"], offload_stats["misses"], offload_stats["picks"]
        ft = _FirstToken()
        out = chunked_generate(model, enc["input_ids"], MAX_NEW[task["type"]], logits_processor=LogitsProcessorList([ft]))
        torch.cuda.synchronize()
        end = time.time()
        text = tok.decode(out[0, n_in:], skip_special_tokens=True)
        n_out = out.shape[1] - n_in
        if offload_stats is not None:
            route = {"copied_gb": (offload_stats["bytes"] - b0) / 1e9, "miss_rate": (offload_stats["misses"] - m0) / max(offload_stats["picks"] - p0, 1)}
        r = {"id": task["id"], "type": task["type"], "bucket": task["bucket"], "input_tokens": n_in, "output_tokens": n_out,
             "ttft_s": ft.t - start, "total_s": end - start, "decode_tok_s": (n_out - 1) / max(end - ft.t, 1e-6),
             "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, **route, "output": text, **score(task, text)}  # fmt: skip
        rows.append(r)
        main_metric = {"extract": "f1", "cite": "value_correct", "reconcile": "value_correct"}[task["type"]]
        print(f"  {i + 1}/{len(tasks)} {task['id']:>12} {n_in:>6} tok in, ttft {r['ttft_s']:5.1f}s, total {r['total_s']:5.1f}s, "
              f"peak {r['peak_gb']:.2f}GB, {main_metric} {r[main_metric]:.2f}" + (f", pack {route['pack']}" if "pack" in route else "")
              + (f", plan {route['plan_s']:.1f}s swap {route['swap_s']:.1f}s changed {route['plan_changed']}" if "plan_s" in route else "")
              + (f", copied {route['copied_gb']:.1f}GB miss {route['miss_rate']:.2f}" if "copied_gb" in route else ""), flush=True)  # fmt: skip

    probe = {}
    filler = "\n\n".join(t["document"] for t in tasks if t["bucket"] == "8k")
    for n in args.probe:
        ids = tok(filler, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :n].to(device)
        if ids.shape[1] < n:
            ids = ids.repeat(1, n // ids.shape[1] + 1)[:, :n]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t1 = time.time()
            chunked_generate(model, ids, 8)
            torch.cuda.synchronize()
            probe[n] = {"peak_gb": torch.cuda.max_memory_allocated() / 1e9, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                        "seconds": time.time() - t1}  # fmt: skip
        except torch.cuda.OutOfMemoryError:
            probe[n] = {"peak_gb": None, "oom": True}
        print(f"  probe {n} tokens: {probe[n]}", flush=True)

    def med(xs):
        return statistics.median(xs) if xs else None

    warm = rows[1:]
    summary = {"config": name, "load_s": load_s, "cold_first_answer_s": load_s + rows[0]["total_s"],
               "warm_ttft_s": {b: med([r["ttft_s"] for r in warm if r["bucket"] == b]) for b in ("2k", "8k", "versions")},
               "warm_total_s": {b: med([r["total_s"] for r in warm if r["bucket"] == b]) for b in ("2k", "8k", "versions")},
               "decode_tok_s": med([r["decode_tok_s"] for r in warm]),
               "peak_gb": {b: max([r["peak_gb"] for r in rows if r["bucket"] == b], default=None) for b in ("2k", "8k", "versions")},
               "peak_reserved_gb": {b: max([r["peak_reserved_gb"] for r in rows if r["bucket"] == b], default=None) for b in ("2k", "8k", "versions")},
               "probe": probe,
               "extract_f1": statistics.mean([r["f1"] for r in rows if r["type"] == "extract"] or [0]),
               "cite_value": statistics.mean([r["value_correct"] for r in rows if r["type"] == "cite"] or [0]),
               "cite_gold_cited": statistics.mean([r["gold_cited"] for r in rows if r["type"] == "cite"] or [0]),
               "cite_outside": statistics.mean([r["cites_outside_excerpt"] for r in rows if r["type"] == "cite"] or [0]),
               "reconcile_value": statistics.mean([r["value_correct"] for r in rows if r["type"] == "reconcile"] or [0]),
               "reconcile_label": statistics.mean([r["label_correct"] for r in rows if r["type"] == "reconcile"] or [0])}  # fmt: skip
    if plan:
        summary["plan_s_median"] = med([r["plan_s"] for r in rows])
        summary["swap_s_median"] = med([r["swap_s"] for r in rows])
        summary["plan_changed_median"] = med([r["plan_changed"] for r in rows])
    if offload_stats is not None:
        summary["copied_gb_median"] = med([r["copied_gb"] for r in rows])
        summary["copied_gb_per_token"] = statistics.mean([r["copied_gb"] / r["output_tokens"] for r in rows])
        summary["miss_rate"] = statistics.mean([r["miss_rate"] for r in rows])
    if gates:
        summary["packs_chosen"] = {d: sum(r["pack"] == d for r in rows) for d in PACKS}
        summary["swaps"] = sum("swap_s" in r for r in rows)
        summary["swap_s_median"] = med([r["swap_s"] for r in rows if "swap_s" in r])
        summary["gate_s_median"] = med([r["gate_s"] for r in rows])
    print(json.dumps({k: v for k, v in summary.items()}, indent=1), flush=True)
    tag = name + (f"_{Path(args.tasks).stem}" if args.tasks != "tasks.jsonl" else "") + (f"_limit{args.limit}" if args.limit else "")
    (ROOT / "results" / f"policy_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
