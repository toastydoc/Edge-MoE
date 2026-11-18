"""Retrain pack gates against hard negatives and write them into pack files.

    python scripts/train_gates.py [--pack-dirs packs]

Round 1 gates were trained "domain vs commonsense/SQuAD", so they learned
"sounds like an exam question" (medicine) or "long passage" (law). Here every
gate is trained against a shared REFERENCE POOL of academic questions from many
subjects, so it has to learn topic:

    MMLU validation + dev, all subjects, labelled by domain (or "other")
    ARC (science), CommonsenseQA, SQuAD                      -> "other"
    MMLU profiling stems (data/prompts, disjoint from data/test) for the pack domains
    1,000 questions from each pack's own training data

Each item is shown with its options half the time and as a bare question the
other half, so a gate cannot key on format. Computer science and history are
left out of the pool entirely: at test time they are domains no gate has ever
seen, which is the honest test of "stays quiet on unrelated questions".

Independence: the pool ships with the base plate; a new pack's gate is trained
against it without touching other packs. A gate's negatives exclude its own
domain and a short list of adjacent subjects.

Prints how often each gate fires on the MMLU test questions (data/test).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.base_plate import BasePlate, format_mc  # noqa: E402
from edge_moe.packs import PackSet, train_gate  # noqa: E402

PACKS = ["medicine", "law", "math"]
MMLU = {
    "medicine": ["clinical_knowledge", "college_medicine", "professional_medicine", "anatomy", "medical_genetics", "virology"],
    "law": ["professional_law", "international_law", "jurisprudence"],
    "math": ["abstract_algebra", "college_mathematics", "high_school_mathematics", "elementary_mathematics", "high_school_statistics"],
}
UNSEEN = ["college_computer_science", "high_school_computer_science", "computer_security", "machine_learning",
          "high_school_us_history", "high_school_world_history", "high_school_european_history", "prehistory"]  # fmt: skip
ADJACENT = {  # neither positive nor negative for that gate
    "medicine": ["nutrition", "human_aging", "college_biology", "high_school_biology"],
    "law": ["business_ethics", "moral_disputes", "high_school_government_and_politics", "us_foreign_policy"],
    "math": ["college_physics", "high_school_physics", "conceptual_physics", "econometrics", "formal_logic"],
}
LAW_PREFIX = "Which holding best fills the blank in this case excerpt?\n\n"


def render(q, choices, rng):
    q = q.removeprefix(LAW_PREFIX)
    return format_mc(q, choices) if choices and rng.random() < 0.5 else q.strip()


def build_pool(rng):
    pool = []  # (text, label)
    subj_label = {s: d for d, ss in MMLU.items() for s in ss}
    for split in ("validation", "dev"):
        for r in load_dataset("cais/mmlu", "all", split=split):
            if r["subject"] in UNSEEN:
                continue
            pool.append((render(r["question"], r["choices"], rng), subj_label.get(r["subject"], r["subject"])))
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for r in load_dataset("allenai/ai2_arc", cfg, split="train").select(range(600)):
            pool.append((render(r["question"], r["choices"]["text"], rng), "other"))
    for line in (ROOT / "data" / "packs" / "generic.jsonl").read_text(encoding="utf-8").splitlines()[:1500]:
        r = json.loads(line)
        pool.append((render(r["question"], r["choices"], rng), "other"))
    for d in PACKS:
        for line in (ROOT / "data" / "prompts" / f"{d}.jsonl").read_text(encoding="utf-8").splitlines():
            pool.append((json.loads(line)["prompt"].strip(), d))
        rows = (ROOT / "data" / "packs" / f"{d}.jsonl").read_text(encoding="utf-8").splitlines()[:-500][:1000]
        for line in rows:  # [:-500] = never the pack's held-out rows
            r = json.loads(line)
            pool.append((render(r["question"], r["choices"], rng), d))
    return pool


@torch.inference_mode()
def featurize(plate, ps, texts, bs=32, max_tokens=640):
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        enc = plate.encode([plate.prompt_text(texts[j]) for j in idx], max_tokens)
        ps.forward_hidden(enc["input_ids"], enc["attention_mask"], mode="features")
        out.append((idx, ps.last_features.cpu()))
    feats = torch.empty(len(texts), out[0][1].shape[1])
    for idx, f in out:
        feats[idx] = f
    return feats


def write_gate(path: Path, gate, stats: dict):
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["meta"])
    t = load_file(str(path))
    for k in ("mean", "std", "w", "b"):
        t[f"gate.{k}"] = getattr(gate, k).float().cpu().contiguous()
    meta["threshold"] = gate.threshold
    meta["gate_version"] = 2
    meta["gate_heldout"] = stats
    save_file(t, str(path), metadata={"meta": json.dumps(meta)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dirs", nargs="+", type=Path, default=[ROOT / "packs"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gates_v2.json")
    args = ap.parse_args()
    rng = random.Random(0)

    pool = build_pool(rng)
    print(f"reference pool: {len(pool)} questions", flush=True)
    plate = BasePlate()
    ps = PackSet(plate)
    X = featurize(plate, ps, [t for t, _ in pool]).to(plate.device)
    labels = [l for _, l in pool]
    perm = torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))
    is_val = torch.zeros(len(pool), dtype=torch.bool)
    is_val[perm[: len(pool) // 5]] = True

    gates, report = {}, {"pool_size": len(pool), "gates": {}}
    for d in PACKS:
        pos = torch.tensor([l == d for l in labels])
        neg = torch.tensor([l != d and l not in ADJACENT[d] for l in labels])
        gate = train_gate(X[(pos & ~is_val).to(plate.device)], X[(neg & ~is_val).to(plate.device)])
        pv, nv = gate.prob(X[(pos & is_val).to(plate.device)]), gate.prob(X[(neg & is_val).to(plate.device)])
        stats = {"tpr": float((pv >= 0.5).float().mean()), "fpr": float((nv >= 0.5).float().mean()),
                 "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}  # fmt: skip
        gates[d] = gate
        report["gates"][d] = {"pool_val": stats}
        print(f"gate {d}: pool held-out TPR {stats['tpr']:.2f} FPR {stats['fpr']:.2f} "
              f"({stats['n_pos']} pos / {stats['n_neg']} neg)", flush=True)  # fmt: skip

    # firing on the MMLU test questions, shown as asked (with options), as in eval_packs
    print("\nfiring rate on MMLU test questions (CS and history never seen by any gate):")
    print(f"{'':10s}" + "".join(f"{d:>18s}" for d in ["medicine", "law", "math", "computer_science", "history"]))
    test = {}
    for dom in ["medicine", "law", "math", "computer_science", "history"]:
        rows = [json.loads(l) for l in (ROOT / "data" / "test" / f"{dom}.jsonl").read_text(encoding="utf-8").splitlines()]
        test[dom] = featurize(plate, ps, [format_mc(r["question"], r["choices"]) for r in rows]).to(plate.device)
    for d in PACKS:
        rates = {dom: float((gates[d].prob(f) >= 0.5).float().mean()) for dom, f in test.items()}
        report["gates"][d]["mmlu_fire_rate"] = rates
        print(f"{d:10s}" + "".join(f"{v:>18.2f}" for v in rates.values()))

    for pack_dir in args.pack_dirs:
        for d in PACKS:
            p = pack_dir / f"{d}.safetensors"
            if p.exists():
                write_gate(p, gates[d], report["gates"][d]["pool_val"])
                print(f"wrote gate into {p}")
    torch.save({d: vars(g) for d, g in gates.items()}, ROOT / "packs" / "gates_v2.pt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
