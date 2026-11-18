"""Training data for route-B domain packs, plus generic data for the gates.

    python scripts/build_pack_data.py [--n 6000]

Each pack is trained on its own domain only (never on another pack's data), so
packs stay independent. Sources are deliberately different from MMLU, which is
kept for evaluation (data/test/*.jsonl):

    medicine  openlifescienceai/medmcqa   4 options, often with an explanation
    law       coastalcph/lex_glue case_hold  5 options (holding of a cited case)
    math      deepmind/aqua_rat            5 options, with a worked rationale
    generic   tau/commonsense_qa + rajpurkar/squad questions
              (negatives for the gates; not used to train any pack)

Output data/packs/<domain>.jsonl rows:
    {"question", "choices", "answer" (index), "explanation" (may be "")}
and data/packs/generic.jsonl rows {"question", "choices" (may be []), ...}.
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def medicine(n, rng):
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    idx = rng.sample(range(len(ds)), min(len(ds), n * 2))
    out = []
    for i in idx:
        r = ds[i]
        if r.get("choice_type") != "single" or r["cop"] not in (0, 1, 2, 3):
            continue
        exp = (r.get("exp") or "").strip()
        exp = re.sub(r"\s+", " ", exp)[:1200]
        out.append({
            "question": r["question"].strip(),
            "choices": [r["opa"], r["opb"], r["opc"], r["opd"]],
            "answer": int(r["cop"]),
            "explanation": exp,
        })  # fmt: skip
        if len(out) == n:
            break
    return out


def law(n, rng):
    ds = load_dataset("coastalcph/lex_glue", "case_hold", split="train")
    idx = rng.sample(range(len(ds)), min(len(ds), n))
    out = []
    for i in idx:
        r = ds[i]
        ctx = r["context"].replace("<HOLDING>", "____").strip()
        out.append({
            "question": "Which holding best fills the blank in this case excerpt?\n\n" + ctx,
            "choices": list(r["endings"]),
            "answer": int(r["label"]),
            "explanation": "",
        })  # fmt: skip
    return out


def math(n, rng):
    ds = load_dataset("deepmind/aqua_rat", "raw", split="train")
    idx = rng.sample(range(len(ds)), min(len(ds), n * 2))
    out = []
    for i in idx:
        r = ds[i]
        opts = [re.sub(r"^[A-E]\)\s*", "", o).strip() for o in r["options"]]
        letter = r["correct"].strip()
        if len(opts) != 5 or letter not in "ABCDE":
            continue
        out.append({
            "question": r["question"].strip(),
            "choices": opts,
            "answer": "ABCDE".index(letter),
            "explanation": r["rationale"].strip()[:1200],
        })  # fmt: skip
        if len(out) == n:
            break
    return out


def generic(n, rng):
    out = []
    cqa = load_dataset("tau/commonsense_qa", split="train")
    for i in rng.sample(range(len(cqa)), min(len(cqa), n // 2)):
        r = cqa[i]
        if not r["answerKey"]:
            continue
        out.append({
            "question": r["question"],
            "choices": list(r["choices"]["text"]),
            "answer": r["choices"]["label"].index(r["answerKey"]),
            "explanation": "",
            "source": "commonsense_qa",
        })  # fmt: skip
    sq = load_dataset("rajpurkar/squad", split="train")
    for i in rng.sample(range(len(sq)), n - len(out)):
        r = sq[i]
        out.append({"question": r["question"], "choices": [], "answer": -1, "explanation": "", "source": "squad"})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()
    out_dir = ROOT / "data" / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in [("medicine", medicine), ("law", law), ("math", math), ("generic", generic)]:
        rows = fn(args.n, random.Random(0))
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with_exp = sum(bool(r["explanation"]) for r in rows)
        print(f"{name}: {len(rows)} rows ({with_exp} with explanation)", flush=True)


if __name__ == "__main__":
    main()
