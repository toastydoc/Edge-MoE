"""Prompts for profiling the 26B base plate: data/prompts_base/universal.jsonl

    python scripts/build_base_prompts.py

"Universal" = general knowledge that is none of the five test domains:
MMLU validation + dev questions from every subject outside medicine, law, math,
computer science and history; ARC science questions; everyday CommonsenseQA /
SQuAD questions. Question stems only, same as data/prompts. No row overlaps
data/test (that is MMLU's test split).
"""

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {  # same as build_datasets.py (not imported: that script runs on import)
    "medicine": ["clinical_knowledge", "college_medicine", "professional_medicine", "anatomy", "medical_genetics", "virology"],
    "law": ["professional_law", "international_law", "jurisprudence"],
    "computer_science": ["college_computer_science", "high_school_computer_science", "computer_security", "machine_learning"],
    "math": ["abstract_algebra", "college_mathematics", "high_school_mathematics", "elementary_mathematics", "high_school_statistics"],
    "history": ["high_school_us_history", "high_school_world_history", "high_school_european_history", "prehistory"],
}
EXCLUDED = {s for ss in DOMAINS.values() for s in ss}


def main() -> None:
    rng = random.Random(0)
    rows = []
    for split in ("validation", "dev"):
        for r in load_dataset("cais/mmlu", "all", split=split):
            if r["subject"] not in EXCLUDED:
                rows.append({"prompt": r["question"].strip(), "subject": r["subject"]})
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for r in load_dataset("allenai/ai2_arc", cfg, split="train").select(range(300)):
            rows.append({"prompt": r["question"].strip(), "subject": "arc"})
    for line in (ROOT / "data" / "packs" / "generic.jsonl").read_text(encoding="utf-8").splitlines()[:600]:
        r = json.loads(line)
        rows.append({"prompt": r["question"].strip(), "subject": r["source"]})
    rows = [r for r in rows if r["prompt"]]
    rng.shuffle(rows)
    out = ROOT / "data" / "prompts_base" / "universal.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} prompts -> {out}")


if __name__ == "__main__":
    main()
