"""Build per-domain profiling prompts and held-out test questions from MMLU.

    python scripts/build_datasets.py

Each domain pools several MMLU subjects (test split). Questions are sampled
round-robin across subjects so no single subject dominates, then split into:

    data/test/<domain>.jsonl     200 multiple-choice questions, never profiled
    data/prompts/<domain>.jsonl  up to 300 question stems (no answer options),
                                 so routing is driven by domain content rather
                                 than the shared A/B/C/D layout
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DOMAINS = {
    "medicine": ["clinical_knowledge", "college_medicine", "professional_medicine", "anatomy", "medical_genetics", "virology"],
    "law": ["professional_law", "international_law", "jurisprudence"],
    "computer_science": ["college_computer_science", "high_school_computer_science", "computer_security", "machine_learning"],
    "math": ["abstract_algebra", "college_mathematics", "high_school_mathematics", "elementary_mathematics", "high_school_statistics"],
    "history": ["high_school_us_history", "high_school_world_history", "high_school_european_history", "prehistory"],
}
N_TEST, N_PROFILE, SEED = 200, 300, 0

root = Path(__file__).resolve().parents[1] / "data"
by_subject = defaultdict(list)
for row in load_dataset("cais/mmlu", "all", split="test"):
    by_subject[row["subject"]].append(row)

for domain, subjects in DOMAINS.items():
    rng = random.Random(SEED)
    pools = []
    seen = set()  # MMLU repeats some question texts; keep the first so test never leaks into profile
    for s in subjects:
        rows = [r for r in by_subject[s] if not (r["question"] in seen or seen.add(r["question"]))]
        assert rows, f"no rows for {s}"
        rng.shuffle(rows)
        pools.append(rows)
    ordered = []  # round-robin across subjects
    while any(pools):
        for p in pools:
            if p:
                ordered.append(p.pop())
    test, profile = ordered[:N_TEST], ordered[N_TEST : N_TEST + N_PROFILE]
    assert not {r["question"] for r in test} & {r["question"] for r in profile}

    for kind, rows in (("test", test), ("prompts", profile)):
        path = root / kind / f"{domain}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                if kind == "test":
                    out = {"question": r["question"], "choices": r["choices"], "answer": r["answer"], "subject": r["subject"]}
                else:
                    out = {"prompt": r["question"], "subject": r["subject"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for r in profile:
        counts[r["subject"]] += 1
    print(f"{domain}: {len(profile)} profile, {len(test)} test | profile by subject: {dict(counts)}")
