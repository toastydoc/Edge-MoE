"""Text for distilling the 26B into packs: data/distill/<domain>.jsonl

    python scripts/build_distill_data.py [--n 4000]

Each row is one chat exchange {"user", "assistant", "letter"}; the teacher's
next-token distribution is recorded at every position, so the pack learns how
the 26B reads and continues domain text, not just which letter it picks.

    medicine  MedMCQA question + options -> letter + explanation
    math      AQuA-RAT question + options -> letter + rationale
    law       half CaseHOLD (letter only), half Law StackExchange questions with
              their top answer: CaseHOLD alone taught "pick the holding", not law
Rows are drawn from the packs' training portion (never their held-out 500).
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.base_plate import LETTERS, format_mc  # noqa: E402


def from_pack(domain, n):
    rows = [json.loads(l) for l in (ROOT / "data" / "packs" / f"{domain}.jsonl").read_text(encoding="utf-8").splitlines()]
    out = []
    for r in rows[:-500][:n]:
        letter = LETTERS[r["answer"]]
        out.append({
            "user": format_mc(r["question"], r["choices"]),
            "assistant": letter + (f"\n\n{r['explanation']}" if r["explanation"] else ""),
            "letter": letter,
        })  # fmt: skip
    return out


def strip_html(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def law_se(n, rng):
    ds = load_dataset("ymoslem/Law-StackExchange", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        answers = sorted(r.get("answers") or [], key=lambda a: -(a.get("score") or 0))
        if not answers or (answers[0].get("score") or 0) < 2:
            continue
        q = strip_html(f"{r['question_title']}\n\n{r['question_body']}")[:1500]
        a = strip_html(answers[0]["body"])[:1800]
        out.append({"user": q, "assistant": a, "letter": ""})
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    rng = random.Random(0)
    out_dir = ROOT / "data" / "distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {
        "medicine": from_pack("medicine", args.n),
        "math": from_pack("math", args.n),
        "law": from_pack("law", args.n // 2) + law_se(args.n - args.n // 2, rng),
    }
    for d, rows in sets.items():
        rng.shuffle(rows)
        with open(out_dir / f"{d}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{d}: {len(rows)} rows ({sum(bool(r['letter']) for r in rows)} multiple choice)")


if __name__ == "__main__":
    main()
