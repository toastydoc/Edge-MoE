"""Prompts for profiling a "reconcile" pack: which of two dated versions of a rule applies on a date.

    python scripts/build_reconcile_prompts.py [--n 1600]

Uses the same non-test CFR titles as the regulation pack (build_reg_prompts.PARTS);
the policy tasks' titles are never read. Each prompt is synthetic: one paragraph
with a numeric value, a second version with that value changed, two effective
dates in random order, a question for a date between or after them, and a worked
answer. The answer is part of the profiled text, because the profiler counts
only prompt tokens and the date reasoning happens in the answer.

Writes data/prompts/reconcile.jsonl ({"prompt": ...}) for scripts/profile_domain.py.
"""

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from build_policy_tasks import API, get, sections  # noqa: E402
from build_reg_prompts import PARTS, TEST_TITLES  # noqa: E402

from edge_moe.policy_scoring import value_tokens  # noqa: E402


def alter(val: str, rng: random.Random) -> str | None:
    """A plausible different value with the same unit/format, or None."""
    digits = "".join(c for c in val if c.isdigit())
    if not digits or len(digits) > 9:
        return None
    n = int(digits)
    m = max(1, round(n * rng.choice([0.8, 0.9, 1.02, 1.05, 1.1, 1.25, 1.5])))
    if m == n:
        m = n + 1
    new = f"{m:,}" if "," in val else str(m)
    i = val.index(digits[0])
    # replace the numeric run (with its commas) in the original string
    j = i
    while j < len(val) and (val[j].isdigit() or val[j] == ","):
        j += 1
    return val[:i] + new + val[j:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    args = ap.parse_args()
    assert not TEST_TITLES & {t for t, _ in PARTS}
    rng = random.Random(1)
    titles = {t["number"]: t["latest_issue_date"] for t in json.loads(get(f"{API}/titles.json"))["titles"]}
    paras = []
    for t, p in PARTS:
        try:
            secs = sections(get(f"{API}/full/{titles[t]}/title-{t}.xml?part={p}"))
        except Exception as e:  # noqa: BLE001
            print(f"skip {t} CFR {p}: {e}", flush=True)
            continue
        for s in secs:
            for para in s["paras"]:
                if 150 <= len(para) <= 1200 and value_tokens(para):
                    paras.append((s["head"], para))
    rng.shuffle(paras)
    print(f"{len(paras)} paragraphs with a value", flush=True)

    out_rows = []
    for head, para in paras:
        if len(out_rows) >= args.n:
            break
        vals = value_tokens(para)
        span, v_old = vals[rng.randrange(len(vals))]
        v_new = alter(para[span[0] : span[1]], rng)
        if not v_new or v_new == para[span[0] : span[1]]:
            continue
        old_txt, new_txt = para, para[: span[0]] + v_new + para[span[1] :]
        d_old = date(2012, 1, 1) + timedelta(days=rng.randrange(3000))
        d_new = d_old + timedelta(days=rng.randrange(120, 1500))
        later = rng.random() < 0.5
        ask = d_new + timedelta(days=rng.randrange(1, 400)) if later else d_old + (d_new - d_old) / 2
        docs = [("old", d_old, old_txt), ("new", d_new, new_txt)]
        rng.shuffle(docs)
        lab = {k: "AB"[i] for i, (k, _, _) in enumerate(docs)}
        doc = "\n\n".join(f"Guidance {'AB'[i]}, in effect from {d}:\n{head}\n{txt}" for i, (_, d, txt) in enumerate(docs))
        doc += "\n\nEach version stays in effect until the next one takes effect."
        masked = new_txt[: span[0]] + "___" + new_txt[span[0] + len(v_new) :]
        q = (f"The two versions above conflict. As of {ask.isoformat()}, what value applies in this provision?\n\"{masked}\"\n"
             "State the value and which guidance (A or B) governs on that date.")  # fmt: skip
        gov, other = ("new", "old") if later else ("old", "new")
        g_date, o_date = (d_new, d_old) if later else (d_old, d_new)
        g_val = v_new if later else para[span[0] : span[1]]
        why = (f"Guidance {lab['new']} took effect on {d_new}, before {ask}, so it replaced Guidance {lab['old']}."
               if later else
               f"Guidance {lab['new']} only takes effect on {d_new}, after {ask}, so Guidance {lab['old']} (in effect from {d_old}) still applies.")  # fmt: skip
        ans = f"As of {ask}, Guidance {lab[gov]} governs. {why} The value is {g_val}."
        out_rows.append({"prompt": f"{doc}\n\n{q}\n\n{ans}"})
        _ = (other, g_date, o_date)

    out = ROOT / "data" / "prompts" / "reconcile.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} prompts to {out}")


if __name__ == "__main__":
    main()
