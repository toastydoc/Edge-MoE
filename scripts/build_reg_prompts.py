"""Prompts for profiling a "regulation" pack, from CFR titles the policy tasks never use.

    python scripts/build_reg_prompts.py [--n 1600]

The policy tasks draw on titles 2, 7, 12, 29, 34, 42 and 45. This uses other
titles (energy, aviation, food and drug, telecom, social security, veterans,
federal acquisition, environment, ethics, immigration, transportation), so the
pack is built from the same kind of text but never the test text. Each prompt
is a ~1,600-character passage followed by one generic instruction of the kind
the device will get (list authorities, find a value, summarise a requirement).

Writes data/prompts/regulation.jsonl ({"prompt": ...}) for scripts/profile_domain.py.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_policy_tasks import API, get, render, sections  # noqa: E402

PARTS = [(10, "50"), (14, "91"), (21, "101"), (47, "15"), (20, "404"), (38, "3"), (48, "52"),
         (40, "262"), (5, "2635"), (8, "214"), (49, "382"), (14, "121")]  # fmt: skip
TEST_TITLES = {2, 7, 12, 29, 34, 42, 45}
INSTRUCTIONS = [
    "List every legal authority cited in the passage above.",
    "What deadline or dollar threshold does the passage above set? Cite the section.",
    "Summarise the requirement in the passage above in two sentences, citing the section.",
    "Which section above governs this situation, and what does it require?",
    "Does the passage above conflict with any earlier version of the rule? Explain briefly.",
    "Extract every numeric limit in the passage above with the section it comes from.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--chars", type=int, default=1600)
    args = ap.parse_args()
    assert not TEST_TITLES & {t for t, _ in PARTS}
    rng = random.Random(0)
    titles = {t["number"]: t["latest_issue_date"] for t in json.loads(get(f"{API}/titles.json"))["titles"]}
    passages = []
    for t, p in PARTS:
        try:
            secs = sections(get(f"{API}/full/{titles[t]}/title-{t}.xml?part={p}"))
        except Exception as e:  # noqa: BLE001
            print(f"skip {t} CFR {p}: {e}", flush=True)
            continue
        text = "\n\n".join(render(s) for s in secs)
        n0 = len(passages)
        for s in range(0, max(len(text) - args.chars, 1), args.chars):
            passages.append(text[s : s + args.chars])
        print(f"{t} CFR {p}: {len(secs)} sections, {len(passages) - n0} passages", flush=True)
    rng.shuffle(passages)
    out = ROOT / "data" / "prompts" / "regulation.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for psg in passages[: args.n]:
            f.write(json.dumps({"prompt": f"{psg.strip()}\n\n{rng.choice(INSTRUCTIONS)}"}) + "\n")
    print(f"wrote {min(args.n, len(passages))} prompts to {out}")


if __name__ == "__main__":
    main()
