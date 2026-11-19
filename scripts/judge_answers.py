"""Blind pairwise judging of results/judge/answers_*.jsonl with Gemini.

    set GEMINI_API_KEY first (never commit it), then:
    python scripts/judge_answers.py --list-models
    python scripts/judge_answers.py [--model gemini-...] [--pairs own:full shared:full]

Each pair is judged twice with the answers in both orders; a configuration wins
only if both verdicts agree, anything else is a tie (cancels position bias).
The judge sees the question and two unlabelled answers, never which
configuration wrote them. Only questions and answers are sent.

Writes results/judge/verdicts_<a>_vs_<b>.jsonl (resumable) and prints win/tie/loss
per domain with a bootstrap 95% interval on the win-or-tie rate.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results" / "judge"
API = "https://generativelanguage.googleapis.com/v1beta"

PROMPT = """You are judging two answers to the same question. Judge which answer is better for an educated reader, considering, in order: factual accuracy, correctly identifying what the term refers to, completeness of the key points, and clarity. Do not prefer an answer for being longer. If they are about equally good, say tie.

Question:
{q}

Answer A:
{a}

Answer B:
{b}

Reply with JSON only: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def key() -> str:
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment first.")
    return k


def call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"x-goog-api-key": key(), "Content-Type": "application/json",
                                          "User-Agent": "edge-moe-research"})  # fmt: skip
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 7:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            raise SystemExit(f"HTTP {e.code}: {e.read()[:300]!r}")
        except urllib.error.URLError:
            time.sleep(10)
    raise SystemExit("network failure")


def verdict(model: str, q: str, a: str, b: str) -> dict:
    r = call(f"{API}/models/{model}:generateContent", {
        "contents": [{"role": "user", "parts": [{"text": PROMPT.format(q=q, a=a, b=b)}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    })  # fmt: skip
    text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        v = {"winner": "tie", "reason": f"unparseable: {text[:80]}"}
    if v.get("winner") not in ("A", "B", "tie"):
        v["winner"] = "tie"
    return v


def load(cfg: str) -> dict:
    return {r["id"]: r["answer"] for r in map(json.loads, (J / f"answers_{cfg}.jsonl").read_text(encoding="utf-8").splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Gemini model id; see --list-models")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--pairs", nargs="+", default=["own:full", "shared:full"])
    args = ap.parse_args()
    if args.list_models:
        for m in call(f"{API}/models?pageSize=200")["models"]:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].removeprefix("models/"), "-", m.get("displayName", ""))
        return
    if not args.model:
        sys.exit("pass --model (see --list-models)")

    qs = [json.loads(l) for l in (J / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(0)
    for pair in args.pairs:
        x, y = pair.split(":")
        ax, ay = load(x), load(y)
        path = J / f"verdicts_{x}_vs_{y}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        for i, q in enumerate(qs):
            if q["id"] in done or q["id"] not in ax or q["id"] not in ay:
                continue
            first_x = rng.random() < 0.5
            a1, b1 = (ax[q["id"]], ay[q["id"]]) if first_x else (ay[q["id"]], ax[q["id"]])
            v1 = verdict(args.model, q["question"], a1, b1)  # x is answer A iff first_x
            v2 = verdict(args.model, q["question"], b1, a1)  # same pair, swapped
            # map each verdict to x / y / tie
            pick = lambda v, x_is_a: "tie" if v["winner"] == "tie" else (x if (v["winner"] == "A") == x_is_a else y)  # noqa: E731
            p1, p2 = pick(v1, first_x), pick(v2, not first_x)
            final = p1 if p1 == p2 else "tie"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": q["id"], "domain": q["domain"], "final": final, "v1": p1, "v2": p2,
                                    "r1": v1.get("reason", ""), "r2": v2.get("reason", "")}, ensure_ascii=False) + "\n")  # fmt: skip
            if i % 20 == 0:
                print(f"  {pair}: {i}/{len(qs)}", flush=True)
        summarize(path, x, y)


def summarize(path: Path, x: str, y: str) -> None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    print(f"\n## {x} vs {y} ({len(rows)} questions; win = both orders agree)\n\n| Domain | {x} wins | ties | {y} wins | {x} win-or-tie |\n|---|---|---|---|---|")
    doms = sorted({r["domain"] for r in rows}) + ["all"]
    rng = random.Random(1)
    for d in doms:
        rs = rows if d == "all" else [r for r in rows if r["domain"] == d]
        w, t, l = (sum(r["final"] == k for r in rs) for k in (x, "tie", y))
        ok = [r["final"] != y for r in rs]
        boots = sorted(sum(rng.choice(ok) for _ in ok) / len(ok) for _ in range(2000))
        print(f"| {d} | {w} | {t} | {l} | {sum(ok) / len(ok):.0%} [{boots[50]:.0%}, {boots[1949]:.0%}] |")
    agree = sum(r["v1"] == r["v2"] for r in rows) / len(rows)
    print(f"\nthe two orders agreed on {agree:.0%} of questions")


if __name__ == "__main__":
    main()
