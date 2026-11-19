"""Profiling prompts and held-out test text for ten domains, from Wikipedia article introductions.

    python scripts/build_wiki_domains.py [--profile 300 --test 100]

Each domain is a set of Wikipedia categories, walked two levels down. Articles
found under more than one domain are dropped from all of them, then each
domain's articles are split by article (never by passage) into:

    data/prompts/wiki_<domain>.jsonl   profiling prompts  {"prompt": intro}
    data/test/wiki_<domain>.jsonl      held-out text      {"title", "text"}

Uses the public MediaWiki API; responses are cached in data/policy/cache/.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_policy_tasks import get  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
DOMAINS = {
    "art": ["Art movements", "Painting techniques", "Sculpture", "Art history", "Visual arts genres"],
    "history": ["Ancient history", "Medieval history", "Early modern history", "History of Europe", "Military history"],
    "biology": ["Cell biology", "Genetics", "Ecology", "Zoology", "Botany"],
    "geology": ["Geology", "Mineralogy", "Plate tectonics", "Petrology", "Sedimentology"],
    "politics": ["Political science", "Political ideologies", "Elections", "Political systems", "Public policy"],
    "computer_science": ["Algorithms", "Data structures", "Programming languages", "Computer security", "Operating systems"],
    "economics": ["Macroeconomics", "Microeconomics", "Economic theories", "International trade", "Finance"],
    "chemistry": ["Chemical reactions", "Organic chemistry", "Inorganic chemistry", "Physical chemistry", "Biochemistry"],
    "physics": ["Classical mechanics", "Quantum mechanics", "Thermodynamics", "Electromagnetism", "Particle physics"],
    "philosophy": ["Epistemology", "Ethics", "Metaphysics", "Philosophy of mind", "Logic"],
}


def q(**params) -> dict:
    """GET with the shared cache; on HTTP 429 back off (honouring Retry-After) and retry."""
    params |= {"format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            text = get(url)
            time.sleep(0.5)  # get() already waits 0.5s after a network fetch; keep well under the API's limits
            return json.loads(text)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 7:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 30 * (attempt + 1)
            print(f"  rate limited; waiting {wait}s", flush=True)
            time.sleep(wait)


def members(cat: str) -> tuple[list[str], list[str]]:
    pages, subcats, cont = [], [], {}
    while True:
        r = q(action="query", list="categorymembers", cmtitle=f"Category:{cat}", cmtype="page|subcat", cmlimit="500", **cont)
        for m in r["query"]["categorymembers"]:
            (subcats if m["ns"] == 14 else pages).append(m["title"].removeprefix("Category:"))
        if "continue" not in r or len(pages) > 3000:
            return pages, subcats
        cont = {"cmcontinue": r["continue"]["cmcontinue"]}


def titles_for(cats: list[str], cap: int = 1500) -> set[str]:
    out = set()
    for root in cats:
        pages, subs = members(root)
        out |= {p for p in pages if not p.startswith(("List of", "Outline of", "Index of", "Glossary of"))}
        for s in subs[:15]:
            p2, _ = members(s)
            out |= {p for p in p2 if not p.startswith(("List of", "Outline of", "Index of", "Glossary of"))}
    return out


def intros(titles: list[str]) -> dict[str, str]:
    out = {}
    for i in range(0, len(titles), 20):
        r = q(action="query", prop="extracts", exintro="1", explaintext="1", exlimit="20", redirects="1", titles="|".join(titles[i : i + 20]))
        for p in r["query"].get("pages", []):
            t = " ".join(p.get("extract", "").split())
            if len(t) >= 600:
                out[p["title"]] = t[:3000]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=int, default=300)
    ap.add_argument("--test", type=int, default=100)
    args = ap.parse_args()
    rng = random.Random(0)
    found = {}
    for d, cats in DOMAINS.items():
        found[d] = titles_for(cats)
        print(f"{d}: {len(found[d])} candidate articles", flush=True)
    seen = {}
    for d, ts in found.items():
        for t in ts:
            seen[t] = seen.get(t, 0) + 1
    need = args.profile + args.test
    for d, ts in found.items():
        cand = sorted(t for t in ts if seen[t] == 1)
        rng.shuffle(cand)
        texts = intros(cand[: int(need * 1.8)])
        keep = [t for t in cand if t in texts][:need]
        if len(keep) < need:
            print(f"  {d}: only {len(keep)} usable intros", flush=True)
        n_test = min(args.test, len(keep) // 4)
        test, prof = keep[:n_test], keep[n_test:]
        with (ROOT / "data" / "test" / f"wiki_{d}.jsonl").open("w", encoding="utf-8") as f:
            for t in test:
                f.write(json.dumps({"title": t, "text": texts[t]}, ensure_ascii=False) + "\n")
        with (ROOT / "data" / "prompts" / f"wiki_{d}.jsonl").open("w", encoding="utf-8") as f:
            for t in prof:
                f.write(json.dumps({"prompt": texts[t]}, ensure_ascii=False) + "\n")
        print(f"  {d}: {len(prof)} profile, {len(test)} test", flush=True)


if __name__ == "__main__":
    main()
