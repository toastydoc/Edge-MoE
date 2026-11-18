"""Held-out policy tasks from the US eCFR, with gold answers checked mechanically.

    python scripts/build_policy_tasks.py [--per-type 24]

Sources (regulations as published by the Office of the Federal Register, via
the public eCFR API): 2 CFR 200 (federal grants), 45 CFR 164 (HIPAA privacy),
29 CFR 825 (FMLA), 7 CFR 273 (SNAP), 34 CFR 668 (student aid). None of it was
used to build packs or gates.

Task types, half at ~2k tokens of context and half at ~8k:
  extract    list every authority cited in one section of a longer excerpt;
             gold = citations found by regex, scored by F1 on normalised keys
  reconcile  two dated versions of a section that disagree on one value;
             "as of <date>, what value applies?"; gold = the value in force
  cite       find the value that fills a blank and cite the supporting
             section; scored on the value, the gold citation, and citations
             to sections that are not in the excerpt

Writes data/policy/tasks.jsonl and caches API responses in data/policy/cache/.
"""

import argparse
import difflib
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.policy_scoring import cite_keys, value_tokens  # noqa: E402

API = "https://www.ecfr.gov/api/versioner/v1"
PARTS = [(2, "200"), (45, "164"), (29, "825"), (7, "273"), (34, "668")]
# extra parts scanned only for dated versions (values adjusted most years)
VERSION_PARTS = PARTS + [(29, "1903"), (12, "1026"), (42, "1003")]
CACHE = ROOT / "data" / "policy" / "cache"
CHARS = {"2k": 8000, "8k": 32000}  # ~4 characters per token


def get(url: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".txt")
    if f.exists():
        return f.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "edge-moe-research", "Accept-Encoding": "gzip"})
    raw = urllib.request.urlopen(req, timeout=180).read()
    text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode("utf-8")
    f.write_text(text, encoding="utf-8")
    time.sleep(0.5)
    return text


def sections(xml: str) -> list[dict]:
    root = ET.fromstring(xml.encode("utf-8"))
    out = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        head = " ".join("".join(div.find("HEAD").itertext()).split()) if div.find("HEAD") is not None else ""
        paras = [" ".join("".join(p.itertext()).split()) for p in div.iter("P")]
        paras = [p for p in paras if p]
        if paras and "[Reserved]" not in head:
            out.append({"n": div.get("N"), "head": head, "paras": paras})
    return out


def render(sec: dict) -> str:
    return sec["head"] + "\n" + "\n".join(sec["paras"])


def window(secs: list[dict], i: int, chars: int, rng: random.Random) -> tuple[str, list[str]]:
    """Contiguous sections around secs[i] up to `chars`, target at a random position."""
    lo = hi = i
    size = len(render(secs[i]))
    cap = chars * 1.15
    while size < chars:
        up = len(render(secs[hi + 1])) if hi < len(secs) - 1 and size + len(render(secs[hi + 1])) <= cap else None
        down = len(render(secs[lo - 1])) if lo > 0 and size + len(render(secs[lo - 1])) <= cap else None
        if up is None and down is None:
            break
        if up is not None and (down is None or rng.random() < 0.5):
            hi += 1
            size += up
        else:
            lo -= 1
            size += down
    chunk = secs[lo : hi + 1]
    return "\n\n".join(render(s) for s in chunk), [s["n"] for s in chunk]


def fit(pool, current, buckets, n):
    """Take candidates in order, keeping the j-th only if its section fits bucket j; one task per section."""
    out, seen = [], set()
    for c in pool:
        if len(out) == n:
            break
        k, i = c[0], c[1]
        if (k, i) in seen or len(render(current[k][i])) > CHARS[buckets[len(out)]] * 0.8:
            continue
        seen.add((k, i))
        out.append(c)
    return out


def mask_candidates(text: str) -> list[tuple[str, str]]:
    """(sentence with one value replaced by ___, gold value) for values unique in the sentence."""
    out = []
    for sent in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text):
        vals = value_tokens(sent)
        if len(vals) != 1 or not 60 <= len(sent) <= 400:
            continue
        span, gold = vals[0]
        out.append((sent[: span[0]] + "___" + sent[span[1] :], gold))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    titles = {t["number"]: t["latest_issue_date"] for t in json.loads(get(f"{API}/titles.json"))["titles"]}

    current = {}
    for t, p in PARTS:
        current[(t, p)] = sections(get(f"{API}/full/{titles[t]}/title-{t}.xml?part={p}"))
        print(f"{t} CFR {p}: {len(current[(t, p)])} sections", flush=True)

    tasks = []
    buckets = ["2k", "8k"] * args.per_type

    # extract
    pool = [(k, i) for k, secs in current.items() for i, s in enumerate(secs)
            if len(cite_keys(" ".join(s["paras"])) - {f"sec:{s['n']}"}) >= 3]  # fmt: skip
    print(f"extract pool {len(pool)}", flush=True)
    rng.shuffle(pool)
    pool = fit(pool, current, buckets, args.per_type)
    for j, (k, i) in enumerate(pool):
        sec = current[k][i]
        doc, in_doc = window(current[k], i, CHARS[buckets[j]], rng)
        gold = sorted(cite_keys(" ".join(sec["paras"])) - {f"sec:{sec['n']}"})
        q = (f"List every legal authority cited in the text of § {sec['n']} (statutes, public laws, executive orders, "
             f"and other CFR sections or parts), one per line. Do not list § {sec['n']} itself and do not list "
             "authorities that appear only in other sections.")  # fmt: skip
        tasks.append({"id": f"extract-{j}", "type": "extract", "bucket": buckets[j], "source": f"{k[0]} CFR {sec['n']}",
                      "document": doc, "question": q, "gold": gold, "self": f"sec:{sec['n']}"})  # fmt: skip

    # cite
    pool = [(k, i, m) for k, secs in current.items() for i, s in enumerate(secs) for m in mask_candidates(" ".join(s["paras"]))]
    print(f"cite pool {len(pool)}", flush=True)
    rng.shuffle(pool)
    pool = fit(pool, current, buckets, args.per_type)
    for j, (k, i, (masked, gold)) in enumerate(pool):
        sec = current[k][i]
        doc, in_doc = window(current[k], i, CHARS[buckets[j]], rng)
        q = (f"Using only the regulations above, what value fills the blank in this provision?\n\"{masked}\"\n"
             "Answer in one or two sentences, and cite the section that supports your answer in the form § N.")  # fmt: skip
        tasks.append({"id": f"cite-{j}", "type": "cite", "bucket": buckets[j], "source": f"{k[0]} CFR {sec['n']}",
                      "document": doc, "question": q, "gold": gold, "gold_section": f"sec:{sec['n']}",
                      "sections_in_doc": [f"sec:{n}" for n in in_doc]})  # fmt: skip

    # reconcile: two dated versions of one section that differ in one value
    cands = []
    for t, p in VERSION_PARTS:
        vers = json.loads(get(f"{API}/versions/title-{t}.json?part={p}"))["content_versions"]
        by_sec = {}
        for v in vers:
            if v["type"] == "section" and v["substantive"] and not v["removed"]:
                by_sec.setdefault(v["identifier"], set()).add(v["date"])
        for n, dates in by_sec.items():
            ds = sorted(dates)
            for a, b in zip(ds, ds[1:]):
                cands.append((t, p, n, a, b))
    rng.shuffle(cands)
    print(f"reconcile: {len(cands)} version pairs to scan", flush=True)
    made, per_section = 0, {}
    for t, p, n, d_old, d_new in cands:
        if made >= args.per_type:
            break
        try:
            old = sections(get(f"{API}/full/{d_old}/title-{t}.xml?part={p}&section={n}"))
            new = sections(get(f"{API}/full/{d_new}/title-{t}.xml?part={p}&section={n}"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {t} CFR {n} {d_old}: {e}", flush=True)
            continue
        if not old or not new or len(render(old[0])) + len(render(new[0])) > CHARS["8k"]:
            continue
        if per_section.get(n, 0) >= 3:
            continue
        found = None
        old_set, new_set = set(old[0]["paras"]), set(new[0]["paras"])
        for po in old[0]["paras"]:
            if po in new_set:  # unchanged paragraph
                continue
            for pn in new[0]["paras"]:
                if pn in old_set or difflib.SequenceMatcher(None, po, pn).ratio() < 0.9:
                    continue
                vo, vn = value_tokens(po), value_tokens(pn)
                if len(vo) != len(vn):
                    continue
                diff = [(a, b) for a, b in zip(vo, vn) if a[1] != b[1]]
                if len(diff) == 1:
                    found = (pn, diff[0][1], diff[0][0][1])
                    break
            if found:
                break
        if not found:
            continue
        para_new, (span, val_new), val_old = found
        masked = para_new[: span[0]] + "___" + para_new[span[1] :]
        mid = date.fromisoformat(d_old) + (date.fromisoformat(d_new) - date.fromisoformat(d_old)) / 2
        later = rng.random() < 0.5
        ask = (date.fromisoformat(d_new) + timedelta(days=30)) if later else mid
        docs = [("A", d_old, render(old[0])), ("B", d_new, render(new[0]))]
        rng.shuffle(docs)
        doc = "\n\n".join(f"Guidance {lab}, in effect from {d}:\n{txt}" for lab, (_, d, txt) in zip("AB", docs))
        doc += "\n\nEach version stays in effect until the next one takes effect."
        q = (f"The two versions above conflict. As of {ask.isoformat()}, what value applies in this provision?\n\"{masked}\"\n"
             "State the value and which guidance (A or B) governs on that date.")  # fmt: skip
        gold_label = "AB"[[d for _, d, _ in docs].index(d_new if later else d_old)]
        tasks.append({"id": f"reconcile-{made}", "type": "reconcile", "bucket": "versions", "source": f"{t} CFR {n}",
                      "document": doc, "question": q, "gold": val_new if later else val_old,
                      "wrong": val_old if later else val_new, "gold_label": gold_label, "ask_date": ask.isoformat()})  # fmt: skip
        made += 1
        per_section[n] = per_section.get(n, 0) + 1
        print(f"  reconcile {made}: {t} CFR {n} {d_old} -> {d_new}: {val_old} -> {val_new}", flush=True)

    out = ROOT / "data" / "policy" / "tasks.jsonl"
    out.write_text("".join(json.dumps(x) + "\n" for x in tasks), encoding="utf-8")
    counts = {}
    for x in tasks:
        counts[x["type"]] = counts.get(x["type"], 0) + 1
    print(f"wrote {len(tasks)} tasks: {counts}")


if __name__ == "__main__":
    main()
