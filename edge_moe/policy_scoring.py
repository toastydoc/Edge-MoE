"""Mechanical answer checking for the eCFR policy tasks.

cite_keys(text)    -> set of normalised authorities: sec:200.318, part:2:200,
                      usc:31:7501, pl:104-191, eo:12549
value_tokens(text) -> [((start, end), key)] for dollar amounts, percentages
                      and durations, e.g. "$250,000" -> "$250000",
                      "10 percent" -> "10%", "30 calendar days" -> "30 day"
score(task, output) -> dict of metrics for that task type
"""

from __future__ import annotations

import re

_SEC_LIST = re.compile(r"§§?\s*((?:\d+\.\d+[a-z]?(?:\([a-z0-9]+\))*(?:\s*(?:,|and|or|through|to)\s*)?)+)")
_SEC_NUM = re.compile(r"\d+\.\d+[a-z]?")
_CFR = re.compile(r"(\d+)\s*(?:CFR|C\.F\.R\.)\s*(?:(parts?|subpart)\s*)?(\d+(?:\.\d+[a-z]?)?)", re.I)
_USC = re.compile(r"(\d+)\s*U\.\s?S\.\s?C\.?\s*(?:§+\s*)?(\d+[a-z0-9\-]*)", re.I)
_PL = re.compile(r"Pub(?:lic)?\.?\s*L(?:aw|\.)?\s*(?:No\.?\s*)?(\d+)\s*[–-]\s*(\d+)", re.I)
_EO = re.compile(r"(?:E\.\s?O\.|Executive Order)\s*(?:No\.?\s*)?(\d{4,5})", re.I)


def cite_keys(text: str) -> set[str]:
    keys = set()
    for m in _SEC_LIST.finditer(text):
        keys.update(f"sec:{n}" for n in _SEC_NUM.findall(m.group(1)))
    for m in _CFR.finditer(text):
        t, kind, n = m.groups()
        keys.add(f"sec:{n}" if "." in n else f"part:{t}:{n}")
    for m in _USC.finditer(text):
        keys.add(f"usc:{m.group(1)}:{m.group(2).rstrip('-').lower()}")
    for m in _PL.finditer(text):
        keys.add(f"pl:{m.group(1)}-{m.group(2)}")
    for m in _EO.finditer(text):
        keys.add(f"eo:{m.group(1)}")
    return keys


_VALUE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s(?:million|billion))?"
    r"|\b\d+(?:\.\d+)?\s?(?:percent\b|%)"
    r"|\b\d+(?:\.\d+)?\s(?:calendar\s|business\s|work\s|working\s)?(?:days?|months?|years?|hours?|weeks?)\b",
    re.I,
)


def _norm(tok: str) -> str:
    t = tok.lower().replace(",", "").replace(" ", "")
    if t.startswith("$"):
        return t
    if "percent" in t or "%" in t:
        return re.sub(r"percent|%", "", t) + "%"
    m = re.match(r"([\d.]+)(?:calendar|business|work|working)?(day|month|year|hour|week)", t)
    return f"{m.group(1)} {m.group(2)}" if m else t


_BARE = re.compile(r"(?<![\d.$])(\d{1,4}\.\d{1,4}[a-z]?)(?![\d.]*\s*(?:percent|%))")


def answer_cite_keys(output: str) -> set[str]:
    """cite_keys plus bare section numbers ("200.512") as a model may write them without §."""
    return cite_keys(output) | {f"sec:{n}" for n in _BARE.findall(output)}


def value_tokens(text: str) -> list[tuple[tuple[int, int], str]]:
    return [((m.start(), m.end()), _norm(m.group(0))) for m in _VALUE.finditer(text)]


def _number(key: str) -> str:
    return re.sub(r"[^\d.]", "", key.split(" ")[0])


def _first_value(output: str, candidates: list[str]) -> str | None:
    """The first of `candidates` the output states (as a value token or a bare number)."""
    best = None
    for c in candidates:
        hits = [s for (s, _), k in value_tokens(output) if k == c]
        hits += [m.start() for m in re.finditer(rf"(?<![\d.]){re.escape(_number(c))}(?![\d])", output.replace(",", ""))]
        if hits and (best is None or min(hits) < best[0]):
            best = (min(hits), c)
    return best and best[1]


def score(task: dict, output: str) -> dict:
    if task["type"] == "extract":
        gold = set(task["gold"])
        pred = answer_cite_keys(output) - {task["self"]}
        tp = len(gold & pred)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold) if gold else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "pred": sorted(pred)}
    if task["type"] == "cite":
        cited = {k for k in cite_keys(output) if k.startswith("sec:")}
        return {"value_correct": float(_first_value(output, [task["gold"]]) == task["gold"]),
                "gold_cited": float(task["gold_section"] in cited),
                "cites_outside_excerpt": float(bool(cited - set(task["sections_in_doc"]))),
                "cited": sorted(cited)}  # fmt: skip
    if task["type"] == "reconcile":
        first = _first_value(output, [task["gold"], task["wrong"]])
        lab = re.search(r"\bGuidance\s+([AB])\b|\b([AB])\b", output)
        label = lab and (lab.group(1) or lab.group(2))
        return {"value_correct": float(first == task["gold"]), "label_correct": float(label == task["gold_label"])}
    raise ValueError(task["type"])
