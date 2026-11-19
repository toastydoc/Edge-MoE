"""Radial domain maps: one picture per domain of which experts the router sends it to.

    python scripts/export_domain_maps.py

Reads every domain profile in profiles/ and writes viz/domain_maps.html from
viz/domain_maps.template.html. In each picture the router is the centre, the 30
MoE layers are rings going outward, and each ring holds that layer's 128 experts.
An expert's position around its ring is set by which domains use it most
(every domain has a fixed direction), so a domain's experts lean its way.

Domains still waiting to be profiled are shown as pending, so the page can be
regenerated as each profile lands.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge_moe.expert_packs import load_manifest, pack_mask  # noqa: E402

import torch  # noqa: E402

# Order around the circle: neighbours are related, so the layout reads sensibly.
ORDER = [
    ("law", "Law", "MMLU law + US Code"),
    ("regulation", "Regulation", "eCFR rule text"),
    ("reconcile", "Reconcile", "two dated rule versions"),
    ("politics", "Politics", "Wikipedia"),
    ("economics", "Economics", "Wikipedia"),
    ("history", "History", "Wikipedia"),
    ("philosophy", "Philosophy", "Wikipedia"),
    ("art", "Art", "Wikipedia"),
    ("medicine", "Medicine", "MMLU + PubMedQA"),
    ("biology", "Biology", "Wikipedia"),
    ("chemistry", "Chemistry", "Wikipedia"),
    ("geology", "Geology", "Wikipedia"),
    ("physics", "Physics", "Wikipedia"),
    ("math", "Math", "MMLU + GSM8K"),
    ("computer_science", "Computer science", "Wikipedia"),
]
WIKI = {"politics", "economics", "history", "philosophy", "art", "biology", "chemistry", "geology", "physics", "computer_science"}
BUDGET = 51


def profile_path(key: str) -> Path | None:
    p = ROOT / "profiles" / (f"wiki_{key}.npz" if key in WIKI else f"{key}.npz")
    return p if p.exists() else None


def main() -> None:
    base = load_manifest(ROOT / "expert_packs" / "base.json")[1]  # [30, 128] bool
    L, E = base.shape
    ready, share = [], {}
    for key, _, _ in ORDER:
        p = profile_path(key)
        if p is None:
            continue
        z = np.load(p)
        c = z["pick_counts"].astype(np.float64)
        share[key] = c / c.sum(1, keepdims=True)  # each layer's routing share per expert, sums to 1
        ready.append(key)

    # Every domain in ORDER gets a fixed direction; only profiled ones pull experts.
    anchor = {k: 2 * math.pi * i / len(ORDER) - math.pi / 2 for i, (k, _, _) in enumerate(ORDER)}
    S = np.stack([share[k] for k in ready])  # [D, L, E]
    lift = S / S.mean(0, keepdims=True).clip(1e-12)  # >1: the domain uses this expert more than the average domain
    w = np.clip(lift - 1, 0, None) ** 1.5
    ang = np.array([anchor[k] for k in ready])[:, None, None]
    x, y = (w * np.cos(ang)).sum(0), (w * np.sin(ang)).sum(0)
    theta = np.arctan2(y, x)
    # Rank each layer's experts by angle and give them evenly spaced slots, so no two overlap.
    slot = np.zeros((L, E), dtype=int)
    for l in range(L):
        start = anchor[ORDER[0][0]]
        order = np.argsort((theta[l] - start) % (2 * math.pi), kind="stable")
        slot[l, order] = np.arange(E)

    domains = []
    for key, label, source in ORDER:
        entry = {"key": key, "label": label, "source": source, "anchor": anchor[key], "ready": key in ready}
        if key in ready:
            pk = pack_mask(torch.as_tensor(share[key]), base, BUDGET / 128).numpy()
            entry["share"] = np.round(share[key] * 1e4).astype(int).tolist()  # basis points
            entry["pack"] = [np.flatnonzero(pk[l]).tolist() for l in range(L)]
            entry["pack_mass"] = float((share[key] * (pk | base.numpy())).sum(1).mean())
        domains.append(entry)

    # Expert labels from the token pass: what text each expert receives.
    labels = None
    tp = ROOT / "profiles" / "expert_tokens.json"
    if tp.exists():
        t = json.loads(tp.read_text(encoding="utf-8"))
        labels = []
        for layer in t["experts"]:
            n = np.array([x["n"] for x in layer], dtype=float)
            dom = np.array([x["dom"] for x in layer])
            overall = (dom * n[:, None]).sum(0) / max(1, n.sum())  # each domain's share of all tokens
            row = []
            for x, dl in zip(layer, dom):
                lift = dl / np.clip(overall, 1e-9, None)
                c = int(np.argmax(x["cls"]))
                row.append({
                    "t": [k["t"] for k in x["top"][:5]],
                    "c": f"{t['classes'][c]} {round(x['cls'][c] * 100)}%",
                    "d": f"{t['domains'][int(np.argmax(lift))]} {lift.max():.1f}x" if x["n"] else "",
                })  # fmt: skip
            labels.append(row)

    data = {
        "labels": labels,
        "layers": L,
        "experts": E,
        "budget": BUDGET,
        "base": [np.flatnonzero(base[l].numpy()).tolist() for l in range(L)],
        "slot": slot.tolist(),
        "domains": domains,
    }
    tpl = (ROOT / "viz" / "domain_maps.template.html").read_text(encoding="utf-8")
    out = ROOT / "viz" / "domain_maps.html"
    out.write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"{len(ready)}/{len(ORDER)} domains mapped -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
