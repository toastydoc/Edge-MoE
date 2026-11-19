"""Print-quality figures for the paper (vector PDF + SVG, and 300 dpi PNG).

    python scripts/make_paper_figures.py

Writes to figures/:
    fig1_fingerprints   radial expert maps of the ten Wikipedia domains (pack experts in colour, base plate grey)
    fig1_fingerprints_dark   the same on a dark field, for slides and covers
    fig2_quality        (a) extra loss per domain for base plate / random / own pack / best other pack,
                        (b) break-even: one fixed 51-expert model shared by k domains
    fig3_routing        what the router's choice tracks, per layer: domain vs token type

Sized for a two-column paper: full text width is 7.2 in.
"""

import colorsys
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_domain_maps import build_data  # noqa: E402

OUT = ROOT / "figures"
WIKI = ["art", "history", "philosophy", "politics", "economics", "biology", "chemistry", "geology", "physics", "computer_science"]
LABEL = {"computer_science": "Computer science"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})  # fmt: skip


def hue(anchor: float) -> float:
    return ((anchor + math.pi / 2) / (2 * math.pi)) % 1.0


def dcol(anchor: float, light: float = 0.42, sat: float = 0.78) -> tuple:
    return colorsys.hls_to_rgb(hue(anchor), light, sat)


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf/.svg/.png")


def radial(ax, data: dict, dom: dict, dark: bool) -> None:
    """One domain's map: router at the centre, layers as rings, pack experts in colour, base plate grey."""
    L, E = data["layers"], data["experts"]
    slot, base = np.array(data["slot"]), [set(b) for b in data["base"]]
    a0 = data["domains"][0]["anchor"]
    r0, R = 0.09, 1.0
    rad = lambda l: r0 + (R - r0) * (l + 0.6) / L  # noqa: E731
    ang = lambda l, e: slot[l][e] / E * 2 * math.pi + a0  # noqa: E731
    col = dcol(dom["anchor"], 0.62 if dark else 0.42)
    grey = (0.85, 0.89, 0.95) if dark else (0.55, 0.57, 0.62)

    # faint layer rings and every expert slot
    th = np.linspace(0, 2 * math.pi, 200)
    for l in range(4, L, 5):
        ax.plot(rad(l) * np.cos(th), rad(l) * np.sin(th), lw=0.25, color=grey, alpha=0.18 if dark else 0.25, zorder=0)
    rr = np.repeat([rad(l) for l in range(L)], E)
    tt = np.tile(np.arange(E) / E * 2 * math.pi, L)
    ax.scatter(rr * np.cos(tt), rr * np.sin(tt), s=0.02, color=grey, alpha=0.35 if dark else 0.3, lw=0, zorder=0, rasterized=True)

    nodes = []
    for l in range(L):
        sh = np.array(dom["share"][l], float)
        mx = sh.max()
        pack = set(dom["pack"][l])
        nodes.append([(l, e, sh[e] / mx, e in pack, ang(l, e)) for e in range(E) if e in pack or e in base[l]])
    segs = {True: [], False: []}
    widths = {True: [], False: []}
    alphas = {True: [], False: []}
    for l in range(L):
        for (_, e, v, is_pack, a) in nodes[l]:
            if l == 0:
                p = (0.0, 0.0)
            else:
                def score(n):
                    d = abs(n[4] - a) % (2 * math.pi)
                    return (0.15 + n[2]) * math.exp(-min(d, 2 * math.pi - d) / 0.3)
                pn = max(nodes[l - 1], key=score)
                p = (rad(l - 1) * math.cos(pn[4]), rad(l - 1) * math.sin(pn[4]))
            segs[is_pack].append([p, (rad(l) * math.cos(a), rad(l) * math.sin(a))])
            widths[is_pack].append(0.25 + 1.1 * v if is_pack else 0.12 + 0.3 * v)
            alphas[is_pack].append(0.3 + 0.7 * v if is_pack else 0.12 + 0.25 * v)
    for is_pack, c in ((False, grey), (True, col)):
        rgba = [(*c, a) for a in alphas[is_pack]]
        ax.add_collection(LineCollection(segs[is_pack], linewidths=widths[is_pack], colors=rgba, capstyle="round", zorder=2 if is_pack else 1))
    for l in range(L):
        pk = [n for n in nodes[l] if n[3]]
        ax.scatter([rad(l) * math.cos(n[4]) for n in pk], [rad(l) * math.sin(n[4]) for n in pk],
                   s=[0.15 + 2.2 * n[2] ** 1.5 for n in pk], color=col, lw=0, zorder=3)  # fmt: skip
    ax.scatter([0], [0], s=14, color=("white" if dark else "black"), zorder=4, lw=0)
    ax.add_patch(plt.Circle((0, 0), 1.04, fill=False, lw=0.6, color=(0.18, 0.83, 0.91) if dark else (0.3, 0.32, 0.36)))
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")


def fig1(data: dict, dark: bool) -> None:
    doms = {d["key"]: d for d in data["domains"]}
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.35), facecolor="#05070d" if dark else "white")
    for ax, k in zip(axes.flat, WIKI):
        radial(ax, data, doms[k], dark)
        ax.set_title(LABEL.get(k, k.capitalize()), color="white" if dark else "black", pad=2, fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.12)
    save(fig, "fig1_fingerprints_dark" if dark else "fig1_fingerprints")


def fig2() -> None:
    s = json.loads((ROOT / "results" / "domain_summary.json").read_text(encoding="utf-8"))
    packs, be = s["packs"], s["break_even"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
    order = sorted(WIKI, key=lambda d: packs[d]["own"])
    y = np.arange(len(order))
    series = [("base", "Base plate alone", (0.70, 0.71, 0.74)), ("random", "Base + 25 random", (0.85, 0.55, 0.35)),
              ("best_other_excess", "Best other domain's pack", (0.45, 0.62, 0.85)), ("own", "Own pack", (0.13, 0.42, 0.62))]  # fmt: skip
    h = 0.2
    for i, (k, lab, c) in enumerate(series):
        a.barh(y + (1.5 - i) * h, [packs[d][k] for d in order], height=h, color=c, label=lab, lw=0)
    a.set_yticks(y, [LABEL.get(d, d.capitalize()) for d in order])
    a.set_xlabel("Extra loss vs full model (nats / token)")
    a.set_title("(a) 51 experts per layer, scored on each domain's held-out text", loc="left")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2)
    a.invert_yaxis()

    runs = {int(k): v for k, v in be["runs"].items()}
    ks = sorted(runs)
    b.axhline(be["random"], ls=":", lw=0.9, color=(0.85, 0.55, 0.35), label="Base + 25 random")
    b.axhline(be["own"], ls="--", lw=0.9, color=(0.13, 0.42, 0.62), label="Each domain's own pack")
    for k in ks:
        b.scatter([k] * len(runs[k]), runs[k], s=9, color=(0.55, 0.57, 0.62), lw=0, zorder=2)
    b.plot([1] + ks, [be["own"]] + [float(np.mean(runs[k])) for k in ks], "o-", ms=3.2, lw=1.1, color="black",
           label="One fixed model for k domains", zorder=3)  # fmt: skip
    b.set_xticks([1] + ks)
    b.set_ylim(0, be["random"] * 1.12)
    b.set_xlabel("Domains served by one 51-expert/layer model (k)")
    b.set_ylabel("Extra loss vs full model (nats / token)")
    b.set_title("(b) Break-even: sharing one model across domains", loc="left")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_quality")


def fig3() -> None:
    s = json.loads((ROOT / "results" / "domain_summary.json").read_text(encoding="utf-8"))
    tv = s.get("token_vs_domain")
    if not tv:
        return
    x = np.arange(1, len(tv["domain_frac"]) + 1)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(x, np.array(tv["domain_frac"]) * 100, "-", lw=1.2, color=(0.13, 0.42, 0.62), label="Domain (15 classes)")
    ax.plot(x, np.array(tv["surface_frac"]) * 100, "-", lw=1.2, color=(0.85, 0.55, 0.35), label="Token type (6 classes)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert-choice entropy explained (%)")
    ax.set_xlim(1, x[-1])
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig3_routing")


def main() -> None:
    data = build_data()
    fig1(data, dark=False)
    fig1(data, dark=True)
    fig2()
    fig3()


if __name__ == "__main__":
    main()
