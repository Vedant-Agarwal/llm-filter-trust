"""Figures 1-3 from the result files (values are read, not typed in). Writes figures/fig_{ndcg,loss,safety}.{pdf,png}."""
import json, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.lines import Line2D
BLUE, GRAY, INK, MUTED, GRID = "#2a78d6", "#b9b8b0", "#1a1a19", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
def save(fig, name):
    fig.tight_layout(pad=0.3); fig.savefig(f"figures/{name}.pdf", bbox_inches="tight"); fig.savefig(f"figures/{name}.png", dpi=220, bbox_inches="tight")
# Fig 1: nDCG@10 with 95% CI (LoRA-3B extractor, hybrid ranker)
def depth(path):
    out, d = {}, None
    for l in open(path):
        m = re.match(r"== (\w+) \(base=\w+, n=(\d+)", l)
        if m: d = m.group(1); out[d] = {"n": int(m.group(2))}; continue
        m = re.match(r"\s+(.+?)\s+(-?[\d.]+) \[\s*(-?[\d.]+),\s*(-?[\d.]+)\]\s+diff", l)
        if m and d: out[d][m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return out
D = depth("results/depth_qwen3b_lora.txt")
keys = ["ignore", "hard filter", "hard filter + title fallback (repair)", "safety-gated + title fallback", "oracle subset of hard filters"]
rows = ["Ignore attributes", "Hard filter", "Hard filter\n+ title fallback", "Safety-gated\n+ title fallback", "Best subset\n(oracle)"]
data = {f"{n} (n={D[d]['n']:,})": [D[d][k] for k in keys] for n, d in (("ESCI", "esci"), ("WANDS", "wands"))}
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.15), sharey=True)
for ax, (name, v) in zip(axes, data.items()):
    m = [x[0] for x in v]; lo = [x[0] - x[1] for x in v]; hi = [x[2] - x[0] for x in v]
    ax.barh(range(len(v)), m, xerr=[lo, hi], color=[GRAY, "#d9534f", BLUE, BLUE, GRAY], edgecolor="white", height=0.62, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK))
    for i, x in enumerate(v): ax.text(x[2] + 2, i, f"{x[0]:.1f}", va="center", fontsize=7.2, color=INK)
    ax.set_title(name, fontsize=8, color=INK); ax.set_xlim(0, 108)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(rows)), rows, fontsize=7.2); axes[0].invert_yaxis()
fig.supxlabel("nDCG@10 (95% bootstrap CI)", fontsize=7.5, color=MUTED, y=0.0)
save(fig, "fig_ndcg")
# Fig 2: why relevant products are dropped (LoRA-3B extractor)
def loss(path):
    out, d = {}, None
    for l in open(path):
        m = re.match(r"== (\w+):", l)
        if m: d = m.group(1); out[d] = {}; continue
        m = re.match(r"\s+\[(\w+)\s*\] n=\s*\d+ kept\s+([\d.]+)%\s+empty\s+([\d.]+)%\s+wording\s+([\d.]+)%\s+disagrees\s+([\d.]+)%", l)
        if m: out[d][m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4, 5))
    return out
Ls = loss("results/loss_qwen3b_lora.txt")
L = [("ESCI brand", *Ls["esci"]["brand"]), ("ESCI color", *Ls["esci"]["color"]), ("ESCI size/material/shape", *Ls["esci"]["size"]), ("WANDS material", *Ls["wands"]["material"]),
     ("WANDS color", *Ls["wands"]["color"]), ("WANDS shape", *Ls["wands"]["shape"]), ("WANDS size", *Ls["wands"]["size"]), ("WANDS brand", *Ls["wands"]["brand"])]
fig, ax = plt.subplots(figsize=(3.5, 2.05))
labs = ["kept", "field empty", "different wording", "field disagrees"]; cols = [BLUE, "#e8a33d", "#9b8ec4", "#d9534f"]; left = np.zeros(len(L))
for k in range(4):
    vals = np.array([r[k + 1] for r in L]); ax.barh(range(len(L)), vals, left=left, color=cols[k], edgecolor="white", height=0.7, label=labs[k]); left += vals
ax.set_yticks(range(len(L)), [r[0] for r in L], fontsize=7.2); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel("% of relevant products (per extracted attribute)", fontsize=7.5, color=MUTED)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(ncol=2, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), handlelength=1)
save(fig, "fig_loss")
# Fig 3: safety score vs kept, all 38 cells, unrounded values from rules.py
rows = [dict(ex=c["extractor"], ret=c["retailer"], t=c["type"], cov=c["coverage"], saf=c["safety"], kept=c["kept"]) for c in json.load(open("results/rules.json"))["cells"]]
COL = {"esci": "#2a78d6", "wands": "#eb6834"}; MK = {"qwen72": "o", "qwen3b_lora": "s", "qwen3b_base": "^"}; NAME = {"qwen72": "72B prompted", "qwen3b_lora": "3B + LoRA", "qwen3b_base": "3B prompted"}
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.axvline(0.5, color=MUTED, lw=0.7, ls="--"); ax.axhline(0.5, color=MUTED, lw=0.7, ls=":")
for r in rows: ax.scatter(r["saf"], r["kept"], s=22, marker=MK[r["ex"]], color=COL[r["ret"]], edgecolor="white", linewidth=0.6, zorder=3)
miss = [r for r in rows if (r["saf"] >= 0.5) != (r["kept"] >= 0.5)]
for r in miss: ax.scatter(r["saf"], r["kept"], s=70, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=4)
ax.annotate("WANDS color", xy=(0.50, 0.16), xytext=(0.62, 0.28), fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
ax.annotate(f"no catalog field\n({sum(r['cov'] == 0 for r in rows)} cells)", xy=(0.0, 0.0), xytext=(0.16, 0.22), fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
ax.set_xlim(-0.03, 0.9); ax.set_ylim(-0.03, 0.75)
ax.set_xlabel("label-free safety score", fontsize=8); ax.set_ylabel("relevant products kept\n(from labels)", fontsize=8)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h = [Line2D([], [], ls="", marker="o", color=COL["esci"], label="ESCI"), Line2D([], [], ls="", marker="o", color=COL["wands"], label="WANDS")] + [Line2D([], [], ls="", marker=MK[k], color=MUTED, label=NAME[k]) for k in MK]
ax.legend(handles=h, fontsize=6.5, frameon=False, loc="upper left", ncol=1, handletextpad=0.3, borderaxespad=0.2)
save(fig, "fig_safety")
print(len(rows), "cells;", sum(r["cov"] == 0 for r in rows), "with no field; misses:", [(r["ex"], r["ret"], r["t"]) for r in miss])
