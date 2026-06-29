#!/usr/bin/env python3
"""
Bar chart: average g-ratio Manual vs ADS, one dot per image.
Replicates the style of the reference figure.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path

BASE          = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve")
OUT_DIR       = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve/analysis")
PIXEL_SIZE_NM = 4.93
MATCH_THRESH  = 75

TARGET_IMAGES = [
    "image_706",  "image_963",  "image_1812", "image_2299",
    "image_2336", "image_2696", "image_4090", "image_5087",
    "image_6107", "image_6294", "image_7422", "image_7559",
    "image_8969", "image_9408",
]


def load_matched_gratio(img_id):
    ads = pd.read_csv(BASE / img_id / "input_Morphometrics.csv", index_col=0)
    border = ads["image_border_touching"].fillna(True).astype(bool)
    ads = ads[~border]

    num = img_id.replace("image_", "")
    raw = pd.read_csv(BASE / f"Results_{num}.csv", index_col=0).to_dict("records")
    records = []
    for i in range(0, len(raw) - 1, 2):
        o, inn = raw[i], raw[i + 1]
        if float(inn["Area"]) > float(o["Area"]):
            o, inn = inn, o
        fa = float(o["Area"]) / 1e6
        aa = float(inn["Area"]) / 1e6
        xnm = (float(o["X"]) + float(inn["X"])) / 2
        ynm = (float(o["Y"]) + float(inn["Y"])) / 2
        records.append({
            "x_px":   xnm / PIXEL_SIZE_NM,
            "y_px":   ynm / PIXEL_SIZE_NM,
            "gratio": math.sqrt(aa / fa),
        })
    man = pd.DataFrame(records)

    tree = KDTree(ads[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(man[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH

    man_gr = man["gratio"].values[mask]
    ads_gr = ads["gratio"].iloc[idxs[mask]].values

    return man_gr.mean(), ads_gr.mean()


# ── Collect per-image means ───────────────────────────────────────────────────
manual_means = []
ads_means    = []

for img_id in TARGET_IMAGES:
    m, a = load_matched_gratio(img_id)
    manual_means.append(m)
    ads_means.append(a)
    print(f"{img_id}: manual={m:.3f}  ADS={a:.3f}")

manual_means = np.array(manual_means)
ads_means    = np.array(ads_means)

# ── Summary stats ─────────────────────────────────────────────────────────────
def stats(arr):
    return arr.mean(), arr.std(), np.median(arr)

m_mean, m_sd, m_med = stats(manual_means)
a_mean, a_sd, a_med = stats(ads_means)

print(f"\nManual — mean={m_mean:.3f}  SD={m_sd:.3f}  median={m_med:.3f}")
print(f"ADS    — mean={a_mean:.3f}  SD={a_sd:.3f}  median={a_med:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 7))

BAR_WIDTH = 0.55
X = [0, 1]
YLIM = (0.600, None)

COLOR_MANUAL = "white"
COLOR_ADS    = "#a78fd0"   # same purple as reference
EDGE_COLOR   = "black"
DOT_COLOR    = "black"
DOT_SIZE     = 50
JITTER       = 0.07

rng = np.random.default_rng(42)

for pos, means, mean_v, sd_v, med_v, facecolor, label in [
    (0, manual_means, m_mean, m_sd, m_med, COLOR_MANUAL, "Manual"),
    (1, ads_means,    a_mean, a_sd, a_med, COLOR_ADS,    "ADS"),
]:
    # Bar
    ax.bar(pos, mean_v, width=BAR_WIDTH,
           color=facecolor, edgecolor=EDGE_COLOR, linewidth=1.5,
           bottom=0, zorder=2)

    # Error bar (SD)
    ax.errorbar(pos, mean_v, yerr=sd_v,
                fmt="none", color=EDGE_COLOR, capsize=6, capthick=1.5,
                lw=1.5, zorder=4)

    # Jittered dots
    jitter = rng.uniform(-JITTER, JITTER, size=len(means))
    ax.scatter(pos + jitter, means,
               s=DOT_SIZE, color=DOT_COLOR, zorder=5, linewidths=0)

    # Stats box at the bottom of the bar
    stats_text = (f"mean    {mean_v:.3f}\n"
                  f"SD       {sd_v:.3f}\n"
                  f"median {med_v:.3f}")
    ax.text(pos, 0.603, stats_text,
            ha="center", va="bottom", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="lightgray", alpha=0.9, lw=0.8),
            zorder=6)

ax.set_xticks(X)
ax.set_xticklabels(["Manual", "Auto\n(ADS)"], fontsize=13)
ax.set_ylabel("average g-ratio", fontsize=13)
ax.set_title("Average g-ratio", fontsize=15, fontweight="bold", pad=12)
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0.600, max(ads_means.max(), manual_means.max()) + 0.06)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
ax.spines[["top", "right"]].set_visible(False)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_linewidth(0.8)
ax.tick_params(axis="both", labelsize=11)

plt.tight_layout()
out = OUT_DIR / "average_gratio_bar.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")
