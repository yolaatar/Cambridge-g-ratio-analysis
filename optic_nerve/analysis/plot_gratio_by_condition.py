#!/usr/bin/env python3
"""
Grouped bar chart: average g-ratio by condition (DREADD vs GFP)
for Manual, ADS circle, and ADS ellipse.
Each dot = one image mean. Error bars = SD across images in that group.
Mann-Whitney U p-value shown for each measurement type.

Output: gratio_bar_by_condition.png
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy import stats
from pathlib import Path

BASE          = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve")
OUT_DIR       = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve/analysis")
PIXEL_SIZE_NM = 4.93
MATCH_THRESH  = 75
DIAM_MIN_UM   = 0.30

SPATIAL_CROPS = {
    "image_2299": dict(x_max=1500, y_min=3500),
    "image_7559": dict(y_min=4300),
}

TARGET_IMAGES = [
    "image_706",  "image_963",  "image_1812", "image_2299",
    "image_2336", "image_2696", "image_4090", "image_5087",
    "image_6107", "image_6294", "image_7422", "image_7559",
    "image_8969", "image_9408",
]

METADATA = {
    "image_706":  {"animal": "TKFG 19.1c", "condition": "DREADD", "injection": "Gi"},
    "image_963":  {"animal": "TKFG 19.1g", "condition": "DREADD", "injection": "Gi"},
    "image_1812": {"animal": "TKFG 19.1g", "condition": "DREADD", "injection": "Gi"},
    "image_2299": {"animal": "TKFG 18.1f", "condition": "DREADD", "injection": "Gq"},
    "image_2336": {"animal": "TKFG 17.1g", "condition": "GFP",    "injection": "Gq"},
    "image_2696": {"animal": "TKFG 17.1g", "condition": "DREADD", "injection": "Gq"},
    "image_4090": {"animal": "TKFG 18.1f", "condition": "GFP",    "injection": "Gq"},
    "image_5087": {"animal": "TKFG 19.1f", "condition": "GFP",    "injection": "Gi"},
    "image_6107": {"animal": "TKFG 19.1c", "condition": "GFP",    "injection": "Gi"},
    "image_6294": {"animal": "TKFG 19.1g", "condition": "GFP",    "injection": "Gi"},
    "image_7422": {"animal": "TKFG 17.1g", "condition": "GFP",    "injection": "Gq"},
    "image_7559": {"animal": "TKFG 19.1e", "condition": "GFP",    "injection": "Gi"},
    "image_8969": {"animal": "TKFG 21.1g", "condition": "GFP",    "injection": "Gq"},
    "image_9408": {"animal": "TKFG 18.1f", "condition": "GFP",    "injection": "Gq"},
}


# ── Loaders (identical to compare_circle_ellipse.py) ─────────────────────────

def load_ads_clean(img_id, mode="circle"):
    fname = "input_Morphometrics.csv" if mode == "circle" else "input_Morphometrics_ellipse.csv"
    df = pd.read_csv(BASE / img_id / fname, index_col=0)
    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    if img_id in SPATIAL_CROPS:
        crop = SPATIAL_CROPS[img_id]
        mask = pd.Series(False, index=df.index)
        if "x_max" in crop: mask |= df["x0 (px)"] < crop["x_max"]
        if "y_min" in crop: mask |= df["y0 (px)"] > crop["y_min"]
        df = df[~mask].copy()
    df = df[df["axon_diam (um)"] >= DIAM_MIN_UM].copy()
    return df


def load_manual(img_id):
    num = img_id.replace("image_", "")
    raw = pd.read_csv(BASE / f"Results_{num}.csv", index_col=0).to_dict("records")
    records = []
    for i in range(0, len(raw) - 1, 2):
        o, inn = raw[i], raw[i + 1]
        if float(inn["Area"]) > float(o["Area"]): o, inn = inn, o
        fa = float(o["Area"]) / 1e6
        aa = float(inn["Area"]) / 1e6
        xnm = (float(o["X"]) + float(inn["X"])) / 2
        ynm = (float(o["Y"]) + float(inn["Y"])) / 2
        records.append({
            "x_px": xnm / PIXEL_SIZE_NM, "y_px": ynm / PIXEL_SIZE_NM,
            "gratio": math.sqrt(aa / fa),
        })
    return pd.DataFrame(records)


def match_gratio(ads_df, man_df):
    tree = KDTree(ads_df[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(man_df[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH
    return man_df["gratio"].values[mask], ads_df["gratio"].iloc[idxs[mask]].values


# ── Collect per-image means ───────────────────────────────────────────────────

rows = []
for img_id in TARGET_IMAGES:
    meta = METADATA[img_id]
    man  = load_manual(img_id)
    circ = load_ads_clean(img_id, "circle")
    elli = load_ads_clean(img_id, "ellipse")
    mg_c, ag_c = match_gratio(circ, man)
    mg_e, ag_e = match_gratio(elli, man)
    if len(mg_c) == 0:
        continue
    rows.append({
        "image":     img_id,
        "condition": meta["condition"],
        "animal":    meta["animal"],
        "manual":    mg_c.mean(),       # same matched axons for circle
        "circle":    ag_c.mean(),
        "ellipse":   ag_e.mean(),
    })

df = pd.DataFrame(rows)
print(df[["image", "condition", "manual", "circle", "ellipse"]].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────

CONDITIONS = ["DREADD", "GFP"]
MEASURES   = ["manual", "circle", "ellipse"]
LABELS     = ["Manual", "ADS circle", "ADS ellipse"]
COLORS     = ["white", "#a78fd0", "#6dbf8a"]
EDGE_COLOR = "black"

BAR_W   = 0.22
GROUP_GAP = 0.9          # distance between DREADD and GFP group centres
OFFSETS = np.array([-BAR_W, 0, BAR_W])   # 3 bars within each group

rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(9, 7))

cond_x = {c: i * GROUP_GAP for i, c in enumerate(CONDITIONS)}

for m_idx, (measure, label, color) in enumerate(zip(MEASURES, LABELS, COLORS)):
    for cond in CONDITIONS:
        x_centre = cond_x[cond] + OFFSETS[m_idx]
        vals = df.loc[df["condition"] == cond, measure].values

        mean_v = vals.mean()
        sd_v   = vals.std(ddof=1)

        ax.bar(x_centre, mean_v, width=BAR_W * 0.92,
               color=color, edgecolor=EDGE_COLOR, linewidth=1.2, zorder=2,
               label=label if cond == "DREADD" else "_nolegend_")
        ax.errorbar(x_centre, mean_v, yerr=sd_v, fmt="none",
                    color=EDGE_COLOR, capsize=5, capthick=1.2, lw=1.2, zorder=4)

        jitter = rng.uniform(-BAR_W * 0.25, BAR_W * 0.25, len(vals))
        ax.scatter(x_centre + jitter, vals,
                   s=45, color="black", zorder=5, linewidths=0)

# Mann-Whitney U between DREADD and GFP for each measure
y_sig_base = df[MEASURES].max().max() + 0.015
for m_idx, (measure, label) in enumerate(zip(MEASURES, LABELS)):
    a = df.loc[df["condition"] == "DREADD", measure].values
    b = df.loc[df["condition"] == "GFP",    measure].values
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    p_txt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"

    x1 = cond_x["DREADD"] + OFFSETS[m_idx]
    x2 = cond_x["GFP"]    + OFFSETS[m_idx]
    y  = y_sig_base + m_idx * 0.012
    ax.plot([x1, x1, x2, x2], [y, y + 0.004, y + 0.004, y],
            lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y + 0.005, p_txt,
            ha="center", va="bottom", fontsize=8)

ax.set_xticks([cond_x[c] for c in CONDITIONS])
ax.set_xticklabels(["DREADD", "GFP"], fontsize=13)
ax.set_ylabel("Average g-ratio (per image)", fontsize=13)
ax.set_title("Average g-ratio by condition\nManual  ·  ADS circle  ·  ADS ellipse",
             fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)

y_max = y_sig_base + len(MEASURES) * 0.012 + 0.02
ax.set_ylim(df[MEASURES].min().min() - 0.02, y_max)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))

n_d = (df["condition"] == "DREADD").sum()
n_g = (df["condition"] == "GFP").sum()
ax.text(0.01, 0.01, f"DREADD n={n_d} images  ·  GFP n={n_g} images",
        transform=ax.transAxes, fontsize=8.5, color="gray", va="bottom")

plt.tight_layout()
out = OUT_DIR / "gratio_bar_by_condition.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out.name}")
