#!/usr/bin/env python3
"""
G-ratio comparison separated by injection type (Gi vs Gq).
Each panel: DREADD vs GFP within one injection group.
Measures: Manual, ADS circle, ADS ellipse.
Mann-Whitney U p-values shown per measure.

Output: gratio_bar_by_injection.png
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


# Collect all matched axon g-ratios pooled per group
pool = {
    (inj, cond, measure): []
    for inj in ["Gi", "Gq"]
    for cond in ["DREADD", "GFP"]
    for measure in ["manual", "circle", "ellipse"]
}

for img_id in TARGET_IMAGES:
    meta = METADATA[img_id]
    man  = load_manual(img_id)
    circ = load_ads_clean(img_id, "circle")
    elli = load_ads_clean(img_id, "ellipse")
    mg_c, ag_c = match_gratio(circ, man)
    mg_e, ag_e = match_gratio(elli, man)
    if len(mg_c) == 0:
        continue
    inj, cond = meta["injection"], meta["condition"]
    pool[(inj, cond, "manual")].extend(mg_c.tolist())
    pool[(inj, cond, "circle")].extend(ag_c.tolist())
    pool[(inj, cond, "ellipse")].extend(ag_e.tolist())

for key, vals in pool.items():
    inj, cond, measure = key
    if vals:
        print(f"  {inj} {cond:6s} {measure:7s}: n={len(vals):4d}  mean={np.mean(vals):.4f}  sd={np.std(vals, ddof=1):.4f}")

# Plot — 2 panels: Gi (left), Gq (right)
INJECTIONS = ["Gi", "Gq"]
CONDITIONS = ["DREADD", "GFP"]
MEASURES   = ["manual", "circle"]
LABELS     = ["Manual", "ADS circle"]
COLORS     = ["white", "#a78fd0"]

BAR_W     = 0.30
GROUP_GAP = 0.85
OFFSETS   = np.array([-BAR_W / 2, BAR_W / 2])

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
fig.suptitle("G-ratio by injection type and condition (all matched axons pooled)\nManual · ADS circle · ADS ellipse",
             fontsize=13, fontweight="bold")

for ax, inj in zip(axes, INJECTIONS):
    cond_x = {c: i * GROUP_GAP for i, c in enumerate(CONDITIONS)}

    for m_idx, (measure, label, color) in enumerate(zip(MEASURES, LABELS, COLORS)):
        for cond in CONDITIONS:
            x_centre = cond_x[cond] + OFFSETS[m_idx]
            vals = np.array(pool[(inj, cond, measure)])
            if len(vals) == 0:
                continue

            mean_v = vals.mean()
            sd_v   = vals.std(ddof=1)

            ax.bar(x_centre, mean_v, width=BAR_W * 0.92,
                   color=color, edgecolor="black", linewidth=1.2, zorder=2,
                   label=label if cond == "DREADD" else "_nolegend_")
            ax.errorbar(x_centre, mean_v, yerr=sd_v, fmt="none",
                        color="black", capsize=5, capthick=1.2, lw=1.2, zorder=4)

    # Stats per measure — Mann-Whitney U on all pooled axons
    y_sig_base = 0.815
    for m_idx, (measure, label) in enumerate(zip(MEASURES, LABELS)):
        a = np.array(pool[(inj, "DREADD", measure)])
        b = np.array(pool[(inj, "GFP",    measure)])
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        p_txt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"

        x1 = cond_x["DREADD"] + OFFSETS[m_idx]
        x2 = cond_x["GFP"]    + OFFSETS[m_idx]
        y  = y_sig_base + m_idx * 0.012
        ax.plot([x1, x1, x2, x2], [y, y + 0.004, y + 0.004, y], lw=1.0, color="black")
        ax.text((x1 + x2) / 2, y + 0.005, p_txt, ha="center", va="bottom", fontsize=8)

    ax.set_xticks([cond_x[c] for c in CONDITIONS])
    ax.set_xticklabels(["DREADD", "GFP"], fontsize=13)
    ax.set_title(f"{inj} group", fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)

    n_d = len(pool[(inj, "DREADD", "manual")])
    n_g = len(pool[(inj, "GFP",    "manual")])
    ax.text(0.01, 0.01, f"DREADD n={n_d}  ·  GFP n={n_g} axons",
            transform=ax.transAxes, fontsize=8.5, color="gray", va="bottom")

axes[0].set_ylabel("G-ratio (all matched axons pooled)", fontsize=13)
axes[0].legend(fontsize=10, framealpha=0.9, loc="lower right")

plt.tight_layout()
out = OUT_DIR / "gratio_bar_by_injection.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out.name}")
