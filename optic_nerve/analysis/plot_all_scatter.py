#!/usr/bin/env python3
"""
Part 1 scatter (ADS vs Manual g-ratio) for all 14 images.
One panel per image, consistent axis limits, same style as part1_ads_vs_manual.png.
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
AXIS_LIM      = (0.35, 1.00)   # consistent across all panels

TARGET_IMAGES = [
    "image_706",  "image_963",  "image_1812", "image_2299",
    "image_2336", "image_2696", "image_4090", "image_5087",
    "image_6107", "image_6294", "image_7422", "image_7559",
    "image_8969", "image_9408",
]

METADATA = {
    "image_706":  {"animal": "TKFG 19.1c", "side": "L", "condition": "DREADD", "injection": "Gi"},
    "image_963":  {"animal": "TKFG 19.1g", "side": "L", "condition": "DREADD", "injection": "Gi"},
    "image_1812": {"animal": "TKFG 19.1g", "side": "L", "condition": "DREADD", "injection": "Gi"},
    "image_2299": {"animal": "TKFG 18.1f", "side": "L", "condition": "DREADD", "injection": "Gq"},
    "image_2336": {"animal": "TKFG 17.1g", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_2696": {"animal": "TKFG 17.1g", "side": "L", "condition": "DREADD", "injection": "Gq"},
    "image_4090": {"animal": "TKFG 18.1f", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_5087": {"animal": "TKFG 19.1f", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_6107": {"animal": "TKFG 19.1c", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_6294": {"animal": "TKFG 19.1g", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_7422": {"animal": "TKFG 17.1g", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_7559": {"animal": "TKFG 19.1e", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_8969": {"animal": "TKFG 21.1g", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_9408": {"animal": "TKFG 18.1f", "side": "R", "condition": "GFP",    "injection": "Gq"},
}

COLOR = {"DREADD": "#2b7bb9", "GFP": "#d62728"}


def load_matched(img_id):
    # ADS
    ads = pd.read_csv(BASE / img_id / "input_Morphometrics.csv", index_col=0)
    border = ads["image_border_touching"].fillna(True).astype(bool)
    ads = ads[~border]

    # Manual
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
            "x_px":  xnm / PIXEL_SIZE_NM,
            "y_px":  ynm / PIXEL_SIZE_NM,
            "gratio": math.sqrt(aa / fa),
        })
    man = pd.DataFrame(records)

    tree = KDTree(ads[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(man[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH

    return (
        man["gratio"].values[mask],
        ads["gratio"].iloc[idxs[mask]].values,
        mask.sum(),
    )


def one_scatter(ax, img_id):
    meta = METADATA[img_id]
    man_gr, ads_gr, n = load_matched(img_id)

    # clip to axis range so outliers don't distort the view
    lo, hi = AXIS_LIM
    in_range = (man_gr >= lo) & (man_gr <= hi) & (ads_gr >= lo) & (ads_gr <= hi)
    n_clipped = n - in_range.sum()

    r, _  = stats.pearsonr(man_gr, ads_gr)
    mae   = np.abs(ads_gr - man_gr).mean()
    pct   = mae / man_gr.mean() * 100

    ax.scatter(
        man_gr[in_range], ads_gr[in_range],
        s=18, alpha=0.65, linewidths=0,
        color=COLOR[meta["condition"]],
    )
    # clipped outliers shown as triangles at the edge
    if n_clipped > 0:
        for mx, ax_v in zip(man_gr[~in_range], ads_gr[~in_range]):
            ax.scatter(
                np.clip(mx, lo + 0.01, hi - 0.01),
                np.clip(ax_v, lo + 0.01, hi - 0.01),
                marker="^", s=20, color="gray", alpha=0.5, linewidths=0,
            )

    ax.plot(AXIS_LIM, AXIS_LIM, "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(AXIS_LIM); ax.set_ylim(AXIS_LIM)
    ax.set_aspect("equal")

    title = (f"{img_id}\n"
             f"{meta['animal']} · {meta['condition']} · {meta['injection']}")
    ax.set_title(title, fontsize=7.5, pad=3)

    label = (f"n={n}  r={r:.2f}\n"
             f"MAE={mae:.3f} ({pct:.1f}%)")
    if n_clipped:
        label += f"\n({n_clipped} outside range)"
    ax.text(0.04, 0.96, label, transform=ax.transAxes,
            va="top", ha="left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, lw=0))

    ax.set_xlabel("Manual g-ratio", fontsize=7)
    ax.set_ylabel("ADS g-ratio", fontsize=7)
    ax.tick_params(labelsize=6)


# ── Combined 2×7 figure ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 7, figsize=(21, 7))
fig.suptitle(
    "ADS vs Manual g-ratio — matched axons — all 14 images\n"
    "Blue = DREADD  ·  Red = GFP  ·  Triangles = outliers clipped to axis range",
    fontsize=11, fontweight="bold", y=1.01,
)

for ax, img_id in zip(axes.flat, TARGET_IMAGES):
    one_scatter(ax, img_id)

plt.tight_layout()
out = OUT_DIR / "part1_all_images.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Individual files (one per image) ─────────────────────────────────────────
for img_id in TARGET_IMAGES:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    one_scatter(ax, img_id)
    fig.tight_layout()
    out_i = OUT_DIR / f"scatter_{img_id}.png"
    fig.savefig(out_i, dpi=160)
    plt.close(fig)
    print(f"Saved → {out_i.name}")

print("Done.")
