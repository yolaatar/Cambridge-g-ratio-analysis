#!/usr/bin/env python3
"""
Request 1: ADS vs Manual g-ratio scatter plots.
3 selected images (best / typical / worst) + 1 pooled scatter of all 14 images.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
from scipy import stats
from pathlib import Path

BASE          = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve")
OUT_DIR       = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve/analysis")
PIXEL_SIZE_NM = 4.93
MATCH_THRESH  = 75
DIAM_MIN_UM   = 0.30
GRATIO_MAX    = 1.0
AXIS_LIM      = (0.40, 1.00)

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

# 3 images picked to span the accuracy range: best, typical, worst
SELECTED = ["image_706", "image_2336", "image_2299"]


def load_clean(img_id):
    df = pd.read_csv(BASE / img_id / "input_Morphometrics.csv", index_col=0)
    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    if img_id in SPATIAL_CROPS:
        crop = SPATIAL_CROPS[img_id]
        mask = pd.Series(True, index=df.index)
        if "x_max" in crop: mask &= df["x0 (px)"] < crop["x_max"]
        if "y_min" in crop: mask &= df["y0 (px)"] > crop["y_min"]
        df = df[~mask].copy()
    df = df[(df["axon_diam (um)"] >= DIAM_MIN_UM) &
            (df["gratio"] > 0) & (df["gratio"] <= GRATIO_MAX)].copy()
    return df


def load_manual(img_id):
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
            "x_px": xnm / PIXEL_SIZE_NM,
            "y_px": ynm / PIXEL_SIZE_NM,
            "gratio": math.sqrt(aa / fa),
        })
    return pd.DataFrame(records)


def get_matched_pairs(img_id):
    ads = load_clean(img_id)
    man = load_manual(img_id)
    tree = KDTree(ads[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(man[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH
    man_gr = man["gratio"].values[mask]
    ads_gr = ads["gratio"].iloc[idxs[mask]].values
    return man_gr, ads_gr


def scatter_panel(ax, man_gr, ads_gr, title, color):
    lo, hi = AXIS_LIM
    in_range = (man_gr >= lo) & (man_gr <= hi) & (ads_gr >= lo) & (ads_gr <= hi)
    n_clip = (~in_range).sum()

    ax.scatter(man_gr[in_range], ads_gr[in_range],
               s=22, alpha=0.70, color=color, linewidths=0)
    if n_clip:
        ax.scatter(np.clip(man_gr[~in_range], lo+.01, hi-.01),
                   np.clip(ads_gr[~in_range], lo+.01, hi-.01),
                   marker="^", s=22, color="gray", alpha=0.5, linewidths=0)

    ax.plot(AXIS_LIM, AXIS_LIM, "k--", lw=0.9, alpha=0.35)
    ax.set_xlim(AXIS_LIM); ax.set_ylim(AXIS_LIM); ax.set_aspect("equal")
    ax.set_xlabel("Manual g-ratio", fontsize=10)
    ax.set_ylabel("ADS g-ratio", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.spines[["top", "right"]].set_visible(False)

    n = len(man_gr)
    r, _ = stats.pearsonr(man_gr, ads_gr)
    mae = np.abs(ads_gr - man_gr).mean()
    pct = mae / man_gr.mean() * 100
    label = f"n = {n}\nr = {r:.2f}\nMAE = {pct:.1f}%"
    ax.text(0.04, 0.96, label, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, lw=0.5))


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    fig.suptitle(
        "ADS (circle) vs Manual g-ratio — matched axons only\n"
        "Dashed line = perfect agreement",
        fontsize=12, fontweight="bold", y=1.03
    )

    colors = {"DREADD": "#2b7bb9", "GFP": "#d62728"}

    # 3 selected individual images
    for ax, img_id in zip(axes[:3], SELECTED):
        man_gr, ads_gr = get_matched_pairs(img_id)
        meta = METADATA[img_id]
        title = f"{img_id.replace('image_', 'Image ')}\n{meta['animal']} · {meta['condition']}"
        scatter_panel(ax, man_gr, ads_gr, title, colors[meta["condition"]])

    # pooled: all 14 images
    all_man, all_ads = [], []
    for img_id in TARGET_IMAGES:
        man_gr, ads_gr = get_matched_pairs(img_id)
        all_man.append(man_gr)
        all_ads.append(ads_gr)
    all_man = np.concatenate(all_man)
    all_ads = np.concatenate(all_ads)

    scatter_panel(axes[3], all_man, all_ads,
                  "All 14 images pooled\n(all matched axons)", color="#555555")

    plt.tight_layout()
    out = OUT_DIR / "request1_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
