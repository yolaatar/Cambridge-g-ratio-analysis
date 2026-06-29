#!/usr/bin/env python3
"""
Request 2 (bar chart): Does measuring all ADS axons reveal a stronger DREADD vs GFP effect
than the matched sample (~75/image)?

Statistical unit = image. Bar height = mean across images. Error bars = SEM.
Circle mode only.
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
GRATIO_MAX    = 1.0

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
    "image_706":  {"condition": "DREADD"},
    "image_963":  {"condition": "DREADD"},
    "image_1812": {"condition": "DREADD"},
    "image_2299": {"condition": "DREADD"},
    "image_2336": {"condition": "GFP"},
    "image_2696": {"condition": "DREADD"},
    "image_4090": {"condition": "GFP"},
    "image_5087": {"condition": "GFP"},
    "image_6107": {"condition": "GFP"},
    "image_6294": {"condition": "GFP"},
    "image_7422": {"condition": "GFP"},
    "image_7559": {"condition": "GFP"},
    "image_8969": {"condition": "GFP"},
    "image_9408": {"condition": "GFP"},
}

COLOR = {"DREADD": "#2b7bb9", "GFP": "#d62728"}


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
            "axon_diam_um": 2 * math.sqrt(aa / math.pi),
            "gratio": math.sqrt(aa / fa),
        })
    return pd.DataFrame(records)


def build_summary():
    rows = []
    for img_id in TARGET_IMAGES:
        cond = METADATA[img_id]["condition"]
        ads = load_clean(img_id)
        man = load_manual(img_id)

        tree = KDTree(ads[["x0 (px)", "y0 (px)"]].values)
        dists, idxs = tree.query(man[["x_px", "y_px"]].values, k=1)
        mask = dists < MATCH_THRESH

        if mask.sum() > 0:
            matched_ads = ads.iloc[idxs[mask]]
            rows.append({
                "image_id": img_id, "condition": cond, "scale": "matched",
                "mean_gratio": matched_ads["gratio"].mean(),
                "mean_diam":   matched_ads["axon_diam (um)"].mean(),
            })

        rows.append({
            "image_id": img_id, "condition": cond, "scale": "all",
            "mean_gratio": ads["gratio"].mean(),
            "mean_diam":   ads["axon_diam (um)"].mean(),
        })

    return pd.DataFrame(rows)


def bar_panel(ax, summary, metric, ylabel, title):
    scales      = ["matched", "all"]
    scale_labels = ["Matched\n(~75/image)", "All ADS\naxons"]
    bar_w = 0.3
    x = np.array([0.0, 1.0])
    rng = np.random.default_rng(42)

    for offset, cond in zip([-bar_w / 2, bar_w / 2], ["DREADD", "GFP"]):
        means, sems, per_image = [], [], []
        for scale in scales:
            vals = summary[(summary["scale"] == scale) &
                           (summary["condition"] == cond)][metric].values
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))
            per_image.append(vals)

        ax.bar(x + offset, means, width=bar_w, color=COLOR[cond],
               alpha=0.80, label=cond, zorder=3)
        ax.errorbar(x + offset, means, yerr=sems, fmt="none",
                    color="black", capsize=4, lw=1.5, zorder=4)

        for xi, vals in zip(x + offset, per_image):
            jitter = rng.uniform(-0.06, 0.06, len(vals))
            ax.scatter(xi + jitter, vals, color=COLOR[cond],
                       s=30, zorder=5, edgecolors="white", linewidths=0.4, alpha=0.9)

    # p-values (Mann-Whitney, image-level)
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
    ax.relim(); ax.autoscale_view()
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    ax.set_ylim(ylo, yhi + span * 0.18)
    ann_y = yhi + span * 0.02

    for xi, scale in zip(x, scales):
        d = summary[(summary["scale"] == scale) & (summary["condition"] == "DREADD")][metric].values
        g = summary[(summary["scale"] == scale) & (summary["condition"] == "GFP")][metric].values
        if len(d) >= 2 and len(g) >= 2:
            _, p = stats.mannwhitneyu(d, g, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(xi, ann_y, f"{sig}  p={p:.2f}",
                    ha="center", va="bottom", fontsize=9, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def main():
    summary = build_summary()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "DREADD vs GFP — matched sample vs all ADS axons\n"
        "Bars = mean across images ± SEM  ·  dots = individual images  ·  circle mode",
        fontsize=11, fontweight="bold"
    )

    bar_panel(ax1, summary, "mean_gratio", "Mean g-ratio (per image)",          "G-ratio")
    bar_panel(ax2, summary, "mean_diam",   "Mean axon diameter µm (per image)", "Axon diameter")

    ax1.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    out = OUT_DIR / "request2_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    print("\nGroup means (image-level):")
    for scale in ("matched", "all"):
        for cond in ("DREADD", "GFP"):
            sub = summary[(summary["scale"] == scale) & (summary["condition"] == cond)]
            gr  = sub["mean_gratio"]
            dm  = sub["mean_diam"]
            print(f"  {scale:8s} {cond:6s}  g-ratio={gr.mean():.3f}±{gr.std():.3f}  "
                  f"diam={dm.mean():.3f}±{dm.std():.3f}  n_images={len(sub)}")


if __name__ == "__main__":
    main()
