#!/usr/bin/env python3
"""
Request 2: DREADD vs GFP distributions — matched sample (~75/image) vs all ADS axons.
Answers: does measuring all axons reveal a stronger DREADD vs GFP effect?
4 curves per panel: DREADD matched, GFP matched, DREADD all, GFP all.
Circle mode only.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
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

# matched = darker + solid + thicker; all ADS = lighter + dashed
STYLES = {
    "DREADD": {
        "matched": dict(color="#1a5fa8", lw=2.8, linestyle="-"),
        "all":     dict(color="#7ab3e0", lw=2.0, linestyle="--"),
    },
    "GFP": {
        "matched": dict(color="#b81c1c", lw=2.8, linestyle="-"),
        "all":     dict(color="#f08080", lw=2.0, linestyle="--"),
    },
}


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
        records.append({"x_px": xnm / PIXEL_SIZE_NM, "y_px": ynm / PIXEL_SIZE_NM})
    return pd.DataFrame(records)


def build_data():
    data = {
        "DREADD": {"matched_gr": [], "matched_diam": [], "all_gr": [], "all_diam": []},
        "GFP":    {"matched_gr": [], "matched_diam": [], "all_gr": [], "all_diam": []},
    }
    for img_id in TARGET_IMAGES:
        cond = METADATA[img_id]["condition"]
        ads = load_clean(img_id)
        man = load_manual(img_id)

        tree = KDTree(ads[["x0 (px)", "y0 (px)"]].values)
        dists, idxs = tree.query(man[["x_px", "y_px"]].values, k=1)
        mask = dists < MATCH_THRESH
        if mask.sum() > 0:
            matched = ads.iloc[idxs[mask]]
            data[cond]["matched_gr"].append(matched["gratio"].values)
            data[cond]["matched_diam"].append(matched["axon_diam (um)"].values)

        data[cond]["all_gr"].append(ads["gratio"].values)
        data[cond]["all_diam"].append(ads["axon_diam (um)"].values)

    return {
        cond: {k: np.concatenate(v) for k, v in d.items()}
        for cond, d in data.items()
    }


def kde_panel(ax, data, metric_matched, metric_all, xlabel, title):
    all_vals = np.concatenate([
        data["DREADD"][metric_matched], data["GFP"][metric_matched],
        data["DREADD"][metric_all],     data["GFP"][metric_all],
    ])
    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    xs = np.linspace(lo, hi, 500)

    for cond in ("DREADD", "GFP"):
        matched = data[cond][metric_matched]
        all_ads = data[cond][metric_all]

        kde = gaussian_kde(matched)
        ys = kde(xs)
        s = STYLES[cond]["matched"]
        ax.plot(xs, ys, label=f"{cond} — matched  (n={len(matched):,})", **s)
        ax.fill_between(xs, ys, alpha=0.10, color=s["color"])

        kde = gaussian_kde(all_ads)
        s = STYLES[cond]["all"]
        ax.plot(xs, kde(xs), label=f"{cond} — all ADS  (n={len(all_ads):,})", **s)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    data = build_data()

    for cond in ("DREADD", "GFP"):
        for key in ("matched_gr", "all_gr"):
            print(f"{cond} {key}: n={len(data[cond][key])}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "DREADD vs GFP — matched sample (~75/image) vs all ADS axons\n"
        "Solid = matched axons · Dashed = all ADS · 14 images · circle mode",
        fontsize=12, fontweight="bold"
    )

    kde_panel(ax1, data, "matched_gr",   "all_gr",   "G-ratio",            "G-ratio distribution")
    kde_panel(ax2, data, "matched_diam", "all_diam", "Axon diameter (µm)", "Axon diameter distribution")

    plt.tight_layout()
    out = OUT_DIR / "request2_distributions.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
