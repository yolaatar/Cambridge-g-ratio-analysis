#!/usr/bin/env python3
"""
Per-image dot plot comparing DREADD vs GFP effect across methods and scales.

Layout: 2 rows (g-ratio, axon diameter) × 2 cols (matched ~75/image, all ADS axons)
Each panel: Manual / Circle / Ellipse side by side, DREADD vs GFP dots per image.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

COLOR = {"DREADD": "#2b7bb9", "GFP": "#d62728"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_clean(img_id, mode="circle"):
    fname = "input_Morphometrics.csv" if mode == "circle" else "input_Morphometrics_ellipse.csv"
    df = pd.read_csv(BASE / img_id / fname, index_col=0)
    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    if img_id in SPATIAL_CROPS:
        crop = SPATIAL_CROPS[img_id]
        mask = pd.Series(True, index=df.index)
        if "x_max" in crop: mask &= df["x0 (px)"] < crop["x_max"]
        if "y_min" in crop: mask &= df["y0 (px)"] > crop["y_min"]
        df = df[~mask].copy()
    df = df[df["axon_diam (um)"] >= DIAM_MIN_UM].copy()
    df = df[(df["gratio"] > 0) & (df["gratio"] <= GRATIO_MAX)].copy()
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
            "axon_diam_um": 2 * math.sqrt(aa / math.pi),
        })
    return pd.DataFrame(records)


def match_mask(ads_df, manual_df):
    tree = KDTree(ads_df[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(manual_df[["x_px", "y_px"]].values, k=1)
    return dists < MATCH_THRESH, idxs


# ── Build per-image summary ───────────────────────────────────────────────────

def build_summary():
    rows = []
    for img_id in TARGET_IMAGES:
        cond = METADATA[img_id]["condition"]
        man_df = load_manual(img_id)

        for mode in ("circle", "ellipse"):
            ads_df = load_clean(img_id, mode)
            mask, idxs = match_mask(ads_df, man_df)

            # matched: ADS values for the ~75 manually measured axons
            if mask.sum() > 0:
                matched_ads = ads_df.iloc[idxs[mask]]
                rows.append({
                    "image_id": img_id, "condition": cond, "mode": mode,
                    "scale": "matched",
                    "mean_gratio": matched_ads["gratio"].mean(),
                    "mean_diam":   matched_ads["axon_diam (um)"].mean(),
                    "n": mask.sum(),
                })

            # all: every ADS axon in the image
            rows.append({
                "image_id": img_id, "condition": cond, "mode": mode,
                "scale": "all",
                "mean_gratio": ads_df["gratio"].mean(),
                "mean_diam":   ads_df["axon_diam (um)"].mean(),
                "n": len(ads_df),
            })

        # manual: ground-truth values for the ~75 annotated axons
        rows.append({
            "image_id": img_id, "condition": cond, "mode": "manual",
            "scale": "matched",
            "mean_gratio": man_df["gratio"].mean(),
            "mean_diam":   man_df["axon_diam_um"].mean(),
            "n": len(man_df),
        })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def dot_group(ax, vals_d, vals_g, x_center, width=0.3, rng=None):
    """Draw DREADD (left) and GFP (right) dot clusters at x_center.
    Returns (y_top, p) so the caller can place annotations after all groups are drawn."""
    if rng is None:
        rng = np.random.default_rng(42)
    for xi_off, vals, cond in [(-width/2, vals_d, "DREADD"), (+width/2, vals_g, "GFP")]:
        x = x_center + xi_off
        jitter = rng.uniform(-0.05, 0.05, len(vals))
        ax.scatter(x + jitter, vals, color=COLOR[cond], s=55, zorder=4,
                   edgecolors="white", linewidths=0.4)
        m = np.mean(vals)
        ax.plot([x - 0.1, x + 0.1], [m, m], color=COLOR[cond], lw=2.5, zorder=5)

    y_top = max(max(vals_d), max(vals_g))
    if len(vals_d) >= 2 and len(vals_g) >= 2:
        _, p = stats.mannwhitneyu(vals_d, vals_g, alternative="two-sided")
    else:
        p = float("nan")
    return y_top, p


def panel(ax, summary, metric_col, scale, title):
    sub = summary[summary["scale"] == scale]

    methods = ["manual", "circle", "ellipse"] if scale == "matched" else ["circle", "ellipse"]
    labels  = ["Manual", "Circle", "Ellipse"] if scale == "matched" else ["Circle", "Ellipse"]
    x_positions = np.arange(len(methods))
    rng = np.random.default_rng(42)

    results = []
    for xi, mode in enumerate(methods):
        grp = sub[sub["mode"] == mode]
        d_vals = grp.loc[grp["condition"] == "DREADD", metric_col].values
        g_vals = grp.loc[grp["condition"] == "GFP",    metric_col].values
        y_top, p = dot_group(ax, d_vals, g_vals, xi, rng=rng)
        results.append((xi, y_top, p))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

    # annotate after axes limits are finalised
    ax.relim(); ax.autoscale_view()
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    ann_y = yhi + span * 0.01   # annotation baseline: just above data
    ax.set_ylim(ylo, yhi + span * 0.14)

    for xi, y_top, p in results:
        if not np.isnan(p):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(xi, ann_y, f"{sig}  p={p:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#555555")


def main():
    summary = build_summary()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        "DREADD vs GFP — per-image mean  ·  Optic Nerve  ·  14 images\n"
        "Each dot = one image. Horizontal bar = group mean.",
        fontsize=12, fontweight="bold", y=1.02
    )

    metrics = [
        ("mean_gratio", "Mean g-ratio (per image)"),
        ("mean_diam",   "Mean axon diameter µm (per image)"),
    ]
    col_configs = [
        ("matched", "Matched axons only (~75/image)\nManual · Circle · Ellipse"),
        ("all",     "All ADS axons (full image)\nCircle · Ellipse"),
    ]

    for row, (metric_col, ylabel) in enumerate(metrics):
        for col, (scale, col_title) in enumerate(col_configs):
            ax = axes[row][col]
            panel(ax, summary, metric_col, scale, col_title)
            ax.set_ylabel(ylabel, fontsize=10)

    # shared legend
    handles = [
        mpatches.Patch(color=COLOR["DREADD"], label="DREADD"),
        mpatches.Patch(color=COLOR["GFP"],    label="GFP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = OUT_DIR / "matched_vs_all_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # quick summary table
    for scale in ("matched", "all"):
        print(f"\n{'─'*60}\n{scale.upper()} — per-image means by condition")
        for mode in (["manual", "circle", "ellipse"] if scale == "matched" else ["circle", "ellipse"]):
            sub = summary[(summary["scale"] == scale) & (summary["mode"] == mode)]
            for cond in ("DREADD", "GFP"):
                v = sub[sub["condition"] == cond]["mean_gratio"].values
                print(f"  {mode:8s} {cond:6s}  g-ratio mean={v.mean():.3f} ± {v.std():.3f}  n_images={len(v)}")


if __name__ == "__main__":
    main()
