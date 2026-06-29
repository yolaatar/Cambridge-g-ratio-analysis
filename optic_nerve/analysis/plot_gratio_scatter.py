#!/usr/bin/env python3
"""
G-ratio vs axon diameter scatter plots (all ADS axons, no manual matching).
Replicates Sebastian's analysis using ADS circle measurements.

Layout (2x2):
  Top-left:    average g-ratio per group (bar + dots)
  Bottom-left: average axon diameter per group (bar + dots)
  Top-right:   g-ratio vs diameter — Gq group
  Bottom-right: g-ratio vs diameter — Gi group

Output: gratio_scatter_by_injection.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE          = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve")
OUT_DIR       = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve/analysis")
PIXEL_SIZE_NM = 4.93
DIAM_MIN_UM   = 0.30

SPATIAL_CROPS = {
    "image_2299": dict(x_max=1500, y_min=3500),
    "image_7559": dict(y_min=4300),
}

TARGET_IMAGES = [
    "image_706",  "image_963",  "image_1812", "image_2299",
    "image_2336", "image_2696", "image_4090", "image_5087",
    "image_6107", "image_6294", "image_7422", "image_7559",
    "image_8969", "image_9408", "image_5767", "image_8002",
    "image_9867",
]

METADATA = {
    "image_706":  {"condition": "DREADD", "injection": "Gi"},
    "image_963":  {"condition": "DREADD", "injection": "Gi"},
    "image_1812": {"condition": "DREADD", "injection": "Gi"},
    "image_2299": {"condition": "DREADD", "injection": "Gq"},
    "image_2336": {"condition": "GFP",    "injection": "Gq"},
    "image_2696": {"condition": "DREADD", "injection": "Gq"},
    "image_4090": {"condition": "GFP",    "injection": "Gq"},
    "image_5087": {"condition": "GFP",    "injection": "Gi"},
    "image_6107": {"condition": "GFP",    "injection": "Gi"},
    "image_6294": {"condition": "GFP",    "injection": "Gi"},
    "image_7422": {"condition": "GFP",    "injection": "Gq"},
    "image_7559": {"condition": "GFP",    "injection": "Gi"},
    "image_8969": {"condition": "GFP",    "injection": "Gq"},
    "image_9408": {"condition": "GFP",    "injection": "Gq"},
    "image_5767": {"condition": "GFP",    "injection": "Gq"},
    "image_8002": {"condition": "GFP",    "injection": "Gi"},
    "image_9867": {"condition": "GFP",    "injection": "Gq"},
}

GROUP_COLORS = {
    ("Gq", "DREADD"): "#2ca02c",
    ("Gq", "GFP"):    "#aec7a8",
    ("Gi", "DREADD"): "#d62728",
    ("Gi", "GFP"):    "#f5c2c2",
}
GROUP_LABELS = {
    ("Gq", "DREADD"): "Gq",
    ("Gq", "GFP"):    "Ctrl_Gq",
    ("Gi", "DREADD"): "Gi",
    ("Gi", "GFP"):    "Ctrl_Gi",
}
BAR_ORDER = [
    ("Gq", "DREADD"), ("Gq", "GFP"),
    ("Gi", "DREADD"), ("Gi", "GFP"),
]


def load_ads_clean(img_id):
    df = pd.read_csv(BASE / img_id / "input_Morphometrics.csv", index_col=0)
    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    if img_id in SPATIAL_CROPS:
        crop = SPATIAL_CROPS[img_id]
        mask = pd.Series(False, index=df.index)
        if "x_max" in crop: mask |= df["x0 (px)"] < crop["x_max"]
        if "y_min" in crop: mask |= df["y0 (px)"] > crop["y_min"]
        df = df[~mask].copy()
    df = df[df["axon_diam (um)"] >= DIAM_MIN_UM].copy()
    df = df[df["gratio"] < 1.0].copy()
    return df


# Load all axons
records = []
for img_id in TARGET_IMAGES:
    meta = METADATA[img_id]
    df = load_ads_clean(img_id)
    df = df[["axon_diam (um)", "gratio"]].copy()
    df["injection"] = meta["injection"]
    df["condition"] = meta["condition"]
    records.append(df)

all_axons = pd.concat(records, ignore_index=True)

# Per-image averages for bar plots
img_rows = []
for img_id in TARGET_IMAGES:
    meta = METADATA[img_id]
    df = load_ads_clean(img_id)
    img_rows.append({
        "injection": meta["injection"],
        "condition": meta["condition"],
        "gratio":    df["gratio"].mean(),
        "diameter":  df["axon_diam (um)"].mean(),
    })
img_df = pd.DataFrame(img_rows)

print(f"Total axons loaded: {len(all_axons)}")
for key in BAR_ORDER:
    inj, cond = key
    n = len(all_axons[(all_axons["injection"] == inj) & (all_axons["condition"] == cond)])
    print(f"  {GROUP_LABELS[key]}: {n} axons")


def bar_panel(ax, img_df, metric, ylabel):
    rng = np.random.default_rng(42)
    for i, (inj, cond) in enumerate(BAR_ORDER):
        vals = img_df[(img_df["injection"] == inj) & (img_df["condition"] == cond)][metric].values
        color = GROUP_COLORS[(inj, cond)]
        mean_v = vals.mean()
        sd_v   = vals.std(ddof=1) if len(vals) > 1 else 0
        ax.bar(i, mean_v, width=0.6, color=color, edgecolor="black", linewidth=1.0, zorder=2)
        ax.errorbar(i, mean_v, yerr=sd_v, fmt="none",
                    color="black", capsize=5, capthick=1.2, lw=1.2, zorder=4)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=40, color="black", zorder=5, linewidths=0)

    ax.set_xticks(range(len(BAR_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[k] for k in BAR_ORDER], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def scatter_panel(ax, all_axons, injection):
    for cond in ["DREADD", "GFP"]:
        key = (injection, cond)
        sub = all_axons[(all_axons["injection"] == injection) & (all_axons["condition"] == cond)]
        is_ctrl = (cond == "GFP")
        color = "gray" if is_ctrl else GROUP_COLORS[key]
        alpha = 0.25 if is_ctrl else 0.5
        label = GROUP_LABELS[key]
        ax.scatter(sub["axon_diam (um)"], sub["gratio"],
                   s=6, color=color, alpha=alpha, linewidths=0, label=label, zorder=2)

        x = sub["axon_diam (um)"].values
        y = sub["gratio"].values
        slope, intercept, r, _, _ = stats.linregress(x, y)
        x_line = np.array([x.min(), x.max()])
        edge_color = GROUP_COLORS[(injection, "DREADD")] if cond == "DREADD" else "gray"
        lw = 1.8 if cond == "DREADD" else 1.2
        ax.plot(x_line, intercept + slope * x_line,
                color=edge_color, linewidth=lw, zorder=4,
                label=f"y = {intercept:.3f} + {slope:.4f}x")
        print(f"  {injection} {GROUP_LABELS[key]:10s}: y = {intercept:.4f} + {slope:.4f}x  (r={r:.3f}, n={len(x)})")

    ax.set_xlabel("axon diameter [μm]", fontsize=10)
    ax.set_ylabel("g-ratio", fontsize=10)
    ax.set_title(f"g-ratio vs. diameter {injection}", fontsize=11, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("G-ratio and axon diameter by injection group (ADS circle, all axons)",
             fontsize=13, fontweight="bold")

bar_panel(axes[0, 0], img_df, "gratio",   "Average g-ratio (per image)")
axes[0, 0].set_title("average g-ratios", fontsize=11, fontweight="bold")
axes[0, 0].set_ylim(0.60, 0.82)

bar_panel(axes[1, 0], img_df, "diameter", "Average axon diameter [μm] (per image)")
axes[1, 0].set_title("average diameter", fontsize=11, fontweight="bold")

scatter_panel(axes[0, 1], all_axons, "Gq")
scatter_panel(axes[1, 1], all_axons, "Gi")

plt.tight_layout()
out = OUT_DIR / "gratio_scatter_by_injection.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out.name}")
