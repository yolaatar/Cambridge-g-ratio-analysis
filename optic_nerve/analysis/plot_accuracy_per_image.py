#!/usr/bin/env python3
"""
Per-image MAE% for ADS circle and ADS ellipse.
Images grouped by condition (DREADD | GFP), sorted by MAE within each group.
Horizontal dashed lines = overall mean for each method.

Output: accuracy_per_image.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve/analysis")

METADATA = {
    "image_706":  "DREADD", "image_963":  "DREADD", "image_1812": "DREADD",
    "image_2299": "DREADD", "image_2696": "DREADD",
    "image_2336": "GFP",    "image_4090": "GFP",    "image_5087": "GFP",
    "image_6107": "GFP",    "image_6294": "GFP",    "image_7422": "GFP",
    "image_7559": "GFP",    "image_8969": "GFP",    "image_9408": "GFP",
}

df = pd.read_csv(OUT_DIR / "circle_vs_ellipse_summary.csv")
df["condition"] = df["image"].map(METADATA)
df["image_short"] = df["image"].str.replace("image_", "", regex=False)

# Sort: DREADD first, then GFP; within each group sort by circle MAE%
df = pd.concat([
    df[df["condition"] == "DREADD"].sort_values("pct_circle"),
    df[df["condition"] == "GFP"].sort_values("pct_circle"),
]).reset_index(drop=True)

x   = np.arange(len(df))
W   = 0.35
fig, ax = plt.subplots(figsize=(13, 6))

bars_c = ax.bar(x - W/2, df["pct_circle"],  width=W, color="#a78fd0",
                edgecolor="black", linewidth=0.8, label="ADS circle",  zorder=2)
bars_e = ax.bar(x + W/2, df["pct_ellipse"], width=W, color="#6dbf8a",
                edgecolor="black", linewidth=0.8, label="ADS ellipse", zorder=2)

mean_c = df["pct_circle"].mean()
mean_e = df["pct_ellipse"].mean()
ax.axhline(mean_c, color="#6a4fa8", lw=1.5, ls="--", zorder=3,
           label=f"Circle mean {mean_c:.1f}%")
ax.axhline(mean_e, color="#3a8f5a", lw=1.5, ls=":",  zorder=3,
           label=f"Ellipse mean {mean_e:.1f}%")

# Condition separator
n_dreadd = (df["condition"] == "DREADD").sum()
ax.axvline(n_dreadd - 0.5, color="gray", lw=1.2, ls="-", alpha=0.5)
ax.text(n_dreadd / 2 - 0.5, ax.get_ylim()[1] if False else 13.5,
        "DREADD", ha="center", fontsize=10, color="#2b7bb9", fontweight="bold")
ax.text(n_dreadd + (len(df) - n_dreadd) / 2 - 0.5, 13.5,
        "GFP", ha="center", fontsize=10, color="#d62728", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{s}\n(n={n})" for s, n in zip(df["image_short"], df["n_circle"])],
    rotation=45, ha="right", fontsize=8.5
)
for tick, cond in zip(ax.get_xticklabels(), df["condition"]):
    tick.set_color("#2b7bb9" if cond == "DREADD" else "#d62728")

ax.set_ylabel("MAE  (% of manual g-ratio)", fontsize=12)
ax.set_title("Per-image ADS accuracy — circle vs ellipse\n(lower = better)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_ylim(0, 14.5)
ax.yaxis.set_major_locator(plt.MultipleLocator(2))
ax.legend(fontsize=9, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = OUT_DIR / "accuracy_per_image.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Overall mean MAE%  —  circle: {mean_c:.2f}%   ellipse: {mean_e:.2f}%")
print(f"Saved → {out.name}")
