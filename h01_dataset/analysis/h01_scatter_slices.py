#!/usr/bin/env python3
"""
Per-slice scatter: ADS (circle & ellipse) vs per-slice manual g-ratio.
Produces three figures:
  h01_scatter_ellipse_per_slice.png  — ellipse only
  h01_scatter_circle_per_slice.png   — circle only
  h01_scatter_circle_vs_ellipse_per_slice.png — side by side
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

RAW     = Path("/Users/yolaatar/Developer/ADS/Timmler_data/H01_dataset_raw")
OUT_DIR = Path("/Users/yolaatar/Developer/ADS/Timmler_data/H01_dataset/analysis")

# ── Manual per-slice data ─────────────────────────────────────────────────────
manual = pd.read_csv(RAW / "All axons/H01_indi_points_complete.txt", sep="\t")
manual["axon"]      = manual["axon_ID"].str.replace("ax_", "Axon", regex=False)
manual["slice_idx"] = manual.groupby("axon").cumcount()
manual_valid = manual[~manual["sheath_ID"].str.startswith("No")].copy()

# ── Helper: load + filter one ADS mode, merge with manual ────────────────────
def load_mode(mode):
    fname = "h01_morphometrics_ellipse.csv" if mode == "ellipse" else "h01_morphometrics_circle.csv"
    ads = pd.read_csv(OUT_DIR / fname)
    ads["slice_idx"] = ads["slice"].str.extract(r"slice_(\d+)").astype(int)
    ads_good = ads[(ads["dist_centre"] <= 100) & (ads["gratio"] <= 0.92)].copy()
    merged = pd.merge(
        manual_valid[["axon", "slice_idx", "g_ratio"]],
        ads_good[["axon", "slice_idx", "gratio"]],
        on=["axon", "slice_idx"], how="inner",
    )
    return merged

merged_e = load_mode("ellipse")
merged_c = load_mode("circle")

# ── Single-panel scatter ──────────────────────────────────────────────────────
LIMS = (0.50, 1.00)

def one_panel(ax, merged, mode_label, color):
    ax.scatter(merged["g_ratio"], merged["gratio"],
               s=45, alpha=0.70, linewidths=0, color=color, zorder=3)
    ax.plot(LIMS, LIMS, "k--", lw=1.0, alpha=0.5, zorder=2)

    r, p   = stats.pearsonr(merged["g_ratio"], merged["gratio"])
    bias   = (merged["gratio"] - merged["g_ratio"]).mean()
    mae    = (merged["gratio"] - merged["g_ratio"]).abs().mean()

    ax.text(0.04, 0.97,
            f"n = {len(merged)} slices  ({merged['axon'].nunique()} axons)\n"
            f"r = {r:.2f}   p = {p:.3f}\n"
            f"mean bias = {bias:+.3f}\n"
            f"MAE = {mae:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9, lw=0.8))

    ax.set_xlim(LIMS); ax.set_ylim(LIMS)
    ax.set_aspect("equal")
    ax.set_xlabel("Manual g-ratio  (per slice)", fontsize=11)
    ax.set_ylabel(f"ADS g-ratio  ({mode_label}, per slice)", fontsize=11)
    ax.set_title(f"ADS {mode_label} vs Manual", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    return r, bias, mae

# Ellipse only
fig, ax = plt.subplots(figsize=(7, 7))
one_panel(ax, merged_e, "ellipse", "#6dbf8a")
plt.tight_layout()
fig.savefig(OUT_DIR / "h01_scatter_ellipse_per_slice.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Circle only
fig, ax = plt.subplots(figsize=(7, 7))
one_panel(ax, merged_c, "circle", "#a78fd0")
plt.tight_layout()
fig.savefig(OUT_DIR / "h01_scatter_circle_per_slice.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("H01 — ADS vs Manual g-ratio per slice · all 20 axons",
             fontsize=13, fontweight="bold", y=1.01)
one_panel(axes[0], merged_c, "circle",  "#a78fd0")
one_panel(axes[1], merged_e, "ellipse", "#6dbf8a")
plt.tight_layout()
fig.savefig(OUT_DIR / "h01_scatter_circle_vs_ellipse_per_slice.png",
            dpi=180, bbox_inches="tight")
plt.close(fig)

print("Saved → h01_scatter_ellipse_per_slice.png")
print("Saved → h01_scatter_circle_per_slice.png")
print("Saved → h01_scatter_circle_vs_ellipse_per_slice.png")

r_c, bias_c, mae_c = [stats.pearsonr(merged_c["g_ratio"], merged_c["gratio"])[0],
                       (merged_c["gratio"]-merged_c["g_ratio"]).mean(),
                       (merged_c["gratio"]-merged_c["g_ratio"]).abs().mean()]
r_e, bias_e, mae_e = [stats.pearsonr(merged_e["g_ratio"], merged_e["gratio"])[0],
                       (merged_e["gratio"]-merged_e["g_ratio"]).mean(),
                       (merged_e["gratio"]-merged_e["g_ratio"]).abs().mean()]
print(f"\nCircle:  n={len(merged_c)}  r={r_c:.3f}  bias={bias_c:+.4f}  MAE={mae_c:.4f}")
print(f"Ellipse: n={len(merged_e)}  r={r_e:.3f}  bias={bias_e:+.4f}  MAE={mae_e:.4f}")
