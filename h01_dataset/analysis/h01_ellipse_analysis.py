#!/usr/bin/env python3
"""
H01 ellipse-mode analysis pipeline.

Steps:
  1. Extract seg_axon / seg_myelin masks from overlay PNGs
  2. Run axondeepseg_morphometrics (circle + ellipse) per slice
  3. Select the central traced axon from each slice
  4. Aggregate per-axon means; compare circle vs ellipse vs manual
  5. Generate figures: bar chart, scatter, per-axon z-profiles

Outputs written to:
  H01_dataset/analysis/
    h01_morphometrics_ellipse.csv   — per-slice ellipse measurements (central axon)
    h01_morphometrics_circle.csv    — per-slice circle measurements (central axon)
    h01_comparison_summary.csv      — per-axon means: manual / circle / ellipse
    h01_gratio_bar.png              — 3-bar chart: Manual / Circle / Ellipse
    h01_scatter_circle_vs_ellipse.png
    h01_zprofiles_ellipse.png       — g-ratio along axon length, all 20 axons
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/Users/yolaatar/Developer/ADS/Timmler_data/H01_dataset")
OUT_DIR  = BASE / "analysis"
ADS_MORPH = Path("/Users/yolaatar/Developer/ADS/axondeepseg/.venv/bin/axondeepseg_morphometrics")
OUT_DIR.mkdir(exist_ok=True)

AXONS = [f"Axon{i}" for i in range(1, 21)]

# Pixel size declared to ADS morphometrics tool
PIXEL_SIZE_UM = 0.13

# Central axon selection criteria
# The traced H01 axon is one of the largest myelinated axons in the image.
# We select the highest-diameter valid axon per slice.
MIN_DIAM_UM  = 1.0    # minimum axon diameter (µm at 0.13 µm/px scale)
MAX_GRATIO   = 0.97   # exclude near-no-myelin artefacts
MIN_GRATIO   = 0.40   # exclude impossible values

# Colours
COL_MANUAL  = "white"
COL_CIRCLE  = "#a78fd0"   # purple
COL_ELLIPSE = "#6dbf8a"   # green

# ── Overlay colours (from metadata) ───────────────────────────────────────────
AXN_COLOR = np.array([220,  40,  40], dtype=np.uint8)   # red   = axon
MYL_COLOR = np.array([ 40,  80, 220], dtype=np.uint8)   # blue  = myelin
COLOR_TOL  = 10   # tolerance for colour matching


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Extract seg masks from overlay PNGs
# ═══════════════════════════════════════════════════════════════════════════════

def extract_masks(axon_dir: Path) -> list[Path]:
    """Extract seg_axon and seg_myelin binary PNGs for every slice in axon_dir/slices/."""
    slices_dir = axon_dir / "slices"
    input_files = sorted(slices_dir.glob("*_input.png"))
    created = []
    for inp in input_files:
        stem = inp.stem                                  # e.g. slice_000_input
        plain_stem = stem.replace("_input", "")         # e.g. slice_000
        overlay_path = slices_dir / f"{plain_stem}_overlay.png"
        # Tool expects: {input_stem}_seg-axon.png  (i.e. slice_000_input_seg-axon.png)
        axon_out   = slices_dir / f"{stem}_seg-axon.png"
        myelin_out = slices_dir / f"{stem}_seg-myelin.png"

        if axon_out.exists() and myelin_out.exists():
            created.append(inp)
            continue

        overlay = np.array(Image.open(overlay_path).convert("RGB"))

        axon_mask  = np.all(np.abs(overlay.astype(int) - AXN_COLOR.astype(int)) <= COLOR_TOL, axis=2)
        myelin_mask = np.all(np.abs(overlay.astype(int) - MYL_COLOR.astype(int)) <= COLOR_TOL, axis=2)

        Image.fromarray(axon_mask.astype(np.uint8) * 255).save(axon_out)
        Image.fromarray(myelin_mask.astype(np.uint8) * 255).save(myelin_out)
        created.append(inp)

    return input_files


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Run morphometrics per slice (both modes)
# ═══════════════════════════════════════════════════════════════════════════════

def run_morphometrics(inp: Path, mode: str) -> Path:
    """Run axondeepseg_morphometrics on one slice. Returns path to output CSV.

    Tool names the output {input_stem}_{-f filename}, so for input
    slice_000_input.png and -f Morphometrics.csv → slice_000_input_Morphometrics.csv.
    """
    suffix = "" if mode == "circle" else "_ellipse"
    f_arg   = f"Morphometrics{suffix}.csv"
    csv_out = inp.parent / f"{inp.stem}_{f_arg}"   # e.g. slice_000_input_Morphometrics_ellipse.csv
    if csv_out.exists():
        return csv_out
    cmd = [
        str(ADS_MORPH),
        "-i", str(inp),
        "-s", str(PIXEL_SIZE_UM),
        "-f", f_arg,
        "-a", mode,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: morphometrics failed for {inp.name} ({mode}): {result.stderr[:200]}")
    return csv_out


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Select the central traced axon from one slice CSV
# ═══════════════════════════════════════════════════════════════════════════════

def pick_central_axon(csv_path: Path, img_w: int, img_h: int):
    """Select the traced axon from one slice CSV.

    The H01 traced axon is one of the largest myelinated axons in the image.
    We select the highest-diameter valid axon per slice (border-excluded, g in
    realistic range), with a slight centre-proximity tiebreaker.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception:
        return None
    if df.empty:
        return None

    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    if df.empty:
        return None

    cx, cy = img_w / 2, img_h / 2
    df["dist_centre"] = np.sqrt((df["x0 (px)"] - cx)**2 + (df["y0 (px)"] - cy)**2)

    candidates = df[
        (df["axon_diam (um)"] >= MIN_DIAM_UM) &
        (df["gratio"] < MAX_GRATIO) &
        (df["gratio"] > MIN_GRATIO) &
        df["gratio"].notna()
    ]
    if candidates.empty:
        return None

    # Score: prefer large diameter; break ties by distance to centre
    candidates = candidates.copy()
    candidates["score"] = candidates["axon_diam (um)"] / (1 + candidates["dist_centre"] / 500)
    best = candidates.loc[candidates["score"].idxmax()]
    return {
        "gratio":      float(best["gratio"]),
        "axon_diam":   float(best["axon_diam (um)"]),
        "dist_centre": float(best["dist_centre"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

circle_records  = []
ellipse_records = []

for axon_id in AXONS:
    axon_dir = BASE / axon_id
    if not axon_dir.exists():
        print(f"[SKIP] {axon_id} — folder not found")
        continue

    print(f"\n── {axon_id} ──")
    input_files = extract_masks(axon_dir)

    for inp in input_files:
        stem = inp.stem.replace("_input", "")
        img = Image.open(inp)
        img_w, img_h = img.size

        for mode, records in [("circle", circle_records), ("ellipse", ellipse_records)]:
            csv = run_morphometrics(inp, mode)
            result = pick_central_axon(csv, img_w, img_h)
            if result is None:
                print(f"  [{mode}] {stem}: no central axon found")
                continue
            records.append({
                "axon":      axon_id,
                "slice":     stem,
                "gratio":    result["gratio"],
                "axon_diam": result["axon_diam"],
                "dist_centre": result["dist_centre"],
            })
            print(f"  [{mode}] {stem}: g={result['gratio']:.3f}  diam={result['axon_diam']:.2f}µm  dist={result['dist_centre']:.1f}px")

circle_df  = pd.DataFrame(circle_records)
ellipse_df = pd.DataFrame(ellipse_records)
circle_df.to_csv(OUT_DIR / "h01_morphometrics_circle.csv", index=False)
ellipse_df.to_csv(OUT_DIR / "h01_morphometrics_ellipse.csv", index=False)
print(f"\nCircle:  {len(circle_df)} slice measurements")
print(f"Ellipse: {len(ellipse_df)} slice measurements")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Aggregate per axon + load manual means from existing summary
# ═══════════════════════════════════════════════════════════════════════════════

manual_summary = pd.read_csv(BASE / "h01_gratio_summary.csv")
manual_means   = dict(zip(manual_summary["axon"], manual_summary["mean_manual"]))

rows = []
for axon_id in AXONS:
    circ  = circle_df[circle_df["axon"] == axon_id]["gratio"]
    elli  = ellipse_df[ellipse_df["axon"] == axon_id]["gratio"]
    man   = manual_means.get(axon_id, np.nan)

    if circ.empty and elli.empty:
        continue

    rows.append({
        "axon":           axon_id,
        "n_circle":       len(circ),
        "mean_circle":    round(circ.mean(), 4) if not circ.empty else np.nan,
        "n_ellipse":      len(elli),
        "mean_ellipse":   round(elli.mean(), 4) if not elli.empty else np.nan,
        "mean_manual":    round(man, 4) if not np.isnan(man) else np.nan,
        "bias_circle":    round(circ.mean() - man, 4) if not circ.empty and not np.isnan(man) else np.nan,
        "bias_ellipse":   round(elli.mean() - man, 4) if not elli.empty and not np.isnan(man) else np.nan,
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "h01_comparison_summary.csv", index=False)
print("\nPer-axon summary:")
print(summary_df[["axon","n_circle","mean_circle","mean_ellipse","mean_manual","bias_circle","bias_ellipse"]].to_string(index=False))

has_manual = summary_df["mean_manual"].notna()
print(f"\nMean bias — Circle:  {summary_df.loc[has_manual,'bias_circle'].mean():.4f}")
print(f"Mean bias — Ellipse: {summary_df.loc[has_manual,'bias_ellipse'].mean():.4f}")
print(f"Mean MAE  — Circle:  {summary_df.loc[has_manual,'bias_circle'].abs().mean():.4f}")
print(f"Mean MAE  — Ellipse: {summary_df.loc[has_manual,'bias_ellipse'].abs().mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 3-bar chart: Manual / Circle / Ellipse (per-axon dots)
# ═══════════════════════════════════════════════════════════════════════════════

df_valid = summary_df.dropna(subset=["mean_manual", "mean_circle", "mean_ellipse"])

man_arr  = df_valid["mean_manual"].values
circ_arr = df_valid["mean_circle"].values
elli_arr = df_valid["mean_ellipse"].values

fig, ax = plt.subplots(figsize=(8, 7))
rng = np.random.default_rng(42)
BAR_W = 0.5

entries = [
    (0, man_arr,  COL_MANUAL,  "Manual",         "black"),
    (1, circ_arr, COL_CIRCLE,  "ADS\n(circle)",  "black"),
    (2, elli_arr, COL_ELLIPSE, "ADS\n(ellipse)", "black"),
]

for pos, vals, fc, label, ec in entries:
    mean_v, sd_v, med_v = vals.mean(), vals.std(), np.median(vals)
    ax.bar(pos, mean_v, width=BAR_W, color=fc, edgecolor=ec, linewidth=1.5, zorder=2)
    ax.errorbar(pos, mean_v, yerr=sd_v, fmt="none", color=ec,
                capsize=6, capthick=1.5, lw=1.5, zorder=4)
    jitter = rng.uniform(-0.07, 0.07, len(vals))
    ax.scatter(pos + jitter, vals, s=50, color="black", zorder=5, linewidths=0)
    stats_txt = f"mean    {mean_v:.3f}\nSD       {sd_v:.3f}\nmedian {med_v:.3f}"
    ax.text(pos, ax.get_ylim()[0] + 0.003 if ax.get_ylim()[0] > 0 else 0.603,
            stats_txt, ha="center", va="bottom", fontsize=8.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9, lw=0.8),
            zorder=6)
    print(f"{label.replace(chr(10),' ')}: mean={mean_v:.3f}  SD={sd_v:.3f}  median={med_v:.3f}")

ylo = min(man_arr.min(), circ_arr.min(), elli_arr.min()) - 0.05
yhi = max(man_arr.max(), circ_arr.max(), elli_arr.max()) + 0.07
ax.set_ylim(max(0.55, ylo), yhi)

# Re-draw stat boxes with correct y position
ax.cla()
for pos, vals, fc, label, ec in entries:
    mean_v, sd_v, med_v = vals.mean(), vals.std(), np.median(vals)
    ax.bar(pos, mean_v, width=BAR_W, color=fc, edgecolor=ec, linewidth=1.5, zorder=2)
    ax.errorbar(pos, mean_v, yerr=sd_v, fmt="none", color=ec,
                capsize=6, capthick=1.5, lw=1.5, zorder=4)
    jitter = rng.uniform(-0.07, 0.07, len(vals))
    ax.scatter(pos + jitter, vals, s=50, color="black", zorder=5, linewidths=0)
    stats_txt = f"mean    {mean_v:.3f}\nSD       {sd_v:.3f}\nmedian {med_v:.3f}"
    ax.text(pos, max(0.55, ylo) + 0.003, stats_txt,
            ha="center", va="bottom", fontsize=8.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9, lw=0.8),
            zorder=6)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Manual", "ADS\n(circle)", "ADS\n(ellipse)"], fontsize=12)
ax.set_ylabel("mean g-ratio per axon", fontsize=13)
ax.set_title("H01 — Average g-ratio\nManual vs ADS circle vs ADS ellipse", fontsize=13,
             fontweight="bold", pad=10)
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(max(0.55, ylo), yhi)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "h01_gratio_bar.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → h01_gratio_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Scatter: circle vs ellipse per axon, coloured by manual reference
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Left: ADS circle vs Manual
ax = axes[0]
ax.scatter(df_valid["mean_manual"], df_valid["mean_circle"],
           s=70, color=COL_CIRCLE, edgecolors="black", linewidths=0.5, zorder=3)
lims = (0.55, 0.95)
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
r_c, _ = stats.pearsonr(df_valid["mean_manual"], df_valid["mean_circle"])
mae_c  = (df_valid["mean_circle"] - df_valid["mean_manual"]).abs().mean()
bias_c = (df_valid["mean_circle"] - df_valid["mean_manual"]).mean()
ax.text(0.05, 0.95, f"r={r_c:.2f}\nMAE={mae_c:.3f}\nbias={bias_c:+.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0))
for _, row in df_valid.iterrows():
    ax.annotate(row["axon"].replace("Axon", ""), (row["mean_manual"], row["mean_circle"]),
                fontsize=6, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
ax.set_xlabel("Manual g-ratio (per axon mean)", fontsize=10)
ax.set_ylabel("ADS circle g-ratio", fontsize=10)
ax.set_title("Circle vs Manual", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

# Right: ADS ellipse vs Manual
ax = axes[1]
ax.scatter(df_valid["mean_manual"], df_valid["mean_ellipse"],
           s=70, color=COL_ELLIPSE, edgecolors="black", linewidths=0.5, zorder=3)
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
r_e, _ = stats.pearsonr(df_valid["mean_manual"], df_valid["mean_ellipse"])
mae_e  = (df_valid["mean_ellipse"] - df_valid["mean_manual"]).abs().mean()
bias_e = (df_valid["mean_ellipse"] - df_valid["mean_manual"]).mean()
ax.text(0.05, 0.95, f"r={r_e:.2f}\nMAE={mae_e:.3f}\nbias={bias_e:+.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0))
for _, row in df_valid.iterrows():
    ax.annotate(row["axon"].replace("Axon", ""), (row["mean_manual"], row["mean_ellipse"]),
                fontsize=6, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
ax.set_xlabel("Manual g-ratio (per axon mean)", fontsize=10)
ax.set_ylabel("ADS ellipse g-ratio", fontsize=10)
ax.set_title("Ellipse vs Manual", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("H01 — ADS vs Manual g-ratio per axon (20 axons)", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "h01_scatter_circle_vs_ellipse.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved → h01_scatter_circle_vs_ellipse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Z-profiles (g-ratio along arc length) all 20 axons, ellipse mode
# ═══════════════════════════════════════════════════════════════════════════════

n_axons = len(AXONS)
ncols = 5
nrows = math.ceil(n_axons / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))

for idx, axon_id in enumerate(AXONS):
    ax = axes.flat[idx]
    elli_slices = ellipse_df[ellipse_df["axon"] == axon_id].copy()
    circ_slices = circle_df[circle_df["axon"] == axon_id].copy()
    man_mean    = manual_means.get(axon_id, np.nan)

    x = np.arange(len(elli_slices))
    ax.plot(x, elli_slices["gratio"].values, color=COL_ELLIPSE, lw=1.5, marker="o", ms=3, label="Ellipse")
    if not circ_slices.empty:
        xc = np.arange(len(circ_slices))
        ax.plot(xc, circ_slices["gratio"].values, color=COL_CIRCLE, lw=1.5, marker="s",
                ms=3, alpha=0.6, label="Circle", linestyle="--")
    if not np.isnan(man_mean):
        ax.axhline(man_mean, color="black", lw=1.0, linestyle=":", alpha=0.7, label="Manual mean")

    ax.set_title(axon_id, fontsize=9, fontweight="bold")
    ax.set_xlabel("Slice index", fontsize=7)
    ax.set_ylabel("g-ratio", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0.5, 1.0)
    ax.spines[["top","right"]].set_visible(False)
    if idx == 0:
        ax.legend(fontsize=6, loc="upper right")

# Hide unused panels
for idx in range(n_axons, nrows * ncols):
    axes.flat[idx].set_visible(False)

fig.suptitle("H01 — G-ratio per slice: Ellipse (green) vs Circle (purple) vs Manual mean (dotted)",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "h01_zprofiles_ellipse.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved → h01_zprofiles_ellipse.png")

print("\nDone.")
