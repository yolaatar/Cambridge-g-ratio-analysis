#!/usr/bin/env python3
"""
Timmler Optic Nerve Analysis
Part 1: ADS vs Manual g-ratio comparison for matched axons (5 images)
Part 2: Axon diameter and g-ratio distributions across all 14 images (all ADS-detected axons)
"""

import os
import subprocess
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import KDTree
from scipy import stats
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve")
MORPH_BIN  = Path("/Users/yolaatar/Developer/ADS/axondeepseg/.venv/bin/axondeepseg_morphometrics")
OUT_DIR    = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve/analysis")
PIXEL_SIZE = 0.00493   # µm/px (target pixel size, consistent across all images)
PIXEL_SIZE_NM = PIXEL_SIZE * 1000  # nm/px

# ── Images with both ADS predictions and manual measurements ──────────────────
TARGET_IMAGES = [
    "image_706",  "image_963",  "image_1812", "image_2299", "image_2336",
    "image_2696", "image_4090", "image_5087", "image_6107", "image_6294",
    "image_7422", "image_7559", "image_8969", "image_9408",
]

# ── Unblinding table (parsed from txt + xlsx) ─────────────────────────────────
# L = DREADD-injected, R = GFP-control
METADATA = {
    "image_706":  {"animal": "TKFG 19.1c", "side": "L", "condition": "DREADD", "injection": "Gi",  "genotype": "Hemi/WT"},
    "image_963":  {"animal": "TKFG 19.1g", "side": "L", "condition": "DREADD", "injection": "Gi",  "genotype": "Hemi/Het"},
    "image_1812": {"animal": "TKFG 19.1g", "side": "L", "condition": "DREADD", "injection": "Gi",  "genotype": "Hemi/Het"},
    "image_2299": {"animal": "TKFG 18.1f", "side": "L", "condition": "DREADD", "injection": "Gq",  "genotype": "Hemi/WT"},
    "image_2336": {"animal": "TKFG 17.1g", "side": "R", "condition": "GFP",    "injection": "Gq",  "genotype": "Hemi/WT"},
    "image_2696": {"animal": "TKFG 17.1g", "side": "L", "condition": "DREADD", "injection": "Gq",  "genotype": "Hemi/WT"},
    "image_4090": {"animal": "TKFG 18.1f", "side": "R", "condition": "GFP",    "injection": "Gq",  "genotype": "Hemi/WT"},
    "image_5087": {"animal": "TKFG 19.1f", "side": "R", "condition": "GFP",    "injection": "Gi",  "genotype": "Hemi/Het"},
    "image_6107": {"animal": "TKFG 19.1c", "side": "R", "condition": "GFP",    "injection": "Gi",  "genotype": "Hemi/WT"},
    "image_6294": {"animal": "TKFG 19.1g", "side": "R", "condition": "GFP",    "injection": "Gi",  "genotype": "Hemi/Het"},
    "image_7422": {"animal": "TKFG 17.1g", "side": "R", "condition": "GFP",    "injection": "Gq",  "genotype": "Hemi/WT"},
    "image_7559": {"animal": "TKFG 19.1e", "side": "R", "condition": "GFP",    "injection": "Gi",  "genotype": "Hemi/WT"},
    "image_8969": {"animal": "TKFG 21.1g", "side": "R", "condition": "GFP",    "injection": "Gq",  "genotype": "Hemi/WT"},
    "image_9408": {"animal": "TKFG 18.1f", "side": "R", "condition": "GFP",    "injection": "Gq",  "genotype": "Hemi/WT"},
}

# Pick 5 images for Part 1 — spread across animals, conditions, and injection types
PART1_IMAGES = ["image_706", "image_2299", "image_2336", "image_2696", "image_6107"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PREPARE SEG FILES  (symlinks: seg_axon.png → input_seg-axon.png)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_seg_symlinks():
    for img in TARGET_IMAGES:
        img_dir = BASE_DIR / img
        for suffix, link_name in [("seg_axon.png", "input_seg-axon.png"),
                                   ("seg_myelin.png", "input_seg-myelin.png")]:
            src  = img_dir / suffix
            link = img_dir / link_name
            if src.exists() and not link.exists():
                link.symlink_to(suffix)
    print("Symlinks ready.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RUN AXONDEEPSEG MORPHOMETRICS
# ═══════════════════════════════════════════════════════════════════════════════

def run_morphometrics():
    for img in TARGET_IMAGES:
        img_path  = BASE_DIR / img / "input.png"
        out_csv   = BASE_DIR / img / "input_Morphometrics.csv"
        if out_csv.exists():
            print(f"  {img}: morphometrics already computed, skipping.")
            continue
        print(f"  Running morphometrics on {img}...")
        result = subprocess.run(
            [str(MORPH_BIN), "-i", str(img_path),
             "-s", str(PIXEL_SIZE), "-f", "Morphometrics.csv"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR on {img}:\n{result.stderr[-500:]}")
        else:
            print(f"  {img}: done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOAD ADS MORPHOMETRICS
# ═══════════════════════════════════════════════════════════════════════════════

def load_ads_morphometrics(img_id: str) -> pd.DataFrame:
    csv_path = BASE_DIR / img_id / "input_Morphometrics.csv"
    df = pd.read_csv(csv_path, index_col=0)
    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    df["image_id"] = img_id
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOAD MANUAL MEASUREMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_manual_measurements(img_id: str) -> pd.DataFrame:
    """
    Parse the Results_XXXX.csv file.
    Rows come in consecutive pairs: outer ROI (larger area), inner ROI (smaller area).
    Coordinates X, Y are in nm (calibrated from original TIF pixel size 1.8625 nm/px).
    Convert to input.png pixels by dividing by PIXEL_SIZE_NM (4.93 nm/px).

    Returns a DataFrame with one row per axon:
      x_px, y_px   — centroid in input.png pixels
      axon_area_um2, fiber_area_um2
      axon_diam_um  — from axon area assuming circular cross-section
      gratio        — sqrt(axon_area / fiber_area)
    """
    num = img_id.replace("image_", "")
    csv_path = BASE_DIR / f"Results_{num}.csv"
    df = pd.read_csv(csv_path, index_col=0)

    records = []
    rows = df.to_dict("records")
    for i in range(0, len(rows) - 1, 2):
        outer = rows[i]
        inner = rows[i + 1]
        # Ensure correct pairing: outer is the larger area
        if float(inner["Area"]) > float(outer["Area"]):
            outer, inner = inner, outer

        fiber_area_nm2 = float(outer["Area"])
        axon_area_nm2  = float(inner["Area"])
        # centroid: average of the two (they're almost identical)
        x_nm = (float(outer["X"]) + float(inner["X"])) / 2
        y_nm = (float(outer["Y"]) + float(inner["Y"])) / 2

        fiber_area_um2 = fiber_area_nm2 / 1e6
        axon_area_um2  = axon_area_nm2  / 1e6
        gratio         = math.sqrt(axon_area_um2 / fiber_area_um2)
        axon_diam_um   = 2 * math.sqrt(axon_area_um2 / math.pi)

        records.append({
            "x_px":          x_nm / PIXEL_SIZE_NM,
            "y_px":          y_nm / PIXEL_SIZE_NM,
            "axon_area_um2": axon_area_um2,
            "fiber_area_um2":fiber_area_um2,
            "axon_diam_um":  axon_diam_um,
            "gratio":        gratio,
            "image_id":      img_id,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MATCH ADS AXONS TO MANUAL AXONS
# ═══════════════════════════════════════════════════════════════════════════════

MATCH_THRESHOLD_PX = 75   # max centroid distance in input.png pixels (~370 nm)

def match_axons(ads_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each manual axon, find the nearest ADS axon by centroid distance.
    Only keep matches within MATCH_THRESHOLD_PX pixels.
    """
    ads_coords    = ads_df[["x0 (px)", "y0 (px)"]].values
    manual_coords = manual_df[["x_px", "y_px"]].values

    tree = KDTree(ads_coords)
    dists, idxs = tree.query(manual_coords, k=1)

    mask = dists < MATCH_THRESHOLD_PX
    matched = pd.DataFrame({
        "manual_gratio":    manual_df["gratio"].values[mask],
        "manual_diam_um":   manual_df["axon_diam_um"].values[mask],
        "ads_gratio":       ads_df["gratio"].iloc[idxs[mask]].values,
        "ads_diam_um":      ads_df["axon_diam (um)"].iloc[idxs[mask]].values,
        "dist_px":          dists[mask],
    })
    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — ADS vs Manual comparison figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_part1():
    print("\n── Part 1: ADS vs Manual comparison ──")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    fig.suptitle("ADS vs Manual g-ratio — matched axons only", fontsize=13, fontweight="bold")

    summary_rows = []
    for ax, img_id in zip(axes, PART1_IMAGES):
        ads_df    = load_ads_morphometrics(img_id)
        manual_df = load_manual_measurements(img_id)
        matched   = match_axons(ads_df, manual_df)
        n = len(matched)

        if n == 0:
            ax.set_title(f"{img_id}\nno matches")
            continue

        r, pval   = stats.pearsonr(matched["manual_gratio"], matched["ads_gratio"])
        mae       = (matched["ads_gratio"] - matched["manual_gratio"]).abs().mean()
        pct_error = mae / matched["manual_gratio"].mean() * 100

        meta = METADATA[img_id]
        title = (f"{img_id}\n{meta['animal']} {meta['side']} "
                 f"({meta['condition']}, {meta['injection']})")

        lims = [
            min(matched["manual_gratio"].min(), matched["ads_gratio"].min()) - 0.05,
            max(matched["manual_gratio"].max(), matched["ads_gratio"].max()) + 0.05,
        ]
        ax.scatter(matched["manual_gratio"], matched["ads_gratio"],
                   alpha=0.6, s=20, edgecolors="none",
                   color="#2b7bb9" if meta["condition"] == "DREADD" else "#d62728")
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Manual g-ratio")
        ax.set_ylabel("ADS g-ratio")
        ax.set_title(title, fontsize=8)
        ax.text(0.05, 0.95,
                f"n={n}\nr={r:.2f}\nMAE={mae:.3f} ({pct_error:.1f}%)",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        summary_rows.append({
            "image": img_id, "n_matched": n, "pearson_r": round(r, 3),
            "MAE": round(mae, 4), "pct_error": round(pct_error, 2),
        })
        print(f"  {img_id}: n={n}, r={r:.3f}, MAE={mae:.4f} ({pct_error:.1f}%)")

    plt.tight_layout()
    out = OUT_DIR / "part1_ads_vs_manual.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

    # also save summary CSV
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "part1_summary.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — All-axon distributions by condition
# ═══════════════════════════════════════════════════════════════════════════════

def plot_part2():
    print("\n── Part 2: All-axon distributions ──")

    # Aggregate all ADS morphometrics
    frames = []
    for img_id in TARGET_IMAGES:
        df = load_ads_morphometrics(img_id)
        for key, val in METADATA[img_id].items():
            df[key] = val
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    print(f"  Total ADS axons: {len(all_df)}")
    for cond, grp in all_df.groupby("condition"):
        print(f"    {cond}: {len(grp)} axons from {grp['image_id'].nunique()} images")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Optic Nerve — ADS morphometrics (all detected axons)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {"DREADD": "#2b7bb9", "GFP": "#d62728"}
    INJ_COLORS = {"Gi": "#e8862a", "Gq": "#7b4da6"}

    # ── Row 0: DREADD vs GFP ─────────────────────────────────────────────────
    for col, (metric, label, unit) in enumerate([
        ("axon_diam (um)", "Axon diameter", "µm"),
        ("gratio",         "G-ratio",       ""),
    ]):
        # Violin
        ax_v = fig.add_subplot(gs[0, col * 2])
        data_groups = [all_df.loc[all_df["condition"] == c, metric].dropna().values
                       for c in ["DREADD", "GFP"]]
        vp = ax_v.violinplot(data_groups, positions=[0, 1], showmedians=True, showextrema=False)
        for patch, color in zip(vp["bodies"], [COLORS["DREADD"], COLORS["GFP"]]):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        vp["cmedians"].set_color("black")
        ax_v.set_xticks([0, 1]); ax_v.set_xticklabels(["DREADD", "GFP"])
        ax_v.set_ylabel(f"{label} ({unit})" if unit else label)
        ax_v.set_title(f"{label}\nDREADD vs GFP")

        # Stat annotation
        u_stat, p_val = stats.mannwhitneyu(*data_groups, alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        y_max = max(np.nanpercentile(d, 99) for d in data_groups)
        ax_v.plot([0, 1], [y_max * 1.03, y_max * 1.03], "k-", lw=0.8)
        ax_v.text(0.5, y_max * 1.05, f"{sig}\np={p_val:.3g}", ha="center", fontsize=8)

        # Histogram
        ax_h = fig.add_subplot(gs[0, col * 2 + 1])
        bins = np.linspace(
            min(all_df[metric].quantile(0.01), 0),
            all_df[metric].quantile(0.99), 40
        )
        for cond in ["DREADD", "GFP"]:
            vals = all_df.loc[all_df["condition"] == cond, metric].dropna()
            ax_h.hist(vals, bins=bins, alpha=0.55, color=COLORS[cond],
                      label=cond, density=True)
        ax_h.set_xlabel(f"{label} ({unit})" if unit else label)
        ax_h.set_ylabel("Density")
        ax_h.set_title(f"{label} distribution")
        ax_h.legend(fontsize=8)

    # ── Row 1: split by injection (Gi vs Gq) ─────────────────────────────────
    for col, (metric, label, unit) in enumerate([
        ("axon_diam (um)", "Axon diameter", "µm"),
        ("gratio",         "G-ratio",       ""),
    ]):
        ax_v = fig.add_subplot(gs[1, col * 2])
        groups_inj = []
        tick_labels = []
        tick_pos = []
        pos = 0
        for cond in ["DREADD", "GFP"]:
            for inj in ["Gi", "Gq"]:
                vals = all_df.loc[(all_df["condition"] == cond) & (all_df["injection"] == inj), metric].dropna().values
                if len(vals) > 0:
                    groups_inj.append((pos, vals, cond, inj))
                    tick_labels.append(f"{cond}\n{inj}")
                    tick_pos.append(pos)
                    pos += 1
            pos += 0.4  # gap between conditions

        for pos_i, vals, cond, inj in groups_inj:
            vp = ax_v.violinplot([vals], positions=[pos_i], showmedians=True, showextrema=False)
            vp["bodies"][0].set_facecolor(INJ_COLORS[inj])
            vp["bodies"][0].set_alpha(0.6)
            vp["cmedians"].set_color("black")

        ax_v.set_xticks(tick_pos); ax_v.set_xticklabels(tick_labels, fontsize=7)
        ax_v.set_ylabel(f"{label} ({unit})" if unit else label)
        ax_v.set_title(f"{label}\nby condition × injection")

        # Histogram split by injection
        ax_h = fig.add_subplot(gs[1, col * 2 + 1])
        bins = np.linspace(
            min(all_df[metric].quantile(0.01), 0),
            all_df[metric].quantile(0.99), 40
        )
        linestyles = {"DREADD": "-", "GFP": "--"}
        for cond in ["DREADD", "GFP"]:
            for inj in ["Gi", "Gq"]:
                vals = all_df.loc[(all_df["condition"] == cond) & (all_df["injection"] == inj), metric].dropna()
                if len(vals) > 0:
                    ax_h.hist(vals, bins=bins, alpha=0.45,
                              color=INJ_COLORS[inj],
                              linestyle=linestyles[cond],
                              label=f"{cond} {inj}", density=True, histtype="step", lw=1.5)
        ax_h.set_xlabel(f"{label} ({unit})" if unit else label)
        ax_h.set_ylabel("Density")
        ax_h.set_title(f"{label} by injection")
        ax_h.legend(fontsize=7)

    out = OUT_DIR / "part2_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # Per-animal summary CSV
    summary = (
        all_df.groupby(["animal", "condition", "injection", "genotype"])
        .agg(
            n_axons      =("gratio", "count"),
            mean_gratio  =("gratio", "mean"),
            median_gratio=("gratio", "median"),
            mean_diam    =("axon_diam (um)", "mean"),
            median_diam  =("axon_diam (um)", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "part2_per_animal_summary.csv", index=False)
    print(f"  Per-animal summary saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Step 1: Prepare symlinks ===")
    prepare_seg_symlinks()

    print("\n=== Step 2: Run morphometrics ===")
    run_morphometrics()

    print("\n=== Step 3: Part 1 — ADS vs Manual ===")
    plot_part1()

    print("\n=== Step 4: Part 2 — All-axon distributions ===")
    plot_part2()

    print("\nDone. Results in:", OUT_DIR)
