#!/usr/bin/env python3
"""
Visual comparison: ADS segmentation vs Manual ROI annotations.

For each of the 5 Part 1 images, produces:
  1. Full-image panel: input + ADS mask overlay + manual circles
  2. Zoomed-patch panel: 15 matched axon pairs side by side

Also produces an improved Part 2 figure with basic quality filtering
(gratio < 0.98 and axon_diam > 0.15 µm) to remove artifact detections.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.spatial import KDTree
from scipy import stats
from pathlib import Path

BASE_DIR        = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve")
OUT_DIR         = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Optic_Nerve/analysis")
PIXEL_SIZE_NM   = 4.93   # nm/px in input.png
PIXEL_SIZE_UM   = 0.00493
MATCH_THRESH_PX = 75

PART1_IMAGES = ["image_706", "image_2299", "image_2336", "image_2696", "image_6107"]

TARGET_IMAGES = [
    "image_706",  "image_963",  "image_1812", "image_2299", "image_2336",
    "image_2696", "image_4090", "image_5087", "image_6107", "image_6294",
    "image_7422", "image_7559", "image_8969", "image_9408",
]

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


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_ads(img_id):
    df = pd.read_csv(BASE_DIR / img_id / "input_Morphometrics.csv", index_col=0)
    border = df["image_border_touching"].fillna(True).astype(bool)
    return df[~border].copy()


def load_manual(img_id):
    num = img_id.replace("image_", "")
    df = pd.read_csv(BASE_DIR / f"Results_{num}.csv", index_col=0)
    rows = df.to_dict("records")
    records = []
    for i in range(0, len(rows) - 1, 2):
        outer, inner = rows[i], rows[i + 1]
        if float(inner["Area"]) > float(outer["Area"]):
            outer, inner = inner, outer
        fiber_nm2 = float(outer["Area"])
        axon_nm2  = float(inner["Area"])
        x_nm = (float(outer["X"]) + float(inner["X"])) / 2
        y_nm = (float(outer["Y"]) + float(inner["Y"])) / 2
        fiber_um2 = fiber_nm2 / 1e6
        axon_um2  = axon_nm2  / 1e6
        records.append({
            "x_px": x_nm / PIXEL_SIZE_NM,
            "y_px": y_nm / PIXEL_SIZE_NM,
            "axon_area_um2":  axon_um2,
            "fiber_area_um2": fiber_um2,
            "axon_r_px":  math.sqrt(axon_nm2  / math.pi) / PIXEL_SIZE_NM,
            "fiber_r_px": math.sqrt(fiber_nm2 / math.pi) / PIXEL_SIZE_NM,
            "axon_diam_um": 2 * math.sqrt(axon_um2 / math.pi),
            "gratio": math.sqrt(axon_um2 / fiber_um2),
        })
    return pd.DataFrame(records)


def match(ads_df, manual_df):
    tree = KDTree(ads_df[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(manual_df[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH_PX
    return mask, idxs, dists


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FULL-IMAGE OVERLAY panels (one file per Part-1 image)
# ═══════════════════════════════════════════════════════════════════════════════

DOWNSAMPLE = 4   # show 1/4 res for the full-image panels

def make_overlay_panel(img_id):
    meta = METADATA[img_id]
    img_dir = BASE_DIR / img_id

    raw      = np.array(Image.open(img_dir / "input.png").convert("RGB"))
    seg_axon = np.array(Image.open(img_dir / "seg_axon.png").convert("L"))
    seg_mye  = np.array(Image.open(img_dir / "seg_myelin.png").convert("L"))

    ads_df    = load_ads(img_id)
    manual_df = load_manual(img_id)
    mask, idxs, dists = match(ads_df, manual_df)

    H, W = raw.shape[:2]

    # Build colored ADS mask: axon=red, myelin=blue
    overlay = raw.copy().astype(float)
    axon_mask = seg_axon > 128
    mye_mask  = (seg_mye  > 128) & ~axon_mask
    overlay[axon_mask] = overlay[axon_mask] * 0.4 + np.array([255, 60, 60]) * 0.6
    overlay[mye_mask]  = overlay[mye_mask]  * 0.4 + np.array([60, 120, 255]) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"{img_id} — {meta['animal']} {meta['side']} ({meta['condition']}, {meta['injection']})",
        fontsize=11, fontweight="bold"
    )

    ds = DOWNSAMPLE
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    # Panel A: raw input
    axes[0].imshow(raw[::ds, ::ds], cmap="gray")
    axes[0].set_title("Raw input image")

    # Panel B: ADS segmentation overlay
    axes[1].imshow(overlay[::ds, ::ds])
    axes[1].set_title("ADS segmentation\n(red=axon, blue=myelin)")

    # Panel C: manual ROI circles + ADS centroids
    axes[2].imshow(raw[::ds, ::ds], cmap="gray")

    for _, row in manual_df.iterrows():
        cx = row["x_px"] / ds
        cy = row["y_px"] / ds
        axon_r  = row["axon_r_px"]  / ds
        fiber_r = row["fiber_r_px"] / ds
        axes[2].add_patch(mpatches.Circle((cx, cy), axon_r,
            fill=False, edgecolor="#00dd00", lw=0.6, alpha=0.8))
        axes[2].add_patch(mpatches.Circle((cx, cy), fiber_r,
            fill=False, edgecolor="#ffdd00", lw=0.6, alpha=0.8))

    matched_ads = ads_df.iloc[idxs[mask]]
    axes[2].scatter(
        ads_df["x0 (px)"].values / ds,
        ads_df["y0 (px)"].values / ds,
        s=2, c="red", alpha=0.4, linewidths=0
    )
    axes[2].scatter(
        matched_ads["x0 (px)"].values / ds,
        matched_ads["y0 (px)"].values / ds,
        s=6, c="cyan", alpha=0.8, linewidths=0
    )
    axes[2].set_title("Manual ROIs (green=axon, yellow=fiber)\n+ ADS centroids (red) / matched (cyan)")
    axes[2].set_xlim(0, W / ds); axes[2].set_ylim(H / ds, 0)

    legend_elems = [
        mpatches.Patch(color="#00dd00", label="Manual axon boundary"),
        mpatches.Patch(color="#ffdd00", label="Manual fiber boundary"),
        mpatches.Patch(color="red",     label="All ADS axons"),
        mpatches.Patch(color="cyan",    label="Matched ADS axons"),
    ]
    axes[2].legend(handles=legend_elems, loc="lower right", fontsize=7)

    plt.tight_layout()
    out = OUT_DIR / f"overlay_{img_id}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ZOOMED PATCH PANEL — matched axon pairs
# ═══════════════════════════════════════════════════════════════════════════════

PATCH_PX = 120   # half-size of patch in input.png pixels

def make_patch_panel(img_id, n_patches=15):
    img_dir = BASE_DIR / img_id
    meta    = METADATA[img_id]

    raw      = np.array(Image.open(img_dir / "input.png").convert("L"))
    seg_axon = np.array(Image.open(img_dir / "seg_axon.png").convert("L"))
    seg_mye  = np.array(Image.open(img_dir / "seg_myelin.png").convert("L"))
    H, W = raw.shape

    ads_df    = load_ads(img_id)
    manual_df = load_manual(img_id)
    mask, idxs, dists = match(ads_df, manual_df)

    matched_manual = manual_df[mask].reset_index(drop=True)
    matched_ads    = ads_df.iloc[idxs[mask]].reset_index(drop=True)
    matched_dists  = dists[mask]

    # Sort by distance so we show the best matches first
    order = np.argsort(matched_dists)
    n = min(n_patches, len(order))

    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 7))
    fig.suptitle(
        f"{img_id} — {meta['animal']} {meta['side']} ({meta['condition']}, {meta['injection']})\n"
        f"Zoomed matched pairs (best {n} by centroid distance)",
        fontsize=10, fontweight="bold"
    )
    row_labels = ["Raw image", "ADS mask", "Manual ROI"]

    for col, idx in enumerate(order[:n]):
        cx_m = int(matched_manual.iloc[idx]["x_px"])
        cy_m = int(matched_manual.iloc[idx]["y_px"])
        cx_a = int(matched_ads.iloc[idx]["x0 (px)"])
        cy_a = int(matched_ads.iloc[idx]["y0 (px)"])
        # centre patch on midpoint
        cx = (cx_m + cx_a) // 2
        cy = (cy_m + cy_a) // 2

        r = PATCH_PX
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)

        patch_raw  = raw[y0:y1, x0:x1]
        patch_axon = seg_axon[y0:y1, x0:x1]
        patch_mye  = seg_mye[y0:y1, x0:x1]

        # Row 0: raw
        axes[0, col].imshow(patch_raw, cmap="gray", vmin=0, vmax=255)
        axes[0, col].axis("off")

        # Row 1: ADS colored overlay
        rgb = np.stack([patch_raw] * 3, axis=-1).astype(float)
        am = patch_axon > 128
        mm = (patch_mye > 128) & ~am
        rgb[am] = rgb[am] * 0.35 + np.array([255, 60, 60]) * 0.65
        rgb[mm] = rgb[mm] * 0.35 + np.array([60, 120, 255]) * 0.65
        axes[1, col].imshow(np.clip(rgb, 0, 255).astype(np.uint8))
        # mark ADS centroid
        axes[1, col].plot(cx_a - x0, cy_a - y0, "r+", ms=8, mew=1.5)
        axes[1, col].axis("off")

        # Row 2: raw + manual ROI circles
        axes[2, col].imshow(patch_raw, cmap="gray", vmin=0, vmax=255)
        axon_r  = matched_manual.iloc[idx]["axon_r_px"]
        fiber_r = matched_manual.iloc[idx]["fiber_r_px"]
        pcx = cx_m - x0
        pcy = cy_m - y0
        axes[2, col].add_patch(mpatches.Circle(
            (pcx, pcy), axon_r,  fill=False, edgecolor="#00dd00", lw=1.2))
        axes[2, col].add_patch(mpatches.Circle(
            (pcx, pcy), fiber_r, fill=False, edgecolor="#ffdd00", lw=1.2))
        # annotate with g-ratio comparison
        gr_m = matched_manual.iloc[idx]["gratio"]
        gr_a = matched_ads.iloc[idx]["gratio"]
        axes[2, col].set_title(
            f"M:{gr_m:.2f}\nA:{gr_a:.2f}", fontsize=6, pad=2)
        axes[2, col].axis("off")

        if col == 0:
            for row_i, lbl in enumerate(row_labels):
                axes[row_i, col].set_ylabel(lbl, fontsize=8, rotation=90, labelpad=4)

    plt.tight_layout()
    out = OUT_DIR / f"patches_{img_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FILTERED PART-2 DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Biologically plausible range for optic nerve myelinated axons
GRATIO_MAX  = 0.98   # exclude unmyelinated / artifact
DIAM_MIN_UM = 0.10   # exclude sub-resolution detections

def plot_filtered_distributions():
    frames = []
    for img_id in TARGET_IMAGES:
        df = load_ads(img_id)
        df["image_id"] = img_id
        for k, v in METADATA[img_id].items():
            df[k] = v
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    total_before = len(all_df)
    filt = (all_df["gratio"] < GRATIO_MAX) & (all_df["axon_diam (um)"] >= DIAM_MIN_UM)
    filt_df = all_df[filt].copy()
    total_after = len(filt_df)
    print(f"  Filtering: {total_before} → {total_after} axons "
          f"({total_before - total_after} removed, "
          f"{(total_before-total_after)/total_before*100:.1f}%)")
    for cond, grp in filt_df.groupby("condition"):
        print(f"    {cond}: {len(grp)} axons, {grp['image_id'].nunique()} images")

    COLORS = {"DREADD": "#2b7bb9", "GFP": "#d62728"}
    INJ_COLORS = {"Gi": "#e8862a", "Gq": "#7b4da6"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Optic Nerve — ADS morphometrics (filtered: gratio < {GRATIO_MAX}, diam ≥ {DIAM_MIN_UM} µm)\n"
        f"{total_after:,} axons across 14 images",
        fontsize=12, fontweight="bold"
    )

    # ── Row 0: DREADD vs GFP ─────────────────────────────────────────────────
    for col, (metric, label, unit) in enumerate([
        ("axon_diam (um)", "Axon diameter", "µm"),
        ("gratio",         "G-ratio",       ""),
    ]):
        ax = axes[0, col]
        groups = [filt_df.loc[filt_df["condition"] == c, metric].dropna().values
                  for c in ["DREADD", "GFP"]]
        vp = ax.violinplot(groups, positions=[0, 1], showmedians=True, showextrema=True)
        for patch, color in zip(vp["bodies"], [COLORS["DREADD"], COLORS["GFP"]]):
            patch.set_facecolor(color); patch.set_alpha(0.65)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_lw(2)

        # overlay individual-animal medians as dots
        for i, cond in enumerate(["DREADD", "GFP"]):
            for animal, grp in filt_df[filt_df["condition"] == cond].groupby("animal"):
                med = grp[metric].median()
                ax.scatter(i, med, s=30, zorder=5, color="black", alpha=0.6)

        ax.set_xticks([0, 1]); ax.set_xticklabels(["DREADD", "GFP"])
        ax.set_ylabel(f"{label} ({unit})" if unit else label)
        ax.set_title(f"{label} — DREADD vs GFP")

        u, p = stats.mannwhitneyu(*groups, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_max = max(np.nanpercentile(d, 99) for d in groups)
        ax.plot([0, 1], [y_max * 1.04, y_max * 1.04], "k-", lw=0.8)
        ax.text(0.5, y_max * 1.07, f"{sig}  p={p:.2e}", ha="center", fontsize=8)

    # Histogram overlay
    ax = axes[0, 2]
    for metric, label in [("axon_diam (um)", "diam"), ("gratio", "g-ratio")]:
        pass  # handled below

    for cond in ["DREADD", "GFP"]:
        vals = filt_df.loc[filt_df["condition"] == cond, "gratio"].dropna()
        ax.hist(vals, bins=50, alpha=0.5, density=True,
                color=COLORS[cond], label=cond)
    ax.set_xlabel("G-ratio"); ax.set_ylabel("Density")
    ax.set_title("G-ratio distribution\nDREADD vs GFP")
    ax.legend(fontsize=9)

    # ── Row 1: split by injection ─────────────────────────────────────────────
    for col, (metric, label, unit) in enumerate([
        ("axon_diam (um)", "Axon diameter", "µm"),
        ("gratio",         "G-ratio",       ""),
    ]):
        ax = axes[1, col]
        combos = [("DREADD", "Gi"), ("DREADD", "Gq"), ("GFP", "Gi"), ("GFP", "Gq")]
        tick_pos, tick_labels, vdata = [], [], []
        for pos_i, (cond, inj) in enumerate(combos):
            vals = filt_df.loc[(filt_df["condition"] == cond) & (filt_df["injection"] == inj), metric].dropna().values
            if len(vals) == 0:
                continue
            vdata.append((pos_i, vals, inj))
            tick_pos.append(pos_i); tick_labels.append(f"{cond}\n{inj}")

        for pos_i, vals, inj in vdata:
            vp = ax.violinplot([vals], positions=[pos_i], showmedians=True, showextrema=True)
            vp["bodies"][0].set_facecolor(INJ_COLORS[inj]); vp["bodies"][0].set_alpha(0.65)
            vp["cmedians"].set_color("black"); vp["cmedians"].set_lw(2)

        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel(f"{label} ({unit})" if unit else label)
        ax.set_title(f"{label} — by condition × injection")

    # Axon diam histogram split by injection
    ax = axes[1, 2]
    linestyles = {"DREADD": "-", "GFP": "--"}
    for cond in ["DREADD", "GFP"]:
        for inj in ["Gi", "Gq"]:
            vals = filt_df.loc[(filt_df["condition"] == cond) & (filt_df["injection"] == inj), "axon_diam (um)"].dropna()
            if len(vals) == 0: continue
            ax.hist(vals, bins=50, density=True, histtype="step",
                    lw=1.8, color=INJ_COLORS[inj], linestyle=linestyles[cond],
                    label=f"{cond} {inj}", alpha=0.9)
    ax.set_xlabel("Axon diameter (µm)"); ax.set_ylabel("Density")
    ax.set_title("Axon diameter distribution\nby condition × injection")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "part2_distributions_filtered.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")

    # Save filtered summary
    summary = (
        filt_df.groupby(["animal", "condition", "injection", "genotype"])
        .agg(
            n_axons       =("gratio", "count"),
            mean_gratio   =("gratio", "mean"),
            median_gratio =("gratio", "median"),
            std_gratio    =("gratio", "std"),
            mean_diam_um  =("axon_diam (um)", "mean"),
            median_diam_um=("axon_diam (um)", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "part2_filtered_summary.csv", index=False)
    return filt_df, summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Full-image overlay panels ===")
    for img_id in PART1_IMAGES:
        make_overlay_panel(img_id)

    print("\n=== Zoomed patch panels ===")
    for img_id in PART1_IMAGES:
        make_patch_panel(img_id)

    print("\n=== Filtered Part-2 distributions ===")
    filt_df, summary = plot_filtered_distributions()
    print("\nFiltered per-animal summary:")
    print(summary.to_string(index=False))

    print("\nDone.")
