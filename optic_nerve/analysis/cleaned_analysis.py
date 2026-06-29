#!/usr/bin/env python3
"""
Cleaned analysis with two-stage artifact removal:
  1. Spatial crop   — removes dense artifact clusters in image_2299 and image_7559
  2. Size filter    — removes axons < 0.3 µm diameter across all images
                      (artifacts from model over-detection in low-contrast regions)

Then regenerates:
  - average_gratio_bar_cleaned.png
  - part1_all_images_cleaned.png   (ADS vs Manual scatter, all 14 images)
  - part2_distributions_cleaned.png
  - artifact_removal_summary.csv
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
DIAM_MIN_UM   = 0.30   # minimum axon diameter — anything below is artifact

# Spatial crop regions to EXCLUDE (x_max, y_min in input.png pixels)
# Axons with centroid inside the crop box are dropped before size filtering
SPATIAL_CROPS = {
    "image_2299": dict(x_max=1500, y_min=3500),   # bottom-left cluster
    "image_7559": dict(y_min=4300),                # bottom strip
}

TARGET_IMAGES = [
    "image_706",  "image_963",  "image_1812", "image_2299",
    "image_2336", "image_2696", "image_4090", "image_5087",
    "image_6107", "image_6294", "image_7422", "image_7559",
    "image_8969", "image_9408",
]

METADATA = {
    "image_706":  {"animal": "TKFG 19.1c", "side": "L", "condition": "DREADD", "injection": "Gi"},
    "image_963":  {"animal": "TKFG 19.1g", "side": "L", "condition": "DREADD", "injection": "Gi"},
    "image_1812": {"animal": "TKFG 19.1g", "side": "L", "condition": "DREADD", "injection": "Gi"},
    "image_2299": {"animal": "TKFG 18.1f", "side": "L", "condition": "DREADD", "injection": "Gq"},
    "image_2336": {"animal": "TKFG 17.1g", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_2696": {"animal": "TKFG 17.1g", "side": "L", "condition": "DREADD", "injection": "Gq"},
    "image_4090": {"animal": "TKFG 18.1f", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_5087": {"animal": "TKFG 19.1f", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_6107": {"animal": "TKFG 19.1c", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_6294": {"animal": "TKFG 19.1g", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_7422": {"animal": "TKFG 17.1g", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_7559": {"animal": "TKFG 19.1e", "side": "R", "condition": "GFP",    "injection": "Gi"},
    "image_8969": {"animal": "TKFG 21.1g", "side": "R", "condition": "GFP",    "injection": "Gq"},
    "image_9408": {"animal": "TKFG 18.1f", "side": "R", "condition": "GFP",    "injection": "Gq"},
}

COLOR = {"DREADD": "#2b7bb9", "GFP": "#d62728"}


# ── Core loader with cleaning ─────────────────────────────────────────────────

def load_clean(img_id) -> pd.DataFrame:
    df = pd.read_csv(BASE / img_id / "input_Morphometrics_ellipse.csv", index_col=0)
    # 1. remove border axons
    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    n_border = border.sum()

    # 2. spatial crop
    n_spatial = 0
    if img_id in SPATIAL_CROPS:
        crop = SPATIAL_CROPS[img_id]
        spatial_mask = pd.Series(True, index=df.index)
        if "x_max" in crop:
            spatial_mask &= df["x0 (px)"] < crop["x_max"]
        if "x_min" in crop:
            spatial_mask &= df["x0 (px)"] > crop["x_min"]
        if "y_min" in crop:
            spatial_mask &= df["y0 (px)"] > crop["y_min"]
        if "y_max" in crop:
            spatial_mask &= df["y0 (px)"] < crop["y_max"]
        n_spatial = spatial_mask.sum()
        df = df[~spatial_mask].copy()

    # 3. size filter
    size_mask = df["axon_diam (um)"] < DIAM_MIN_UM
    n_size = size_mask.sum()
    df = df[~size_mask].copy()

    return df, n_border, n_spatial, n_size


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
            "x_px":  xnm / PIXEL_SIZE_NM,
            "y_px":  ynm / PIXEL_SIZE_NM,
            "gratio": math.sqrt(aa / fa),
            "axon_diam_um": 2 * math.sqrt(aa / math.pi),
        })
    return pd.DataFrame(records)


def match(ads_df, manual_df):
    tree = KDTree(ads_df[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(manual_df[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH
    return mask, idxs, dists


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{'image':<15} {'raw':>6} {'border':>7} {'spatial':>8} {'size':>6} {'kept':>6} {'kept%':>6}")
    print("─" * 65)
    rows = []
    for img_id in TARGET_IMAGES:
        raw_df = pd.read_csv(BASE / img_id / "input_Morphometrics.csv", index_col=0)
        border = raw_df["image_border_touching"].fillna(True).astype(bool)
        raw_df = raw_df[~border]
        n_raw = len(raw_df)
        clean_df, nb, ns, nz = load_clean(img_id)
        n_kept = len(clean_df)
        print(f"{img_id:<15} {n_raw:>6} {nb:>7} {ns:>8} {nz:>6} {n_kept:>6} {n_kept/n_raw*100:>5.1f}%")
        rows.append({"image": img_id, "raw": n_raw, "removed_border": nb,
                     "removed_spatial": ns, "removed_size": nz, "kept": n_kept,
                     "kept_pct": round(n_kept/n_raw*100, 1)})
    print("─" * 65)
    pd.DataFrame(rows).to_csv(OUT_DIR / "artifact_removal_summary.csv", index=False)
    print("Saved artifact_removal_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Average g-ratio bar chart (cleaned)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_average_gratio():
    manual_means, ads_means = [], []
    for img_id in TARGET_IMAGES:
        ads_df, *_ = load_clean(img_id)
        man_df = load_manual(img_id)
        mask, idxs, _ = match(ads_df, man_df)
        if mask.sum() == 0:
            continue
        manual_means.append(man_df["gratio"].values[mask].mean())
        ads_means.append(ads_df["gratio"].iloc[idxs[mask]].values.mean())

    manual_means = np.array(manual_means)
    ads_means    = np.array(ads_means)

    def stats(a): return a.mean(), a.std(), np.median(a)
    mm, ms, mmed = stats(manual_means)
    am, as_, amed = stats(ads_means)

    print(f"\nManual — mean={mm:.3f}  SD={ms:.3f}  median={mmed:.3f}")
    print(f"ADS    — mean={am:.3f}  SD={as_:.3f}  median={amed:.3f}")

    fig, ax = plt.subplots(figsize=(6, 7))
    rng = np.random.default_rng(42)

    for pos, means, mean_v, sd_v, med_v, fc in [
        (0, manual_means, mm,  ms,   mmed, "white"),
        (1, ads_means,    am,  as_,  amed, "#a78fd0"),
    ]:
        ax.bar(pos, mean_v, width=0.55, color=fc, edgecolor="black", linewidth=1.5, zorder=2)
        ax.errorbar(pos, mean_v, yerr=sd_v, fmt="none", color="black",
                    capsize=6, capthick=1.5, lw=1.5, zorder=4)
        jitter = rng.uniform(-0.07, 0.07, len(means))
        ax.scatter(pos + jitter, means, s=50, color="black", zorder=5, linewidths=0)
        stats_text = f"mean    {mean_v:.3f}\nSD       {sd_v:.3f}\nmedian {med_v:.3f}"
        ax.text(pos, 0.603, stats_text, ha="center", va="bottom", fontsize=9,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray",
                          alpha=0.9, lw=0.8), zorder=6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Manual", "Auto\n(ADS)"], fontsize=13)
    ax.set_ylabel("average g-ratio", fontsize=13)
    ax.set_title("Average g-ratio\n(after artifact removal)", fontsize=14,
                 fontweight="bold", pad=10)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.600, max(ads_means.max(), manual_means.max()) + 0.06)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "average_gratio_bar_ellipse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved average_gratio_bar_ellipse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ADS vs Manual scatter, all 14 images (cleaned)
# ═══════════════════════════════════════════════════════════════════════════════

AXIS_LIM = (0.35, 1.00)

def one_scatter(ax, img_id):
    meta = METADATA[img_id]
    ads_df, *_ = load_clean(img_id)
    man_df = load_manual(img_id)
    mask, idxs, dists = match(ads_df, man_df)

    man_gr = man_df["gratio"].values[mask]
    ads_gr = ads_df["gratio"].iloc[idxs[mask]].values
    n = len(man_gr)
    if n < 2:
        ax.set_title(f"{img_id}\nno matches"); return

    r, _  = stats.pearsonr(man_gr, ads_gr)
    mae   = np.abs(ads_gr - man_gr).mean()
    pct   = mae / man_gr.mean() * 100

    lo, hi = AXIS_LIM
    in_range = (man_gr >= lo) & (man_gr <= hi) & (ads_gr >= lo) & (ads_gr <= hi)
    n_clip = n - in_range.sum()

    ax.scatter(man_gr[in_range], ads_gr[in_range],
               s=18, alpha=0.65, linewidths=0, color=COLOR[meta["condition"]])
    if n_clip:
        ax.scatter(
            np.clip(man_gr[~in_range], lo+.01, hi-.01),
            np.clip(ads_gr[~in_range], lo+.01, hi-.01),
            marker="^", s=20, color="gray", alpha=0.5, linewidths=0)

    ax.plot(AXIS_LIM, AXIS_LIM, "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(AXIS_LIM); ax.set_ylim(AXIS_LIM); ax.set_aspect("equal")
    ax.set_title(f"{img_id}\n{meta['animal']} · {meta['condition']} · {meta['injection']}",
                 fontsize=7.5, pad=3)
    label = f"n={n}  r={r:.2f}\nMAE={mae:.3f} ({pct:.1f}%)"
    if n_clip: label += f"\n({n_clip} clipped)"
    ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, lw=0))
    ax.set_xlabel("Manual g-ratio", fontsize=7)
    ax.set_ylabel("ADS g-ratio", fontsize=7)
    ax.tick_params(labelsize=6)


def plot_scatter_all():
    fig, axes = plt.subplots(2, 7, figsize=(21, 7))
    fig.suptitle(
        "ADS vs Manual g-ratio — matched axons — all 14 images  (after artifact removal)\n"
        "Blue = DREADD  ·  Red = GFP  ·  Triangles = outliers clipped to axis range",
        fontsize=11, fontweight="bold", y=1.01)
    for ax, img_id in zip(axes.flat, TARGET_IMAGES):
        one_scatter(ax, img_id)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "part1_all_images_ellipse.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved part1_all_images_ellipse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — All-axon distributions (cleaned)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_distributions():
    frames = []
    for img_id in TARGET_IMAGES:
        df, *_ = load_clean(img_id)
        df["image_id"] = img_id
        for k, v in METADATA[img_id].items():
            df[k] = v
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    print(f"\nTotal axons after cleaning: {len(all_df)}")
    for cond, grp in all_df.groupby("condition"):
        print(f"  {cond}: {len(grp)} axons from {grp['image_id'].nunique()} images")

    COLORS_C = {"DREADD": "#2b7bb9", "GFP": "#d62728"}
    INJ_C    = {"Gi": "#e8862a", "Gq": "#7b4da6"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Optic Nerve — ADS morphometrics ellipse mode (cleaned: spatial crop + diam ≥ {DIAM_MIN_UM} µm)\n"
        f"{len(all_df):,} axons across 14 images",
        fontsize=12, fontweight="bold")

    def violin_row(row_axes, metric, label, unit):
        ax_v, ax_h, ax_inj = row_axes

        # violin DREADD vs GFP
        groups = [all_df.loc[all_df["condition"] == c, metric].dropna().values
                  for c in ["DREADD", "GFP"]]
        vp = ax_v.violinplot(groups, positions=[0, 1], showmedians=True, showextrema=True)
        for patch, color in zip(vp["bodies"], [COLORS_C["DREADD"], COLORS_C["GFP"]]):
            patch.set_facecolor(color); patch.set_alpha(0.65)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_lw(2)
        for i, cond in enumerate(["DREADD", "GFP"]):
            for _, grp in all_df[all_df["condition"] == cond].groupby("animal"):
                ax_v.scatter(i, grp[metric].median(), s=35, zorder=5,
                             color="black", alpha=0.7)
        ax_v.set_xticks([0, 1]); ax_v.set_xticklabels(["DREADD", "GFP"])
        ax_v.set_ylabel(f"{label} ({unit})" if unit else label)
        ax_v.set_title(f"{label} — DREADD vs GFP")
        u, p = stats.mannwhitneyu(*groups, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_max = max(np.nanpercentile(d, 99) for d in groups)
        ax_v.set_ylim(bottom=0)
        ax_v.plot([0, 1], [y_max*1.05, y_max*1.05], "k-", lw=0.8)
        ax_v.text(0.5, y_max*1.08, f"{sig}  p={p:.2e}", ha="center", fontsize=8)

        # histogram DREADD vs GFP
        vals_all = all_df[metric].dropna()
        bins = np.linspace(vals_all.quantile(0.01), vals_all.quantile(0.99), 45)
        for cond in ["DREADD", "GFP"]:
            v = all_df.loc[all_df["condition"] == cond, metric].dropna()
            ax_h.hist(v, bins=bins, alpha=0.5, density=True,
                      color=COLORS_C[cond], label=cond)
        ax_h.set_xlabel(f"{label} ({unit})" if unit else label)
        ax_h.set_ylabel("Density")
        ax_h.set_title(f"{label} distribution")
        ax_h.legend(fontsize=9)

        # histogram by injection × condition
        ls = {"DREADD": "-", "GFP": "--"}
        for cond in ["DREADD", "GFP"]:
            for inj in ["Gi", "Gq"]:
                v = all_df.loc[(all_df["condition"] == cond) & (all_df["injection"] == inj),
                               metric].dropna()
                if len(v) == 0: continue
                ax_inj.hist(v, bins=bins, density=True, histtype="step",
                            lw=1.8, color=INJ_C[inj], linestyle=ls[cond],
                            label=f"{cond} {inj}", alpha=0.9)
        ax_inj.set_xlabel(f"{label} ({unit})" if unit else label)
        ax_inj.set_ylabel("Density")
        ax_inj.set_title(f"{label} by injection")
        ax_inj.legend(fontsize=8)

    violin_row(axes[0], "axon_diam (um)", "Axon diameter", "µm")
    violin_row(axes[1], "gratio",         "G-ratio",       "")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "part2_distributions_ellipse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved part2_distributions_ellipse.png")

    # per-animal summary
    summary = (
        all_df.groupby(["animal", "condition", "injection"])
        .agg(n=("gratio","count"), mean_gr=("gratio","mean"),
             median_gr=("gratio","median"), std_gr=("gratio","std"),
             mean_diam=("axon_diam (um)","mean"), median_diam=("axon_diam (um)","median"))
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "part2_ellipse_summary.csv", index=False)
    print("Saved part2_ellipse_summary.csv")
    print(summary.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Artifact removal summary ===")
    print_summary()

    print("\n=== Figure 1: Average g-ratio bar chart ===")
    plot_average_gratio()

    print("\n=== Figure 2: ADS vs Manual scatter (all 14 images) ===")
    plot_scatter_all()

    print("\n=== Figure 3: All-axon distributions ===")
    plot_distributions()

    print("\nAll done. Results in:", OUT_DIR)
