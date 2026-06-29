#!/usr/bin/env python3
"""
Compare circle vs ellipse axon shape for ADS morphometrics.
Applies the same two-stage cleaning (spatial crop + size filter) to both.
Produces:
  - gratio_bar_circle_vs_ellipse.png   — 3-bar chart: Manual / ADS circle / ADS ellipse
  - scatter_circle_vs_ellipse.png      — side-by-side scatter for all 14 images
  - circle_vs_ellipse_summary.csv      — per-image accuracy metrics for both modes
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

COLOR_COND = {"DREADD": "#2b7bb9", "GFP": "#d62728"}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ads_clean(img_id, mode="circle"):
    fname = "input_Morphometrics.csv" if mode == "circle" else "input_Morphometrics_ellipse.csv"
    df = pd.read_csv(BASE / img_id / fname, index_col=0)
    border = df["image_border_touching"].fillna(True).astype(bool)
    df = df[~border].copy()
    if img_id in SPATIAL_CROPS:
        crop = SPATIAL_CROPS[img_id]
        mask = pd.Series(False, index=df.index)
        if "x_max" in crop: mask |= df["x0 (px)"] < crop["x_max"]
        if "y_min" in crop: mask |= df["y0 (px)"] > crop["y_min"]
        df = df[~mask].copy()
    df = df[df["axon_diam (um)"] >= DIAM_MIN_UM].copy()
    return df


def load_manual(img_id):
    num = img_id.replace("image_", "")
    raw = pd.read_csv(BASE / f"Results_{num}.csv", index_col=0).to_dict("records")
    records = []
    for i in range(0, len(raw) - 1, 2):
        o, inn = raw[i], raw[i + 1]
        if float(inn["Area"]) > float(o["Area"]): o, inn = inn, o
        fa = float(o["Area"]) / 1e6
        aa = float(inn["Area"]) / 1e6
        xnm = (float(o["X"]) + float(inn["X"])) / 2
        ynm = (float(o["Y"]) + float(inn["Y"])) / 2
        records.append({
            "x_px": xnm / PIXEL_SIZE_NM, "y_px": ynm / PIXEL_SIZE_NM,
            "gratio": math.sqrt(aa / fa),
            "axon_diam_um": 2 * math.sqrt(aa / math.pi),
        })
    return pd.DataFrame(records)


def match_gratio(ads_df, man_df):
    tree = KDTree(ads_df[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(man_df[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH
    return man_df["gratio"].values[mask], ads_df["gratio"].iloc[idxs[mask]].values, mask.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 3-bar chart: Manual / Circle / Ellipse
# ═══════════════════════════════════════════════════════════════════════════════

def plot_3bar():
    manual_m, circle_m, ellipse_m = [], [], []
    for img_id in TARGET_IMAGES:
        man = load_manual(img_id)
        circ = load_ads_clean(img_id, "circle")
        elli = load_ads_clean(img_id, "ellipse")
        mg, cg, _ = match_gratio(circ, man)
        mg2, eg, _ = match_gratio(elli, man)
        if len(mg) == 0: continue
        manual_m.append(mg.mean())
        circle_m.append(cg.mean())
        ellipse_m.append(eg.mean())

    manual_m  = np.array(manual_m)
    circle_m  = np.array(circle_m)
    ellipse_m = np.array(ellipse_m)

    def st(a): return a.mean(), a.std(), np.median(a)

    fig, ax = plt.subplots(figsize=(8, 7))
    rng = np.random.default_rng(42)
    BAR_W = 0.5

    entries = [
        (0, manual_m,  "white",   "Manual",         "black"),
        (1, circle_m,  "#a78fd0", "ADS\n(circle)",  "black"),
        (2, ellipse_m, "#6dbf8a", "ADS\n(ellipse)", "black"),
    ]

    for pos, means, fc, label, ec in entries:
        mean_v, sd_v, med_v = st(means)
        ax.bar(pos, mean_v, width=BAR_W, color=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.errorbar(pos, mean_v, yerr=sd_v, fmt="none", color=ec,
                    capsize=6, capthick=1.5, lw=1.5, zorder=4)
        jitter = rng.uniform(-0.07, 0.07, len(means))
        ax.scatter(pos + jitter, means, s=50, color="black", zorder=5, linewidths=0)
        stats_txt = f"mean    {mean_v:.3f}\nSD       {sd_v:.3f}\nmedian {med_v:.3f}"
        ax.text(pos, 0.603, stats_txt, ha="center", va="bottom", fontsize=8.5,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray",
                          alpha=0.9, lw=0.8), zorder=6)
        print(f"{label.replace(chr(10),' ')}: mean={mean_v:.3f}  SD={sd_v:.3f}  median={med_v:.3f}")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Manual", "ADS\n(circle)", "ADS\n(ellipse)"], fontsize=12)
    ax.set_ylabel("average g-ratio", fontsize=13)
    ax.set_title("Average g-ratio\nManual vs ADS circle vs ADS ellipse", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.600, max(ellipse_m.max(), circle_m.max(), manual_m.max()) + 0.07)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.025))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    out = OUT_DIR / "gratio_bar_circle_vs_ellipse.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — side-by-side scatter: circle (top) vs ellipse (bottom), all 14
# ═══════════════════════════════════════════════════════════════════════════════

AXIS_LIM = (0.35, 1.00)

def one_scatter(ax, img_id, mode):
    meta = METADATA[img_id]
    ads  = load_ads_clean(img_id, mode)
    man  = load_manual(img_id)
    mg, ag, n = match_gratio(ads, man)
    if n < 2: ax.set_title(f"{img_id}\nno matches"); return

    r, _  = stats.pearsonr(mg, ag)
    mae   = np.abs(ag - mg).mean()
    pct   = mae / mg.mean() * 100

    lo, hi = AXIS_LIM
    in_r = (mg >= lo) & (mg <= hi) & (ag >= lo) & (ag <= hi)
    ax.scatter(mg[in_r], ag[in_r], s=14, alpha=0.6,
               color=COLOR_COND[meta["condition"]], linewidths=0)
    if (~in_r).sum():
        ax.scatter(np.clip(mg[~in_r], lo+.01, hi-.01),
                   np.clip(ag[~in_r], lo+.01, hi-.01),
                   marker="^", s=16, color="gray", alpha=0.5, linewidths=0)
    ax.plot(AXIS_LIM, AXIS_LIM, "k--", lw=0.7, alpha=0.4)
    ax.set_xlim(AXIS_LIM); ax.set_ylim(AXIS_LIM); ax.set_aspect("equal")
    ax.set_title(f"{img_id}\n{meta['animal']} · {meta['condition']}", fontsize=7, pad=2)
    ax.text(0.04, 0.96, f"n={n} r={r:.2f}\nMAE={mae:.3f}({pct:.1f}%)",
            transform=ax.transAxes, va="top", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, lw=0))
    ax.set_xlabel("Manual g-ratio", fontsize=6)
    ax.set_ylabel("ADS g-ratio", fontsize=6)
    ax.tick_params(labelsize=5)
    return r, mae, pct, n


def plot_scatter_comparison():
    fig, axes = plt.subplots(2, 14, figsize=(28, 5.5))
    fig.suptitle(
        "ADS vs Manual g-ratio — Circle (top) vs Ellipse (bottom) — all 14 images\n"
        "Blue = DREADD  ·  Red = GFP",
        fontsize=11, fontweight="bold", y=1.02)

    rows = []
    for col, img_id in enumerate(TARGET_IMAGES):
        for row, mode in enumerate(["circle", "ellipse"]):
            res = one_scatter(axes[row, col], img_id, mode)
            if res and row == 0:
                r_c, mae_c, pct_c, n_c = res
            if res and row == 1:
                r_e, mae_e, pct_e, n_e = res
                rows.append({"image": img_id,
                             "n_circle": n_c, "r_circle": round(r_c, 3),
                             "mae_circle": round(mae_c, 4), "pct_circle": round(pct_c, 2),
                             "n_ellipse": n_e, "r_ellipse": round(r_e, 3),
                             "mae_ellipse": round(mae_e, 4), "pct_ellipse": round(pct_e, 2)})
        axes[0, col].set_ylabel("Circle\nADS g-ratio", fontsize=6)
        axes[1, col].set_ylabel("Ellipse\nADS g-ratio", fontsize=6)

    # row labels
    axes[0, 0].set_ylabel("Circle\nADS g-ratio", fontsize=7)
    axes[1, 0].set_ylabel("Ellipse\nADS g-ratio", fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scatter_circle_vs_ellipse.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "circle_vs_ellipse_summary.csv", index=False)
    print("\nPer-image comparison:")
    print(df[["image","r_circle","mae_circle","pct_circle","r_ellipse","mae_ellipse","pct_ellipse"]].to_string(index=False))
    print(f"\nMean MAE — Circle: {df['mae_circle'].mean():.4f}  Ellipse: {df['mae_ellipse'].mean():.4f}")
    print(f"Mean r   — Circle: {df['r_circle'].mean():.3f}   Ellipse: {df['r_ellipse'].mean():.3f}")
    print(f"Saved → scatter_circle_vs_ellipse.png")


if __name__ == "__main__":
    print("=== 3-bar chart ===")
    plot_3bar()
    print("\n=== Scatter comparison ===")
    plot_scatter_comparison()
    print("\nDone.")
