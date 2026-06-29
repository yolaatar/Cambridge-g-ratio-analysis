#!/usr/bin/env python3
"""
Two figures:

1. overall_scatter.png
   All matched axon pairs from all 14 images pooled into one scatter.
   Left = circle, right = ellipse. Colored by condition (DREADD/GFP).

2. circle_vs_ellipse_tradeoff.png
   Left panel  = mean g-ratio (bias vs manual) — ellipse wins here
   Right panel = MAE% per image (error per axon) — circle wins here
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
    "image_706":  {"condition": "DREADD"}, "image_963":  {"condition": "DREADD"},
    "image_1812": {"condition": "DREADD"}, "image_2299": {"condition": "DREADD"},
    "image_2696": {"condition": "DREADD"}, "image_2336": {"condition": "GFP"},
    "image_4090": {"condition": "GFP"},    "image_5087": {"condition": "GFP"},
    "image_6107": {"condition": "GFP"},    "image_6294": {"condition": "GFP"},
    "image_7422": {"condition": "GFP"},    "image_7559": {"condition": "GFP"},
    "image_8969": {"condition": "GFP"},    "image_9408": {"condition": "GFP"},
}

COLOR_COND = {"DREADD": "#2b7bb9", "GFP": "#d62728"}


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
    return df[df["axon_diam (um)"] >= DIAM_MIN_UM].copy()


def load_manual(img_id):
    num = img_id.replace("image_", "")
    raw = pd.read_csv(BASE / f"Results_{num}.csv", index_col=0).to_dict("records")
    records = []
    for i in range(0, len(raw) - 1, 2):
        o, inn = raw[i], raw[i + 1]
        if float(inn["Area"]) > float(o["Area"]): o, inn = inn, o
        fa, aa = float(o["Area"]) / 1e6, float(inn["Area"]) / 1e6
        xnm = (float(o["X"]) + float(inn["X"])) / 2
        ynm = (float(o["Y"]) + float(inn["Y"])) / 2
        records.append({"x_px": xnm / PIXEL_SIZE_NM, "y_px": ynm / PIXEL_SIZE_NM,
                        "gratio": math.sqrt(aa / fa)})
    return pd.DataFrame(records)


def match_gratio(ads_df, man_df):
    tree = KDTree(ads_df[["x0 (px)", "y0 (px)"]].values)
    dists, idxs = tree.query(man_df[["x_px", "y_px"]].values, k=1)
    mask = dists < MATCH_THRESH
    return man_df["gratio"].values[mask], ads_df["gratio"].iloc[idxs[mask]].values


# ── Pool all matched pairs ────────────────────────────────────────────────────

all_c = {"manual": [], "ads": [], "condition": []}
all_e = {"manual": [], "ads": [], "condition": []}

for img_id in TARGET_IMAGES:
    cond = METADATA[img_id]["condition"]
    man  = load_manual(img_id)
    mg_c, ag_c = match_gratio(load_ads_clean(img_id, "circle"),  man)
    mg_e, ag_e = match_gratio(load_ads_clean(img_id, "ellipse"), man)
    all_c["manual"].extend(mg_c); all_c["ads"].extend(ag_c)
    all_c["condition"].extend([cond] * len(mg_c))
    all_e["manual"].extend(mg_e); all_e["ads"].extend(ag_e)
    all_e["condition"].extend([cond] * len(mg_e))

for d in (all_c, all_e):
    d["manual"] = np.array(d["manual"])
    d["ads"]    = np.array(d["ads"])

print(f"Total matched pairs — circle: {len(all_c['manual'])}  ellipse: {len(all_e['manual'])}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — overall scatter
# ══════════════════════════════════════════════════════════════════════════════

LIMS = (0.35, 1.00)

def scatter_panel(ax, data, title):
    for cond in ["DREADD", "GFP"]:
        idx = np.array(data["condition"]) == cond
        ax.scatter(data["manual"][idx], data["ads"][idx],
                   s=12, alpha=0.4, linewidths=0, color=COLOR_COND[cond],
                   label=cond, zorder=2)

    ax.plot(LIMS, LIMS, "k--", lw=1.0, alpha=0.5, zorder=1)
    r, _  = stats.pearsonr(data["manual"], data["ads"])
    bias  = (data["ads"] - data["manual"]).mean()
    mae   = np.abs(data["ads"] - data["manual"]).mean()
    pct   = mae / data["manual"].mean() * 100

    ax.text(0.04, 0.97,
            f"n = {len(data['manual'])} axons\n"
            f"r = {r:.3f}\n"
            f"bias = {bias:+.4f}\n"
            f"MAE = {mae:.4f}  ({pct:.1f}%)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray",
                      alpha=0.9, lw=0.8))
    ax.set_xlim(LIMS); ax.set_ylim(LIMS); ax.set_aspect("equal")
    ax.set_xlabel("Manual g-ratio", fontsize=11)
    ax.set_ylabel("ADS g-ratio", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, markerscale=2, framealpha=0.8)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("ADS vs Manual g-ratio — all 14 optic nerve images pooled",
             fontsize=13, fontweight="bold", y=1.01)
scatter_panel(axes[0], all_c, "Circle")
scatter_panel(axes[1], all_e, "Ellipse")
plt.tight_layout()
out1 = OUT_DIR / "overall_scatter.png"
fig.savefig(out1, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out1.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — trade-off: what each method does better
# ══════════════════════════════════════════════════════════════════════════════

summary = pd.read_csv(OUT_DIR / "circle_vs_ellipse_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
fig.suptitle("Circle vs Ellipse — what each method does better",
             fontsize=13, fontweight="bold", y=1.01)

rng = np.random.default_rng(42)

# LEFT: mean g-ratio (bias) — pooled
ax = axes[0]
bias_c = (all_c["ads"] - all_c["manual"]).mean()
bias_e = (all_e["ads"] - all_e["manual"]).mean()
bars = ax.bar([0, 1], [bias_c, bias_e], width=0.5,
              color=["#a78fd0", "#6dbf8a"], edgecolor="black", linewidth=1.2)
ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Circle", "Ellipse"], fontsize=12)
ax.set_ylabel("Mean bias  (ADS − Manual g-ratio)", fontsize=11)
ax.set_title("Systematic bias\n(closer to 0 = better)", fontsize=11, fontweight="bold")
for bar, val in zip(bars, [bias_c, bias_e]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f"{val:+.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(-0.005, max(bias_c, bias_e) + 0.01)

# RIGHT: per-image MAE% — jitter dots + mean bar
ax = axes[1]
for xi, (col_name, color) in enumerate(zip(["pct_circle", "pct_ellipse"],
                                            ["#a78fd0", "#6dbf8a"])):
    vals = summary[col_name].values
    mean_v = vals.mean()
    ax.bar(xi, mean_v, width=0.5, color=color, edgecolor="black", linewidth=1.2, zorder=2)
    ax.errorbar(xi, mean_v, yerr=vals.std(ddof=1), fmt="none",
                color="black", capsize=6, capthick=1.2, lw=1.2, zorder=4)
    jitter = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(xi + jitter, vals, s=45, color="black", zorder=5, linewidths=0)
    ax.text(xi, mean_v + vals.std(ddof=1) + 0.3, f"{mean_v:.1f}%",
            ha="center", fontsize=10, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Circle", "Ellipse"], fontsize=12)
ax.set_ylabel("MAE  (% of manual g-ratio)", fontsize=11)
ax.set_title("Per-axon error\n(lower = better)", fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 15)

plt.tight_layout()
out2 = OUT_DIR / "circle_vs_ellipse_tradeoff.png"
fig.savefig(out2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out2.name}")
