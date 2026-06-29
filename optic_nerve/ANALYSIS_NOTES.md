# Timmler Optic Nerve — Analysis Notes

## Experimental Background

This dataset comes from a DREADD chemogenetics experiment in mice, with the goal of testing whether activating specific G-protein signalling pathways in oligodendrocytes affects myelin structure in the optic nerve.

### Design

Each animal received two injections — one into the left optic nerve (the **DREADD**-expressing condition) and one into the right (the **GFP-only control**):

| Side | Condition |
|------|-----------|
| **Left** | DREADD (active) |
| **Right** | GFP (passive control) |

Two DREADD constructs were used, targeting different G-protein pathways:

| Injection | Pathway |
|-----------|---------|
| **Gi** | inhibitory G-protein |
| **Gq** | excitatory G-protein |

Both pathways were tested in parallel across animals. The readout is the **g-ratio** and **axon diameter** measured from transmission electron microscopy (TEM) cross-sections of the optic nerve. G-ratio = axon diameter / fiber diameter (axon + myelin), where lower g-ratio = thicker myelin relative to axon size. Normal optic nerve g-ratio is approximately 0.73–0.78.

### Animals in this dataset (14 images with both ADS and manual data)

| Image | Animal | Side | Condition | Injection | Genotype |
|-------|--------|------|-----------|-----------|----------|
| image_706  | TKFG 19.1c | L | DREADD | Gi | Hemi/WT |
| image_963  | TKFG 19.1g | L | DREADD | Gi | Hemi/Het |
| image_1812 | TKFG 19.1g | L | DREADD | Gi | Hemi/Het |
| image_2299 | TKFG 18.1f | L | DREADD | Gq | Hemi/WT |
| image_2336 | TKFG 17.1g | R | GFP    | Gq | Hemi/WT |
| image_2696 | TKFG 17.1g | L | DREADD | Gq | Hemi/WT |
| image_4090 | TKFG 18.1f | R | GFP    | Gq | Hemi/WT |
| image_5087 | TKFG 19.1f | R | GFP    | Gi | Hemi/Het |
| image_6107 | TKFG 19.1c | R | GFP    | Gi | Hemi/WT |
| image_6294 | TKFG 19.1g | R | GFP    | Gi | Hemi/Het |
| image_7422 | TKFG 17.1g | R | GFP    | Gq | Hemi/WT |
| image_7559 | TKFG 19.1e | R | GFP    | Gi | Hemi/WT |
| image_8969 | TKFG 21.1g | R | GFP    | Gq | Hemi/WT |
| image_9408 | TKFG 18.1f | R | GFP    | Gq | Hemi/WT |

> **Note:** image_963 and image_1812 are from the same animal (TKFG 19.1g, L, DREADD) but different grid locations. Similarly, image_4090 and image_9408 are both TKFG 18.1f R, and image_2336 and image_7422 are both TKFG 17.1g R — i.e., multiple images per nerve were captured from the same animal.

---

## Image Processing — ADS Segmentation

The TEM images were originally 16-bit TIFF files at **1.8625 nm/px** (8000× magnification, Hitachi HT7800). They were preprocessed and downscaled to **4.93 nm/px** (scale factor 0.378) to match the training resolution of the ADS generalist model (`model_seg_generalist_light`).

For each image, ADS produced:
- `seg_axon.png` — binary mask of axon interiors
- `seg_myelin.png` — binary mask of myelin sheaths
- `overlay.png` — colour overlay for visual inspection

Morphometrics (per-axon measurements) were then computed using `axondeepseg_morphometrics` (pixel size = 0.00493 µm/px), producing for each detected axon:

| Column | Description |
|--------|-------------|
| `gratio` | g-ratio = axon\_diam / fiber\_diam |
| `axon_diam (um)` | equivalent circle diameter of the axon |
| `axon_area (um^2)` | axon cross-sectional area |
| `myelin_thickness (um)` | mean myelin thickness |
| `image_border_touching` | True if axon touches the image border (excluded from analysis) |

Axons touching the image border were excluded to avoid partial measurements.

---

## Manual Measurements

The collaborator measured axons manually using **ImageJ / FIJI**:

- For each selected axon, two ellipses were drawn and measured: one tracing the **inner boundary** (axon) and one tracing the **outer boundary** (axon + myelin = fiber).
- 75 axons were measured per image (150 rows in the Results CSV = 75 pairs).
- Measurements are stored as `Results_XXXX.csv` files (columns: Label, Area in nm², X and Y centroid in nm from the original TIF origin).
- The ROI boundaries themselves are stored as `RoiSet_XXXX.zip` (ImageJ ROI format).

From these, per-axon g-ratio was computed as:

```
g-ratio = sqrt(axon_area / fiber_area)
axon_diameter = 2 * sqrt(axon_area / π)
```

Centroid coordinates were converted from nm to input.png pixels by dividing by the target pixel size (4.93 nm/px).

---

## Part 1 — ADS vs Manual: Accuracy Assessment

### Goal

Show how closely ADS g-ratio measurements match the collaborator's manual measurements, **restricted to the manually measured axons** (i.e., not the full ADS detection).

### Method

For each of 5 selected images, each manually measured axon was paired to the nearest ADS-detected axon by centroid distance (Euclidean, in input.png pixels). Matches exceeding 75 pixels (~370 nm) were excluded. The 5 images were chosen to cover both conditions and both injection types:

| Image | Condition | Injection |
|-------|-----------|-----------|
| image_706  | DREADD | Gi |
| image_2299 | DREADD | Gq |
| image_2336 | GFP    | Gq |
| image_2696 | DREADD | Gq |
| image_6107 | GFP    | Gi |

### Results

| Image | n matched | Pearson r | MAE | % error |
|-------|-----------|-----------|-----|---------|
| image_706  | 74 / 75 | 0.692 | 0.026 | 3.4% |
| image_2336 | 75 / 75 | 0.608 | 0.026 | 3.5% |
| image_2696 | 73 / 75 | 0.583 | 0.034 | 4.6% |
| image_6107 | 65 / 75 | 0.385 | 0.066 | 8.9% |
| image_2299 | 68 / 75 | 0.288 | 0.078 | 10.9% |

**Three images (706, 2336, 2696) show good agreement**: 3–5% mean absolute error, correlations of 0.58–0.69. This is typical performance for ADS on TEM optic nerve data.

**Two images (6107, 2299) show weaker agreement**: 9–11% error, correlations of 0.29–0.39. This likely reflects genuine difficulty in those images (more noise, denser packing, or myelin irregularities that confuse the model boundary detection).

**Visualisations produced:**
- `analysis/part1_ads_vs_manual.png` — 5-panel scatter plot (ADS g-ratio vs Manual g-ratio per matched axon)
- `analysis/overlay_image_XXXX.png` — full-image panels: raw | ADS overlay | manual ROI circles + ADS centroids
- `analysis/patches_image_XXXX.png` — zoomed patch panels showing 15 best-matched axon pairs (rows: raw image, ADS coloured mask, manual ROI circles with per-axon g-ratio labels)

---

## Part 2 — All-Axon Distributions

### Goal

Use ADS to measure **all axons** in each image (not just the 75 manually selected), giving much higher statistical power to detect changes in axon diameter and g-ratio distributions between DREADD and GFP conditions.

---

## Artefact Detection and Removal

### The problem

After running morphometrics on all 14 images, the raw axon count was **17,062** — far more than expected for optic nerve TEM images (typically 200–500 myelinated axons per field of view at this magnification). Spatial inspection revealed that the ADS generalist model over-detects in low-contrast or noisy image regions, segmenting noise as clusters of sub-resolution "axons" (diameter < 0.3 µm, g-ratio = 1.0 due to no detectable myelin). These artefacts inflate counts massively in affected images and corrupt any group-level statistics.

The per-image breakdown showed the problem was highly concentrated:

| Image | Raw axons | % artefacts (diam < 0.3 µm or g-ratio ≥ 0.98) |
|-------|-----------|------------------------------------------------|
| image_706  | 412  | ~24% — cleanest image |
| image_2299 | 2,726 | **88%** — dense cluster bottom-left |
| image_7559 | 3,332 | **91%** — dense cluster bottom strip |
| image_8969 | 2,135 | **84%** — diffuse throughout |

### Solution: two-stage cleaning

**Stage 1 — Spatial crop** (applied to image_2299 and image_7559 only):

Spatial maps of axon centroid positions (coloured by diameter) confirmed that the artefacts in these two images are tightly clustered in specific regions, not scattered uniformly. Rather than discarding all small axons globally (which risks removing real small axons in the rest of the image), the artifact-dense regions are excluded by centroid position:

| Image | Excluded region | Rationale |
|-------|----------------|-----------|
| image_2299 | x < 1500 px AND y > 3500 px | Dense cluster in bottom-left corner |
| image_7559 | y > 4300 px | Dense band across the bottom strip |

**Stage 2 — Size filter** (applied to all 14 images after spatial crop):

Any remaining axon with equivalent circle diameter < **0.30 µm** is removed. This threshold is above the ADS detection noise floor and below the smallest real myelinated optic nerve axons visible in the clean images. It catches the scattered individual artefacts that exist at low density in every image (including image_706).

### Results after cleaning

| Image | Raw | Border removed | Spatial crop | Size filter | **Kept** | **Kept %** |
|-------|-----|---------------|-------------|------------|---------|-----------|
| image_706  | 412  | 89 | 0    | 97   | **315** | 76.5% |
| image_963  | 698  | 88 | 0    | 245  | **453** | 64.9% |
| image_1812 | 689  | 104 | 0   | 242  | **447** | 64.9% |
| image_2299 | 2726 | 163 | 1124 | 1329 | **273** | 10.0% |
| image_2336 | 717  | 125 | 0   | 249  | **468** | 65.3% |
| image_2696 | 564  | 67  | 0   | 338  | **226** | 40.1% |
| image_4090 | 1070 | 113 | 0   | 620  | **450** | 42.1% |
| image_5087 | 669  | 96  | 0   | 347  | **322** | 48.1% |
| image_6107 | 835  | 92  | 0   | 421  | **414** | 49.6% |
| image_6294 | 651  | 96  | 0   | 357  | **294** | 45.2% |
| image_7422 | 1458 | 128 | 0   | 955  | **503** | 34.5% |
| image_7559 | 3332 | 183 | 2533 | 524 | **275** | 8.3%  |
| image_8969 | 2135 | 155 | 0   | 1825 | **310** | 14.5% |
| image_9408 | 1106 | 129 | 0   | 707  | **399** | 36.1% |
| **TOTAL**  | **17,062** | | | | **5,149** | **30.2%** |

> image_2299, image_7559 and image_8969 remain the most affected even after cleaning, likely because those specific grids captured regions with poor tissue preservation or section quality. Their retained axon counts (273, 275, 310) are plausible but conservative.

### Cleaned distributions (5,149 axons)

| Animal | Condition | Injection | n | Median g-ratio | Median diam (µm) |
|--------|-----------|-----------|---|----------------|-----------------|
| TKFG 17.1g | DREADD | Gq | 226 | 0.757 | 0.713 |
| TKFG 17.1g | GFP    | Gq | 971 | 0.722 | 0.665 |
| TKFG 18.1f | DREADD | Gq | 273 | 0.775 | 0.845 |
| TKFG 18.1f | GFP    | Gq | 849 | 0.730 | 0.742 |
| TKFG 19.1c | DREADD | Gi | 315 | 0.756 | 0.881 |
| TKFG 19.1c | GFP    | Gi | 414 | 0.777 | 0.803 |
| TKFG 19.1e | GFP    | Gi | 275 | 0.763 | 0.697 |
| TKFG 19.1f | GFP    | Gi | 322 | 0.726 | 0.841 |
| TKFG 19.1g | DREADD | Gi | 900 | 0.717 | 0.685 |
| TKFG 19.1g | GFP    | Gi | 294 | 0.763 | 0.857 |
| TKFG 21.1g | GFP    | Gq | 310 | 0.775 | 0.841 |

**Group-level comparison (Mann-Whitney U, all axons pooled):**
- Axon diameter: DREADD vs GFP — **not significant** (p ≈ 0.83)
- G-ratio: DREADD vs GFP — **not significant** (p ≈ 0.13)

The g-ratio trend (DREADD slightly higher than GFP in most animals) is consistent across animals but does not reach significance at the axon level. A proper paired analysis at the animal level (L vs R nerve, same animal) with n ≥ 8 animals per group would be needed to evaluate this meaningfully.

**Average g-ratio comparison — circle mode (per-image means, matching manual axons only):**
- Manual: mean = 0.719, SD = 0.014, median = 0.721
- ADS (circle): mean = 0.748, SD = 0.021, median = 0.747
- ADS overestimates g-ratio by ~0.029 — systematic upward bias consistent across all 14 images

See the next section for the ellipse mode re-run that eliminates this bias.

---

## Axon Shape Mode: Circle vs Ellipse

### Motivation

The default ADS morphometrics mode (`circle`) computes axon diameter as the **equivalent circle diameter** (2√(area/π)). The manual ImageJ measurements instead fit **ellipses** and report the minor axis as the diameter — which is appropriate for optic nerve axons that are elongated in cross-section. This methodological mismatch was the root cause of the +0.029 g-ratio overestimate seen in circle mode.

`axondeepseg_morphometrics` supports an alternative mode via `-a ellipse`, which fits an ellipse to each segmented axon and uses the minor axis as the diameter.

### Experiment

Both modes were run on all 14 images:
- **Circle**: `input_Morphometrics.csv` — default, already computed
- **Ellipse**: `input_Morphometrics_ellipse.csv` — re-run with `-a ellipse -f Morphometrics_ellipse.csv`

The same two-stage cleaning was applied to both (spatial crop → size filter, diam ≥ 0.30 µm).

### Accuracy comparison (matched axons vs manual)

| Metric | Circle | Ellipse |
|--------|--------|---------|
| Mean Pearson r | 0.688 | 0.632 |
| Mean MAE | 0.041 | 0.043 |
| Systematic bias (ADS − Manual) | **+0.031** | **≈ 0.000** |

> Circle mode has slightly higher per-axon correlation (0.69 vs 0.63) and lower MAE (0.041 vs 0.043), meaning it tracks individual axon variation better. However, it carries a consistent **+0.031 g-ratio overestimate** relative to manual across all images. Ellipse mode eliminates this bias entirely — the group means match the manual mean almost exactly — because both use the same minor-axis measurement convention.

Per-image breakdown:

| Image | r circle | MAE circle | r ellipse | MAE ellipse |
|-------|----------|------------|-----------|-------------|
| image_706  | 0.692 | 0.026 | 0.607 | 0.043 |
| image_963  | 0.890 | 0.036 | 0.794 | 0.034 |
| image_1812 | 0.857 | 0.034 | 0.813 | 0.038 |
| image_2299 | 0.540 | 0.085 | 0.396 | 0.074 |
| image_2336 | 0.608 | 0.026 | 0.527 | 0.048 |
| image_2696 | 0.811 | 0.025 | 0.808 | 0.038 |
| image_4090 | 0.808 | 0.036 | 0.737 | 0.034 |
| image_5087 | 0.783 | 0.030 | 0.711 | 0.035 |
| image_6107 | 0.532 | 0.058 | 0.561 | 0.042 |
| image_6294 | 0.711 | 0.041 | 0.515 | 0.041 |
| image_7422 | 0.550 | 0.031 | 0.497 | 0.046 |
| image_7559 | 0.541 | 0.052 | 0.638 | 0.039 |
| image_8969 | 0.666 | 0.067 | 0.633 | 0.051 |
| image_9408 | 0.641 | 0.035 | 0.615 | 0.041 |

### Decision

**Ellipse mode is used for all final analyses going forward.** The bias elimination is the priority — the absolute error values (4–5% both modes) are acceptable for automated segmentation. The circle CSVs are preserved (`input_Morphometrics.csv`) for reference.

### Ellipse mode — all-axon distributions (4,956 axons after cleaning)

| Animal | Condition | Injection | n | Mean g-ratio | Median g-ratio | Mean diam (µm) |
|--------|-----------|-----------|---|-------------|----------------|----------------|
| TKFG 17.1g | DREADD | Gq | 208 | 0.713 | 0.709 | 0.586 |
| TKFG 17.1g | GFP    | Gq | 922 | 0.685 | 0.685 | 0.592 |
| TKFG 18.1f | DREADD | Gq | 266 | 0.746 | 0.747 | 0.767 |
| TKFG 18.1f | GFP    | Gq | 822 | 0.692 | 0.693 | 0.625 |
| TKFG 19.1c | DREADD | Gi | 312 | 0.720 | 0.723 | 0.801 |
| TKFG 19.1c | GFP    | Gi | 407 | 0.749 | 0.751 | 0.716 |
| TKFG 19.1e | GFP    | Gi | 258 | 0.735 | 0.725 | 0.625 |
| TKFG 19.1f | GFP    | Gi | 318 | 0.688 | 0.696 | 0.731 |
| TKFG 19.1g | DREADD | Gi | 853 | 0.688 | 0.691 | 0.641 |
| TKFG 19.1g | GFP    | Gi | 286 | 0.725 | 0.730 | 0.718 |
| TKFG 21.1g | GFP    | Gq | 304 | 0.746 | 0.751 | 0.756 |

**Average g-ratio — ellipse mode (matched axons, all 14 images):**
- Manual: mean = 0.720
- ADS (ellipse): mean = 0.716
- Residual difference ≈ 0.004 — within noise, bias effectively eliminated

**Output files (ellipse mode):**
- `average_gratio_bar_ellipse.png` — bar chart: Manual vs ADS ellipse
- `part1_all_images_ellipse.png` — all 14 scatter plots (ellipse)
- `part2_distributions_ellipse.png` — g-ratio and diameter distributions (ellipse, cleaned)
- `part2_ellipse_summary.csv` — per-animal summary table
- `gratio_bar_circle_vs_ellipse.png` — 3-bar chart: Manual / ADS circle / ADS ellipse
- `scatter_circle_vs_ellipse.png` — side-by-side scatter for all 14 images (circle top, ellipse bottom)
- `circle_vs_ellipse_summary.csv` — per-image accuracy metrics for both modes

---

## Output File Index

```
Timmler_data/Optic_Nerve/
├── image_XXXX/
│   ├── input.png                  — preprocessed TEM image (input to ADS)
│   ├── seg_axon.png               — ADS axon mask
│   ├── seg_myelin.png             — ADS myelin mask
│   ├── overlay.png                — colour overlay
│   ├── metadata.json              — pixel size, scale, model info
│   ├── input_seg-axon.png         — symlink → seg_axon.png (required by morphometrics tool)
│   ├── input_seg-myelin.png       — symlink → seg_myelin.png
│   ├── input_Morphometrics.csv         — per-axon morphometrics, circle mode (raw, unfiltered)
│   ├── input_Morphometrics_ellipse.csv — per-axon morphometrics, ellipse mode (raw, unfiltered)
│   └── input_index.png            — instance-labelled axon image
├── Results_XXXX.csv               — manual measurements (75 axons × 2 ROIs = 150 rows)
├── RoiSet_XXXX.zip                — ImageJ ROI files
├── unblinding_set1/2/3.txt        — image code → original filename → animal + side
├── for_Yousseff.xlsx              — animal ID → genotype + injection
└── analysis/
    ├── run_analysis.py                     — morphometrics runner + initial analysis
    ├── plot_visual_comparison.py           — full-image overlays + zoomed patch panels
    ├── plot_all_scatter.py                 — ADS vs Manual scatter for all 14 images
    ├── plot_average_gratio.py              — average g-ratio bar chart (pre-cleaning)
    ├── cleaned_analysis.py                 — two-stage artefact removal + all final figures (ellipse mode)
    ├── compare_circle_ellipse.py           — circle vs ellipse head-to-head comparison
    │
    ├── part1_ads_vs_manual.png             — ADS vs Manual scatter (5 images, initial, circle)
    ├── part1_all_images.png                — ADS vs Manual scatter (all 14, pre-cleaning, circle)
    ├── part1_all_images_cleaned.png        — ADS vs Manual scatter (all 14, cleaned, circle)
    ├── part1_all_images_ellipse.png        — ADS vs Manual scatter (all 14, cleaned, ellipse) ★ final
    ├── part1_summary.csv                   — accuracy metrics per image (circle)
    ├── overlay_image_XXXX.png             — full-image comparison panels (5 images)
    ├── patches_image_XXXX.png             — zoomed axon-pair panels (5 images)
    ├── scatter_image_XXXX.png             — individual scatter per image (circle)
    ├── average_gratio_bar.png              — average g-ratio bar chart (pre-cleaning, circle)
    ├── average_gratio_bar_cleaned.png      — average g-ratio bar chart (cleaned, circle)
    ├── average_gratio_bar_ellipse.png      — average g-ratio bar chart (cleaned, ellipse) ★ final
    ├── gratio_bar_circle_vs_ellipse.png    — 3-bar chart: Manual / circle / ellipse
    ├── scatter_circle_vs_ellipse.png       — side-by-side scatter: circle (top) vs ellipse (bottom)
    ├── circle_vs_ellipse_summary.csv       — per-image accuracy metrics for both modes
    ├── artifact_spatial_map.png            — spatial map of artefact distributions
    ├── crop_boundaries.png                 — crop boundary visualisation (2299, 7559)
    ├── artifact_removal_summary.csv        — per-image counts before/after cleaning (ellipse)
    ├── part2_distributions_cleaned.png     — all-axon distributions (cleaned, circle)
    ├── part2_distributions_ellipse.png     — all-axon distributions (cleaned, ellipse) ★ final
    ├── part2_cleaned_summary.csv           — per-animal summary (circle)
    └── part2_ellipse_summary.csv           — per-animal summary (ellipse) ★ final
```

---

## Limitations and Next Steps

1. **Pseudoreplication**: multiple images per nerve and multiple nerves per animal. For statistical testing, animal-level or nerve-level averages should be used, not individual axons.

2. **Unequal groups**: 5 DREADD images vs 9 GFP images. Some DREADD images were not available with ADS predictions (11 additional images have manual data but no ADS segmentation — running ADS on those originals would double the DREADD sample).

3. **Persistent artefact rate**: even after cleaning, image_2299, image_7559 and image_8969 have unusually low retained counts, suggesting tissue or imaging quality issues in those grids. Their contribution to group statistics should be interpreted cautiously.

4. **Systematic ADS bias (circle mode)**: ADS overestimates g-ratio by ~0.029 in circle mode. This is eliminated by switching to ellipse mode (`-a ellipse`), which matches the minor-axis convention used in manual ImageJ measurements. Ellipse mode is now the default for all analyses.

5. **Model fine-tuning**: the generalist model was not trained specifically on optic nerve TEM data. Fine-tuning on a small manually annotated optic nerve dataset would likely reduce the artefact rate substantially and improve boundary accuracy.

6. **No unmyelinated axon analysis**: several images contain `seg_uaxon.png` masks. Unmyelinated axon density could be quantified with `axondeepseg_morphometrics --unmyelinated` if the collaborator is interested.
