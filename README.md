# Cambridge g-ratio analysis

Collaboration with Sebastian Timmler (University of Cambridge). Validation of AxonDeepSeg (ADS) automatic g-ratio and axon diameter measurements against manual annotations on two EM datasets.

## Datasets

### Optic Nerve (`optic_nerve/`)

14 TEM images from 7 mice. Paired design: left optic nerve = DREADD-injected condition, right = GFP control. Two DREADD constructs (Gi inhibitory, Gq excitatory). Native pixel size: 1.8625 nm/px (8000x), downsampled to 4.93 nm/px for ADS.

Manual ground truth: 75 axons measured per image in ImageJ (inner + outer ROI pairs). ADS detects 500-3500 axons per image; only the 75 manually measured ones are used for validation.

Key results:
- Circle mode: mean Pearson r = 0.69, MAE = 0.041, systematic bias = +0.031 g-ratio
- Ellipse mode: mean Pearson r = 0.63, MAE = 0.043, systematic bias = ~0.000
- Ellipse mode is preferred for final analyses as it eliminates the systematic overestimate

Full details: `optic_nerve/ANALYSIS_NOTES.md`

### H01 dataset (`h01_dataset/`)

20 axons tracked across multiple serial TEM slices. Per-slice g-ratio comparison: ADS (circle + ellipse) vs manual. Ellipse has lower systematic bias; circle competitive on MAE.

## Scripts

### Optic Nerve

| Script | Description |
|--------|-------------|
| `run_analysis.py` | Run ADS morphometrics on all images, initial scatter plots |
| `cleaned_analysis.py` | Two-stage artifact removal + all final figures (ellipse mode) |
| `compare_circle_ellipse.py` | Circle vs ellipse head-to-head accuracy comparison |
| `plot_visual_comparison.py` | Full-image overlays + zoomed patch panels |
| `plot_all_scatter.py` | ADS vs manual scatter for all 14 images |
| `plot_average_gratio.py` | Average g-ratio bar chart |
| `plot_gratio_by_condition.py` | G-ratio by DREADD vs GFP condition |
| `plot_gratio_by_injection.py` | G-ratio by Gi vs Gq injection |
| `plot_gratio_scatter.py` | G-ratio scatter plots |
| `plot_matched_vs_all.py` | Matched axons vs all-axon distributions |
| `plot_overall_scatter_and_tradeoff.py` | Overall scatter + circle/ellipse tradeoff |
| `plot_request1.py` | Sebastian's request 1 figures |
| `plot_request2.py` | Sebastian's request 2 figures |
| `plot_accuracy_per_image.py` | Per-image accuracy breakdown |

### H01 dataset

| Script | Description |
|--------|-------------|
| `h01_ellipse_analysis.py` | Ellipse mode analysis, per-axon z-profiles |
| `h01_scatter_slices.py` | Circle vs ellipse scatter per slice |

### Root

| Script | Description |
|--------|-------------|
| `upsample_masks_to_fullres.py` | Upsample ADS masks from 4.93 nm/px back to native resolution |

## Dependencies

```bash
pip install axondeepseg pandas matplotlib scipy scikit-learn numpy
```

Uses the ADS venv at `~/Developer/ADS/axondeepseg/.venv`.
