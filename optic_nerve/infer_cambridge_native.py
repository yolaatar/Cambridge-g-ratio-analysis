"""
Run Dataset005 inference on all 38 Cambridge TIFs at native 1.86 nm/px resolution.
Converts 16-bit TIF -> 8-bit via percentile normalization, no resampling.
Saves seg_axon_native.png and seg_myelin_native.png in each cambridge_inputs folder.

Usage (on tassan):
    source ~/venvs/multires/bin/activate
    export nnUNet_results=~/nnunet_results
    python3 optic_nerve/infer_cambridge_native.py
"""

import os, tempfile, torch
from pathlib import Path
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

MODEL_DIR  = Path.home() / "nnunet_results/Dataset005_TEM12_multires/nnUNetTrainer__nnUNetPlans__2d"
RAW_DIR    = Path.home() / "Optic nerve"          # unzipped from optic_nerve_june2026.zip
OUT_BASE   = Path.home() / "cambridge_inputs"
CHECKPOINT = "checkpoint_final.pth"

os.environ.setdefault("nnUNet_raw",          "/tmp/nnunet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
os.environ.setdefault("nnUNet_results",      str(Path.home() / "nnunet_results"))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}")

    tifs = sorted(RAW_DIR.glob("image_*.tif"),
                  key=lambda p: int(p.stem.replace("image_", "")))
    print(f"Found {len(tifs)} TIFs in {RAW_DIR}\n")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR),
        use_folds=("all",),
        checkpoint_name=CHECKPOINT,
    )
    print("Model loaded.\n")

    for i, tif_path in enumerate(tifs, 1):
        img_id  = tif_path.stem
        out_dir = OUT_BASE / img_id
        out_axon = out_dir / "seg_axon_native.png"
        out_mye  = out_dir / "seg_myelin_native.png"

        if out_axon.exists() and out_mye.exists():
            print(f"[{i}/{len(tifs)}] {img_id}: already done, skipping")
            continue

        if not out_dir.exists():
            print(f"[{i}/{len(tifs)}] {img_id}: no output folder, skipping")
            continue

        print(f"[{i}/{len(tifs)}] {img_id}...", end=" ", flush=True)

        img_8 = to_uint8(np.array(Image.open(tif_path)))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "inp").mkdir()
            (tmp / "out").mkdir()
            Image.fromarray(img_8).save(tmp / "inp" / "case_0000.png")

            predictor.predict_from_files(
                [[str(tmp / "inp" / "case_0000.png")]],
                [str(tmp / "out" / "case")],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            seg = np.array(Image.open(tmp / "out" / "case.png").convert("L"))

        Image.fromarray(((seg == 1).astype(np.uint8) * 255)).save(out_axon)
        Image.fromarray(((seg == 2).astype(np.uint8) * 255)).save(out_mye)
        print(f"done  axon={100*(seg==1).mean():.1f}%  myelin={100*(seg==2).mean():.1f}%")

    print("\nAll done.")
