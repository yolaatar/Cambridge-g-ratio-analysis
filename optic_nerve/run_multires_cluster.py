"""
Multires inference on all Optic Nerve images.
Input:  ~/cambridge_data/Optic nerve raw/image_*.tif
Output: ~/cambridge_on/image_<id>/seg_axon_multires.png
                                   seg_myelin_multires.png

Run on tassan:
    python optic_nerve/run_multires_cluster.py
"""

import os, shutil, tempfile
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_DIR = Path.home() / "multires_model" / "nnUNetTrainer__nnUNetPlans__2d"
RAW_DIR   = Path.home() / "cambridge_data" / "Optic nerve raw"
OUT_DIR   = Path.home() / "cambridge_on"
CHECKPOINT = "checkpoint_best.pth"

os.environ["nnUNet_raw"]          = str(Path.home() / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(Path.home() / "nnUNet_preprocessed")
os.environ["nnUNet_results"]      = str(Path.home() / "nnUNet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch


def run_image(tif: Path, predictor, tmp_root: Path):
    name = tif.stem  # e.g. image_706
    img_out = OUT_DIR / name
    img_out.mkdir(parents=True, exist_ok=True)

    axon_out   = img_out / "seg_axon_multires.png"
    myelin_out = img_out / "seg_myelin_multires.png"
    if axon_out.exists() and myelin_out.exists():
        print(f"  {name}: already done, skipping.")
        return

    tmp_in  = tmp_root / "in"  / name
    tmp_out = tmp_root / "out" / name
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    Image.open(tif).convert("L").save(tmp_in / f"{name}_0000.png")

    print(f"  {name}: running inference...", flush=True)
    predictor.predict_from_files(
        str(tmp_in), str(tmp_out),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=2, num_processes_segmentation_export=2)

    pred_file = tmp_out / f"{name}.png"
    if not pred_file.exists():
        print(f"  {name}: ERROR - prediction output not found.")
        return

    pred   = np.array(Image.open(pred_file))
    axon   = ((pred == 1) | (pred == 2)).astype(np.uint8) * 255
    myelin = (pred == 2).astype(np.uint8) * 255
    Image.fromarray(axon).save(axon_out)
    Image.fromarray(myelin).save(myelin_out)
    print(f"  {name}: done.")

    shutil.rmtree(tmp_in,  ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    tifs = sorted(
        [f for f in RAW_DIR.glob("image_*.tif")],
        key=lambda f: int(f.stem.replace("image_", ""))
    )
    print(f"Found {len(tifs)} images in {RAW_DIR}")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=device, verbose=False)
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR), use_folds=("all",), checkpoint_name=CHECKPOINT)
    print("Model loaded.\n")

    tmp_root = Path(tempfile.mkdtemp(prefix="multires_"))
    try:
        for tif in tifs:
            run_image(tif, predictor, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nAll done.")
