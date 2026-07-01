"""
Multires inference + morphometrics on all Optic Nerve images.
Input:  ~/cambridge_data/Optic nerve raw/image_*.tif
Output: ~/cambridge_on/image_<id>/input.png
                                   input_seg-axon.png
                                   input_seg-myelin.png
                                   input_axon_morphometrics.xlsx

Run on tassan:
    python optic_nerve/run_multires_cluster.py
"""

import os, sys, shutil, tempfile
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_DIR  = Path.home() / "multires_model" / "nnUNetTrainer__nnUNetPlans__2d"
RAW_DIR    = Path.home() / "cambridge_data" / "Optic nerve raw"
OUT_DIR    = Path.home() / "cambridge_on"
CHECKPOINT = "checkpoint_best.pth"
PIXEL_SIZE = 0.00493  # um/px at 4.93 nm/px
TARGET_W   = 6700     # downsample to match 4.93 nm/px working resolution

os.environ["nnUNet_raw"]          = str(Path.home() / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(Path.home() / "nnUNet_preprocessed")
os.environ["nnUNet_results"]      = str(Path.home() / "nnUNet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
from AxonDeepSeg.morphometrics.launch_morphometrics_computation import main as ads_morphometrics_main

Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check for large TIFs


def run_inference(tif: Path, predictor, tmp_root: Path):
    name    = tif.stem
    img_out = OUT_DIR / name
    img_out.mkdir(parents=True, exist_ok=True)

    input_png  = img_out / "input.png"
    axon_out   = img_out / "input_seg-axon.png"
    myelin_out = img_out / "input_seg-myelin.png"

    if axon_out.exists() and myelin_out.exists():
        print(f"  {name}: segmentation already done, skipping inference.")
        return

    tmp_in  = tmp_root / "in"  / name
    tmp_out = tmp_root / "out" / name
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    img = Image.open(tif).convert("L")
    if img.width > TARGET_W:
        scale = TARGET_W / img.width
        img = img.resize((TARGET_W, int(img.height * scale)), Image.LANCZOS)
    img.save(input_png)
    img.save(tmp_in / f"{name}_0000.png")

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
    print(f"  {name}: inference done.")

    shutil.rmtree(tmp_in,  ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)


def run_morphometrics(name: str):
    img_out   = OUT_DIR / name
    xlsx_out  = img_out / "input_axon_morphometrics.xlsx"
    input_png = img_out / "input.png"

    if xlsx_out.exists():
        print(f"  {name}: morphometrics already done, skipping.")
        return
    if not input_png.exists():
        print(f"  {name}: no input.png, skipping morphometrics.")
        return

    print(f"  {name}: running morphometrics...", flush=True)
    try:
        ads_morphometrics_main(["-i", str(input_png), "-s", str(PIXEL_SIZE)])
    except SystemExit:
        pass
    except Exception as e:
        print(f"  {name}: morphometrics ERROR - {e}")
        return
    print(f"  {name}: morphometrics done.")


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

    print("=== Inference ===")
    tmp_root = Path(tempfile.mkdtemp(prefix="multires_"))
    try:
        for tif in tifs:
            run_inference(tif, predictor, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n=== Morphometrics ===")
    for tif in tifs:
        run_morphometrics(tif.stem)

    print("\nAll done.")
