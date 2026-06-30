"""
Multires inference on all Optic Nerve images.
Run on tassan from ~/cambridge_on/ with:
    python run_multires_cluster.py
Outputs: seg_axon_multires.png, seg_myelin_multires.png per image dir.
"""

import os, sys, shutil, tempfile
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_DIR = Path("/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset005_TEM12_multires")
DATA_DIR  = Path.home() / "cambridge_on"
CHECKPOINT = "checkpoint_best.pth"

os.environ["nnUNet_raw"]          = str(Path.home() / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(Path.home() / "nnUNet_preprocessed")
os.environ["nnUNet_results"]      = str(Path.home() / "nnUNet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

def run_image(img_dir: Path, predictor, tmp_root: Path):
    name = img_dir.name
    inp  = img_dir / "input.png"
    if not inp.exists():
        print(f"  {name}: no input.png, skipping.")
        return

    axon_out   = img_dir / "seg_axon_multires.png"
    myelin_out = img_dir / "seg_myelin_multires.png"
    if axon_out.exists() and myelin_out.exists():
        print(f"  {name}: already done, skipping.")
        return

    tmp_in  = tmp_root / "in"  / name
    tmp_out = tmp_root / "out" / name
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(inp, tmp_in / f"{name}_0000.png")

    print(f"  {name}: running inference...", flush=True)
    predictor.predict_from_files(
        str(tmp_in), str(tmp_out),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=2, num_processes_segmentation_export=2)

    pred_file = tmp_out / f"{name}.png"
    if not pred_file.exists():
        print(f"  {name}: ERROR - no prediction output found.")
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
    img_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("image_")],
        key=lambda d: int(d.name.replace("image_", ""))
    )
    print(f"Found {len(img_dirs)} image directories.")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_gpu=True, device=device, verbose=False)
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR), use_folds=(0,), checkpoint_name=CHECKPOINT)
    print("Model loaded.\n")

    tmp_root = Path(tempfile.mkdtemp(prefix="multires_"))
    try:
        for img_dir in img_dirs:
            run_image(img_dir, predictor, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nAll done.")
