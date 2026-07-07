"""
Test Dataset005 inference on one Cambridge image at native 1.86 nm/px resolution.
Converts 16-bit TIF -> 8-bit via percentile normalization, runs nnUNet, saves overlay.

Usage (on tassan):
    source ~/venvs/multires/bin/activate
    export nnUNet_results=~/nnunet_results
    python3 optic_nerve/test_native_res.py
"""

import os, tempfile, torch
from pathlib import Path
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

MODEL_DIR   = Path.home() / "nnunet_results/Dataset005_TEM12_multires/nnUNetTrainer__nnUNetPlans__2d"
RAW_DIR     = Path.home() / "cambridge_inputs_raw"   # where you put the TIFs on tassan
TEST_IMAGE  = "image_2336.tif"
OUT_DIR     = Path.home() / "cambridge_inputs" / "image_2336"
CHECKPOINT  = "checkpoint_final.pth"

os.environ.setdefault("nnUNet_raw",          "/tmp/nnunet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
os.environ.setdefault("nnUNet_results",      str(Path.home() / "nnunet_results"))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Percentile-based normalization from 16-bit to 8-bit."""
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255
    return arr.astype(np.uint8)


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}")

    tif_path = RAW_DIR / TEST_IMAGE
    print(f"Loading {tif_path}...")
    img_16 = np.array(Image.open(tif_path))
    print(f"  Shape: {img_16.shape}, dtype: {img_16.dtype}, range: [{img_16.min()}, {img_16.max()}]")
    img_8 = to_uint8(img_16)
    print(f"  Converted to uint8, range: [{img_8.min()}, {img_8.max()}]")

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
    print("Model loaded. Running inference...")

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

    axon  = ((seg == 1).astype(np.uint8) * 255)
    myelin = ((seg == 2).astype(np.uint8) * 255)
    print(f"axon={100*(seg==1).mean():.1f}%  myelin={100*(seg==2).mean():.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(axon).save(OUT_DIR / "seg_axon_native.png")
    Image.fromarray(myelin).save(OUT_DIR / "seg_myelin_native.png")

    # Quick overlay thumbnail for visual check
    rgb = np.stack([img_8, img_8, img_8], axis=-1)
    rgb[myelin > 0] = (rgb[myelin > 0] * 0.35 + np.array([60, 100, 255]) * 0.65).astype(np.uint8)
    rgb[axon  > 0] = (rgb[axon  > 0]  * 0.30 + np.array([255, 60, 60])  * 0.70).astype(np.uint8)
    scale = 0.08
    h, w = rgb.shape[:2]
    thumb = Image.fromarray(rgb).resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    thumb.save(OUT_DIR / "overlay_native_thumb.jpg", quality=85)

    print(f"\nSaved to {OUT_DIR}:")
    print("  seg_axon_native.png")
    print("  seg_myelin_native.png")
    print("  overlay_native_thumb.jpg  <- pull this to check visually")
