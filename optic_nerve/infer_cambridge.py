"""
Run Dataset005 (TEM1+TEM2 multires) inference on all 38 Cambridge Optic Nerve images.
Saves seg_axon.png and seg_myelin.png in each image folder (overwrites existing).
No resampling -- Cambridge data is at 0.00493 um/px = TEM2 native resolution.

Usage (on tassan):
    source ~/venvs/<VENV>/bin/activate
    export nnUNet_results=~/nnunet_results
    export nnUNet_raw=/tmp/dummy_raw
    export nnUNet_preprocessed=/tmp/dummy_preprocessed
    python3 infer_cambridge.py
"""

import os, tempfile, torch
from pathlib import Path
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

MODEL_DIR  = Path.home() / "nnunet_results/Dataset005_TEM12_multires/nnUNetTrainer__nnUNetPlans__2d"
INPUT_BASE = Path.home() / "cambridge_inputs"
CHECKPOINT = "checkpoint_final.pth"

# nnUNet needs these env vars even for inference
os.environ.setdefault("nnUNet_raw",           "/tmp/nnunet_raw")
os.environ.setdefault("nnUNet_preprocessed",  "/tmp/nnunet_preprocessed")
os.environ.setdefault("nnUNet_results",       str(Path.home() / "nnunet_results"))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

print(f"Model : {MODEL_DIR}")
print(f"Images: {INPUT_BASE}")
print(f"GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}")
print()

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

images = sorted(
    [d for d in INPUT_BASE.iterdir() if d.is_dir() and d.name.startswith("image_")],
    key=lambda x: int(x.name.replace("image_", ""))
)
print(f"Found {len(images)} image folders.\n")

for i, img_dir in enumerate(images, 1):
    inp = img_dir / "input.png"
    if not inp.exists():
        print(f"[{i}/{len(images)}] {img_dir.name}: no input.png, skipping")
        continue

    out_axon = img_dir / "seg_axon.png"
    out_mye  = img_dir / "seg_myelin.png"

    print(f"[{i}/{len(images)}] {img_dir.name}...", end=" ", flush=True)

    img_arr = np.array(Image.open(inp).convert("L"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inp_dir = tmp / "inp"
        out_dir = tmp / "out"
        inp_dir.mkdir()
        out_dir.mkdir()

        Image.fromarray(img_arr).save(inp_dir / "case_0000.png")

        predictor.predict_from_files(
            [[str(inp_dir / "case_0000.png")]],
            [str(out_dir / "case")],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        seg = np.array(Image.open(out_dir / "case.png").convert("L"))

    Image.fromarray(((seg == 1).astype(np.uint8) * 255)).save(out_axon)
    Image.fromarray(((seg == 2).astype(np.uint8) * 255)).save(out_mye)

    axon_pct = 100 * (seg == 1).mean()
    mye_pct  = 100 * (seg == 2).mean()
    print(f"done  axon={axon_pct:.1f}%  myelin={mye_pct:.1f}%")

print("\nAll done.")
