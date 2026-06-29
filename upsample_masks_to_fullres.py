"""
For each Cambridge CC image:
  - Finds the full-res TIF from timmler_nnunet/fullsizedata (recursive search)
  - Upsamples all masks to match the full-res dimensions (nearest-neighbor)
"""

from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

CAMBRIDGE_CC = Path("/Users/yolaatar/Developer/ADS/Timmler_data/Corpus_Callosum")
FULLRES_DIR  = Path("/Users/yolaatar/Developer/ADS/timmler_nnunet/fullsizedata/Corpus_Callosum")
MASK_NAMES   = ["seg_axon.png", "seg_myelin.png", "seg_uaxon.png"]

def find_tif(name: str) -> Path | None:
    matches = list(FULLRES_DIR.rglob(f"{name}.tif"))
    # Prefer the one without a pixel-size suffix
    exact = [m for m in matches if m.stem == name]
    return exact[0] if exact else (matches[0] if matches else None)

for folder in sorted(CAMBRIDGE_CC.iterdir()):
    if not folder.is_dir():
        continue

    tif = find_tif(folder.name)
    if tif is None:
        print(f"MISSING full-res TIF for {folder.name}, skipping")
        continue

    with Image.open(tif) as fullres:
        target_size = fullres.size  # (W, H)

    for mask_name in MASK_NAMES:
        mask_path = folder / mask_name
        if not mask_path.exists():
            print(f"  {folder.name}: no {mask_name}, skipping")
            continue

        with Image.open(mask_path) as mask:
            if mask.size == target_size:
                print(f"  {folder.name}/{mask_name}: already full-res, skipping")
                continue
            src_size = mask.size
            upsampled = mask.resize(target_size, Image.NEAREST)

        out_path = folder / f"{mask_path.stem}_fullres.png"
        upsampled.save(out_path)
        print(f"  {folder.name}/{mask_name}: {src_size} -> {target_size}")

    print(f"{folder.name}: done")

print("\nAll done.")
