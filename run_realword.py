import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage import img_as_float32
from piq import brisque

from src.algorithms.clahe import dehaze as clahe_dehaze
from src.algorithms.dcp import dehaze as dcp_dehaze
from src.algorithms.ridcp import dehaze as ridcp_dehaze


RAW_DIR = Path("data/raw/real_haze")

OUT_BASE = Path("data/results/real_haze")
OUT_CLAHE = OUT_BASE / "clahe"
OUT_DCP = OUT_BASE / "dcp"
OUT_RIDCP = OUT_BASE / "ridcp"

for d in [OUT_BASE, OUT_CLAHE, OUT_DCP, OUT_RIDCP]:
    d.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUT_BASE / "real_haze_brisque.csv"

RIDCP_ROOT = "external/RIDCP_dehazing"
RIDCP_WEIGHT = "pretrained_models/pretrained_RIDCP.pth"
RIDCP_ALPHA = -21.25
RIDCP_USE_WEIGHT = True

MAX_SIDE = 1024


def resize_long_side(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_torch_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_as_float32(img_rgb)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    return tensor


def compute_brisque(img_bgr: np.ndarray) -> float:
    x = bgr_to_torch_rgb(img_bgr)
    x = x.to("cpu")
    with torch.no_grad():
        score = brisque(x, data_range=1.0).item()
    return score


def main():
    hazy_files = sorted(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))

    if not hazy_files:
        print(f"No images found in {RAW_DIR}")
        return

    rows = []

    print(f"Found {len(hazy_files)} real-world hazy images. Processing...")

    for img_path in hazy_files:
        fname = img_path.name
        print(f"\nProcessing {fname} ...")

        hazy = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if hazy is None:
            print(f"Cannot read {img_path}, skipping.")
            continue

        orig_h, orig_w = hazy.shape[:2]
        hazy_small = resize_long_side(hazy, MAX_SIDE)
        new_h, new_w = hazy_small.shape[:2]
        print(f"  Original size: {orig_w}x{orig_h}, resized to: {new_w}x{new_h}")

        hazy_brisque = compute_brisque(hazy_small)
        print(f"  Hazy     : BRISQUE={hazy_brisque:.3f}")

        row = {
            "filename": fname,
            "hazy_brisque": hazy_brisque,
        }

        clahe_out = clahe_dehaze(hazy_small, tile_grid_size=(8, 8), clip_limit=2.0)
        cv2.imwrite(str(OUT_CLAHE / fname), clahe_out)
        clahe_brisque = compute_brisque(clahe_out)
        print(f"  CLAHE    : BRISQUE={clahe_brisque:.3f}")
        row["clahe_brisque"] = clahe_brisque

        dcp_out = dcp_dehaze(
            hazy_small,
            patch_size=15,
            omega=0.95,
            t0=0.1,
            use_guided_filter=True,
            guided_radius=40,
            guided_eps=0.001,
        )
        cv2.imwrite(str(OUT_DCP / fname), dcp_out)
        dcp_brisque = compute_brisque(dcp_out)
        print(f"  DCP      : BRISQUE={dcp_brisque:.3f}")
        row["dcp_brisque"] = dcp_brisque

        ridcp_out = ridcp_dehaze(
            hazy_small,
            ridcp_root=RIDCP_ROOT,
            weight_relpath=RIDCP_WEIGHT,
            alpha=RIDCP_ALPHA,
            use_weight=RIDCP_USE_WEIGHT,
            python_exec=sys.executable,
        )
        cv2.imwrite(str(OUT_RIDCP / fname), ridcp_out)
        ridcp_brisque = compute_brisque(ridcp_out)
        print(f"  RIDCP    : BRISQUE={ridcp_brisque:.3f}")
        row["ridcp_brisque"] = ridcp_brisque

        rows.append(row)

    if rows:
        with open(METRICS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "filename",
                "hazy_brisque",
                "clahe_brisque",
                "dcp_brisque",
                "ridcp_brisque",
            ]
            writer.writerow(header)
            for r in rows:
                writer.writerow([r.get(col, "") for col in header])

        print(f"\nMetrics saved to {METRICS_CSV}")
    else:
        print("\nNo images were successfully processed.")


if __name__ == "__main__":
    main()
    print("\nDone. (Real-world haze evaluation with BRISQUE and resized images)")
