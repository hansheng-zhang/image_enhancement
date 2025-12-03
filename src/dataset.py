# src/dataset.py
import os
from glob import glob

def list_pairs(processed_dir, dataset_name, split="test"):

    base = os.path.join(processed_dir, split, dataset_name)
    hazy_dir = os.path.join(base, "hazy")
    gt_dir   = os.path.join(base, "gt")

    hazy_files = sorted(glob(os.path.join(hazy_dir, "*.*")))
    pairs = []

    for hpath in hazy_files:
        fname = os.path.basename(hpath)
        gpath = os.path.join(gt_dir, fname)
        if os.path.exists(gpath):
            pairs.append((hpath, gpath))
        else:
            print(f"[WARN] GT not found for {hpath}")
    return pairs
