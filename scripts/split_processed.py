# scripts/split_processed.py
import os
import shutil
from glob import glob
import math

def split_one_dataset(raw_root, processed_root, name, train_ratio=0.7):
    hazy_raw = sorted(glob(os.path.join(raw_root, name, "hazy", "*.*")))
    gt_raw   = sorted(glob(os.path.join(raw_root, name, "gt", "*.*")))
    assert len(hazy_raw) == len(gt_raw), "hazy / gt 数量不一致"

    n = len(hazy_raw)
    n_train = math.floor(n * train_ratio)

    print(f"[{name}] total = {n}, train = {n_train}, test = {n - n_train}")

    for split, idxs in [("train", range(0, n_train)),
                        ("test", range(n_train, n))]:

        hazy_dst_dir = os.path.join(processed_root, split, name, "hazy")
        gt_dst_dir   = os.path.join(processed_root, split, name, "gt")

        os.makedirs(hazy_dst_dir, exist_ok=True)
        os.makedirs(gt_dst_dir, exist_ok=True)

        for i in idxs:
            hf = hazy_raw[i]
            gf = gt_raw[i]
            fname = os.path.basename(hf)

            dst_hazy = os.path.join(hazy_dst_dir, fname)
            dst_gt   = os.path.join(gt_dst_dir, fname)

            # 如果目标文件存在，先删除，防止 PermissionError
            if os.path.exists(dst_hazy):
                os.remove(dst_hazy)
            if os.path.exists(dst_gt):
                os.remove(dst_gt)

            shutil.copy2(hf, dst_hazy)
            shutil.copy2(gf, dst_gt)

        print(f"  → {split}: {len(list(idxs))} files copied.")


if __name__ == "__main__":
    raw_root = "data/raw"
    processed_root = "data/processed"

    split_one_dataset(raw_root, processed_root, "ohaze", train_ratio=0.7)
    split_one_dataset(raw_root, processed_root, "ihaze", train_ratio=0.7)
