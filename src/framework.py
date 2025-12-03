# src/framework.py
import os
import cv2
import numpy as np

from .dataset import list_pairs
from .metrics import psnr, ssim, deltaE00
from .utils import ensure_dir

def get_algorithm_fn(name):
    name = name.lower()
    if name == "clahe":
        from .algorithms import clahe as algo
    elif name == "dcp":
        from .algorithms import dcp as algo
    elif name == "ridcp":
        from .algorithms import ridcp as algo
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    return algo.dehaze


class DehazeRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processed_dir = cfg["data"]["processed_dir"]
        self.results_dir = cfg["data"]["results_dir"]

    def run(self):
        for ds in self.cfg["data"]["datasets"]:
            name = ds["name"]
            split = ds.get("split", "test")
            self._run_on_dataset(name, split)

    def _run_on_dataset(self, dataset_name, split):
        pairs = list_pairs(self.processed_dir, dataset_name, split)
        print(f"[{dataset_name}/{split}] {len(pairs)} pairs found.")

        for algo_cfg in self.cfg["algorithms"]:
            algo_name = algo_cfg["name"]
            params = algo_cfg.get("params", {})
            self._run_algo_on_dataset(dataset_name, split, algo_name, params, pairs)

    def _run_algo_on_dataset(self, dataset_name, split, algo_name, params, pairs):
        print(f"  >> Algorithm: {algo_name}")
        dehaze_fn = get_algorithm_fn(algo_name)

        save_dir = os.path.join(self.results_dir, algo_name, dataset_name, split)
        ensure_dir(save_dir)

        psnrs, ssims, des = [], [], []

        for hazy_path, gt_path in pairs:
            fname = os.path.basename(hazy_path)
            print(f"    - {fname}")

            hazy_bgr = cv2.imread(hazy_path, cv2.IMREAD_COLOR)
            gt_bgr   = cv2.imread(gt_path, cv2.IMREAD_COLOR)

            out_bgr = dehaze_fn(hazy_bgr, **params)
            cv2.imwrite(os.path.join(save_dir, fname), out_bgr)

            # 转 RGB 计算指标
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            gt_rgb  = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

            psnrs.append(psnr(out_rgb, gt_rgb))
            ssims.append(ssim(out_rgb, gt_rgb))
            des.append(deltaE00(out_rgb, gt_rgb))

        print(f"    [{dataset_name}/{algo_name}] "
              f"PSNR={np.mean(psnrs):.2f}, SSIM={np.mean(ssims):.4f}, ΔE00={np.mean(des):.2f}")
