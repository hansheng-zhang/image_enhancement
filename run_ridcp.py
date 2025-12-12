import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

# 使用你自己的 RIDCP 封装
from src.algorithms.ridcp import dehaze


# ================== 通用配置 ==================

# 统一缩放的长边像素（和 CLAHE / DCP 保持一致）
LONG_SIDE = 720   # 可以视算力改成 512 / 1024

# RIDCP 所在目录 & 权重（按你之前用的路径）
RIDCP_ROOT = "external/RIDCP_dehazing"
RIDCP_WEIGHT = "pretrained_models/pretrained_RIDCP.pth"
RIDCP_ALPHA = -21.25
RIDCP_USE_WEIGHT = True

# 两个数据集的路径配置（分别输出到各自目录&CSV）
DATASETS = {
    "ihaze": {
        "hazy_dir": Path("data/raw/ihaze/hazy"),
        "gt_dir":   Path("data/raw/ihaze/gt"),
        "out_dir":  Path("data/results/ridcp_full_ihaze"),
        "csv":      Path("data/results/ridcp_full_ihaze/metrics_ridcp_ihaze.csv"),
    },
    "ohaze": {
        "hazy_dir": Path("data/raw/ohaze/hazy"),
        "gt_dir":   Path("data/raw/ohaze/gt"),
        "out_dir":  Path("data/results/ridcp_full_ohaze"),
        "csv":      Path("data/results/ridcp_full_ohaze/metrics_ridcp_ohaze.csv"),
    },
}


# ================== 工具函数 ==================

def resize_long_side(img, long_side=LONG_SIDE):
    """按长边缩放到 long_side，保持宽高比"""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    scale = long_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def psnr(pred, gt):
    return peak_signal_noise_ratio(gt, pred, data_range=255)


def ssim(pred, gt):
    return structural_similarity(gt, pred, channel_axis=-1, data_range=255)


def deltaE00(pred, gt):
    pred_lab = color.rgb2lab(pred / 255.0)
    gt_lab   = color.rgb2lab(gt   / 255.0)
    delta = color.deltaE_ciede2000(gt_lab, pred_lab)
    return float(delta.mean())


def run_one_dataset(
    name: str,
    hazy_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    csv_path: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    hazy_files = sorted(hazy_dir.glob("*.jpg"))
    if not hazy_files:
        print(f"[{name}] Error: 找不到 hazy 图像: {hazy_dir}")
        return

    rows = []
    print(f"\n========== RIDCP on {name.upper()} ==========")
    print(f"[{name}] 共发现 {len(hazy_files)} 张 hazy 图，开始处理...")

    for hazy_path in hazy_files:
        fname = hazy_path.name

        # 假设命名规则：xxx_hazy.jpg → xxx_GT.jpg
        gt_name = fname.replace("_hazy", "_GT")
        gt_path = gt_dir / gt_name

        if not gt_path.exists():
            print(f"[{name}] [Warn] 找不到 GT：{gt_path}，跳过")
            continue

        hazy = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
        gt   = cv2.imread(str(gt_path),   cv2.IMREAD_COLOR)

        if hazy is None or gt is None:
            print(f"[{name}] [Warn] 读取失败：{hazy_path} 或 {gt_path}，跳过")
            continue

        print(f"\n[{name}] Processing {fname} ...")
        print(f"  原始尺寸 hazy={hazy.shape}, gt={gt.shape}")

        # 统一缩放
        hazy_small = resize_long_side(hazy, LONG_SIDE)
        gt_small   = resize_long_side(gt,   LONG_SIDE)
        print(f"  缩放后尺寸 hazy={hazy_small.shape}, gt={gt_small.shape}")

        # ---- RIDCP 去雾 ----
        out = dehaze(
            hazy_small,
            ridcp_root=RIDCP_ROOT,
            weight_relpath=RIDCP_WEIGHT,
            alpha=RIDCP_ALPHA,
            use_weight=RIDCP_USE_WEIGHT,
            python_exec=sys.executable,   # 用当前 env 的 python
        )

        # 保存结果
        save_path = out_dir / fname
        cv2.imwrite(str(save_path), out)
        print(f"  Saved result to {save_path}")

        # 转 RGB，用于算指标（和缩放后的 GT 对比）
        out_rgb = cv2.cvtColor(out,      cv2.COLOR_BGR2RGB)
        gt_rgb  = cv2.cvtColor(gt_small, cv2.COLOR_BGR2RGB)

        cur_psnr = psnr(out_rgb, gt_rgb)
        cur_ssim = ssim(out_rgb, gt_rgb)
        cur_de   = deltaE00(out_rgb, gt_rgb)

        print(f"  PSNR = {cur_psnr:.2f}")
        print(f"  SSIM = {cur_ssim:.4f}")
        print(f"  ΔE00 = {cur_de:.2f}")

        rows.append([fname, cur_psnr, cur_ssim, cur_de])

    # 写 CSV
    if rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "PSNR", "SSIM", "DeltaE00"])
            writer.writerows(rows)
        print(f"\n[{name}] RIDCP 指标已写入: {csv_path}")
    else:
        print(f"\n[{name}] 没有成功处理的样本。")


# ================== 主入口 ==================

if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        run_one_dataset(
            name=name,
            hazy_dir=cfg["hazy_dir"],
            gt_dir=cfg["gt_dir"],
            out_dir=cfg["out_dir"],
            csv_path=cfg["csv"],
        )

    print("\nAll done! (RIDCP on I-HAZE + O-HAZE)")
