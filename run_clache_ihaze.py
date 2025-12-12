import os
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

# 你自己的 CLAHE 实现
from src.algorithms.clahe import dehaze


# ======== 配置区域 ========

# 数据集路径（按你现在的结构）
HAZY_DIR = Path("data/raw/ihaze/hazy")
GT_DIR   = Path("data/raw/ihaze/gt")

# 输出去雾结果
OUTPUT_DIR = Path("data/results/clahe_full_ihaze")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 结果指标保存成 CSV
METRICS_CSV = OUTPUT_DIR / "metrics_clahe_ihaze.csv"

# 统一缩放的长边像素
LONG_SIDE = 720   # 可以改成 512/1024 等，看你算力和需求

# CLAHE 参数（照你之前的）
CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_CLIP_LIMIT = 2.0


# ======== 工具函数 ========

def resize_long_side(img, long_side=LONG_SIDE):
    """把图像按长边缩放到 long_side，保持宽高比"""
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


# ======== 主流程：遍历整个 ihaze/hazy 目录 ========

def main():
    hazy_files = sorted(HAZY_DIR.glob("*.jpg"))

    if not hazy_files:
        print(f"[Error] 找不到 hazy 图像: {HAZY_DIR}")
        return

    rows = []
    print(f"共发现 {len(hazy_files)} 张 hazy 图，开始处理...")

    for hazy_path in hazy_files:
        fname = hazy_path.name
        # 假设文件名是 01_indoor_hazy.jpg → 01_indoor_GT.jpg
        gt_name = fname.replace("_hazy", "_GT")
        gt_path = GT_DIR / gt_name

        if not gt_path.exists():
            print(f"[Warn] 找不到 GT：{gt_path}，跳过")
            continue

        hazy = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
        gt   = cv2.imread(str(gt_path),   cv2.IMREAD_COLOR)

        if hazy is None or gt is None:
            print(f"[Warn] 读取失败：{hazy_path} 或 {gt_path}，跳过")
            continue

        print(f"\nProcessing {fname} ...")
        print(f"  原始尺寸 hazy={hazy.shape}, gt={gt.shape}")

        # 统一缩放
        hazy_small = resize_long_side(hazy, LONG_SIDE)
        gt_small   = resize_long_side(gt,   LONG_SIDE)
        print(f"  缩放后尺寸 hazy={hazy_small.shape}, gt={gt_small.shape}")

        # CLAHE 去雾（对缩小后的 hazy）
        out = dehaze(
            hazy_small,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
            clip_limit=CLAHE_CLIP_LIMIT,
        )

        # 保存结果（文件名保持一致）
        save_path = OUTPUT_DIR / fname
        cv2.imwrite(str(save_path), out)
        print(f"  Saved result to {save_path}")

        # 转 RGB 用于算指标（和缩小后的 GT 对比）
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
        with open(METRICS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "PSNR", "SSIM", "DeltaE00"])
            writer.writerows(rows)
        print(f"\n所有结果已写入: {METRICS_CSV}")
    else:
        print("\n没有成功处理的样本。")


if __name__ == "__main__":
    main()
    print("\nDone!")
