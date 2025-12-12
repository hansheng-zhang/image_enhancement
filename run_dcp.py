import os
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

# 使用你自己的 DCP 算法
from src.algorithms.dcp import dehaze


# ======== 配置区域（I-HAZE） ========

HAZY_DIR = Path("data/raw/ihaze/hazy")
GT_DIR   = Path("data/raw/ihaze/gt")

OUTPUT_DIR = Path("data/results/dcp_full_ihaze")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUTPUT_DIR / "metrics_dcp_ihaze.csv"

# 统一缩放的长边像素（和 CLAHE/RIDCP 保持一致）
LONG_SIDE = 720   # 可改 512/1024 等

# DCP 参数（和你之前测试保持一致）
DCP_PATCH_SIZE = 15
DCP_OMEGA = 0.95
DCP_T0 = 0.1
DCP_USE_GUIDED = True
DCP_GUIDED_RADIUS = 40
DCP_GUIDED_EPS = 0.001


# ======== 工具函数 ========

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


# ======== 主流程：遍历 ihaze/hazy 目录 ========

def main():
    hazy_files = sorted(HAZY_DIR.glob("*.jpg"))

    if not hazy_files:
        print(f"[I-HAZE] Error: 找不到 hazy 图像: {HAZY_DIR}")
        return

    rows = []
    print(f"[I-HAZE] 共发现 {len(hazy_files)} 张 hazy 图，开始处理...")

    for hazy_path in hazy_files:
        fname = hazy_path.name

        # 01_indoor_hazy.jpg → 01_indoor_GT.jpg
        gt_name = fname.replace("_hazy", "_GT")
        gt_path = GT_DIR / gt_name

        if not gt_path.exists():
            print(f"[I-HAZE] [Warn] 找不到 GT：{gt_path}，跳过")
            continue

        hazy = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
        gt   = cv2.imread(str(gt_path),   cv2.IMREAD_COLOR)

        if hazy is None or gt is None:
            print(f"[I-HAZE] [Warn] 读取失败：{hazy_path} 或 {gt_path}，跳过")
            continue

        print(f"\n[I-HAZE] Processing {fname} ...")
        print(f"  原始尺寸 hazy={hazy.shape}, gt={gt.shape}")

        # 统一缩放
        hazy_small = resize_long_side(hazy, LONG_SIDE)
        gt_small   = resize_long_side(gt,   LONG_SIDE)
        print(f"  缩放后尺寸 hazy={hazy_small.shape}, gt={gt_small.shape}")

        # ---- DCP 去雾 ----
        out = dehaze(
            hazy_small,
            patch_size=DCP_PATCH_SIZE,
            omega=DCP_OMEGA,
            t0=DCP_T0,
            use_guided_filter=DCP_USE_GUIDED,
            guided_radius=DCP_GUIDED_RADIUS,
            guided_eps=DCP_GUIDED_EPS,
        )

        # 保存结果
        save_path = OUTPUT_DIR / fname
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
        with open(METRICS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "PSNR", "SSIM", "DeltaE00"])
            writer.writerows(rows)
        print(f"\n[I-HAZE] 所有结果已写入: {METRICS_CSV}")
    else:
        print("\n[I-HAZE] 没有成功处理的样本。")


if __name__ == "__main__":
    main()
    print("\nDone! (DCP on I-HAZE)")
