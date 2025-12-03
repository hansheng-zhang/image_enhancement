import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

# 使用你自己的 DCP 算法
from src.algorithms.dcp import dehaze


# ---- 配置两张测试图 ----
pairs = [
    (
        "data/raw/ihaze/hazy/01_indoor_hazy.jpg",
        "data/raw/ihaze/gt/01_indoor_GT.jpg"
    ),
    (
        "data/raw/ihaze/hazy/02_indoor_hazy.jpg",
        "data/raw/ihaze/gt/02_indoor_GT.jpg"
    ),
]

output_dir = "data/results/dcp_two_test"
os.makedirs(output_dir, exist_ok=True)


# ---- 指标函数 ----
def psnr(pred, gt):
    return peak_signal_noise_ratio(gt, pred, data_range=255)

def ssim(pred, gt):
    return structural_similarity(gt, pred, channel_axis=-1, data_range=255)

def deltaE00(pred, gt):
    pred_lab = color.rgb2lab(pred / 255.0)
    gt_lab   = color.rgb2lab(gt / 255.0)
    delta = color.deltaE_ciede2000(gt_lab, pred_lab)
    return float(delta.mean())


# ---- 主循环 ----
for hazy_path, gt_path in pairs:
    fname = os.path.basename(hazy_path)

    hazy = cv2.imread(hazy_path, cv2.IMREAD_COLOR)
    gt   = cv2.imread(gt_path, cv2.IMREAD_COLOR)

    print(f"\nProcessing {fname} ...")

    # 执行 DCP（参数与你 config.yaml 中一致）
    out = dehaze(
        hazy,
        patch_size=15,
        omega=0.95,
        t0=0.1,
        use_guided_filter=True,
        guided_radius=40,
        guided_eps=0.001
    )

    # 保存
    save_path = os.path.join(output_dir, fname)
    cv2.imwrite(save_path, out)
    print(f"Saved result to {save_path}")

    # 转 RGB，用于计算指标
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    gt_rgb  = cv2.cvtColor(gt,  cv2.COLOR_BGR2RGB)

    print(f" PSNR = {psnr(out_rgb, gt_rgb):.2f}")
    print(f" SSIM = {ssim(out_rgb, gt_rgb):.4f}")
    print(f" ΔE00 = {deltaE00(out_rgb, gt_rgb):.2f}")

print("\nDone!")
