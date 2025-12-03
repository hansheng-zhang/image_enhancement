import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

# 使用你已有的 clahe 算法
from src.algorithms.clahe import dehaze


# ---- 配置两张测试图 ----
pairs = [
    (
        "data\\raw\\ihaze\\hazy\\01_indoor_hazy.jpg",
        "data\\raw\\ihaze\\gt\\01_indoor_GT.jpg"
        
    ),
    (
        "data\\raw\\ihaze\\hazy\\02_indoor_hazy.jpg",
        "data\\raw\\ihaze\\gt\\02_indoor_GT.jpg"
    )
]

output_dir = "data/results/clahe_two_test"
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

    hazy = cv2.imread(hazy_path)
    gt   = cv2.imread(gt_path)

    print(f"\nProcessing {fname} ...")

    # 执行 CLAHE
    out = dehaze(hazy, tile_grid_size=(8,8), clip_limit=2.0)

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

print("\nDone! ")
