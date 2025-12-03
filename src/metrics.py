import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

def psnr(pred_rgb, gt_rgb):
    return peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=255)

def ssim(pred_rgb, gt_rgb):
    return structural_similarity(gt_rgb, pred_rgb,
                                 channel_axis=-1, data_range=255)

def deltaE00(pred_rgb, gt_rgb):
    pred_lab = color.rgb2lab(pred_rgb / 255.0)
    gt_lab   = color.rgb2lab(gt_rgb / 255.0)
    delta = color.deltaE_ciede2000(gt_lab, pred_lab)
    return float(delta.mean())
