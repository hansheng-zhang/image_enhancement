from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def evaluate(original, enhanced):
    psnr = peak_signal_noise_ratio(original, enhanced, data_range=1.0)
    ssim = structural_similarity(original, enhanced, channel_axis=2)
    return {"PSNR": psnr, "SSIM": ssim}
