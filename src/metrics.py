import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from piq import brisque

def calculate_metrics(result, gt_img=None):
    """
    Calculate image quality metrics.
    
    Args:
        result: Enhanced image (numpy array, float32, [0, 1])
        gt_img: Ground truth image (numpy array, float32, [0, 1]), optional
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    metrics = {}
    
    # PSNR & SSIM (requires GT)
    if gt_img is not None:
        # Ensure same shape/type if necessary, though dataset loader should handle it
        # result and gt_img are float32 [0,1]
        metrics["psnr"] = f"{psnr_func(gt_img, result, data_range=1.0):.4f}"
        metrics["ssim"] = f"{ssim_func(gt_img, result, data_range=1.0, channel_axis=2):.4f}"
    
    # BRISQUE (No Reference)
    # Convert to torch tensor: (N, C, H, W), range [0, 1]
    try:
        tensor_img = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0).float()
        # piq expects inputs in [0, 1]
        metrics["brisque"] = f"{brisque(tensor_img, data_range=1.0).item():.4f}"
    except Exception as e:
        print(f"Error calculating PIQ metrics: {e}")
        metrics["brisque"] = "Error"
        
    return metrics
