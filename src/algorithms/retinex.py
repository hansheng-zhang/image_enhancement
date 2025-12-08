import cv2
import numpy as np

def single_scale_retinex(img, sigma):
    """
    Single Scale Retinex (SSR) using log10.
    """
    # Use log10 to match dongb5/Retinex
    return np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

def multi_scale_retinex(img, sigma_list):
    """
    Multi-Scale Retinex (MSR)
    """
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    
    retinex = retinex / len(sigma_list)
    return retinex

def color_restoration(img, alpha, beta):
    """
    Color Restoration Function (CRF)
    """
    img_sum = np.sum(img, axis=2, keepdims=True)
    # Match dongb5/Retinex: beta * (log10(alpha * img) - log10(img_sum))
    # Note: img here is already (original + 1.0)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

def simplest_color_balance(img, low_clip, high_clip, stretch=False):
    """
    Simulate the simplestColorBalance from the reference, 
    but using numpy percentiles for efficiency and adding stretching 
    (which is standard for this algorithm, even if the reference implementation 
    might be doing just clipping or implicit stretching elsewhere).
    """
    out = np.zeros_like(img)
    for i in range(img.shape[2]):
        c = img[:, :, i]
        low_val = np.percentile(c, low_clip * 100)
        high_val = np.percentile(c, high_clip * 100)
        
        c = np.clip(c, low_val, high_val)
        
        if stretch:
            # Stretch to 0-255
            denom = high_val - low_val
            if denom == 0: denom = 1e-6
            c = (c - low_val) / denom * 255.0
        out[:, :, i] = c
        
    return out

def apply_retinex(img, sigma_list=[15, 80, 250], G=5.0, b=25.0, alpha=125.0, beta=46.0, low_clip=0.01, high_clip=0.99, stretch_color_balance=False):
    """
    MSRCR: Multi-Scale Retinex with Color Restoration
    
    Parameters aligned with dongb5/Retinex:
    G=5.0, b=25.0, alpha=125.0, beta=46.0
    """
    
    # Ensure input is float
    if img.max() <= 1.0:
        img = img * 255.0
    
    # Add 1.0 to avoid log(0) and match reference implementation logic
    img = np.float64(img) + 1.0
    
    # MSR
    img_msr = multi_scale_retinex(img, sigma_list)
    
    # Color Restoration
    img_cr = color_restoration(img, alpha, beta)
    
    # MSRCR
    # Formula: G * (MSR * CR + b)
    img_msrcr = G * (img_msr * img_cr + b)
    
    # Normalization
    # The reference implementation does:
    # 1. Min-Max normalize to 0-255
    # 2. Simplest Color Balance (clip percentiles)
    
    for i in range(img_msrcr.shape[2]):
        c = img_msrcr[:, :, i]
        c_min = np.min(c)
        c_max = np.max(c)
        
        if c_max - c_min == 0:
            c = np.zeros_like(c)
        else:
            c = (c - c_min) / (c_max - c_min) * 255.0
        img_msrcr[:, :, i] = c
        
    img_msrcr = np.clip(img_msrcr, 0, 255)
    
    # Apply Simplest Color Balance (Percentile Clipping & Stretching)
    img_msrcr = simplest_color_balance(img_msrcr, low_clip, high_clip, stretch=stretch_color_balance)
    
    img_msrcr = np.clip(img_msrcr, 0, 255).astype(np.uint8)
    
    return img_msrcr.astype(np.float32) / 255.0
