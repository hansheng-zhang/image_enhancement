import cv2, numpy as np

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # handle uint8 or float
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # return float32 image
    return enhanced.astype(np.float32) / 255.0
