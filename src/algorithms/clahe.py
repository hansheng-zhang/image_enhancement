# src/algorithms/clahe.py
import cv2
import numpy as np

def dehaze(img_bgr, tile_grid_size=(8, 8), clip_limit=2.0, **kwargs):

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size)
    )
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    out_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return out_bgr
