import cv2, os
import numpy as np

def preprocess_dataset(in_dir, out_dir, size=(512, 512)):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = cv2.imread(os.path.join(in_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        cv2.imwrite(os.path.join(out_dir, fname), img)
