import os, cv2, numpy as np

def save_image(img, out_dir, name, method):
    os.makedirs(f"{out_dir}/{method}", exist_ok=True)
    cv2.imwrite(f"{out_dir}/{method}/{name}", (img * 255).astype(np.uint8)[..., ::-1])
