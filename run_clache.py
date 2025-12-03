import cv2
import os
from src.algorithms.clahe import dehaze

input_dir = "data/raw/ohaze/hazy"
output_dir = "data/results/clahe_simple"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(input_dir, fname)
    img = cv2.imread(path)

    out = dehaze(img, tile_grid_size=(16,16), clip_limit=2.0)

    cv2.imwrite(os.path.join(output_dir, fname), out)

print("CLAHE is done, result is stored at：", output_dir)
