import cv2, os, numpy as np

class ImageDataset:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]

    def __iter__(self):
        for path in self.files:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield path, img.astype(np.float32) / 255.0
