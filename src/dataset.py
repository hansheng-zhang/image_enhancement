import cv2, os, numpy as np

class ImageDataset:
    def __init__(self, input_dir, gt_dir=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])

    def __iter__(self):
        for f in self.files:
            low_path = os.path.join(self.input_dir, f)
            low_img = cv2.imread(low_path)
            low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
            low_img = low_img.astype(np.float32) / 255.0

            gt_img = None
            if self.gt_dir:
                gt_path = os.path.join(self.gt_dir, f)
                if os.path.exists(gt_path):
                    gt_img = cv2.imread(gt_path)
                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                    gt_img = gt_img.astype(np.float32) / 255.0
            
            yield low_path, low_img, gt_img
