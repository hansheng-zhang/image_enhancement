import os
from src.algorithms import clahe, retinex
from src.utils import save_image
from tqdm import tqdm

class ImageEnhancementFramework:
    def __init__(self, config):
        self.config = config
        self.methods = {
            "clahe": clahe.apply_clahe,
            "retinex": retinex.apply_retinex
        }

    def run(self, dataset):
        for path, img in tqdm(dataset, desc="Processing Images"):
            name = os.path.basename(path)
            for method in self.config["algorithms"]:
                func = self.methods[method]
                result = func(img, **self.config.get(method, {}))
                save_image(result, self.config["data"]["output_dir"], name, method)
