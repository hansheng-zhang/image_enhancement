import os
from src.algorithms import clahe, dcp, ridcp
from src.utils import save_image

class ImageDahazingFramework:
    def __init__(self, config):
        self.config = config
        self.methods = {
            "clahe": clahe.apply_clahe,
            "dcp": dcp.apply_dcp,
            "ridcp": ridcp.apply_ridcp
        }

    def run(self, dataset):
        for path, img in dataset:
            name = os.path.basename(path)
            for method in self.config["algorithms"]:
                func = self.methods[method]
                result = func(img, **self.config.get(method, {}))
                save_image(result, self.config["data"]["output_dir"], name, method)
