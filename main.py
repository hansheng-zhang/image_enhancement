import yaml
from src.dataset import ImageDataset
from src.framework import ImageEnhancementFramework

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    dataset = ImageDataset(config["data"]["input_dir"], config["data"].get("gt_dir"))
    framework = ImageEnhancementFramework(config)
    framework.run(dataset)
