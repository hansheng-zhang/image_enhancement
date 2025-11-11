import yaml
from src.dataset import ImageDataset
from src.framework import ImageDahazingFramework

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    dataset = ImageDataset(config["data"]["input_dir"])
    framework = ImageDahazingFramework(config)
    framework.run(dataset)
