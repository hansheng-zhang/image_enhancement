# main.py
import argparse
from src.utils import load_config
from src.framework import DehazeRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Image Dehazing Benchmark")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    runner = DehazeRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()


