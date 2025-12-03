# src/utils.py
import os
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
