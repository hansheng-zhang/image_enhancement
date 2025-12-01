import os
import time
import csv
from src.algorithms import clahe, retinex
from src.utils import save_image
from src.metrics import calculate_metrics
from tqdm import tqdm

class ImageEnhancementFramework:
    def __init__(self, config):
        self.config = config
        self.methods = {
            "clahe": clahe.apply_clahe,
            "retinex": retinex.apply_retinex
        }
        self.results_file = os.path.join(config["data"]["output_dir"], "metrics.csv")
        os.makedirs(config["data"]["output_dir"], exist_ok=True)
        
        # Initialize CSV with headers
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "method", "runtime_sec", "psnr", "ssim", "brisque"])

    def run(self, dataset):
        for path, low_img, gt_img in tqdm(dataset, desc="Processing Images"):
            name = os.path.basename(path)
            
            for method in self.config["algorithms"]:
                func = self.methods[method]
                
                # Measure Runtime
                start_time = time.time()
                result = func(low_img, **self.config.get(method, {}))
                end_time = time.time()
                runtime = end_time - start_time
                
                # Save Result
                save_image(result, self.config["data"]["output_dir"], name, method)
                
                # Calculate Metrics
                metrics = calculate_metrics(result, gt_img)
                
                # Log to CSV
                with open(self.results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name, 
                        method, 
                        f"{runtime:.4f}", 
                        metrics.get("psnr", "N/A"), 
                        metrics.get("ssim", "N/A"), 
                        metrics.get("brisque", "N/A")
                    ])
