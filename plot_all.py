# plot_all.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

COMPARE_FILES = {
    "I-HAZE": "data/results/compare_ihaze.csv",
    "O-HAZE": "data/results/compare_ohaze.csv",
}


def extract_metrics(csv_path):
    df = pd.read_csv(csv_path)
    return {
        "CLAHE": {
            "PSNR": df["CLAHE_PSNR"].mean(),
            "SSIM": df["CLAHE_SSIM"].mean(),
            "DE":   df["CLAHE_DE"].mean(),
        },
        "DCP": {
            "PSNR": df["DCP_PSNR"].mean(),
            "SSIM": df["DCP_SSIM"].mean(),
            "DE":   df["DCP_DE"].mean(),
        },
        "RIDCP": {
            "PSNR": df["RIDCP_PSNR"].mean(),
            "SSIM": df["RIDCP_SSIM"].mean(),
            "DE":   df["RIDCP_DE"].mean(),
        }
    }


def plot_single_dataset(name, metrics, out_prefix):
    methods = list(metrics.keys())
    psnr = [metrics[m]["PSNR"] for m in methods]
    ssim = [metrics[m]["SSIM"] for m in methods]
    de   = [metrics[m]["DE"]   for m in methods]

    # PSNR
    plt.figure()
    plt.bar(methods, psnr)
    plt.title(f"{name}: Average PSNR")
    plt.ylabel("PSNR (dB)")
    plt.tight_layout()
    plt.savefig(f"data/results/{out_prefix}_psnr.png")
    plt.close()

    # SSIM
    plt.figure()
    plt.bar(methods, ssim)
    plt.title(f"{name}: Average SSIM")
    plt.ylabel("SSIM")
    plt.tight_layout()
    plt.savefig(f"data/results/{out_prefix}_ssim.png")
    plt.close()

    # ΔE00
    plt.figure()
    plt.bar(methods, de)
    plt.title(f"{name}: ΔE00 (Lower is better)")
    plt.ylabel("ΔE00")
    plt.tight_layout()
    plt.savefig(f"data/results/{out_prefix}_de.png")
    plt.close()


def plot_combined(all_metrics, metric_name, out_name):
    datasets = list(all_metrics.keys())
    methods = ["CLAHE", "DCP", "RIDCP"]

    for method in methods:
        vals = [all_metrics[d][method][metric_name] for d in datasets]
        plt.plot(datasets, vals, marker="o", label=method)

    plt.title(f"{metric_name} Across Datasets")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"data/results/{out_name}.png")
    plt.close()


def main():
    all_metrics = {}

    for dataset, csv_path in COMPARE_FILES.items():
        metrics = extract_metrics(csv_path)
        all_metrics[dataset] = metrics

        prefix = dataset.lower().replace("-", "")
        plot_single_dataset(dataset, metrics, prefix)

    # 画组合图：I-HAZE + O-HAZE
    plot_combined(all_metrics, "PSNR", "all_psnr")
    plot_combined(all_metrics, "SSIM", "all_ssim")
    plot_combined(all_metrics, "DE",   "all_de")

    print("已输出 ihaze + ohaze 的全部图表到 data/results/")

if __name__ == "__main__":
    main()
