# compare_all.py
from pathlib import Path
import pandas as pd


DATASETS = {
    "ihaze": {
        "clahe": "data/results/clahe_full_ihaze/metrics_clahe_ihaze.csv",
        "dcp":   "data/results/dcp_full_ihaze/metrics_dcp_ihaze.csv",
        "ridcp": "data/results/ridcp_full_ihaze/metrics_ridcp_ihaze.csv",
        "out_compare": "data/results/compare_ihaze.csv",
        "out_avg":     "data/results/compare_ihaze_avg.csv",
    },
    "ohaze": {
        "clahe": "data/results/clahe_full_ohaze/metrics_clahe_ohaze.csv",
        "dcp":   "data/results/dcp_full_ohaze/metrics_dcp_ohaze.csv",
        "ridcp": "data/results/ridcp_full_ohaze/metrics_ridcp_ohaze.csv",
        "out_compare": "data/results/compare_ohaze.csv",
        "out_avg":     "data/results/compare_ohaze_avg.csv",
    }
}

def rename_cols(df, prefix):
    return df.rename(columns={
        "PSNR": f"{prefix}_PSNR",
        "SSIM": f"{prefix}_SSIM",
        "DeltaE00": f"{prefix}_DE"
    })

def combine_one_dataset(name, paths):
    df_c = rename_cols(pd.read_csv(paths["clahe"]), "CLAHE")
    df_d = rename_cols(pd.read_csv(paths["dcp"]), "DCP")
    df_r = rename_cols(pd.read_csv(paths["ridcp"]), "RIDCP")

    df = df_c.merge(df_d, on="filename").merge(df_r, on="filename")
    df.to_csv(paths["out_compare"], index=False)
    print(f"[{name}] 保存对比表 → {paths['out_compare']}")

    avg = df.mean(numeric_only=True)
    avg.to_csv(paths["out_avg"], header=["value"])
    print(f"[{name}] 保存平均表 → {paths['out_avg']}")
    return avg

def main():
    summary = {}

    for name, paths in DATASETS.items():
        print(f"\n===== 处理 {name.upper()} =====")
        avg = combine_one_dataset(name, paths)
        summary[name] = avg

    # === 合并成一个总表（ihaze + ohaze）====
    df_all = pd.DataFrame(summary)
    df_all.to_csv("data/results/compare_all_summary.csv")
    print("\n已生成总表：data/results/compare_all_summary.csv")

if __name__ == "__main__":
    main()
