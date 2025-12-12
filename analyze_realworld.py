import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


CSV_PATH = Path("data/results/real_haze/real_haze_brisque.csv")
OUT_FIG = Path("data/results/real_haze/real_haze_brisque_bar.png")


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    cols = ["hazy_brisque", "clahe_brisque", "dcp_brisque", "ridcp_brisque"]
    means = df[cols].mean()

    print("Mean BRISQUE (lower is better):")
    for k, v in means.items():
        print(f"  {k}: {v:.3f}")

    methods = ["Hazy", "CLAHE", "DCP", "RIDCP"]
    values = [
        means["hazy_brisque"],
        means["clahe_brisque"],
        means["dcp_brisque"],
        means["ridcp_brisque"],
    ]

    plt.figure()
    plt.bar(methods, values)
    plt.ylabel("Mean BRISQUE (lower is better)")
    plt.title("Real-World Hazy Images: BRISQUE Comparison")
    plt.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print(f"\nBar plot saved to: {OUT_FIG}")


if __name__ == "__main__":
    main()
