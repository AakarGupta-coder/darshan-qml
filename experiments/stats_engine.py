import os

import pandas as pd


def generate_statistics_report(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    print(f"\n{'=' * 50}")
    print(f"Statistical Significance Report for {csv_path}")
    print(f"{'=' * 50}")
    summary = (
        df.groupby(["Dataset", "Model"])
        .agg(
            Accuracy_Mean=("Accuracy", "mean"),
            Accuracy_Std=("Accuracy", "std"),
            AUC_ROC_Mean=("AUC_ROC", "mean"),
            AUC_ROC_Std=("AUC_ROC", "std"),
        )
        .reset_index()
    )
    print("\nAggregate Metrics:")
    print(summary.to_string(index=False))
    latex_table = summary.to_latex(index=False)
    latex_path = csv_path.replace(".csv", "_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {latex_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        generate_statistics_report(sys.argv[1])
    else:
        for f in os.listdir("results/metrics"):
            if f.endswith(".csv"):
                generate_statistics_report(os.path.join("results/metrics", f))
