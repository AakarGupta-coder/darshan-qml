import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data.loader import load_dataset
from models.ananta_pro import AnantaPro
from models.ananta_vqc import AnantaVQC
from models.parampara_pro import ParamparaPro
from models.parampara_svm import ParamparaLegacy
from models.samyoga_go import SamyogaGo
from models.samyoga_pro import SamyogaPro
from models.samyoga_shadow import SamyogaShadow
from models.samyoga_svm import SamyogaLegacySVM

sns.set_theme(style="whitegrid")


def run_noise_study(dataset_name="iris", noise_levels=[0.0, 0.01, 0.05, 0.1], epochs_vqc=50):
    print(f"\n{'=' * 50}")
    print(f"Running Noise Study on: {dataset_name.upper()}")
    print(f"{'=' * 50}")
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    data = load_dataset(dataset_name)
    n_features = data["n_features"]
    X_train, y_train = (data["X_train"], data["y_train"])
    X_test, y_test = (data["X_test"], data["y_test"])
    all_results = []
    for p in noise_levels:
        print(f"\n--- Depolarizing Noise Probability: p = {p} ---")
        models = {
            "Parampara Legacy": ParamparaLegacy(tuning_mode="fast"),
            "Parampara Pro": ParamparaPro(mode="fair", n_qubits=n_features, tuning_mode="fast"),
            "Parampara Pro+": ParamparaPro(mode="industry", n_qubits=n_features, tuning_mode="fast"),
            "Ananta Legacy": AnantaVQC(n_qubits=n_features, epochs=epochs_vqc, noise_prob=p),
            "Ananta Pro": AnantaPro(
                n_qubits=n_features, head_search="fast", quantum_feature_seeds=(11,), noise_aware=True, noise_prob=p
            ),
            "Samyoga Legacy": SamyogaLegacySVM(
                n_qubits=n_features, epochs_pretrain=10, dataset_name=dataset_name, noise_prob=p, tuning_mode="fast"
            ),
            "Samyoga Shadow": SamyogaShadow(d_model=n_features, d_state=8, num_experts=3),
            "Samyoga Pro": SamyogaPro(d_model=n_features, d_state=8, qubits=4, num_experts=3),
            "Samyoga Go": SamyogaGo(d_model=n_features, d_state=8, qubits=4, num_experts=3),
        }
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            try:
                model.fit(X_train, y_train)
                res = model.evaluate(X_test, y_test)
                all_results.append({"Noise_Prob": p, "Model": model_name, "Accuracy": res["accuracy"]})
                print(f"  Accuracy: {res['accuracy']:.4f}")
            except Exception as e:
                print(f"  [ERROR] {model_name} failed: {e}")
    df = pd.DataFrame(all_results)
    csv_path = f"results/metrics/noise_study_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Noise_Prob", y="Accuracy", hue="Model", marker="s", linewidth=2.5, markersize=8)
    classical_rows = df[(df["Model"] == "Parampara Legacy") & (df["Noise_Prob"] == 0.0)]
    if not classical_rows.empty:
        classical_accuracy = classical_rows["Accuracy"].values[0]
        plt.axhline(y=classical_accuracy, color="black", linestyle="--", label="Classical SVM (Noiseless Baseline)")
    plt.title(f"Impact of Depolarizing Noise on Quantum Models ({dataset_name.upper()})", fontsize=14)
    plt.xlabel("Depolarizing Noise Probability (p)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.legend(title="Model Type")
    plt.tight_layout()
    fig_path = f"results/figures/noise_study_{dataset_name}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"\nPlot saved to {fig_path}")
    return df


if __name__ == "__main__":
    run_noise_study("iris", noise_levels=[0.0, 0.05], epochs_vqc=15)
