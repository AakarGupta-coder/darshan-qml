import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from data.loader import load_dataset, subsample_train
from models.ananta_pro import AnantaPro
from models.samyoga_svm import SamyogaLegacySVM


def run_ablation(datasets=None, config=None, callbacks=None):
    if datasets is None:
        datasets = ["iris", "breast_cancer"]
    if config is None:
        config = {"epochs": 10, "samples": 100, "qubits": 4, "binary": False}
    os.makedirs("results/metrics", exist_ok=True)
    csv_path = "results/metrics/ablation_analysis.csv"
    all_results = []
    existing_df = None
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        print(f"Loaded existing results from {csv_path}")
    models_to_retrain = set()
    if existing_df is not None:
        trained_models = set()
        for ds_name in datasets:
            match = existing_df[existing_df["Dataset"] == ds_name]
            if not match.empty:
                trained_models.update(match["Model"].tolist())
        if trained_models:
            if callbacks and hasattr(callbacks[0], "progress"):
                callbacks[0].progress.stop()
                print("")
            import questionary

            from ui.theme import get_q_style

            families = {"Parampara": [], "Ananta": [], "Samyoga": []}
            for m in trained_models:
                if "Samyoga" in m:
                    families["Samyoga"].append(m)
                elif "Parampara" in m:
                    families["Parampara"].append(m)
                elif "Ananta" in m:
                    families["Ananta"].append(m)
                else:
                    families["Samyoga"].append(m)
            for fam, mods in families.items():
                if mods:
                    choices = [questionary.Choice(title=m, value=m, checked=False) for m in sorted(mods)]
                    ans = questionary.checkbox(
                        f"Existing {fam} models found. Select any to RETRAIN (Space to select, Enter to confirm):",
                        choices=choices,
                        style=get_q_style(),
                    ).ask()
                    if ans:
                        models_to_retrain.update(ans)
            if callbacks and hasattr(callbacks[0], "progress"):
                print("")
                callbacks[0].progress.start()
    for ds_name in datasets:
        print(f"\n{'=' * 50}")
        print(f"Running ABLATION Analysis on: {ds_name.upper()}")
        print(f"{'=' * 50}")
        if config.get("binary") and ds_name in ["digits", "pendigits"]:
            data = load_dataset(ds_name, binary_classes=(0, 1), n_pca=config.get("qubits", 4))
        else:
            data = load_dataset(ds_name, n_pca=config.get("qubits", 4))
        max_samples = config.get("samples", len(data["X_train"]))
        data = subsample_train(data, max_samples)
        X_train, y_train = (data["X_train"], data["y_train"])
        X_test, y_test = (data["X_test"], data["y_test"])
        n_features = data["n_features"]
        models = {
            "Samyoga Legacy (Full)": SamyogaLegacySVM(
                n_qubits=n_features,
                epochs_pretrain=config.get("epochs", 10),
                use_interactions=True,
                feature_selection=True,
                tuning_mode="fast",
                callbacks=callbacks,
            ),
            "Samyoga Legacy (No Interactions)": SamyogaLegacySVM(
                n_qubits=n_features,
                epochs_pretrain=config.get("epochs", 10),
                use_interactions=False,
                feature_selection=True,
                tuning_mode="fast",
                callbacks=callbacks,
            ),
            "Samyoga Legacy (No Selection)": SamyogaLegacySVM(
                n_qubits=n_features,
                epochs_pretrain=config.get("epochs", 10),
                use_interactions=True,
                feature_selection=False,
                tuning_mode="fast",
                callbacks=callbacks,
            ),
            "AnantaPro (Full Hybrid)": AnantaPro(
                n_qubits=n_features, ablation_mode=None, use_interactions=True, head_search="fast", callbacks=callbacks
            ),
            "AnantaPro (Classical Only)": AnantaPro(
                n_qubits=n_features, ablation_mode="classical_only", head_search="fast", callbacks=callbacks
            ),
            "AnantaPro (Quantum Only)": AnantaPro(
                n_qubits=n_features, ablation_mode="quantum_only", head_search="fast", callbacks=callbacks
            ),
            "AnantaPro (Shuffled Quantum)": AnantaPro(
                n_qubits=n_features, ablation_mode="shuffled_quantum", head_search="fast", callbacks=callbacks
            ),
            "AnantaPro (No Interactions)": AnantaPro(
                n_qubits=n_features, ablation_mode=None, use_interactions=False, head_search="fast", callbacks=callbacks
            ),
        }
        for model_name, model in models.items():
            if existing_df is not None:
                match = existing_df[(existing_df["Dataset"] == ds_name) & (existing_df["Model"] == model_name)]
                if not match.empty and model_name not in models_to_retrain:
                    print(f"  [SKIPPED] {model_name} (Already trained).")
                    all_results.append(match.iloc[-1].to_dict())
                    continue
            print(f"\nTraining {model_name}...")
            try:
                model.fit(X_train, y_train)
                res = model.evaluate(X_test, y_test)
                all_results.append(
                    {"Dataset": ds_name, "Model": model_name, "Accuracy": res["accuracy"], "AUC_ROC": res["auc_roc"]}
                )
                print(f"  Accuracy: {res['accuracy']:.4f}")
            except Exception as e:
                print(f"  [ERROR] {model_name} failed: {e}")
    run_df = pd.DataFrame(all_results)
    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, run_df], ignore_index=True)
        df = df.drop_duplicates(subset=["Dataset", "Model"], keep="last")
    else:
        df = run_df
    if not df.empty:
        df.to_csv(csv_path, index=False)
        ananta_df = df[df["Model"].str.contains("Ananta")]
        if not ananta_df.empty:
            ananta_df.to_csv("results/metrics/ananta_pro_ablation.csv", index=False)
        print(f"\nAblation results saved to {csv_path}")
    return run_df


if __name__ == "__main__":
    run_ablation()
