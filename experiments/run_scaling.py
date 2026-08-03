import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from data.loader import load_dataset, subsample_train
from models.ananta_pro import AnantaPro
from models.ananta_vqc import AnantaVQC
from models.parampara_pro import ParamparaPro
from models.parampara_svm import ParamparaLegacy
from models.samyoga_go import SamyogaGo
from models.samyoga_pro import SamyogaPro
from models.samyoga_shadow import SamyogaShadow
from models.samyoga_svm import SamyogaLegacySVM


def run_scaling(datasets=None, config=None, callbacks=None):
    if datasets is None:
        datasets = ["iris", "breast_cancer"]
    if config is None:
        config = {"epochs": 10, "qubits": 4, "binary": False}
    sample_sizes = [10, 20, 50, 100, 200]
    os.makedirs("results/metrics", exist_ok=True)
    csv_path = "results/metrics/scaling_analysis.csv"
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
        print(f"Running SCALING Analysis on: {ds_name.upper()}")
        print(f"{'=' * 50}")
        if config.get("binary") and ds_name in ["digits", "pendigits"]:
            full_data = load_dataset(ds_name, binary_classes=(0, 1), n_pca=config.get("qubits", 4))
        else:
            full_data = load_dataset(ds_name, n_pca=config.get("qubits", 4))
        n_features = full_data["n_features"]
        for samples in sample_sizes:
            if samples > len(full_data["X_train"]):
                print(f"Skipping {samples} samples (dataset only has {len(full_data['X_train'])})")
                continue
            print(f"\n--- Training Size: {samples} ---")
            data = subsample_train(full_data, samples)
            X_train, y_train = (data["X_train"], data["y_train"])
            X_test, y_test = (data["X_test"], data["y_test"])
            models = {
                "Parampara Legacy": ParamparaLegacy(tuning_mode="fast"),
                "Parampara Pro": ParamparaPro(mode="fair", n_qubits=n_features, tuning_mode="fast"),
                "Parampara Pro+": ParamparaPro(mode="industry", n_qubits=n_features, tuning_mode="fast"),
                "Ananta Legacy": AnantaVQC(n_qubits=n_features, epochs=config.get("epochs", 20), callbacks=callbacks),
                "Ananta Pro": AnantaPro(
                    n_qubits=n_features, head_search="fast", quantum_feature_seeds=(11,), callbacks=callbacks
                ),
                "Samyoga Legacy": SamyogaLegacySVM(
                    n_qubits=n_features,
                    epochs_pretrain=config.get("epochs", 10),
                    dataset_name=ds_name,
                    tuning_mode="fast",
                    callbacks=callbacks,
                ),
                "Samyoga Shadow": SamyogaShadow(d_model=n_features, d_state=8, num_experts=3, callbacks=callbacks),
                "Samyoga Pro": SamyogaPro(d_model=n_features, d_state=8, qubits=4, num_experts=3, callbacks=callbacks),
                "Samyoga Go": SamyogaGo(d_model=n_features, d_state=8, qubits=4, num_experts=3, callbacks=callbacks),
            }
            for model_name, model in models.items():
                if existing_df is not None:
                    match = existing_df[
                        (existing_df["Dataset"] == ds_name)
                        & (existing_df["Model"] == model_name)
                        & (existing_df["Train_Size"] == samples)
                    ]
                    if not match.empty and model_name not in models_to_retrain:
                        print(f"  [SKIPPED] {model_name} (Already trained).")
                        all_results.append(match.iloc[-1].to_dict())
                        continue
                print(f"Training {model_name}...")
                try:
                    model.fit(X_train, y_train)
                    res = model.evaluate(X_test, y_test)
                    all_results.append(
                        {
                            "Dataset": ds_name,
                            "Model": model_name,
                            "Train_Size": samples,
                            "Accuracy": res["accuracy"],
                            "AUC_ROC": res["auc_roc"],
                        }
                    )
                    print(f"  Accuracy: {res['accuracy']:.4f}")
                except Exception as e:
                    print(f"  [ERROR] {model_name} failed: {e}")
    run_df = pd.DataFrame(all_results)
    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, run_df], ignore_index=True)
        df = df.drop_duplicates(subset=["Dataset", "Model", "Train_Size"], keep="last")
    else:
        df = run_df
    if not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"\nScaling results saved to {csv_path}")
    return run_df


if __name__ == "__main__":
    run_scaling()
