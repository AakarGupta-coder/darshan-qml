import os

import numpy as np
import pandas as pd

from data.loader import DATASET_CONFIGS, load_dataset, subsample_train
from models.ananta_pro import AnantaPro
from models.ananta_vqc import AnantaVQC
from models.parampara_pro import ParamparaPro
from models.parampara_svm import ParamparaLegacy
from models.samyoga_go import SamyogaGo
from models.samyoga_pro import SamyogaPro
from models.samyoga_shadow import SamyogaShadow
from models.samyoga_svm import SamyogaLegacySVM


def run_comparison(datasets=None, mode="full", fair_mode=False, config=None, callbacks=None):
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    if config is None:
        config = {"epochs": 10, "samples": 200, "qubits": 4, "binary": False, "seeds": [42, 43, 44]}
    os.makedirs("results/metrics", exist_ok=True)
    csv_path = "results/metrics/model_comparison.csv"
    current_run_results = []
    existing_df = None
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        print(f"Loaded existing results from {csv_path}")
    seeds = config.get("seeds", [42, 43, 44])
    models_to_retrain = set()
    if existing_df is not None:
        trained_models = set()
        for ds_name in datasets:
            for seed in seeds:
                match = existing_df[(existing_df["Dataset"] == ds_name) & (existing_df["Seed"] == seed)]
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
    complexity_wall_action = "No"
    if "complexity_wall" in datasets:
        import questionary

        from ui.theme import get_q_style

        if callbacks and hasattr(callbacks[0], "progress"):
            callbacks[0].progress.stop()
            print("")
        complexity_wall_action = questionary.select(
            "The 'complexity_wall' dataset has a known PCA bottleneck. Do you want to remove the bottleneck (Yes), keep it as is (No), or skip the dataset (Skip)?",
            choices=["Yes", "No", "Skip"],
            style=get_q_style(),
        ).ask()
        if callbacks and hasattr(callbacks[0], "progress"):
            print("")
            callbacks[0].progress.start()
    for ds_name in datasets:
        if ds_name == "complexity_wall" and complexity_wall_action == "Skip":
            print(f"Skipping {ds_name} as requested.")
            continue
        print(f"\n{'=' * 50}")
        print(f"Running comparison on: {ds_name.upper()}")
        print(f"{'=' * 50}")
        n_pca_val = config.get("qubits", 4)
        if ds_name == "complexity_wall" and complexity_wall_action == "Yes":
            n_pca_val = None
        if config.get("binary") and ds_name in ["digits", "pendigits"]:
            full_data = load_dataset(ds_name, binary_classes=(0, 1), n_pca=n_pca_val)
        else:
            full_data = load_dataset(ds_name, n_pca=n_pca_val)
        n_features = full_data["n_features"]
        for seed in seeds:
            print(f"\n--- Random Seed: {seed} ---")
            max_samples = config.get("samples", len(full_data["X_train"]))
            if len(full_data["X_train"]) > max_samples:
                print(f"Subsampling to {max_samples} for low-data/quantum-safe regime...")
                np.random.seed(seed)
                data = subsample_train(full_data, max_samples)
            else:
                data = full_data
            X_train, y_train = (data["X_train"], data["y_train"])
            X_test, y_test = (data["X_test"], data["y_test"])
            models = {}
            models["Parampara Legacy"] = (
                ParamparaLegacy(tuning_mode="fast")
                if (mode in ["fast", "low_data", "quantum_safe"] and len(full_data["X_train"]) > max_samples)
                else ParamparaLegacy(tuning_mode=mode if mode in ["fast", "full", "research"] else "fast")
            )
            p_mode = "research" if mode in ["full", "research"] else "fast"
            models["Parampara Pro"] = ParamparaPro(mode="fair", n_qubits=n_features, tuning_mode=p_mode)
            models["Parampara Pro+"] = ParamparaPro(mode="industry", n_qubits=n_features, tuning_mode=p_mode)
            models["Ananta Legacy"] = AnantaVQC(
                n_qubits=n_features, epochs=config.get("epochs", 30), callbacks=callbacks
            )
            models["Ananta Pro"] = AnantaPro(
                n_qubits=n_features,
                quantum_feature_seeds=(11, 23, 37) if p_mode == "research" else (42,),
                head_search=p_mode,
                callbacks=callbacks,
            )
            models["Samyoga Legacy"] = SamyogaLegacySVM(
                n_qubits=n_features,
                epochs_pretrain=config.get("epochs", 15),
                dataset_name=ds_name,
                callbacks=callbacks,
                random_state=seed,
                tuning_mode="fast" if mode == "fast" else "full",
            )
            if fair_mode:
                models["Samyoga Shadow"] = SamyogaShadow(
                    d_model=n_features, d_state=8, num_experts=3, callbacks=callbacks
                )
            models["Samyoga Pro"] = SamyogaPro(
                d_model=n_features, d_state=8, qubits=4, num_experts=3, callbacks=callbacks
            )
            models["Samyoga Go"] = SamyogaGo(
                d_model=n_features, d_state=8, qubits=4, num_experts=3, callbacks=callbacks
            )
            for model_name, model in models.items():
                skip = False
                if existing_df is not None:
                    match = existing_df[
                        (existing_df["Dataset"] == ds_name)
                        & (existing_df["Seed"] == seed)
                        & (existing_df["Model"] == model_name)
                    ]
                    if not match.empty and model_name not in models_to_retrain:
                        print(f"  [SKIPPED] {model_name} (Already trained).")
                        current_run_results.append(match.iloc[-1].to_dict())
                        skip = True
                if skip:
                    continue
                print(f"\nTraining {model_name}...")
                if "Samyoga Pro" in model_name:
                    print("  [NOTE: Press Ctrl+C to skip this model if training takes too long]")
                try:
                    if "Full Data" in model_name:
                        model.fit(full_data["X_train"], full_data["y_train"])
                    else:
                        model.fit(X_train, y_train)
                    res = model.evaluate(X_test, y_test)
                    record = {
                        "Dataset": ds_name,
                        "Seed": seed,
                        "Model": model_name,
                        "Accuracy": res["accuracy"],
                        "F1_Macro": res["f1_macro"],
                        "AUC_ROC": res["auc_roc"],
                        "Brier_Score": res.get("brier_score", float("nan")),
                        "Log_Loss": res.get("log_loss", float("nan")),
                        "Train_Time_s": res["train_time"],
                        "Predict_Time_s": res.get("predict_time", float("nan")),
                    }
                    current_run_results.append(record)
                    print(f"  Accuracy: {res['accuracy']:.4f}")
                    print(f"  AUC-ROC:  {res['auc_roc']:.4f}")
                    print(f"  Time:     {res['train_time']:.1f}s")
                except KeyboardInterrupt:
                    print(f"\n  [SKIPPED] User manually aborted {model_name}. No results recorded.")
                    continue
                except Exception as e:
                    print(f"  [ERROR] {model_name} failed on {ds_name} (Seed {seed}): {e}")
    run_df = pd.DataFrame(current_run_results)
    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, run_df], ignore_index=True)
        df = df.drop_duplicates(subset=["Dataset", "Seed", "Model"], keep="last")
    else:
        df = run_df
    if not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    else:
        print("\nNo results to save.")
    return run_df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    run_comparison(datasets=["iris", "breast_cancer"])
