import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASET_CONFIGS = {
    "moons": {
        "size_label": "[Micro]",
        "raw_features": 2,
        "pca_features": None,
        "n_pca": None,
        "n_qubits": 2,
        "recommended_quantum_samples": 100,
        "recommended_epochs": 5,
        "supports_full_quantum_run": True,
        "is_large": False,
        "description": "A tiny 100-sample synthetic dataset of two interleaving half-circles. Perfect for ultra-fast testing of non-linear classification.",
    },
    "wine": {
        "size_label": "[Small]",
        "raw_features": 13,
        "pca_features": 4,
        "n_pca": 4,
        "n_qubits": 4,
        "recommended_quantum_samples": 178,
        "recommended_epochs": 10,
        "supports_full_quantum_run": True,
        "is_large": False,
        "description": "The Wine dataset contains 178 samples of 13 chemical constituents across 3 wine cultivars. Fast real-world benchmark.",
    },
    "iris": {
        "size_label": "[Small]",
        "raw_features": 4,
        "pca_features": None,
        "n_pca": None,
        "n_qubits": 4,
        "recommended_quantum_samples": 150,
        "recommended_epochs": 10,
        "supports_full_quantum_run": True,
        "is_large": False,
        "description": "The famous Iris dataset is a classic multivariate dataset consisting of 150 samples across 3 species of Iris flowers.",
    },
    "breast_cancer": {
        "size_label": "[Medium]",
        "raw_features": 30,
        "pca_features": 8,
        "n_pca": 8,
        "n_qubits": 8,
        "recommended_quantum_samples": 200,
        "recommended_epochs": 15,
        "supports_full_quantum_run": True,
        "is_large": False,
        "description": "The Breast Cancer Wisconsin diagnostic dataset features 569 samples of cell nuclei measurements.",
    },
    "complexity_wall": {
        "size_label": "[Medium]",
        "raw_features": 16,
        "pca_features": 4,
        "n_pca": 4,
        "n_qubits": 4,
        "recommended_quantum_samples": 200,
        "recommended_epochs": 15,
        "supports_full_quantum_run": True,
        "is_large": False,
        "description": "A mathematically chaotic dataset intentionally generated with highly entangled, non-linear informative features.",
    },
    "digits": {
        "size_label": "[Large]",
        "raw_features": 64,
        "pca_features": 4,
        "n_pca": 4,
        "n_qubits": 4,
        "recommended_quantum_samples": 200,
        "recommended_epochs": 15,
        "supports_full_quantum_run": False,
        "is_large": True,
        "description": "The Digits dataset is a collection of 1797 small 8x8 images of hand-written digits.",
    },
    "pendigits": {
        "size_label": "[Very Large]",
        "raw_features": 16,
        "pca_features": 4,
        "n_pca": 4,
        "n_qubits": 4,
        "recommended_quantum_samples": 200,
        "recommended_epochs": 15,
        "supports_full_quantum_run": False,
        "is_large": True,
        "description": "The Pen-Based Recognition of Handwritten Digits dataset contains 10,992 samples.",
    },
}


def _load_pendigits():
    try:
        from sklearn.datasets import fetch_openml

        pendigits = fetch_openml("pendigits", version=1, as_frame=False, parser="auto")
        X = pendigits.data.astype(np.float64)
        y = pendigits.target.astype(int)
        return (X, y)
    except Exception as e:
        print(f"[WARNING] Could not fetch Pendigits from OpenML: {e}")
        print("[WARNING] Using Digits dataset as a stand-in for Pendigits.")
        data = load_digits()
        return (data.data, data.target)


from typing import Any, Dict, List, Optional, Tuple


def load_dataset(
    name: str,
    n_pca: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_range: Tuple[float, float] = (0, np.pi),
    binary_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    name = name.lower().strip()
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_CONFIGS.keys())}")
    config = DATASET_CONFIGS[name]
    if n_pca is None:
        n_pca = config["n_pca"]
    if name == "iris":
        raw = load_iris()
        X, y = (raw.data, raw.target)
        class_names = list(raw.target_names)
    elif name == "breast_cancer":
        raw = load_breast_cancer()
        X, y = (raw.data, raw.target)
        class_names = list(raw.target_names)
    elif name == "digits":
        raw = load_digits()
        X, y = (raw.data, raw.target)
        class_names = [str(i) for i in range(10)]
    elif name == "wine":
        raw = load_wine()
        X, y = (raw.data, raw.target)
        class_names = list(raw.target_names)
    elif name == "moons":
        X, y = make_moons(n_samples=100, noise=0.15, random_state=random_state)
        class_names = ["Moon 0", "Moon 1"]
    elif name == "pendigits":
        X, y = _load_pendigits()
        class_names = [str(i) for i in range(10)]
    elif name == "complexity_wall":
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=500,
            n_features=16,
            n_informative=12,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=4,
            flip_y=0.1,
            class_sep=0.5,
            random_state=random_state,
        )
        class_names = ["Class 0", "Class 1"]
    if binary_classes is not None and len(binary_classes) == 2:
        mask = np.isin(y, binary_classes)
        X = X[mask]
        y_orig = y[mask]
        y = np.where(y_orig == binary_classes[0], 0, 1)
        class_names = [str(binary_classes[0]), str(binary_classes[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler_std = StandardScaler()
    X_train = scaler_std.fit_transform(X_train)
    X_test = scaler_std.transform(X_test)
    pca_model = None
    if n_pca is not None and n_pca < X_train.shape[1]:
        pca_model = PCA(n_components=n_pca, random_state=random_state)
        X_train = pca_model.fit_transform(X_train)
        X_test = pca_model.transform(X_test)
    scaler_mm = MinMaxScaler(feature_range=scale_range)
    X_train = scaler_mm.fit_transform(X_train)
    X_test = scaler_mm.transform(X_test)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y))
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_features": n_features,
        "n_classes": n_classes,
        "class_names": class_names,
        "name": name,
        "description": config.get("description", ""),
        "raw_features": config.get("raw_features", n_features),
        "pca_features": config.get("pca_features"),
        "n_qubits": config.get("n_qubits", n_features),
        "recommended_quantum_samples": config.get("recommended_quantum_samples", 200),
        "recommended_epochs": config.get("recommended_epochs", 10),
        "supports_full_quantum_run": config.get("supports_full_quantum_run", False),
        "is_large": config.get("is_large", False),
        "preprocessing_steps": ["StandardScaler", f"PCA({n_pca})" if pca_model else "None", "MinMaxScaler(0, pi)"],
        "_scaler_std": scaler_std,
        "_scaler_mm": scaler_mm,
        "_pca": pca_model,
    }


def subsample_train(dataset: Dict[str, Any], n_samples: int, random_state: int = 42) -> Dict[str, Any]:
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    if n_samples >= len(X_train):
        return dataset
    rng = np.random.RandomState(random_state)
    classes = np.unique(y_train)
    indices = []
    samples_per_class = max(1, n_samples // len(classes))
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        n_take = min(samples_per_class, len(cls_indices))
        chosen = rng.choice(cls_indices, size=n_take, replace=False)
        indices.extend(chosen)
    remaining = n_samples - len(indices)
    if remaining > 0:
        all_indices = set(range(len(y_train)))
        available = list(all_indices - set(indices))
        extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
        indices.extend(extra)
    indices = np.array(indices[:n_samples])
    rng.shuffle(indices)
    sub = dict(dataset)
    sub["X_train"] = X_train[indices]
    sub["y_train"] = y_train[indices]
    return sub


if __name__ == "__main__":
    for ds_name in DATASET_CONFIGS:
        print(f"\nLoading {ds_name}...")
        data = load_dataset(ds_name)
        print(f"  {data['description']}")
        print(f"  X_train: {data['X_train'].shape}  X_test: {data['X_test'].shape}")
        print(f"  Features: {data['n_features']}  Classes: {data['n_classes']}")
        print(f"  Feature range: [{data['X_train'].min():.3f}, {data['X_train'].max():.3f}]")
        sub = subsample_train(data, n_samples=50)
        print(f"  Subsampled to 50: X_train: {sub['X_train'].shape}")
