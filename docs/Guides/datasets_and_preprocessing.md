# Datasets & Preprocessing Guide

This guide documents every dataset available in the Darshan framework, the preprocessing pipeline applied to each, and instructions for adding custom datasets.

---

## 1. Dataset Registry

All datasets are defined in `data/loader.py` within the `DATASET_CONFIGS` dictionary. Each entry specifies metadata used throughout the framework.

### Registry Fields

| Field | Type | Description |
|:---|:---|:---|
| `size_label` | string | UI display tag: `[Micro]`, `[Small]`, `[Medium]`, `[Large]`, `[Very Large]` |
| `raw_features` | int | Number of features in the original dataset before any preprocessing |
| `pca_features` | int\|None | Target PCA dimensionality (None = no PCA needed) |
| `n_pca` | int\|None | PCA components to retain when loading (same as `pca_features`) |
| `n_qubits` | int | Default number of qubits for quantum models on this dataset |
| `recommended_quantum_samples` | int | Suggested max training samples for quantum-safe execution |
| `recommended_epochs` | int | Suggested VQC training epochs |
| `supports_full_quantum_run` | bool | Whether the full dataset can run through quantum models without subsampling |
| `is_large` | bool | If true, dataset is unchecked by default in sweep selections |
| `description` | string | Human-readable description displayed in the UI |

---

## 2. Dataset Catalog

### 2.1 Moons (Synthetic)

| Property | Value |
|:---|:---|
| **Source** | `sklearn.datasets.make_moons(n_samples=100, noise=0.15, random_state=42)` |
| **Raw Features** | 2 |
| **PCA Applied** | No (2D = fits quantum circuits directly) |
| **Samples** | 100 (80 train / 20 test) |
| **Classes** | 2 (Moon 0, Moon 1) |
| **Task** | Binary classification |
| **Topological Characteristics** | Two interlocking crescent-shaped manifolds. Requires non-linear decision boundaries. |
| **Benchmarking Objective** | Testing non-linear boundary separation in shallow VQCs. Fast iteration dataset. |

### 2.2 Iris (Empirical)

| Property | Value |
|:---|:---|
| **Source** | `sklearn.datasets.load_iris()` |
| **Raw Features** | 4 (sepal length, sepal width, petal length, petal width) |
| **PCA Applied** | No (4 features = 4 qubits, exact mapping) |
| **Samples** | 150 (120 train / 30 test) |
| **Classes** | 3 (setosa, versicolor, virginica) |
| **Task** | Multiclass classification |
| **Topological Characteristics** | One linearly separable class; two overlapping classes. |
| **Benchmarking Objective** | 4-qubit exact mapping without PCA information loss. Standard ML benchmark. |

### 2.3 Wine (Empirical)

| Property | Value |
|:---|:---|
| **Source** | `sklearn.datasets.load_wine()` |
| **Raw Features** | 13 (chemical constituents: alcohol, malic acid, ash, etc.) |
| **PCA Applied** | Yes → 4 components |
| **Samples** | 178 (142 train / 36 test) |
| **Classes** | 3 (cultivar 1, 2, 3) |
| **Task** | Multiclass classification |
| **Topological Characteristics** | High-variance features across 3 wine cultivars. PCA compression from 13→4 retains ~75% variance. |
| **Benchmarking Objective** | Testing quantum models under moderate PCA compression. Primary benchmark dataset. |

### 2.4 Breast Cancer (Empirical)

| Property | Value |
|:---|:---|
| **Source** | `sklearn.datasets.load_breast_cancer()` |
| **Raw Features** | 30 (digitized cell nucleus measurements: radius, texture, perimeter, etc.) |
| **PCA Applied** | Yes → 8 components |
| **Samples** | 569 (455 train / 114 test) |
| **Classes** | 2 (malignant, benign) |
| **Task** | Binary classification |
| **Topological Characteristics** | High-dimensional with many correlated features. Severe PCA compression (30→8). |
| **Benchmarking Objective** | Stress-testing PCA compression and 8-qubit quantum circuits. |

### 2.5 Complexity Wall (Synthetic)

| Property | Value |
|:---|:---|
| **Source** | `sklearn.datasets.make_classification(n_samples=500, n_features=16, n_informative=12, n_redundant=0, n_classes=2, n_clusters_per_class=4, flip_y=0.1, class_sep=0.5)` |
| **Raw Features** | 16 |
| **PCA Applied** | Yes → 4 components |
| **Samples** | 500 (400 train / 100 test) |
| **Classes** | 2 |
| **Task** | Binary classification (chaotic) |
| **Topological Characteristics** | Intentionally difficult: 12 informative features entangled across 4 clusters per class, 10% label noise, low class separation. |
| **Benchmarking Objective** | Testing whether quantum entanglement captures complex feature interactions better than classical kernels under PCA bottleneck. |

### 2.6 Digits (Empirical)

| Property | Value |
|:---|:---|
| **Source** | `sklearn.datasets.load_digits()` |
| **Raw Features** | 64 (8×8 pixel images) |
| **PCA Applied** | Yes → 4 components |
| **Samples** | 1797 (1437 train / 360 test) |
| **Classes** | 10 (digits 0–9) |
| **Task** | Multiclass classification (10 classes) |
| **Topological Characteristics** | Image-based, high-dimensional. Extreme PCA compression (64→4). |
| **Benchmarking Objective** | Stress-testing quantum models with severe dimensionality reduction and many classes. Marked as `is_large`. |

### 2.7 Pendigits (Empirical)

| Property | Value |
|:---|:---|
| **Source** | `sklearn.datasets.fetch_openml('pendigits')` (falls back to digits if network unavailable) |
| **Raw Features** | 16 |
| **PCA Applied** | Yes → 4 components |
| **Samples** | 10,992 |
| **Classes** | 10 (digits 0–9) |
| **Task** | Multiclass classification |
| **Topological Characteristics** | Large-scale handwritten digit recognition from pen-tablet input. |
| **Benchmarking Objective** | Testing classical scalability (HistGradientBoosting) and quantum feasibility limits. Marked as `is_large`. |

---

## 3. Preprocessing Pipeline

The function `load_dataset(name, n_pca=None, test_size=0.2, random_state=42, scale_range=(0, np.pi), binary_classes=None)` applies the following pipeline:

### Step 1: Data Loading
- Dataset-specific loading via scikit-learn or synthetic generation
- Optional binary class filtering via `binary_classes=(class_a, class_b)`

### Step 2: Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
- **Stratified split** ensures class proportions are maintained in both sets
- Default split ratio: 80% train / 20% test
- Fixed `random_state=42` for reproducibility

### Step 3: Z-Score Normalization
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train only
X_test = scaler.transform(X_test)  # Transform test with train statistics
```
- Centers each feature to mean=0, std=1
- **Fit only on training data** to prevent data leakage

### Step 4: PCA Compression (Conditional)
```python
if n_pca is not None and n_pca < X_train.shape[1]:
    pca = PCA(n_components=n_pca, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
```
- Applied only when `n_pca < raw_features`
- For iris (4 features, 4 qubits): no PCA needed
- For wine (13 features, 4 qubits): PCA reduces 13→4
- For breast_cancer (30 features, 8 qubits): PCA reduces 30→8

### Step 5: Angle Scaling
```python
scaler_mm = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler_mm.fit_transform(X_train)
X_test = scaler_mm.transform(X_test)
```
- Maps all features to $[0, \pi]$ range
- Required for quantum angle embedding: `RY(x_i)` expects angles in this range

### Pipeline Summary

```
Raw Data → Stratified Split (80/20) → StandardScaler → [PCA(N)] → MinMaxScaler(0, π)
```

---

## 4. Subsampling

The `subsample_train(dataset, n_samples, random_state=42)` function provides stratified subsampling for low-data regime experiments:

1. Computes `samples_per_class = n_samples // n_classes`
2. Samples that many instances from each class
3. Fills remaining quota from the pool randomly
4. Returns a modified dataset dict with reduced `X_train` and `y_train`

This is used by `/test scaling` to evaluate models at training sizes like N ∈ {10, 20, 50, 100, 200}.

---

## 5. Return Value

`load_dataset()` returns a dictionary with:

| Key | Type | Description |
|:---|:---|:---|
| `X_train` | ndarray | Training features, shape `(n_train, n_features)` |
| `X_test` | ndarray | Test features, shape `(n_test, n_features)` |
| `y_train` | ndarray | Training labels |
| `y_test` | ndarray | Test labels |
| `n_features` | int | Number of features after preprocessing |
| `n_classes` | int | Number of unique classes |
| `class_names` | list[str] | Human-readable class names |
| `name` | str | Dataset identifier |
| `description` | str | Description text from config |
| `raw_features` | int | Original feature count before PCA |
| `pca_features` | int\|None | PCA target (None if no PCA) |
| `n_qubits` | int | Recommended qubit count |
| `preprocessing_steps` | list[str] | Applied transformations as strings |
| `_scaler_std` | StandardScaler | Fitted StandardScaler (internal) |
| `_scaler_mm` | MinMaxScaler | Fitted MinMaxScaler (internal) |
| `_pca` | PCA\|None | Fitted PCA transformer (internal) |

---

## 6. Adding a New Dataset

### Step-by-step:

1. **Add config entry** in `DATASET_CONFIGS`:
```python
'my_dataset': {
    'size_label': '[Small]',
    'raw_features': 10,
    'pca_features': 4,
    'n_pca': 4,
    'n_qubits': 4,
    'recommended_quantum_samples': 200,
    'recommended_epochs': 10,
    'supports_full_quantum_run': True,
    'is_large': False,
    'description': 'Description of my dataset.'
}
```

2. **Add loading logic** in `load_dataset()`:
```python
elif name == 'my_dataset':
    X, y = load_my_data()  # Your loading function
    class_names = ['Class A', 'Class B']
```

3. **Preprocessing is automatic:** The StandardScaler → PCA → MinMaxScaler pipeline applies based on the config values.

4. **Test:**
```powershell
python -c "from data.loader import load_dataset; d = load_dataset('my_dataset'); print(d['X_train'].shape)"
```

---

## 7. Reproducibility

| Mechanism | Implementation |
|:---|:---|
| **Train/test split** | `random_state=42` in `train_test_split()` |
| **PCA** | `random_state=42` in `PCA()` |
| **Synthetic data** | `random_state=42` in `make_moons()`, `make_classification()` |
| **Subsampling** | `RandomState(random_state)` in `subsample_train()` |
| **Multi-seed experiments** | Seeds `[42, 43, 44]` passed to models and subsamplers |

All random operations are seeded, ensuring identical results across runs on the same hardware.
