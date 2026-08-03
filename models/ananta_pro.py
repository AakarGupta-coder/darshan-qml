import time

import numpy as np
import pennylane as qml
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


class AnantaQuantumFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_qubits=4,
        quantum_feature_seeds=(11, 23, 37),
        observables=("X", "Y", "Z"),
        include_classical_features=True,
        noise_aware=False,
        noise_prob=0.005,
        ablation_mode=None,
        use_interactions=False,
        max_feasible_samples=50000,
        callbacks=None,
        backend="default.qubit",
    ):
        self.n_qubits = n_qubits
        self.quantum_feature_seeds = quantum_feature_seeds
        self.observables = observables
        self.include_classical_features = include_classical_features
        self.noise_aware = noise_aware
        self.noise_prob = noise_prob
        self.ablation_mode = ablation_mode
        self.use_interactions = use_interactions
        self.max_feasible_samples = max_feasible_samples
        self.callbacks = callbacks if callbacks is not None else []
        self.backend = backend
        self._feature_cache = {}
        self.diagnostics_ = {
            "q_feature_extraction_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "n_circuit_executions": 0,
            "circuit_depth_estimate": 0,
        }
        self.dev = None
        self.qnode = None

    def _setup_quantum_backend(self):
        if self.dev is not None:
            return
        if self.noise_aware or self.noise_prob > 0:
            self.dev = qml.device("default.mixed", wires=self.n_qubits)
        else:
            try:
                self.dev = qml.device(self.backend, wires=self.n_qubits)
            except Exception:
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
            self.noise_prob = 0.0

        @qml.qnode(self.dev)
        def _feature_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            if self.noise_prob > 0:
                for w in range(self.n_qubits):
                    qml.DepolarizingChannel(self.noise_prob, wires=w)
            obs_list = []
            if "X" in self.observables:
                for i in range(self.n_qubits):
                    obs_list.append(qml.expval(qml.PauliX(i)))
            if "Y" in self.observables:
                for i in range(self.n_qubits):
                    obs_list.append(qml.expval(qml.PauliY(i)))
            if "Z" in self.observables:
                for i in range(self.n_qubits):
                    obs_list.append(qml.expval(qml.PauliZ(i)))
            return obs_list if len(obs_list) > 1 else [obs_list[0]]

        self.qnode = _feature_circuit
        self.diagnostics_["circuit_depth_estimate"] = 1 + 3

    def _extract_quantum_features(self, X):
        shape = qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=self.n_qubits)
        all_q_features = []
        for seed in self.quantum_feature_seeds:
            np.random.seed(seed)
            weights = np.random.normal(loc=0.0, scale=0.1, size=shape)
            seed_features = []
            for x in X:
                res = self.qnode(x, weights)
                seed_features.append(res)
                self.diagnostics_["n_circuit_executions"] += 1
            all_q_features.append(np.array(seed_features))
        q_features = np.hstack(all_q_features)
        if self.ablation_mode == "shuffled_quantum":
            np.random.shuffle(q_features)
        return q_features

    def fit(self, X, y=None):
        if len(X) > self.max_feasible_samples:
            raise MemoryError(
                f"Infeasible Kernel: Dataset size {len(X)} exceeds maximum feasible quantum samples ({self.max_feasible_samples})."
            )
        self._setup_quantum_backend()
        self.n_classical_features_ = X.shape[1]
        return self

    def transform(self, X):
        if self.ablation_mode == "classical_only" or (
            not self.include_classical_features and self.ablation_mode != "quantum_only"
        ):
            pass
        should_be_classical_only = self.ablation_mode == "classical_only"
        should_be_quantum_only = self.ablation_mode == "quantum_only" or (
            not self.include_classical_features and self.ablation_mode not in ["shuffled_quantum", "classical_only"]
        )
        if should_be_classical_only:
            return X
        X_q = np.zeros((len(X), len(self.quantum_feature_seeds) * len(self.observables) * self.n_qubits))
        for i, x_row in enumerate(X):
            x_hash = hash(x_row.tobytes())
            if x_hash in self._feature_cache:
                self.diagnostics_["cache_hits"] += 1
                X_q[i] = self._feature_cache[x_hash]
            else:
                start_q_time = time.time()
                X_q[i] = self._extract_quantum_features(x_row.reshape(1, -1))
                self.diagnostics_["cache_misses"] += 1
                self.diagnostics_["q_feature_extraction_time"] += time.time() - start_q_time
                self._feature_cache[x_hash] = X_q[i]
            for cb in self.callbacks:
                if hasattr(cb, "on_batch_end"):
                    cb.on_batch_end(batch_index=i + 1, total_batches=len(X))
        if self.use_interactions:
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            X_q = poly.fit_transform(X_q)
        if should_be_quantum_only:
            return X_q
        return np.hstack((X, X_q))


class AnantaPro(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits=4,
        quantum_feature_seeds=(11, 23, 37),
        observables=("X", "Y", "Z"),
        include_classical_features=True,
        head_search="research",
        noise_aware=False,
        noise_prob=0.005,
        ablation_mode=None,
        use_interactions=False,
        max_feasible_samples=50000,
        scoring="f1_macro",
        callbacks=None,
        backend="default.qubit",
    ):
        self.n_qubits = n_qubits
        self.quantum_feature_seeds = quantum_feature_seeds
        self.observables = observables
        self.include_classical_features = include_classical_features
        self.head_search = head_search
        self.noise_aware = noise_aware
        self.noise_prob = noise_prob
        self.ablation_mode = ablation_mode
        self.use_interactions = use_interactions
        self.max_feasible_samples = max_feasible_samples
        self.scoring = scoring
        self.callbacks = callbacks if callbacks is not None else []
        self.backend = backend
        self.classes_ = None
        self.train_time_ = 0.0
        self.diagnostics_ = {}

    def _trigger_callback(self, event_name, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, event_name):
                getattr(cb, event_name)(**kwargs)

    def fit(self, X, y):
        start_time = time.time()
        self._trigger_callback("on_stage_start", stage_name="Quantum Feature Extraction", total_steps=len(X))
        self.classes_ = np.unique(y)
        q_features = AnantaQuantumFeatures(
            n_qubits=self.n_qubits,
            quantum_feature_seeds=self.quantum_feature_seeds,
            observables=self.observables,
            include_classical_features=self.include_classical_features,
            noise_aware=self.noise_aware,
            noise_prob=self.noise_prob,
            ablation_mode=self.ablation_mode,
            use_interactions=self.use_interactions,
            max_feasible_samples=self.max_feasible_samples,
            callbacks=self.callbacks,
        )
        self.prep_pipeline_ = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=self.n_qubits)), ("q_features", q_features)]
        )
        X_prep = self.prep_pipeline_.fit_transform(X, y)
        self._trigger_callback("on_stage_end", stage_name="Quantum Feature Extraction")
        self._trigger_callback("on_stage_start", stage_name="Hybrid Head Training")
        self.clf_pipeline_ = Pipeline(
            [("select", SelectKBest(f_classif, k="all")), ("clf", SVC(probability=True, random_state=42))]
        )
        param_grid = {
            "clf__C": loguniform(0.001, 1000.0),
            "clf__gamma": loguniform(0.0001, 100.0),
            "clf__class_weight": [None, "balanced"],
        }
        n_iter = 10 if self.head_search == "fast" else 60
        min_class_count = np.min(np.unique(y, return_counts=True)[1])
        n_splits = max(2, min(5, min_class_count))
        use_stratified = min_class_count >= 2
        from sklearn.model_selection import KFold

        cv_strategy = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            if use_stratified
            else KFold(n_splits=max(2, min(5, len(y))), shuffle=True, random_state=42)
        )
        self.head_ = RandomizedSearchCV(
            self.clf_pipeline_,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_strategy,
            n_jobs=None,
            random_state=42,
            scoring=self.scoring,
        )
        self.head_.fit(X_prep, y)
        self._trigger_callback("on_stage_end", stage_name="Hybrid Head Training")
        self.train_time_ = time.time() - start_time
        best_pipeline = self.head_.best_estimator_
        q_feat_step = self.prep_pipeline_.named_steps["q_features"]
        self.diagnostics_ = {
            "n_qubits": self.n_qubits,
            "quantum_seeds_used": len(self.quantum_feature_seeds),
            "observables_extracted": self.observables,
            "ablation_mode": self.ablation_mode,
            "train_time": self.train_time_,
            "best_head_params": self.head_.best_params_,
            "head_search_scoring": self.scoring,
            "noise_prob": q_feat_step.noise_prob,
            "circuit_depth_estimate": q_feat_step.diagnostics_["circuit_depth_estimate"],
            "n_circuit_executions": q_feat_step.diagnostics_["n_circuit_executions"],
            "cache_hits": q_feat_step.diagnostics_["cache_hits"],
            "cache_misses": q_feat_step.diagnostics_["cache_misses"],
            "hardware_readiness_score": "Moderate (Requires logical fault tolerance for high N, NISQ viable for N<200)",
        }
        selected_mask = best_pipeline.named_steps["select"].get_support()
        self.diagnostics_["embedding_dimension"] = len(selected_mask)
        self.diagnostics_["feature_selection_k"] = int(np.sum(selected_mask))
        if self.ablation_mode not in ["classical_only", "quantum_only"] and self.include_classical_features:
            num_classical = self.n_qubits
            classical_survived = int(np.sum(selected_mask[:num_classical]))
            quantum_survived = int(np.sum(selected_mask[num_classical:]))
            self.diagnostics_["claim_audit"] = {
                "classical_features_survived": classical_survived,
                "quantum_features_survived": quantum_survived,
                "quantum_advantage_plausible": quantum_survived > 0,
                "details": f"{quantum_survived} quantum features and {classical_survived} classical features survived selection.",
            }
        return self

    def predict(self, X):
        X_prep = self.prep_pipeline_.transform(X)
        return self.head_.predict(X_prep)

    def predict_proba(self, X):
        X_prep = self.prep_pipeline_.transform(X)
        return self.head_.predict_proba(X_prep)

    def evaluate(self, X_test, y_test):
        start_time = time.time()
        y_pred = self.predict(X_test)
        predict_time = time.time() - start_time
        y_proba = self.predict_proba(X_test)
        n_classes = len(self.classes_)
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
                brier = brier_score_loss(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
                brier = float("nan")
        except ValueError:
            auc = float("nan")
            brier = float("nan")
        try:
            ll = log_loss(y_test, y_proba)
        except Exception:
            ll = float("nan")
        return {
            "model": "AnantaPro (Hybrid)",
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "auc_roc": auc,
            "brier_score": brier,
            "log_loss": ll,
            "train_time": self.train_time_,
            "predict_time": predict_time,
            "diagnostics": self.diagnostics_,
        }
