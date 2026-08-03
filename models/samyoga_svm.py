import hashlib
import json
import os
import time

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SamyogaLegacySVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits,
        n_layers=3,
        learning_rate=0.05,
        epochs_pretrain=15,
        batch_size=32,
        noise_prob=0.0,
        cv_folds=5,
        random_state=42,
        max_train_samples=None,
        dataset_name="unknown",
        callbacks=None,
        use_interactions=True,
        feature_selection=True,
        calibration=True,
        tuning_mode="fast",
        warm_start=False,
        n_quantum_seeds=1,
        use_contrastive_loss=True,
        use_pca_hybrid=False,
        backend="default.qubit",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs_pretrain = epochs_pretrain
        self.batch_size = batch_size
        self.noise_prob = noise_prob
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.dataset_name = dataset_name
        self.callbacks = callbacks or []
        self.use_interactions = use_interactions
        self.feature_selection = feature_selection
        self.calibration = calibration
        self.tuning_mode = tuning_mode
        self.warm_start = warm_start
        self.n_quantum_seeds = n_quantum_seeds
        self.use_contrastive_loss = use_contrastive_loss
        self.use_pca_hybrid = use_pca_hybrid
        self.backend = backend
        np.random.seed(self.random_state)
        if self.noise_prob > 0:
            self.dev = qml.device("default.mixed", wires=self.n_qubits)
        else:
            try:
                self.dev = qml.device(self.backend, wires=self.n_qubits)
            except Exception:
                self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="autograd", diff_method="best")
        def _qnode(inputs, weights):
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(inputs[i], wires=i)
                    qml.RY(inputs[i], wires=i)
                    qml.RZ(inputs[i], wires=i)
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(self.n_qubits))
            if self.noise_prob > 0:
                for w in range(self.n_qubits):
                    qml.DepolarizingChannel(self.noise_prob, wires=w)
            return (
                [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)]
                + [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)]
                + [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            )

        self.qnode = _qnode
        self.weights = None
        self.readout_w = None
        self.readout_b = None
        self.classes_ = None
        self._is_multiclass = False
        self.loss_history_ = []
        self.val_score_history_ = []
        self.train_score_history_ = []
        self.total_epochs_trained_ = 0
        self.training_sessions_ = 0
        self.best_weights_ = None
        self.best_epoch_ = -1
        self.selected_head_name_ = None
        self.selected_head_ = None
        self.head_scores_ = {}
        self.best_params_ = None
        self.train_time_ = None
        self.scaler = StandardScaler()
        self.selector = None
        self.pca_hybrid = None

    def _initialize_weights(self):
        weights_list = []
        rw_list = []
        rb_list = []
        n_outputs = 1 if len(self.classes_) == 2 else len(self.classes_)
        n_features = 3 * self.n_qubits
        for s in range(self.n_quantum_seeds):
            np.random.seed(self.random_state + s)
            shape = (self.n_layers,) + qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=self.n_qubits)
            w = np.random.normal(loc=0.0, scale=0.1, size=shape)
            rw = np.random.normal(loc=0.0, scale=0.1, size=(n_features, n_outputs))
            rb = np.zeros((n_outputs,))
            weights_list.append(qml.numpy.array(w, requires_grad=True))
            rw_list.append(qml.numpy.array(rw, requires_grad=True))
            rb_list.append(qml.numpy.array(rb, requires_grad=True))
        np.random.seed(self.random_state)
        return (weights_list, rw_list, rb_list)

    def _trigger_callback(self, event_name, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, event_name):
                getattr(cb, event_name)(**kwargs)

    def _forward(self, X, weights, rw, rb):
        q_out = pnp.stack([self.qnode(x, weights) for x in X])
        q_out = pnp.reshape(q_out, (len(X), -1))
        logits = pnp.dot(q_out, rw) + rb
        return (logits, q_out)

    def _cost(self, weights, rw, rb, X, y):
        logits, q_out = self._forward(X, weights, rw, rb)
        if len(self.classes_) == 2:
            probs = 1 / (1 + pnp.exp(-logits[:, 0]))
            ce_loss = -pnp.mean(y * pnp.log(probs + 1e-10) + (1 - y) * pnp.log(1 - probs + 1e-10))
        else:
            exp_logits = pnp.exp(logits - pnp.max(logits, axis=1, keepdims=True))
            probs = exp_logits / pnp.sum(exp_logits, axis=1, keepdims=True)
            y_one_hot = pnp.zeros_like(probs)
            for i in range(len(y)):
                y_one_hot[i, y[i]] = 1
            ce_loss = -pnp.mean(pnp.sum(y_one_hot * pnp.log(probs + 1e-10), axis=1))
        if getattr(self, "use_contrastive_loss", False):
            unique_classes = pnp.unique(y)
            if len(unique_classes) > 1:
                centroids = []
                for c in unique_classes:
                    mask = y == c
                    centroid = pnp.mean(q_out[mask], axis=0)
                    centroids.append(centroid)
                intra_loss = 0
                for c, centroid in zip(unique_classes, centroids):
                    mask = y == c
                    intra_loss += pnp.mean(pnp.sum((q_out[mask] - centroid) ** 2, axis=1))
                intra_loss /= len(unique_classes)
                inter_loss = 0
                pairs = 0
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        inter_loss += pnp.sum((centroids[i] - centroids[j]) ** 2)
                        pairs += 1
                inter_loss = inter_loss / pairs if pairs > 0 else 1.0
                return ce_loss + 0.1 * (intra_loss / (inter_loss + 1e-08))
        return ce_loss

    def _get_cache_path(self, X, y=None):
        config = {
            "dataset": self.dataset_name,
            "q": self.n_qubits,
            "l": self.n_layers,
            "ep": self.epochs_pretrain,
            "rs": self.random_state,
            "noise": self.noise_prob,
            "interact": self.use_interactions,
        }
        config_str = json.dumps(config, sort_keys=True)
        h = hashlib.sha256()
        h.update(config_str.encode("utf-8"))
        h.update(X.tobytes())
        if y is not None:
            h.update(y.tobytes())
        if self.weights is not None:
            for w in self.weights:
                weights_np = w.numpy() if hasattr(w, "numpy") else w
                h.update(weights_np.tobytes())
        digest = h.hexdigest()[:16]
        prefix = "train" if y is not None else "test"
        filename = f"cache_samyoga_legacy_{self.dataset_name}_{prefix}_{digest}.npy"
        os.makedirs("results/cache", exist_ok=True)
        return os.path.join("results/cache", filename)

    def _build_features(self, X):
        n_batches = int(np.ceil(len(X) / self.batch_size))
        self._trigger_callback("on_stage_start", stage_name="Extracting Quantum Features", total_steps=n_batches)
        q_features = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            batch_q_out = []
            for w in self.weights:
                q_out = np.array([self.qnode(x, w) for x in batch])
                batch_q_out.append(q_out.reshape(len(batch), -1))
            q_features.append(np.hstack(batch_q_out))
            self._trigger_callback("on_batch_end", batch_index=i // self.batch_size + 1, total_batches=n_batches)
        q_features = np.vstack(q_features)
        self._trigger_callback("on_stage_end", stage_name="Extracting Quantum Features")
        if self.use_interactions:
            n = q_features.shape[1]
            interacts = []
            for i in range(n):
                for j in range(i + 1, n):
                    interacts.append(q_features[:, i] * q_features[:, j])
            if interacts:
                q_features = np.hstack([q_features, np.column_stack(interacts)])
        return np.hstack((X, q_features))

    def _extract_quantum_features(self, X, y=None, use_cache=True):
        cache_path = self._get_cache_path(X, y)
        if use_cache and os.path.exists(cache_path):
            return np.load(cache_path)
        features = self._build_features(X)
        if use_cache:
            np.save(cache_path, features)
        return features

    def fit(self, X, y):
        start_time = time.time()
        self.classes_ = np.unique(y)
        self._trigger_callback("on_stage_start", stage_name="Quantum Feature Pre-training")
        if len(self.classes_) > 2:
            self._is_multiclass = True
            y_train = np.array([np.where(self.classes_ == label)[0][0] for label in y])
        else:
            y_train = np.where(y == self.classes_[1], 1, 0)
        X_pre = X
        y_pre = y_train
        if self.max_train_samples and len(X) > self.max_train_samples:
            indices = np.random.choice(len(X), self.max_train_samples, replace=False)
            X_pre = X[indices]
            y_pre = y_train[indices]
        if not (self.warm_start and self.weights is not None):
            self.weights, self.readout_w, self.readout_b = self._initialize_weights()
            self.total_epochs_trained_ = 0
            self.loss_history_ = []
            self.val_score_history_ = []
            self.train_score_history_ = []
        for s in range(self.n_quantum_seeds):
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)

            def cost_wrapper(w, rw, rb, X_b, y_b):
                return self._cost(w, rw, rb, X_b, y_b)

            for epoch in range(self.epochs_pretrain):
                indices = np.arange(len(X_pre))
                np.random.shuffle(indices)
                X_shuffled = X_pre[indices]
                y_shuffled = y_pre[indices]
                for i in range(0, len(X_pre), self.batch_size):
                    batch_X = X_shuffled[i : i + self.batch_size]
                    batch_y = y_shuffled[i : i + self.batch_size]
                    params = opt.step(
                        lambda w, rw, rb: cost_wrapper(w, rw, rb, batch_X, batch_y),
                        self.weights[s],
                        self.readout_w[s],
                        self.readout_b[s],
                    )
                    self.weights[s], self.readout_w[s], self.readout_b[s] = params
                epoch_loss = cost_wrapper(self.weights[s], self.readout_w[s], self.readout_b[s], X_pre, y_pre)
                self.loss_history_.append(epoch_loss)
                logits, _ = self._forward(X_pre, self.weights[s], self.readout_w[s], self.readout_b[s])
                if len(self.classes_) == 2:
                    preds = (logits[:, 0] > 0).astype(int)
                else:
                    preds = np.argmax(logits, axis=1)
                train_acc = accuracy_score(y_pre, preds)
                self.train_score_history_.append(train_acc)
                self.total_epochs_trained_ += 1
                self._trigger_callback(
                    "on_epoch_end",
                    epoch=epoch,
                    loss=epoch_loss,
                    model_name=f"Samyoga Legacy Q-Encoder (Seed {s + 1})",
                    total_epochs=self.epochs_pretrain,
                )
        self._trigger_callback("on_stage_end", stage_name="Quantum Feature Pre-training")
        self._trigger_callback("on_stage_start", stage_name="Hybrid Head Training")
        H_train = self._extract_quantum_features(X, y)
        H_train = self.scaler.fit_transform(H_train)
        if self.feature_selection:
            vt = VarianceThreshold(threshold=1e-05)
            H_train = vt.fit_transform(H_train)
            k = max(self.n_qubits, H_train.shape[1] // 2)
            self.selector = SelectKBest(mutual_info_classif, k=k)
            H_train = self.selector.fit_transform(H_train, y)
        if self.use_pca_hybrid:
            self.pca_hybrid = PCA(n_components=min(16, H_train.shape[1]), random_state=self.random_state)
            H_train = self.pca_hybrid.fit_transform(H_train)
        heads = {
            "SVM": (
                SVC(probability=True, random_state=self.random_state),
                {"C": [0.1, 1, 10], "gamma": ["scale", 0.1], "kernel": ["rbf", "linear"]},
            ),
            "LR": (LogisticRegression(max_iter=1000, random_state=self.random_state), {"C": [0.1, 1, 10]}),
            "RF": (
                RandomForestClassifier(random_state=self.random_state),
                {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
            ),
            "GB": (
                GradientBoostingClassifier(random_state=self.random_state),
                {"n_estimators": [50], "learning_rate": [0.1, 0.05]},
            ),
            "ET": (ExtraTreesClassifier(random_state=self.random_state), {"n_estimators": [50, 100]}),
            "Nystroem+SGD": (
                Pipeline(
                    [
                        ("nystroem", Nystroem(gamma=0.2, random_state=self.random_state, n_components=100)),
                        ("sgd", SGDClassifier(loss="log_loss", max_iter=1000, random_state=self.random_state)),
                    ]
                ),
                {"nystroem__gamma": [0.1, 0.2, 0.5], "sgd__alpha": [0.0001, 0.001, 0.01]},
            ),
        }
        if self.tuning_mode == "fast":
            heads = {
                "SVM": (
                    SVC(probability=True, random_state=self.random_state, kernel="rbf", gamma="scale"),
                    {"C": [1, 10]},
                )
            }
        best_score = -1
        best_head_name = None
        best_estimator = None
        min_class_count = np.min(np.unique(y, return_counts=True)[1])
        n_splits = max(2, min(self.cv_folds, min_class_count))
        use_stratified = min_class_count >= 2
        from sklearn.model_selection import KFold

        cv = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            if use_stratified
            else KFold(n_splits=max(2, min(self.cv_folds, len(y))), shuffle=True, random_state=self.random_state)
        )
        for name, (model, grid) in heads.items():
            search = GridSearchCV(model, grid, cv=cv, scoring="accuracy", n_jobs=None)
            search.fit(H_train, y)
            self.head_scores_[name] = search.best_score_
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_head_name = name
                best_estimator = search.best_estimator_
                best_params = search.best_params_
        self.selected_head_name_ = best_head_name
        self.best_params_ = best_params
        if self.calibration:
            self.selected_head_ = CalibratedClassifierCV(best_estimator, method="sigmoid", cv="prefit")
            self.selected_head_.fit(H_train, y)
        else:
            self.selected_head_ = best_estimator
        self._trigger_callback("on_stage_end", stage_name="Hybrid Head Training")
        self.train_time_ = time.time() - start_time
        self.training_sessions_ += 1
        return self

    def continue_training(self, X, y, additional_epochs):
        old_warm_start = self.warm_start
        self.warm_start = True
        self.epochs_pretrain = additional_epochs
        self.fit(X, y)
        self.warm_start = old_warm_start
        return self

    def get_training_summary(self):
        return {
            "total_epochs": self.total_epochs_trained_,
            "training_sessions": self.training_sessions_,
            "best_head": self.selected_head_name_,
            "best_epoch": self.best_epoch_,
            "quantum_seeds": self.n_quantum_seeds,
            "classes": len(self.classes_) if self.classes_ is not None else 0,
        }

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        w_list = [w.numpy() if hasattr(w, "numpy") else w for w in self.weights]
        rw_list = [rw.numpy() if hasattr(rw, "numpy") else rw for rw in self.readout_w]
        rb_list = [rb.numpy() if hasattr(rb, "numpy") else rb for rb in self.readout_b]
        np.savez(
            path,
            weights=w_list,
            readout_w=rw_list,
            readout_b=rb_list,
            classes=self.classes_,
            total_epochs_trained=self.total_epochs_trained_,
            loss_history=self.loss_history_,
            val_score_history=self.val_score_history_,
            train_score_history=self.train_score_history_,
            training_sessions=self.training_sessions_,
        )

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        data = np.load(path, allow_pickle=True)
        self.weights = [pnp.array(w, requires_grad=True) for w in data["weights"]]
        self.readout_w = [pnp.array(w, requires_grad=True) for w in data["readout_w"]]
        self.readout_b = [pnp.array(w, requires_grad=True) for w in data["readout_b"]]
        self.classes_ = data["classes"]
        self.total_epochs_trained_ = int(data["total_epochs_trained"])
        self.loss_history_ = data["loss_history"].tolist()
        self.val_score_history_ = data["val_score_history"].tolist() if "val_score_history" in data else []
        self.train_score_history_ = data["train_score_history"].tolist() if "train_score_history" in data else []
        self.training_sessions_ = int(data["training_sessions"]) if "training_sessions" in data else 1
        return self

    def _prepare_features(self, X):
        H = self._extract_quantum_features(X)
        H = self.scaler.transform(H)
        if self.feature_selection:
            if self.selector:
                H = self.selector.transform(H)
        if self.pca_hybrid:
            H = self.pca_hybrid.transform(H)
        return H

    def predict(self, X):
        H = self._prepare_features(X)
        return self.selected_head_.predict(H)

    def predict_proba(self, X):
        H = self._prepare_features(X)
        return self.selected_head_.predict_proba(H)

    def evaluate(self, X_test, y_test):
        start = time.time()
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        predict_time = time.time() - start
        n_classes = len(np.unique(y_test))
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
            "model": f"Samyoga Legacy (Hybrid Ensemble - {self.selected_head_name_})",
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "auc_roc": auc,
            "brier_score": brier,
            "log_loss": ll,
            "train_time": self.train_time_,
            "predict_time": predict_time,
            "selected_head": self.selected_head_name_,
            "best_params": self.best_params_,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "loss_history": self.loss_history_,
        }


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import load_dataset

    data = load_dataset("iris", binary_classes=(0, 1))
    model = SamyogaLegacySVM(n_qubits=4, epochs_pretrain=2, tuning_mode="full", max_train_samples=30)
    model.fit(data["X_train"], data["y_train"])
    results = model.evaluate(data["X_test"], data["y_test"])
    print(f"Selected Head: {results['selected_head']}")
    print(f"Accuracy:      {results['accuracy']:.4f}")
    print(f"AUC-ROC:       {results['auc_roc']:.4f}")
    print(f"Train time:    {results['train_time']:.2f}s")
