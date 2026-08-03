import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


class SamyogaShadow:
    """
    Classical Shadow of Samyoga Pro (Trainable Version).
    This model mathematically matches the parameter footprint of Samyoga Pro's
    quantum architecture using a purely classical Multi-Layer Perceptron (MLP).

    It serves as the 'Fair Track' baseline to prove whether the Quantum Overhead
    (TNQE, QMoE, etc.) provides true Advantage over an identically sized classical net.
    """

    def __init__(self, d_model=16, d_state=8, num_experts=3, callbacks=None):
        self.d_model = d_model
        self.callbacks = callbacks if callbacks is not None else []
        h1 = max(128, d_model * 16)
        h2 = max(64, d_model * 8 * num_experts)
        h3 = d_state * 8
        self.model = MLPClassifier(
            hidden_layer_sizes=(h1, h2, h3),
            activation="relu",
            solver="adam",
            learning_rate_init=0.002,
            max_iter=1000,
            early_stopping=False,
            random_state=42,
        )

    def _trigger_callback(self, event_name, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, event_name):
                getattr(cb, event_name)(**kwargs)

    def fit(self, X, y):
        start_t = time.time()
        print(f"Samyoga Shadow: Training Classical Backbone (Layer sizes: {self.model.hidden_layer_sizes})...")
        self._trigger_callback("on_stage_start", stage_name="Training Classical Backbone (MLP)")
        self.model.fit(X, y)
        self._trigger_callback("on_stage_end", stage_name="Training Classical Backbone (MLP)")
        self.loss_history_ = self.model.loss_curve_
        self.train_time_ = time.time() - start_t
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import f1_score, roc_auc_score

        start_t = time.time()
        preds = self.predict(X_test)
        predict_t = time.time() - start_t
        acc = accuracy_score(y_test, preds)
        try:
            f1 = f1_score(y_test, preds, average="macro")
        except Exception:
            f1 = float("nan")
        try:
            proba = self.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
        return {
            "model": "Samyoga Shadow",
            "accuracy": acc,
            "f1_macro": f1,
            "auc_roc": auc,
            "train_time": getattr(self, "train_time_", 0.0),
            "predict_time": predict_t,
        }
