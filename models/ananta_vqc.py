import time
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, brier_score_loss, log_loss

class AnantaVQC(BaseEstimator, ClassifierMixin):

    def __init__(self, n_qubits, n_layers=3, learning_rate=0.01, epochs=30, batch_size=32, patience=10, min_delta=0.0001, noise_prob=0.0, random_state=42, max_train_samples=None, callbacks=None, optimizer='adam', data_reuploading=True, warm_start=False):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.noise_prob = noise_prob
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.callbacks = callbacks or []
        self.optimizer = optimizer
        self.data_reuploading = data_reuploading
        self.warm_start = warm_start
        self.training_sessions_ = 0
        np.random.seed(self.random_state)
        if self.noise_prob > 0:
            self.dev = qml.device('default.mixed', wires=self.n_qubits)
        else:
            self.dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(self.dev, interface='autograd', diff_method='best')
        def _qnode(inputs, weights):
            if self.data_reuploading:
                for layer in range(self.n_layers):
                    qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
                    qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=range(self.n_qubits))
            else:
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            if self.noise_prob > 0:
                for w in range(self.n_qubits):
                    qml.DepolarizingChannel(self.noise_prob, wires=w)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.qnode = _qnode
        self.weights = None
        self.readout_w = None
        self.readout_b = None
        self.classes_ = None
        self._is_multiclass = False
        self.train_time_ = None
        self.loss_history_ = []
        self.val_score_history_ = []
        self.total_epochs_trained_ = 0
        self.circuit_depth_ = self.n_layers * 2 if self.data_reuploading else self.n_layers + 1

    def _initialize_weights(self):
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        weights = np.random.normal(loc=0.0, scale=0.1, size=shape)
        n_outputs = 1 if len(self.classes_) == 2 else len(self.classes_)
        readout_w = np.random.normal(loc=0.0, scale=0.1, size=(self.n_qubits, n_outputs))
        readout_b = np.zeros((n_outputs,))
        return (qml.numpy.array(weights, requires_grad=True), qml.numpy.array(readout_w, requires_grad=True), qml.numpy.array(readout_b, requires_grad=True))

    def _trigger_callback(self, event_name, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, event_name):
                getattr(cb, event_name)(**kwargs)

    def _forward(self, X, weights, rw, rb):
        q_out = pnp.stack([self.qnode(x, weights) for x in X])
        logits = pnp.dot(q_out, rw) + rb
        return logits

    def _cost(self, weights, rw, rb, X, y):
        logits = self._forward(X, weights, rw, rb)
        if len(self.classes_) == 2:
            probs = 1 / (1 + pnp.exp(-logits[:, 0]))
            loss = -pnp.mean(y * pnp.log(probs + 1e-10) + (1 - y) * pnp.log(1 - probs + 1e-10))
            return loss
        else:
            exp_logits = pnp.exp(logits - pnp.max(logits, axis=1, keepdims=True))
            probs = exp_logits / pnp.sum(exp_logits, axis=1, keepdims=True)
            y_one_hot = pnp.zeros_like(probs)
            for i in range(len(y)):
                y_one_hot[i, y[i]] = 1
            loss = -pnp.mean(pnp.sum(y_one_hot * pnp.log(probs + 1e-10), axis=1))
            return loss

    def fit(self, X, y, additional_epochs=None):
        start_time = time.time()
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            self._is_multiclass = True
            y_train = np.array([np.where(self.classes_ == label)[0][0] for label in y])
        else:
            y_train = np.where(y == self.classes_[1], 1, 0)
        epochs_to_run = additional_epochs if additional_epochs is not None else self.epochs
        self._trigger_callback('on_stage_start', stage_name='Quantum VQC Training', total_steps=epochs_to_run)
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
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            self.scaler_ = StandardScaler()
            self.pca_ = PCA(n_components=self.n_qubits) if X.shape[1] > self.n_qubits else None
            X_pre = self.scaler_.fit_transform(X_pre)
            if self.pca_ is not None:
                X_pre = self.pca_.fit_transform(X_pre)
        else:
            X_pre = self.scaler_.transform(X_pre)
            if hasattr(self, 'pca_') and self.pca_ is not None:
                X_pre = self.pca_.transform(X_pre)

        self.n_parameters_ = self.weights.size + self.readout_w.size + self.readout_b.size
        if self.optimizer.lower() == 'gradientdescent':
            opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        elif self.optimizer.lower() == 'spsa':
            opt = qml.SPSAOptimizer(maxiter=self.epochs)
        else:
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        best_loss = np.inf
        patience_counter = 0

        def cost_wrapper(w, rw, rb, X_b, y_b):
            return self._cost(w, rw, rb, X_b, y_b)
        for epoch in range(epochs_to_run):
            indices = np.arange(len(X_pre))
            np.random.shuffle(indices)
            X_shuffled = X_pre[indices]
            y_shuffled = y_pre[indices]
            n_batches = int(np.ceil(len(X_pre) / self.batch_size))
            for i in range(0, len(X_pre), self.batch_size):
                batch_X = X_shuffled[i:i + self.batch_size]
                batch_y = y_shuffled[i:i + self.batch_size]
                params = opt.step(lambda w, rw, rb: cost_wrapper(w, rw, rb, batch_X, batch_y), self.weights, self.readout_w, self.readout_b)
                self.weights, self.readout_w, self.readout_b = params
                self._trigger_callback('on_batch_end', batch_index=i // self.batch_size + 1, total_batches=n_batches)
            epoch_loss = cost_wrapper(self.weights, self.readout_w, self.readout_b, X_pre, y_pre)
            self.loss_history_.append(epoch_loss)
            self.total_epochs_trained_ += 1
            self._trigger_callback('on_epoch_end', epoch=epoch, loss=epoch_loss, model_name='AnantaVQC', total_epochs=epochs_to_run)
            if epoch_loss < best_loss - self.min_delta:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break
        self._trigger_callback('on_stage_end', stage_name='Quantum VQC Training')
        self.train_time_ = time.time() - start_time
        self.training_sessions_ += 1
        return self

    def continue_training(self, X, y, additional_epochs):
        old_warm_start = self.warm_start
        self.warm_start = True
        self.fit(X, y, additional_epochs=additional_epochs)
        self.warm_start = old_warm_start
        return self

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, weights=self.weights.numpy() if hasattr(self.weights, 'numpy') else self.weights, readout_w=self.readout_w.numpy() if hasattr(self.readout_w, 'numpy') else self.readout_w, readout_b=self.readout_b.numpy() if hasattr(self.readout_b, 'numpy') else self.readout_b, classes=self.classes_, total_epochs_trained=self.total_epochs_trained_, loss_history=self.loss_history_, training_sessions=self.training_sessions_)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found at {path}')
        data = np.load(path)
        self.weights = pnp.array(data['weights'], requires_grad=True)
        self.readout_w = pnp.array(data['readout_w'], requires_grad=True)
        self.readout_b = pnp.array(data['readout_b'], requires_grad=True)
        self.classes_ = data['classes']
        self.total_epochs_trained_ = int(data['total_epochs_trained'])
        self.loss_history_ = data['loss_history'].tolist()
        self.training_sessions_ = int(data['training_sessions']) if 'training_sessions' in data else 1
        return self

    def predict_proba(self, X):
        X_pre = self.scaler_.transform(X)
        if hasattr(self, 'pca_') and self.pca_ is not None:
            X_pre = self.pca_.transform(X_pre)
        logits = self._forward(X_pre, self.weights, self.readout_w, self.readout_b).numpy()
        if len(self.classes_) == 2:
            probs = 1 / (1 + np.exp(-logits[:, 0]))
            return np.vstack((1 - probs, probs)).T
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

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
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                brier = float('nan')
        except ValueError:
            auc = float('nan')
            brier = float('nan')
        try:
            ll = log_loss(y_test, y_proba)
        except:
            ll = float('nan')
        return {'model': 'AnantaVQC', 'accuracy': accuracy_score(y_test, y_pred), 'f1_macro': f1_score(y_test, y_pred, average='macro'), 'auc_roc': auc, 'brier_score': brier, 'log_loss': ll, 'train_time': self.train_time_, 'predict_time': predict_time, 'classification_report': classification_report(y_test, y_pred, output_dict=True), 'loss_history': self.loss_history_}
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.loader import load_dataset
    data = load_dataset('iris')
    print(f"Dataset: {data['description']}")
    model = AnantaVQC(n_qubits=4, epochs=2, batch_size=10, max_train_samples=30)
    model.fit(data['X_train'], data['y_train'])
    results = model.evaluate(data['X_test'], data['y_test'])
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"F1 (macro):  {results['f1_macro']:.4f}")
    print(f"AUC-ROC:     {results['auc_roc']:.4f}")
    print(f"Train time:  {results['train_time']:.2f}s")