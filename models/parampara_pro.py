import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from scipy.stats import loguniform, uniform
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, log_loss

class ParamparaPro(BaseEstimator, ClassifierMixin):

    def __init__(self, mode='fair', n_qubits=4, tuning_mode='fast'):
        self.mode = mode
        self.n_qubits = n_qubits
        self.tuning_mode = tuning_mode
        self.pipeline_ = None
        self.train_time_ = 0.0
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        start_time = time.time()
        n_samples = len(X)
        steps = [('scaler', StandardScaler())]
        if self.mode == 'fair':
            steps.append(('pca', PCA(n_components=self.n_qubits)))
        if self.mode == 'industry' and n_samples > 10000:
            clf = HistGradientBoostingClassifier(random_state=42)
            param_grid = {'classifier__learning_rate': loguniform(0.001, 1.0), 'classifier__max_iter': [100, 200, 500, 1000], 'classifier__max_leaf_nodes': [15, 31, 63, 127], 'classifier__l2_regularization': loguniform(1e-06, 10.0), 'classifier__min_samples_leaf': [1, 5, 10, 20, 50]}
        else:
            clf = SVC(kernel='rbf', probability=True, random_state=42)
            param_grid = {'classifier__C': loguniform(0.001, 1000.0), 'classifier__gamma': loguniform(0.0001, 100.0), 'classifier__class_weight': [None, 'balanced']}
        steps.append(('classifier', clf))
        base_pipeline = Pipeline(steps=steps)
        min_class_count = np.min(np.unique(y, return_counts=True)[1])
        n_splits = max(2, min(5, min_class_count))
        use_stratified = (min_class_count >= 2)
        from sklearn.model_selection import KFold

        if self.tuning_mode == 'fast':
            cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if use_stratified else KFold(n_splits=max(2, min(5, len(y))), shuffle=True, random_state=42)
            n_iter = 30
        elif self.tuning_mode == 'research':
            cv_strategy = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=42) if use_stratified else KFold(n_splits=max(2, min(5, len(y))), shuffle=True, random_state=42)
            n_iter = 100
        elif self.tuning_mode == 'champion':
            cv_strategy = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42) if use_stratified else KFold(n_splits=max(2, min(5, len(y))), shuffle=True, random_state=42)
            n_iter = 500
        else:
            cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if use_stratified else KFold(n_splits=max(2, min(5, len(y))), shuffle=True, random_state=42)
            n_iter = 30
        self.pipeline_ = RandomizedSearchCV(base_pipeline, param_distributions=param_grid, n_iter=n_iter, cv=cv_strategy, n_jobs=None, random_state=42, scoring='accuracy')
        self.pipeline_.fit(X, y)
        self.train_time_ = time.time() - start_time
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.pipeline_.best_estimator_.named_steps['classifier'], 'predict_proba'):
            return self.pipeline_.predict_proba(X)
        else:
            preds = self.pipeline_.predict(X)
            proba = np.zeros((len(X), len(self.classes_)))
            for i, p in enumerate(preds):
                idx = np.where(self.classes_ == p)[0][0]
                proba[i, idx] = 1.0
            return proba

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
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                brier = float('nan')
        except ValueError:
            auc = float('nan')
            brier = float('nan')
        try:
            ll = log_loss(y_test, y_proba)
        except:
            ll = float('nan')
        return {'model': f'ParamparaPro ({self.mode.capitalize()} Track)', 'accuracy': accuracy_score(y_test, y_pred), 'f1_macro': f1_score(y_test, y_pred, average='macro'), 'auc_roc': auc, 'brier_score': brier, 'log_loss': ll, 'train_time': self.train_time_, 'predict_time': predict_time, 'best_params': self.pipeline_.best_params_}