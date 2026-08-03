import time
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, brier_score_loss, log_loss

class ParamparaLegacy(BaseEstimator, ClassifierMixin):

    def __init__(self, tuning_mode='fast', n_pca=None, cv_folds=5, random_state=42):
        self.tuning_mode = tuning_mode
        self.n_pca = n_pca
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.best_params_ = None
        self.best_cv_score_ = None
        self.train_time_ = None
        self.calibrated = False

    def _get_param_grid(self):
        if self.tuning_mode == 'fast':
            return {'svc__C': [1, 10], 'svc__gamma': ['scale', 0.1], 'svc__kernel': ['rbf']}
        elif self.tuning_mode == 'research':
            return {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1], 'svc__kernel': ['rbf', 'poly', 'sigmoid'], 'svc__class_weight': [None, 'balanced']}
        else:
            return {'svc__C': np.logspace(-2, 3, 6).tolist(), 'svc__gamma': ['scale', 'auto'] + np.logspace(-3, 1, 5).tolist(), 'svc__kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'svc__degree': [2, 3], 'svc__class_weight': [None, 'balanced']}

    def fit(self, X_train, y_train):
        start = time.time()
        steps = [('scaler', StandardScaler())]
        if self.n_pca is not None:
            steps.append(('pca', PCA(n_components=self.n_pca, random_state=self.random_state)))
        if self.tuning_mode == 'large':
            steps.append(('nystroem', Nystroem(random_state=self.random_state)))
            steps.append(('clf', SGDClassifier(loss='log_loss', random_state=self.random_state)))
            param_grid = {'nystroem__gamma': ['scale', 0.1, 1], 'nystroem__n_components': [50, 100], 'clf__alpha': [0.0001, 0.001, 0.01], 'clf__class_weight': [None, 'balanced']}
        else:
            steps.append(('svc', SVC(probability=True, random_state=self.random_state)))
            param_grid = self._get_param_grid()
        pipeline = Pipeline(steps)
        min_class_count = np.min(np.unique(y_train, return_counts=True)[1])
        if min_class_count >= 2:
            from sklearn.model_selection import StratifiedKFold
            cv_strategy = StratifiedKFold(n_splits=min(self.cv_folds, min_class_count), shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv_strategy = KFold(n_splits=max(2, min(self.cv_folds, len(y_train))), shuffle=True, random_state=self.random_state)

        if self.tuning_mode in ['full', 'large']:
            search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=20, cv=cv_strategy, scoring='accuracy', n_jobs=None, random_state=self.random_state, refit=True)
        else:
            search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=None, refit=True)
        search.fit(X_train, y_train)
        self.best_params_ = search.best_params_
        self.best_cv_score_ = search.best_score_
        best_estimator = search.best_estimator_
        if self.tuning_mode in ['research', 'full', 'large']:
            self.model = CalibratedClassifierCV(best_estimator, method='sigmoid', cv='prefit')
            self.model.fit(X_train, y_train)
            self.calibrated = True
        else:
            self.model = best_estimator
        self.train_time_ = time.time() - start
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

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
        return {'model': 'ParamparaLegacy', 'accuracy': accuracy_score(y_test, y_pred), 'f1_macro': f1_score(y_test, y_pred, average='macro'), 'auc_roc': auc, 'brier_score': brier, 'log_loss': ll, 'train_time': self.train_time_, 'predict_time': predict_time, 'best_params': self.best_params_, 'best_cv_score': self.best_cv_score_, 'calibrated': self.calibrated, 'classification_report': classification_report(y_test, y_pred, output_dict=True)}
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.loader import load_dataset
    data = load_dataset('iris', binary_classes=(0, 1))
    model = ParamparaLegacy(tuning_mode='fast')
    model.fit(data['X_train'], data['y_train'])
    results = model.evaluate(data['X_test'], data['y_test'])
    print(f"Best params: {results['best_params']}")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Brier score: {results['brier_score']:.4f}")
    print(f"Log loss:    {results['log_loss']:.4f}")
    print(f"Train time:  {results['train_time']:.2f}s")