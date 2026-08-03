import numpy as np
import pytest

from models.ananta_vqc import AnantaVQC
from models.parampara_pro import ParamparaPro
from models.samyoga_svm import SamyogaLegacySVM


@pytest.fixture
def mock_data():
    np.random.seed(42)
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, 10)
    return X, y


def test_parampara_pro_fit_predict(mock_data):
    X, y = mock_data
    model = ParamparaPro(mode="fair", n_qubits=4)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})


def test_ananta_vqc_fit_predict(mock_data):
    X, y = mock_data
    model = AnantaVQC(n_qubits=4, epochs=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)
    probs = model.predict_proba(X)
    assert probs.shape == (10, 2)
    assert np.all((probs >= 0) & (probs <= 1))


def test_samyoga_legacy_fit_predict(mock_data):
    X, y = mock_data
    model = SamyogaLegacySVM(n_qubits=4, epochs_pretrain=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)
