import numpy as np
import pytest

from data.loader import load_dataset, subsample_train


@pytest.mark.parametrize("dataset_name", ["moons", "iris", "wine"])
def test_load_dataset_basic(dataset_name):
    data = load_dataset(dataset_name)
    assert "X_train" in data
    assert "X_test" in data
    assert "y_train" in data
    assert "y_test" in data
    assert isinstance(data["X_train"], np.ndarray)
    assert isinstance(data["y_train"], np.ndarray)
    # Check scaling bounds
    assert data["X_train"].min() >= -1e-5
    assert data["X_train"].max() <= np.pi + 1e-5


def test_load_dataset_invalid_name():
    with pytest.raises(ValueError):
        load_dataset("invalid_dataset_name")


def test_pca_reduction():
    data = load_dataset("wine", n_pca=2)
    assert data["X_train"].shape[1] == 2
    assert data["X_test"].shape[1] == 2


def test_subsample_train():
    data = load_dataset("wine")
    n_samples_orig = data["X_train"].shape[0]
    sub_data = subsample_train(data, n_samples=10)
    assert sub_data["X_train"].shape[0] == 10
    assert sub_data["y_train"].shape[0] == 10
    # Check if more samples than available returns all
    sub_data_all = subsample_train(data, n_samples=n_samples_orig + 100)
    assert sub_data_all["X_train"].shape[0] == n_samples_orig
