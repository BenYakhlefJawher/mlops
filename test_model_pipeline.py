"""Tests unitaires de model_pipeline.py."""

import pytest

from model_pipeline import load_model, prepare_data, save_model, train_model


def test_prepare_data_shapes():
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert x_train.shape == (8000, 9)
    assert x_test.shape == (2000, 9)
    assert len(y_train) == 8000


def test_prepare_data_missing_file():
    with pytest.raises(FileNotFoundError):
        prepare_data("inexistant.csv")


def test_save_and_load_model(tmp_path):
    x_train, x_test, y_train, _ = prepare_data("Churn_Modelling.csv")
    model = train_model(x_train, y_train, n_estimators=10)
    path = tmp_path / "model.joblib"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert (loaded.predict(x_test) == model.predict(x_test)).all()
