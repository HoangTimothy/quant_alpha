"""Tests for ML/DL models."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.model_factory import create_model, list_models


class TestModelFactory:
    """Tests for model factory."""

    def test_list_models(self):
        models = list_models()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "logistic" in models
        assert len(models) == 8

    def test_create_all_models(self):
        for name in list_models():
            model = create_model(name)
            assert model.name == name
            assert not model.is_fitted

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")


class TestSklearnModels:
    """Tests for sklearn-based models."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series((X["f0"] > 0).astype(int))
        return X[:150], y[:150], X[150:], y[150:]

    @pytest.mark.parametrize("model_name", ["logistic", "random_forest"])
    def test_fit_predict(self, model_name, simple_data):
        X_train, y_train, X_test, y_test = simple_data
        model = create_model(model_name)
        model.fit(X_train, y_train)
        assert model.is_fitted

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

        probs = model.predict_proba(X_test)
        assert len(probs) == len(X_test)
        assert probs.min() >= 0 and probs.max() <= 1

    @pytest.mark.parametrize("model_name", ["logistic", "random_forest"])
    def test_save_load(self, model_name, simple_data):
        X_train, y_train, X_test, _ = simple_data
        model = create_model(model_name)
        model.fit(X_train, y_train)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)
            assert path.exists()

            loaded = create_model(model_name)
            loaded.load(path)
            assert loaded.is_fitted

            preds_orig = model.predict(X_test)
            preds_loaded = loaded.predict(X_test)
            np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestTreeModels:
    """Tests for XGBoost and LightGBM."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(300, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series((X["f0"] > 0).astype(int))
        return X[:200], y[:200], X[200:250], y[200:250], X[250:], y[250:]

    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
    def test_fit_with_early_stopping(self, model_name, simple_data):
        X_train, y_train, X_val, y_val, X_test, y_test = simple_data
        params = {"n_estimators": 50, "max_depth": 3, "early_stopping_rounds": 10}
        model = create_model(model_name, params=params)
        model.fit(X_train, y_train, X_val, y_val)
        assert model.is_fitted

        probs = model.predict_proba(X_test)
        assert len(probs) == len(X_test)

    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
    def test_feature_importance(self, model_name, simple_data):
        X_train, y_train, _, _, _, _ = simple_data
        model = create_model(model_name, params={"n_estimators": 20})
        model.fit(X_train, y_train)
        imp = model.get_feature_importance()
        assert imp is not None
        assert len(imp) == 10


class TestDeepLearningModels:
    """Tests for PyTorch DL models."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series((X["f0"] > 0).astype(int))
        return X[:150], y[:150], X[150:], y[150:]

    def test_mlp_fit_predict(self, simple_data):
        X_train, y_train, X_test, y_test = simple_data
        model = create_model("mlp", params={"hidden_dims": [32, 16], "dropout": 0.1})
        model.fit(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
        assert model.is_fitted

        probs = model.predict_proba(X_test)
        assert len(probs) == len(X_test)

    def test_tcnn_fit_predict(self, simple_data):
        X_train, y_train, X_test, y_test = simple_data
        model = create_model("tcnn", params={"seq_length": 10, "channels": [16, 8], "kernel_sizes": [3, 3]})
        model.fit(X_train, y_train, epochs=3, batch_size=32)
        assert model.is_fitted

        probs = model.predict_proba(X_test)
        assert len(probs) == len(X_test)

    def test_transformer_fit_predict(self, simple_data):
        X_train, y_train, X_test, y_test = simple_data
        model = create_model("transformer", params={"seq_length": 10, "d_model": 16, "nhead": 2, "num_layers": 1})
        model.fit(X_train, y_train, epochs=3, batch_size=32)
        assert model.is_fitted

        probs = model.predict_proba(X_test)
        assert len(probs) == len(X_test)

    def test_tft_fit_predict(self, simple_data):
        X_train, y_train, X_test, y_test = simple_data
        model = create_model("tft", params={"seq_length": 10, "d_model": 16, "nhead": 2})
        model.fit(X_train, y_train, epochs=3, batch_size=32)
        assert model.is_fitted
