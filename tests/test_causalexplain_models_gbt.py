import numpy as np
import pandas as pd
import pytest

from causalexplain.models import gbt


def _dataframe():
    return pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})


def test_gbt_fit_predict_and_score(monkeypatch):
    monkeypatch.setattr(gbt, "ProgBar", lambda *_, **__: type("PB", (), {"start_subtask": lambda *a, **k: None, "update_subtask": lambda *a, **k: None, "remove": lambda *a, **k: None})())
    df = _dataframe()
    model = gbt.GBTRegressor(random_state=0, n_estimators=10, prog_bar=False)
    model.fit(df)

    assert "x" in model.regressor and "y" in model.regressor
    # Binary target should trigger the classifier path.
    assert model.regressor["y"].__class__.__name__ == "GradientBoostingClassifier"

    preds = model.predict(df)
    assert preds.shape[0] == len(df.columns)

    scores = model.score(df)
    assert scores.shape[0] == len(df.columns)
    assert np.all(scores <= 1.0)


def test_gbt_predict_requires_fit():
    model = gbt.GBTRegressor()
    with pytest.raises(AttributeError):
        model.predict(_dataframe())


def test_gbt_correlation_filter(monkeypatch):
    class FakeHierarchies:
        @staticmethod
        def compute_correlation_matrix(X):
            return X

        @staticmethod
        def compute_correlated_features(matrix, _th, feature_names, verbose=False):
            return {name: [] for name in feature_names}

    monkeypatch.setattr(gbt, "Hierarchies", FakeHierarchies)
    monkeypatch.setattr(gbt, "ProgBar", lambda *_, **__: None)

    df = _dataframe()
    model = gbt.GBTRegressor(correlation_th=0.5, prog_bar=False, n_estimators=5)
    model.fit(df)

    # Every model should have been trained on a single predictor after dropping correlations.
    assert all(reg.n_features_in_ == 1 for reg in model.regressor.values())
