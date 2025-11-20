import numpy as np
import pandas as pd
import pytest

from causalexplain.independence import regressors


class DummyGPR:
    def __init__(self, *args, **kwargs):
        self.fitted = False

    def fit(self, x, y):
        self.fitted = True

    def predict(self, x):
        return np.zeros((x.shape[0],))


class DummyGAM:
    def gridsearch(self, x, y, progress=False):
        return self

    def predict(self, x):
        return np.ones_like(x).flatten()


def test_fit_and_get_residuals_branches(monkeypatch):
    monkeypatch.setattr(regressors, "gpr", lambda normalize_y: DummyGPR())
    monkeypatch.setattr(regressors, "LinearGAM", DummyGAM)
    monkeypatch.setattr(np.random, "normal", lambda loc, scale, size: np.zeros(size))

    X = np.array([0.0, 1.0, 2.0])
    Y = np.array([1.0, 2.0, 3.0])

    res_gpr = regressors.fit_and_get_residuals(X, Y, method="gpr")
    assert res_gpr.shape == (3, 1)
    assert np.allclose(res_gpr, Y.reshape(-1, 1))

    res_gam = regressors.fit_and_get_residuals(X, Y, method="gam")
    assert res_gam.shape == (3, 1)
    assert np.allclose(res_gam, X.reshape(-1, 1) - 1.0)

    with pytest.raises(ValueError):
        regressors.fit_and_get_residuals(X, Y, method="bad")


def test_run_feature_selection_uses_hsic(monkeypatch):
    class DummyHSIC:
        def fit(self, X, y):
            self.stat = np.array([[0.0, 0.5], [0.5, 1.0]])
            return self

    monkeypatch.setattr(regressors, "HSIC", DummyHSIC)
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "target": [1, 0]})
    stat = regressors.run_feature_selection(df, "target")
    assert np.allclose(stat, [[0.0, 0.5], [0.5, 1.0]])
