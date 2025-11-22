import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.lingam import base


class DummyLiNGAM(base._BaseLiNGAM):
    def fit(self, X):
        X = np.asarray(X)
        self._causal_order = list(range(X.shape[1]))
        self._adjacency_matrix = np.eye(X.shape[1])
        return self


def test_estimate_total_effect_warns_on_reverse_order(monkeypatch):
    model = DummyLiNGAM()
    model._causal_order = [0, 2, 1]
    model._adjacency_matrix = np.array([[0, 1, 0], [0, 0, 0], [0, 0.3, 0]])

    coefs = np.array([0.7, 0.2])
    monkeypatch.setattr(base, "predict_adaptive_lasso", lambda X, predictors, target: coefs)

    X = pd.DataFrame([[1.0, 2.0, 3.0]])
    with pytest.warns(UserWarning):
        effect = model.estimate_total_effect(X, from_index=2, to_index=1)

    assert effect == pytest.approx(coefs[0])


def test_estimate_adjacency_matrix_respects_prior_knowledge(monkeypatch):
    model = DummyLiNGAM()
    model._causal_order = [0, 1, 2]
    X = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])

    def fake_predict(X_arr, predictors, target):
        return np.full(len(predictors), target + 1.0)

    monkeypatch.setattr(base, "predict_adaptive_lasso", fake_predict)

    prior = np.ones((3, 3))
    prior[1, 0] = 0  # forbid edge 0 -> 1
    model._estimate_adjacency_matrix(X, prior_knowledge=prior)

    assert model.adjacency_matrix_[1, 0] == 0
    assert model.adjacency_matrix_[2, 0] != 0
    assert model.adjacency_matrix_[2, 1] != 0


def test_get_error_independence_p_values(monkeypatch):
    class FakeHSIC:
        def fit(self, x, y):
            self.p_value = float(np.mean(x) + np.mean(y))

    monkeypatch.setattr(base, "HSIC", FakeHSIC)

    model = DummyLiNGAM()
    model._adjacency_matrix = np.zeros((2, 2))

    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    p_values = model.get_error_independence_p_values(X)

    assert p_values.shape == (2, 2)
    assert p_values[0, 1] == p_values[1, 0]
