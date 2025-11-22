import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from causalexplain.estimators.cam import train_gam as train_gam_module


def test_train_gam_fallback_when_model_fit_fails(monkeypatch):
    lam_calls = []

    def fake_build(p, num_splines, lam=False):
        lam_calls.append(lam)
        return f"formula-{p}-{num_splines}-{lam}"

    class FakeGAM:
        fail_once = True

        def __init__(self, formula):
            self.formula = formula
            self.statistics_ = {"n_samples": 2, "edof": 1, "p_values": np.array([0.2, 0.4])}
            self._fit_calls = 0

        def fit(self, X, y):
            self._fit_calls += 1
            if FakeGAM.fail_once:
                FakeGAM.fail_once = False
                raise RuntimeError("force fallback")

        def predict(self, X):
            return np.ones((X.shape[0], 1))

    monkeypatch.setattr(train_gam_module, "_build_gam_formula", fake_build)
    monkeypatch.setattr(train_gam_module, "GAM", FakeGAM)

    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    result = train_gam_module.train_gam(X, y, pars={"num_basis_fcts": 2})

    assert lam_calls == [False, True]
    assert result["residuals"].shape == (2, 1)
    assert np.allclose(result["residuals"].flatten(), y - 1)


def test_build_gam_formula_stacks_terms():
    reloadable = importlib.reload(train_gam_module)
    formula = reloadable._build_gam_formula(3, 5, lam=True)

    assert hasattr(formula, "__add__") or isinstance(formula, SimpleNamespace)
