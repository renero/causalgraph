import numpy as np

from causalexplain.estimators.cam import selLasso as sel_lasso_module


def test_sel_lasso_marks_nonzero_coefficients(monkeypatch):
    monkeypatch.setattr(
        sel_lasso_module,
        "train_lasso",
        lambda X, y, pars=None: {"model": type("M", (), {"coef_": np.array([1.0, 0.0])})()},
    )

    X = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    selection = sel_lasso_module.selLasso(X, k=1)

    assert selection == [True, False, False]


def test_sel_lasso_single_variable():
    X = np.ones((2, 1))
    assert sel_lasso_module.selLasso(X, k=0) == []
