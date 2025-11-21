import numpy as np

from causalexplain.estimators.cam import selGamBoost as sel_gam_boost_module


def test_sel_gamboost_returns_mask(monkeypatch):
    monkeypatch.setattr(
        sel_gam_boost_module,
        "train_GAMboost",
        lambda X, y, pars=None: {"model": type("M", (), {"feature_importances_": np.array([0.5, 0.2])})()},
    )

    X = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    selection = sel_gam_boost_module.selGamBoost(X, pars={"atLeastThatMuchSelected": 0.1, "atMostThatManyNeighbors": 2}, k=1)

    assert len(selection) == X.shape[1]
    assert selection == [False, False, False]


def test_sel_gamboost_single_variable():
    X = np.ones((2, 1))
    assert sel_gam_boost_module.selGamBoost(X) == []
