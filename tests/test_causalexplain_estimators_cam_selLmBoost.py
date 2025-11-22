import importlib
import sys
import types

import numpy as np


def _load_module_with_stub(monkeypatch, feature_importances):
    def fake_train(X, y, pars=None):
        return {"model": type("M", (), {"feature_importances_": np.array(feature_importances)})()}

    python_pkg = types.ModuleType("Python")
    cam_pkg = types.ModuleType("Python.CAM")
    train_pkg = types.ModuleType("Python.CAM.train_LMboost")
    train_pkg.train_LMboost = fake_train
    cam_pkg.train_LMboost = train_pkg
    python_pkg.CAM = cam_pkg

    monkeypatch.setitem(sys.modules, "Python", python_pkg)
    monkeypatch.setitem(sys.modules, "Python.CAM", cam_pkg)
    monkeypatch.setitem(sys.modules, "Python.CAM.train_LMboost", train_pkg)

    module = importlib.import_module("causalexplain.estimators.cam.selLmBoost")
    monkeypatch.setattr(module, "train_LMboost", fake_train)
    return module


def test_sel_lmboost_selects_top_features(monkeypatch):
    module = _load_module_with_stub(monkeypatch, [0.2, 0.8])

    X = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    selection = module.selLmBoost(X, pars={"atLeastThatMuchSelected": 0.1, "atMostThatManyNeighbors": 2}, k=1)

    assert len(selection) == X.shape[1]
    assert selection.count(True) == 2


def test_sel_lmboost_single_variable(monkeypatch):
    module = _load_module_with_stub(monkeypatch, [0.5])
    X = np.ones((2, 1))

    assert module.selLmBoost(X) == []
