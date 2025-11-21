import numpy as np

from causalexplain.estimators.cam import selGam as sel_gam_module


def test_sel_gam_marks_significant_predictors(monkeypatch):
    monkeypatch.setattr(
        sel_gam_module,
        "train_gam",
        lambda X, y, pars=None, verbose=False: {"p_values": np.array([0.0001, 0.5, 0.9])},
    )

    X = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    selection = sel_gam_module.selGam(X, pars={"cutOffPVal": 0.001, "numBasisFcts": 5}, k=2)

    assert bool(selection[0]) is True
    assert bool(selection[1]) is False
    assert len(selection) == 3


def test_sel_gam_no_selection_for_single_feature():
    X = np.array([[1.0], [2.0]])
    assert sel_gam_module.selGam(X) == []
