import numpy as np

from causalexplain.estimators.cam import selLm as sel_lm_module


def test_sel_lm_uses_p_values(monkeypatch):
    class FakeResult:
        pvalue = np.array([0.5, 0.0001, 0.7])

    monkeypatch.setattr(sel_lm_module.stats, "linregress", lambda X, y: FakeResult())

    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    selection = sel_lm_module.selLm(X, pars={"cut_off_p_val": 0.001}, k=1)

    assert selection == [True, False, False]


def test_sel_lm_single_variable():
    X = np.ones((2, 1))
    assert sel_lm_module.selLm(X, k=0) == []
