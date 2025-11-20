import numpy as np
import pytest

import causalexplain.independence.hsic as hsic_mod
from causalexplain.independence.hsic import HSIC


def test_hsic_fit_wraps_independence_test(monkeypatch):
    called = {}

    def fake_independence_test(X, Y, conditioned_on=None, method=None):
        called["args"] = (X.copy(), Y.copy(), conditioned_on, method)
        return 0.2

    monkeypatch.setattr(hsic_mod, "independence_test", fake_independence_test)
    h = HSIC()
    result = h.fit(np.array([1, 2]), np.array([2, 3]))

    assert called["args"][2] is None and called["args"][3] == "kernel"
    assert result.independence
    assert pytest.approx(result.p_value) == 0.2
