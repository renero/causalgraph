import types

import numpy as np
import pytest
import torch

from causalexplain.estimators.notears import lbfgsb_scipy


def test_init_requires_single_param_group():
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([2.0]))
    with pytest.raises(ValueError):
        lbfgsb_scipy.LBFGSBScipy([{"params": [p1]}, {"params": [p2]}])


def test_step_invokes_closure_and_applies_solution(monkeypatch):
    linear = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight[:] = torch.tensor([[0.5]])

    called = {}

    def fake_minimize(fun, x0, method=None, jac=None, bounds=None):
        # emulate scipy returning an updated flat parameter vector
        loss, grad = fun(x0)
        called["loss"] = loss
        called["grad"] = grad
        result = types.SimpleNamespace()
        result.x = x0 + 0.25  # pretend optimizer nudged parameters
        return result

    monkeypatch.setattr(lbfgsb_scipy.sopt, "minimize", fake_minimize)

    optimizer = lbfgsb_scipy.LBFGSBScipy(linear.parameters())

    def closure():
        optimizer.zero_grad()
        output = linear(torch.tensor([[2.0]]))
        loss = (output ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    assert pytest.approx(called["loss"]) == 1.0
    assert called["grad"].shape == (1,)
    assert np.allclose(linear.weight.detach().numpy(), np.array([[0.75]]))
