import numpy as np
import torch

from causalexplain.estimators.notears import nonlinear


def test_notears_mlp_forward_and_regularizers():
    torch.set_default_dtype(torch.double)
    model = nonlinear.NotearsMLP(dims=[2, 2, 1], bias=True)
    with torch.no_grad():
        model.fc1_pos.weight[:] = 0.5
        model.fc1_neg.weight[:] = 0.1
    x = torch.tensor([[1.0, 2.0], [0.5, -0.5]], dtype=torch.double)
    output = model(x)
    assert output.shape == (2, 2)
    assert model.h_func().dtype == torch.double
    assert model.l2_reg() >= 0
    assert model.fc1_l1_reg() >= 0
    adj = model.fc1_to_adj()
    assert adj.shape == (2, 2)
    assert np.all(adj >= 0)


def test_notears_sobolev_forward_and_adj():
    torch.set_default_dtype(torch.double)
    model = nonlinear.NotearsSobolev(d=2, k=2)
    x = torch.tensor([[0.1, -0.2]], dtype=torch.double)
    out = model(x)
    assert out.shape == (1, 2)
    assert model.h_func().dtype == torch.double
    assert model.l2_reg() >= 0
    assert model.fc1_l1_reg() >= 0
    adj = model.fc1_to_adj()
    assert adj.shape == (2, 2)


def test_dual_ascent_step_uses_model(monkeypatch):
    class DummyOptim:
        def __init__(self, _):
            self.steps = 0

        def step(self, closure):
            self.steps += 1
            closure()

        def zero_grad(self):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([[0.0]], dtype=torch.double))
            self.h_val = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

        def forward(self, x):
            return x * self.w

        def h_func(self):
            return self.h_val

        def l2_reg(self):
            return torch.sum(self.w ** 2)

        def fc1_l1_reg(self):
            return torch.sum(torch.abs(self.w))

    monkeypatch.setattr(nonlinear, "LBFGSBScipy", DummyOptim)

    model = DummyModel()
    X = np.zeros((2, 1))
    rho, alpha, h_new = nonlinear.dual_ascent_step(
        model, X, lambda1=0.0, lambda2=0.0, rho=1.0, alpha=0.0, h=np.inf, rho_max=10.0)

    assert rho == 1.0
    assert alpha == 0.0
    assert h_new == 0.0


def test_notears_nonlinear_stops_when_constraint_small(monkeypatch):
    calls = {"iterations": 0}

    def fake_dual_ascent(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
        calls["iterations"] += 1
        return rho, alpha, 0.0  # immediately satisfy h <= h_tol

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

        def fc1_to_adj(self):
            return np.array([[0.05, 0.0], [0.0, 0.0]])

    monkeypatch.setattr(nonlinear, "dual_ascent_step", fake_dual_ascent)

    model = DummyModel()
    X = np.zeros((1, 2))
    W_est = nonlinear.notears_nonlinear(
        model, X, lambda1=0.0, lambda2=0.0, max_iter=5, h_tol=1e-8, w_threshold=0.1)

    assert calls["iterations"] == 1
    assert np.array_equal(W_est, np.zeros((2, 2)))
