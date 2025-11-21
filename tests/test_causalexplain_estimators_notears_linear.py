import numpy as np
import pytest

from causalexplain.estimators.notears import linear


@pytest.mark.parametrize("loss_type", ["l2", "logistic", "poisson"])
def test_notears_linear_supported_losses(monkeypatch, loss_type):
    calls = {"minimize": 0}

    def fake_minimize(func, w_est, method=None, jac=None, bounds=None):
        calls["minimize"] += 1
        # Call func once to exercise _loss/_h branches
        func(w_est)
        result = type("Result", (), {})()
        result.x = w_est
        return result

    monkeypatch.setattr(linear.sopt, "minimize", fake_minimize)
    monkeypatch.setattr(linear.slin, "expm",
                        lambda mat: np.eye(mat.shape[0]) + mat)

    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    W_est = linear.notears_linear(
        X, lambda1=0.0, loss_type=loss_type, max_iter=2, h_tol=1e-8)

    assert W_est.shape == (2, 2)
    assert np.allclose(W_est, np.zeros_like(W_est))
    assert calls["minimize"] >= 1


def test_notears_linear_unknown_loss(monkeypatch):
    def fake_minimize(func, w_est, method=None, jac=None, bounds=None):
        func(w_est)
        return type("Res", (), {"x": w_est})

    monkeypatch.setattr(linear.sopt, "minimize", fake_minimize)
    monkeypatch.setattr(linear.slin, "expm",
                        lambda mat: np.eye(mat.shape[0]) + mat)

    X = np.array([[0.1]])
    with pytest.raises(ValueError):
        linear.notears_linear(X, lambda1=0.0, loss_type="invalid")
